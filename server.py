from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional, Dict, Any
import csv
import sys
import uuid
import pathlib
import subprocess

APP_VERSION = "0.1.5"  # serves UI at / and /ui + hr_bpm-only API

BASE_DIR = pathlib.Path(__file__).resolve().parent

WEBUI_DIR = BASE_DIR / "webui"
INDEX_HTML = WEBUI_DIR / "index.html"

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"

THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"

WEBUI_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for demo; later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONDITIONS = {"rest", "breath", "exercise"}
ALLOWED_MODALITIES = {"face", "palm"}
ALLOWED_METHODS = {"thesis_precomputed", "fft"}  # extend later


# ---------- UI routes ----------
@app.get("/", include_in_schema=False)
def root():
    # Serve UI if present, otherwise go to docs
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML), media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/ui", include_in_schema=False)
def ui():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML), media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {
        "ok": True,
        "version": APP_VERSION,
        "__file__": str(pathlib.Path(__file__).resolve()),
        "cwd": str(pathlib.Path().resolve()),
        "ui_path": str(INDEX_HTML),
        "ui_exists": INDEX_HTML.exists(),
        "thesis_csv": str(THESIS_CSV),
        "thesis_csv_exists": THESIS_CSV.exists(),
    }


@app.get("/downloads/{name}", include_in_schema=False)
def download_file(name: str):
    path = DOWNLOAD_DIR / name
    if not path.exists():
        return JSONResponse(status_code=404, content={"ok": 0, "error": "file_not_found"})
    return FileResponse(str(path), media_type="text/csv", filename=name)


def _to_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _to_int(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def thesis_lookup(stem: str) -> Optional[Dict[str, Any]]:
    """
    Look up a row from thesis CSV by stem.
    CSV stores HR as 'median_bpm' (legacy) but API returns ONLY 'hr_bpm'.
    """
    if not THESIS_CSV.exists():
        return None

    with THESIS_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row_stem = (row.get("stem") or "").strip()
            if row_stem != stem:
                continue

            hr_bpm = _to_float(row.get("median_bpm") or row.get("Median_Bpm") or row.get("MEDIAN_BPM"))

            selected_imf = _to_int(row.get("selected_imf"))
            valid_pct_masked = _to_float(row.get("valid_pct_masked"))
            q1_bpm = _to_float(row.get("q1_bpm"))
            q3_bpm = _to_float(row.get("q3_bpm"))
            iqr_bpm = _to_float(row.get("iqr_bpm"))

            if iqr_bpm is None and (q1_bpm is not None) and (q3_bpm is not None):
                iqr_bpm = q3_bpm - q1_bpm

            return {
                "stem": stem,
                "hr_bpm": hr_bpm,
                "valid_pct_masked": valid_pct_masked,
                "q1_bpm": q1_bpm,
                "q3_bpm": q3_bpm,
                "iqr_bpm": iqr_bpm,
                "selected_imf": selected_imf,
            }

    return None


@app.post("/upload_video")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    subject_id: str = Form(...),
    condition: str = Form(...),
    modality: str = Form(...),
    method: str = Form("thesis_precomputed"),
    min_valid_pct: float = Form(50.0),
    save: int = Form(0),
    timeout_sec: int = Form(120),
):
    subject_id = (subject_id or "S01").strip()
    condition = (condition or "rest").strip().lower()
    modality = (modality or "face").strip().lower()
    method = (method or "thesis_precomputed").strip()

    if condition not in ALLOWED_CONDITIONS:
        condition = "rest"
    if modality not in ALLOWED_MODALITIES:
        modality = "face"
    if method not in ALLOWED_METHODS:
        method = "thesis_precomputed"

    stem = f"{subject_id}_{condition}_{modality}"

    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        ext = ".mp4"

    safe_name = f"{stem}_{uuid.uuid4().hex}{ext}"
    dst_path = UPLOAD_DIR / safe_name

    # Save upload (even for thesis_precomputed, client sends it)
    with dst_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    out: Dict[str, Any] = {
        "ok": 1,
        "subject": subject_id,
        "condition": condition,
        "modality": modality,
        "stem": stem,
        "hr_method": method,
        "min_valid_pct": float(min_valid_pct),
        "trusted": 1,
        "hr_bpm": None,
        "thesis": None,
    }

    if method == "thesis_precomputed":
        thesis_row = thesis_lookup(stem)
        if thesis_row is None:
            out["trusted"] = 0
            out["error"] = "stem_not_found_in_thesis_csv"
        else:
            out["thesis"] = thesis_row
            vpm = thesis_row.get("valid_pct_masked")
            if (vpm is not None) and (float(vpm) < float(min_valid_pct)):
                out["trusted"] = 0
                out["error"] = "below_min_valid_pct"
            else:
                out["hr_bpm"] = thesis_row.get("hr_bpm")
                if out["hr_bpm"] is None:
                    out["trusted"] = 0
                    out["error"] = "hr_missing_in_thesis_csv_row"
    else:
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--video", str(dst_path),
            "--subject", subject_id,
            "--condition", condition,
            "--modality", modality,
            "--outdir", str(OUTPUT_DIR),
            "--method", method,
            "--min_valid_pct", str(min_valid_pct),
            "--save", str(int(save)),
            "--thesis_root", str(THESIS_ROOT),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_sec))
        except subprocess.TimeoutExpired:
            return JSONResponse(status_code=500, content={"ok": 0, "error": "pipeline_timeout", "timeout_sec": int(timeout_sec)})

        if proc.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": 0,
                    "error": "pipeline_failed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-3000:],
                    "stderr": proc.stderr[-3000:],
                    "cmd": cmd,
                },
            )

        out["trusted"] = 0
        out["hr_bpm"] = None

    # Always write a small CSV + download link
    csv_name = f"{stem}_{uuid.uuid4().hex}.csv"
    csv_path = DOWNLOAD_DIR / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "condition", "modality", "stem", "hr_bpm", "trusted", "method"])
        w.writerow([subject_id, condition, modality, stem, out.get("hr_bpm"), out.get("trusted"), method])

    base = str(request.base_url).rstrip("/")
    out["saved"] = {
        "summary_metrics_csv": str(csv_path),
        "download_url": f"{base}/downloads/{csv_name}",
    }

    # Cleanup upload unless you explicitly want to keep it
    if int(save) == 0:
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(content=out)
