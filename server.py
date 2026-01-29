from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import csv
import sys
import uuid
import pathlib
import subprocess

APP_VERSION = "0.1.6"  # HR + SpO2 precomputed support + serve UI at /

BASE_DIR = pathlib.Path(__file__).resolve().parent
WEBUI_DIR = BASE_DIR / "webui"
INDEX_HTML = WEBUI_DIR / "index.html"

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"

THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_HR_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"
THESIS_SPO2_DIR = THESIS_ROOT / "06_spo2"

WEBUI_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your UI domain if you want
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
    if INDEX_HTML.exists():
        return FileResponse(
            str(INDEX_HTML),
            media_type="text/html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    return RedirectResponse(url="/docs")


@app.get("/ui", include_in_schema=False)
def ui():
    if INDEX_HTML.exists():
        return FileResponse(
            str(INDEX_HTML),
            media_type="text/html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    return RedirectResponse(url="/docs")


@app.get("/healthz", include_in_schema=False)
def healthz():
    spo2_files = []
    if THESIS_SPO2_DIR.exists():
        spo2_files = sorted([p.name for p in THESIS_SPO2_DIR.glob("*_spo2_trend.csv")])

    return {
        "ok": True,
        "version": APP_VERSION,
        "cwd": str(pathlib.Path.cwd()),
        "__file__": str(pathlib.Path(__file__).resolve()),
        "ui_index": str(INDEX_HTML),
        "ui_exists": INDEX_HTML.exists(),
        "thesis_hr_csv": str(THESIS_HR_CSV),
        "thesis_hr_csv_exists": THESIS_HR_CSV.exists(),
        "thesis_spo2_dir": str(THESIS_SPO2_DIR),
        "thesis_spo2_dir_exists": THESIS_SPO2_DIR.exists(),

        # NEW debug
        "spo2_files_count": len(spo2_files),
        "spo2_files_sample": spo2_files[:10],
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


def _median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float(vals[mid - 1] + vals[mid]) / 2.0


def thesis_hr_lookup(stem: str):
    """
    Read HR from thesis HR summary CSV.
    Thesis CSV may store HR under 'median_bpm' (legacy).
    API OUTPUT uses ONLY 'hr_bpm'.
    """
    if not THESIS_HR_CSV.exists():
        return None

    with THESIS_HR_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row_stem = (row.get("stem") or "").strip()
            if row_stem != stem:
                continue

            hr_bpm = _to_float(
                row.get("median_bpm")
                or row.get("Median_Bpm")
                or row.get("MEDIAN_BPM")
                or row.get("hr_bpm")
            )

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


def _pick_spo2_column(fieldnames):
    if not fieldnames:
        return None
    lowered = [c.lower() for c in fieldnames]
    candidates = ["spo2", "spo2_pct", "spo2_percent", "spo2_trend", "spo2_est", "spo2_value"]
    for cand in candidates:
        for i, col in enumerate(lowered):
            if col == cand:
                return fieldnames[i]
    for i, col in enumerate(lowered):
        if "spo2" in col:
            return fieldnames[i]
    return None


def thesis_spo2_lookup(stem: str):
    """
    Read SpO2 from per-clip trend file:
      thesis_pipeline/06_spo2/{stem}_spo2_trend.csv
    Return a single summary value (median) + basic stats.
    """
    if not THESIS_SPO2_DIR.exists():
        return None

    path = THESIS_SPO2_DIR / f"{stem}_spo2_trend.csv"
    if not path.exists():
        return None

    values = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        spo2_col = _pick_spo2_column(r.fieldnames)

        if spo2_col is None:
            return {
                "stem": stem,
                "file": str(path),
                "spo2_pct": None,
                "n": 0,
                "error": "no_spo2_column_found",
                "columns": r.fieldnames,
            }

        for row in r:
            v = _to_float(row.get(spo2_col))
            if v is None:
                continue
            values.append(v)

    if not values:
        return {
            "stem": stem,
            "file": str(path),
            "spo2_pct": None,
            "n": 0,
            "error": "no_numeric_spo2_values",
        }

    med = _median(values)
    mn = float(min(values))
    mx = float(max(values))
    mean = float(sum(values)) / float(len(values))

    return {
        "stem": stem,
        "file": str(path),
        "spo2_pct": med,
        "mean_pct": mean,
        "min_pct": mn,
        "max_pct": mx,
        "n": len(values),
    }


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

    with dst_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    out = {
        "ok": 1,
        "subject": subject_id,
        "condition": condition,
        "modality": modality,
        "stem": stem,
        "hr_method": method,
        "spo2_method": method,
        "min_valid_pct": float(min_valid_pct),

        "trusted": 1,
        "hr_bpm": None,
        "thesis": None,

        "spo2_trusted": 0,
        "spo2_pct": None,
        "spo2": None,
    }

    if method == "thesis_precomputed":
        # ---- HR ----
        hr_row = thesis_hr_lookup(stem)
        if hr_row is None:
            out["trusted"] = 0
            out["error"] = "stem_not_found_in_hr_csv"
        else:
            out["thesis"] = hr_row
            vpm = hr_row.get("valid_pct_masked")
            if (vpm is not None) and (float(vpm) < float(min_valid_pct)):
                out["trusted"] = 0
                out["hr_bpm"] = None
                out["error"] = "below_min_valid_pct"
            else:
                out["hr_bpm"] = hr_row.get("hr_bpm")
                if out["hr_bpm"] is None:
                    out["trusted"] = 0
                    out["error"] = "hr_missing_in_hr_csv_row"

        # ---- SpO2 ----
        spo2_row = thesis_spo2_lookup(stem)
        if spo2_row is not None:
            out["spo2"] = spo2_row
            out["spo2_pct"] = spo2_row.get("spo2_pct")

            n = spo2_row.get("n") or 0
            s = spo2_row.get("spo2_pct")
            if (s is not None) and (n >= 5) and (70.0 <= float(s) <= 100.5):
                out["spo2_trusted"] = 1

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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_sec))
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
        out["spo2_trusted"] = 0
        out["spo2_pct"] = None

    csv_name = f"{stem}_{uuid.uuid4().hex}.csv"
    csv_path = DOWNLOAD_DIR / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "subject", "condition", "modality", "stem",
            "hr_bpm", "trusted", "hr_method",
            "spo2_pct", "spo2_trusted", "spo2_method"
        ])
        w.writerow([
            subject_id, condition, modality, stem,
            out.get("hr_bpm"), out.get("trusted"), out.get("hr_method"),
            out.get("spo2_pct"), out.get("spo2_trusted"), out.get("spo2_method")
        ])

    base = str(request.base_url).rstrip("/")
    out["saved"] = {
        "summary_metrics_csv": str(csv_path),
        "download_url": f"{base}/downloads/{csv_name}",
    }

    if int(save) == 0:
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(content=out)
