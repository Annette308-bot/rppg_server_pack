from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import csv
import json
import sys
import uuid
import pathlib
import subprocess
from typing import Dict, Any, Optional, Tuple

APP_VERSION = "0.1.2"

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"

# thesis lookup (Render-safe paths)
THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your web domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONDITIONS = {"rest", "breath", "exercise"}
ALLOWED_MODALITIES = {"face", "palm"}
ALLOWED_METHODS = {"thesis_precomputed", "fft"}

# --------- Thesis CSV cache (fast lookups) ----------
_THESIS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_THESIS_MTIME: Optional[float] = None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _load_thesis_cache() -> Dict[str, Dict[str, Any]]:
    global _THESIS_CACHE, _THESIS_MTIME

    if not THESIS_CSV.exists():
        _THESIS_CACHE = {}
        _THESIS_MTIME = None
        return _THESIS_CACHE

    mtime = THESIS_CSV.stat().st_mtime
    if _THESIS_CACHE is not None and _THESIS_MTIME == mtime:
        return _THESIS_CACHE

    cache: Dict[str, Dict[str, Any]] = {}
    with THESIS_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # headers in your file: stem, selected_imf, median_bpm, q1_bpm, q3_bpm, iqr_bpm, ..., valid_pct_masked, ...
            stem = (row.get("stem") or row.get("Stem") or "").strip()
            if not stem:
                continue

            # IMPORTANT FIX: use median_bpm as hr_bpm
            median_bpm = _safe_float(row.get("median_bpm") or row.get("Median_Bpm") or row.get("MEDIAN_BPM"))
            valid_pct = _safe_float(row.get("valid_pct_masked") or row.get("Valid_Pct_Masked") or row.get("valid_pct"))
            q1 = _safe_float(row.get("q1_bpm") or row.get("Q1_Bpm"))
            q3 = _safe_float(row.get("q3_bpm") or row.get("Q3_Bpm"))
            iqr = _safe_float(row.get("iqr_bpm") or row.get("Iqr_Bpm"))
            selected_imf = _safe_float(row.get("selected_imf") or row.get("Selected_Imf"))

            cache[stem] = {
                "stem": stem,
                "median_bpm": median_bpm,
                "valid_pct_masked": valid_pct,
                "q1_bpm": q1,
                "q3_bpm": q3,
                "iqr_bpm": iqr,
                "selected_imf": selected_imf,
            }

    _THESIS_CACHE = cache
    _THESIS_MTIME = mtime
    return cache


def _parse_last_json_line(stdout_text: str) -> dict:
    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return {"ok": 0, "error": "No JSON line found in stdout", "stdout_tail": lines[-10:]}


# ------------------- Routes -------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {
        "ok": True,
        "version": APP_VERSION,
        "thesis_root": str(THESIS_ROOT),
        "thesis_csv": str(THESIS_CSV),
        "thesis_csv_exists": THESIS_CSV.exists(),
        "cached_rows": len(_load_thesis_cache()) if THESIS_CSV.exists() else 0,
    }


@app.get("/downloads/{name}", include_in_schema=False)
def download_file(name: str):
    path = DOWNLOAD_DIR / name
    if not path.exists():
        return JSONResponse(status_code=404, content={"ok": 0, "error": "file_not_found"})
    return FileResponse(str(path), media_type="text/csv", filename=name)


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
    timeout_sec: int = Form(240),
):
    # normalize inputs
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

    # Base response
    data: Dict[str, Any] = {
        "ok": 1,
        "subject": subject_id,
        "condition": condition,
        "modality": modality,
        "stem": stem,
        "hr_method": method,
        "trusted": 1,
        "min_valid_pct": float(min_valid_pct),
    }

    # ---------- FAST PATH: thesis_precomputed (NO subprocess) ----------
    if method == "thesis_precomputed":
        if not THESIS_CSV.exists():
            return JSONResponse(
                status_code=500,
                content={
                    "ok": 0,
                    "error": "thesis_summary_missing_on_server",
                    "looked_for": str(THESIS_CSV),
                },
            )

        cache = _load_thesis_cache()
        row = cache.get(stem)

        if not row:
            data["trusted"] = 0
            data["hr_bpm"] = None
            data["error"] = "stem_not_found_in_thesis_csv"
        else:
            hr = row.get("median_bpm")
            valid_pct = row.get("valid_pct_masked")

            data["hr_bpm"] = hr
            data["thesis"] = row

            # reliability gate
            if hr is None or (valid_pct is not None and float(valid_pct) < float(min_valid_pct)):
                data["trusted"] = 0
                data["hr_bpm"] = None

        # ALWAYS write a small CSV + return a download URL
        csv_name = f"{stem}_{uuid.uuid4().hex}.csv"
        csv_path = DOWNLOAD_DIR / csv_name
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["subject", "condition", "modality", "stem", "hr_bpm", "trusted"])
            w.writerow([subject_id, condition, modality, stem, data.get("hr_bpm"), data.get("trusted")])

        base = str(request.base_url).rstrip("/")
        data["saved"] = {
            "summary_metrics_csv": str(csv_path),
            "download_url": f"{base}/downloads/{csv_name}",
        }

        return JSONResponse(content=data)

    # ---------- Other methods: save upload + run script ----------
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        ext = ".mp4"

    safe_name = f"{stem}_{uuid.uuid4().hex}{ext}"
    dst_path = UPLOAD_DIR / safe_name

    try:
        with dst_path.open("wb") as f_out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

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
                    "error": "Pipeline failed",
                    "returncode": proc.returncode,
                    "cmd": cmd,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                },
            )

        result = _parse_last_json_line(proc.stdout)
        data.update(result)

        # reliability-aware: if trusted==0 -> hr_bpm = null
        try:
            if float(data.get("trusted", 0)) == 0.0:
                data["hr_bpm"] = None
        except Exception:
            pass

        # ALWAYS write a small CSV + return a download URL
        csv_name = f"{stem}_{uuid.uuid4().hex}.csv"
        csv_path = DOWNLOAD_DIR / csv_name
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["subject", "condition", "modality", "stem", "hr_bpm", "trusted"])
            w.writerow([subject_id, condition, modality, stem, data.get("hr_bpm"), data.get("trusted")])

        base = str(request.base_url).rstrip("/")
        data["saved"] = {
            "summary_metrics_csv": str(csv_path),
            "download_url": f"{base}/downloads/{csv_name}",
        }

        # cleanup upload if save==0
        if int(save) == 0:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=data)

    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"ok": 0, "error": "processing_timeout"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": 0, "error": str(e)})
