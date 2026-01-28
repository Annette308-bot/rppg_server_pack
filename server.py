from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import csv
import json
import sys
import uuid
import pathlib
import subprocess

APP_VERSION = "0.2.0"

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
DOWNLOAD_DIR = OUTPUT_DIR / "downloads"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"

THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

# Serve saved CSVs
app.mount("/downloads", StaticFiles(directory=str(DOWNLOAD_DIR)), name="downloads")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "script_path": str(SCRIPT_PATH),
        "script_exists": SCRIPT_PATH.exists(),
    }

def _parse_last_json_line(stdout_text: str) -> dict:
    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return {"ok": 0, "error": "No JSON line found in stdout", "stdout_tail": lines[-10:]}

def _row_get_ci(row: dict, *keys: str):
    """Case-insensitive dict key lookup."""
    lower_map = {str(k).lower(): k for k in row.keys()}
    for key in keys:
        k = lower_map.get(key.lower())
        if k is not None:
            return row.get(k)
    return None

def _to_float_or_none(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def lookup_thesis_precomputed(subject_id: str, condition: str, modality: str, min_valid_pct: float):
    if not THESIS_CSV.exists():
        return {
            "ok": 0,
            "error": "thesis_summary_missing_on_server",
            "looked_for": str(THESIS_CSV),
        }

    stem = f"{subject_id}_{condition}_{modality}"

    with THESIS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Try match by 'stem' first, then by (subject, condition, modality)
    match = None
    for row in rows:
        row_stem = _row_get_ci(row, "stem")
        if row_stem and str(row_stem).strip() == stem:
            match = row
            break

    if match is None:
        for row in rows:
            r_sub = _row_get_ci(row, "subject", "subject_id")
            r_cond = _row_get_ci(row, "condition")
            r_mod = _row_get_ci(row, "modality")
            if (str(r_sub).strip() == subject_id) and (str(r_cond).strip() == condition) and (str(r_mod).strip() == modality):
                match = row
                break

    if match is None:
        return {
            "ok": 0,
            "error": "no_matching_row_in_thesis_csv",
            "stem": stem,
            "hint": "Check if your CSV has 'stem' or subject/condition/modality columns, and that values match exactly.",
        }

    hr_bpm = _to_float_or_none(_row_get_ci(match, "hr_bpm", "Hr_Bpm", "HR_BPM"))
    trusted = _to_float_or_none(_row_get_ci(match, "trusted", "Trusted"))
    valid_pct_masked = _to_float_or_none(_row_get_ci(match, "valid_pct_masked", "Valid_Pct_Masked"))
    iqr_bpm = _to_float_or_none(_row_get_ci(match, "iqr_bpm", "Iqr_Bpm"))

    # Apply your reliability rule
    if trusted is not None and float(trusted) == 0.0:
        hr_bpm = None

    # If CSV has selected_imf (optional)
    selected_imf = _to_float_or_none(_row_get_ci(match, "selected_imf", "Selected_Imf"))

    return {
        "ok": 1,
        "subject": subject_id,
        "condition": condition,
        "modality": modality,
        "file": None,
        "stem": stem,
        "hr_method": "thesis_precomputed",
        "hr_bpm": hr_bpm,
        "trusted": trusted if trusted is not None else 1,
        "min_valid_pct": float(min_valid_pct),
        "valid_pct_masked": valid_pct_masked,
        "iqr_bpm": iqr_bpm,
        "selected_imf": selected_imf if selected_imf is not None else 1,
        "saved": {"summary_metrics_csv": None, "download_url": None},
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
):
    # Save upload to disk (still useful for traceability)
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        ext = ".mp4"

    safe_name = f"{subject_id}_{condition}_{modality}_{uuid.uuid4().hex}{ext}"
    dst_path = UPLOAD_DIR / safe_name

    try:
        with dst_path.open("wb") as f_out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

        # ✅ IMPORTANT: thesis_precomputed path should NOT run the script
        if method == "thesis_precomputed":
            data = lookup_thesis_precomputed(subject_id, condition, modality, min_valid_pct)
            data["file"] = str(dst_path)

            if int(save) == 1 and data.get("ok") == 1:
                csv_name = f"summary_{data['stem']}_{uuid.uuid4().hex[:8]}.csv"
                csv_path = DOWNLOAD_DIR / csv_name

                with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=list(data.keys()))
                    writer.writeheader()
                    writer.writerow(data)

                base = str(request.base_url).rstrip("/")
                data["saved"]["summary_metrics_csv"] = str(csv_path)
                data["saved"]["download_url"] = f"{base}/downloads/{csv_name}"

            # cleanup upload if save==0
            if int(save) == 0:
                try:
                    dst_path.unlink(missing_ok=True)
                except Exception:
                    pass

            return JSONResponse(content=data)

        # Otherwise run the pipeline script (for non-precomputed methods)
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--video", str(dst_path),
            "--subject", str(subject_id),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

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

        data = _parse_last_json_line(proc.stdout)

        # reliability-aware: if trusted==0 -> hr_bpm = null
        try:
            trusted_val = float(data.get("trusted", 0))
        except Exception:
            trusted_val = 0.0

        if trusted_val == 0.0:
            data["hr_bpm"] = None

        if int(save) == 0:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": 0, "error": str(e)})
