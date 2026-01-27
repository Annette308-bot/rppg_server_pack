from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import csv
import json
import sys
import uuid
import pathlib
import subprocess
from datetime import datetime

APP_VERSION = "0.2.0"

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"

# Thesis precomputed CSV
THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

# Serve saved CSVs (and anything else you drop in OUTPUT_DIR)
app.mount("/downloads", StaticFiles(directory=str(OUTPUT_DIR)), name="downloads")

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
        "thesis_csv_exists": THESIS_CSV.exists(),
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
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

def _public_base_url(request: Request) -> str:
    # Render usually provides this; fallback to request base URL
    env_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return str(request.base_url).rstrip("/")

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

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
    # Fail fast if thesis_precomputed is requested but CSV isn't present
    if method == "thesis_precomputed" and not THESIS_CSV.exists():
        return JSONResponse(
            status_code=500,
            content={
                "ok": 0,
                "error": "thesis_summary_missing_on_server",
                "looked_for": str(THESIS_CSV),
                "hint": "Confirm thesis_pipeline is committed & deployed and THESIS_ROOT is correct.",
            },
        )

    # Save upload
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        ext = ".mp4"

    safe_name = f"{subject_id}_{condition}_{modality}_{uuid.uuid4().hex}{ext}"
    dst_path = UPLOAD_DIR / safe_name

    try:
        with dst_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--video", str(dst_path),
            "--subject", str(subject_id),
            "--condition", str(condition),
            "--modality", str(modality),
            "--outdir", str(OUTPUT_DIR),
            "--method", str(method),
            "--min_valid_pct", str(min_valid_pct),
            "--save", str(int(save)),
            "--thesis_root", str(THESIS_ROOT),
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
        trusted_val = _safe_float(data.get("trusted", 0), 0.0)
        if trusted_val == 0.0:
            data["hr_bpm"] = None

        # --- NEW: SAVE CSV + RETURN DOWNLOAD URL ---
        saved = {}
        if int(save) == 1:
            base = _public_base_url(request)

            # create a per-upload metrics CSV
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            stem = str(data.get("stem") or f"{subject_id}_{condition}_{modality}")
            csv_name = f"metrics_{stem}_{ts}.csv".replace(" ", "_")
            csv_path = OUTPUT_DIR / csv_name

            row = {
                "timestamp_utc": ts,
                "subject": data.get("subject", subject_id),
                "condition": data.get("condition", condition),
                "modality": data.get("modality", modality),
                "hr_method": data.get("hr_method", method),
                "hr_bpm": data.get("hr_bpm"),
                "trusted": data.get("trusted"),
                "min_valid_pct": data.get("min_valid_pct", min_valid_pct),
                "valid_pct_masked": data.get("valid_pct_masked"),
                "iqr_bpm": data.get("iqr_bpm"),
                "selected_imf": data.get("selected_imf"),
                "video_file_saved": str(dst_path),
            }

            with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            per_upload_url = f"{base}/downloads/{csv_name}"
            saved["summary_metrics_csv"] = per_upload_url

            # append to a global log
            log_path = OUTPUT_DIR / "uploads_log.csv"
            log_exists = log_path.exists()
            with log_path.open("a", newline="", encoding="utf-8") as flog:
                writer = csv.DictWriter(flog, fieldnames=list(row.keys()))
                if not log_exists:
                    writer.writeheader()
                writer.writerow(row)

            saved["uploads_log_csv"] = f"{base}/downloads/uploads_log.csv"

        # attach saved info (even if empty)
        data["saved"] = saved

        # cleanup upload if save==0
        if int(save) == 0:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": 0, "error": str(e)})
