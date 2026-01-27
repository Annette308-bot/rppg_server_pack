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

APP_VERSION = "0.1.0"

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"

# ✅ thesis precomputed CSV
THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"

# ✅ NEW: where we save downloadable CSVs
DOWNLOADS_DIR = BASE_DIR / "downloads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

# ✅ Serve saved files at /files/<filename>
app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# CORS (allow your web app to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://rppg-web.onrender.com"]
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
        "downloads_dir": str(DOWNLOADS_DIR),
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

def _flatten_dict(d, prefix=""):
    """Flatten nested dicts into 1-level dict suitable for CSV."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out

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
    # Fail-fast if thesis_precomputed is requested but CSV isn't present
    if method == "thesis_precomputed" and not THESIS_CSV.exists():
        return JSONResponse(
            status_code=500,
            content={
                "ok": 0,
                "error": "thesis_summary_missing_on_server",
                "looked_for": str(THESIS_CSV),
                "hint": "Confirm thesis_pipeline is committed and deployed, and that THESIS_ROOT points to the correct folder.",
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

        # Run pipeline script
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
        try:
            trusted_val = float(data.get("trusted", 0))
        except Exception:
            trusted_val = 0.0

        if trusted_val == 0.0:
            data["hr_bpm"] = None

        # ✅ NEW: if save==1, write a CSV + return download URL
        if int(save) == 1 and isinstance(data, dict):
            flat = _flatten_dict(data)

            csv_name = f"summary_{subject_id}_{condition}_{modality}_{uuid.uuid4().hex[:8]}.csv"
            csv_path = DOWNLOADS_DIR / csv_name

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
                writer.writeheader()
                writer.writerow(flat)

            base = str(request.base_url).rstrip("/")
            data["saved"] = {
                "summary_metrics_csv": csv_name,
                "download_url": f"{base}/files/{csv_name}",
            }

        # cleanup upload if save==0
        if int(save) == 0:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": 0, "error": str(e)})
