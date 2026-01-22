from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

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

# ✅ Add this here:
THESIS_ROOT = BASE_DIR / "thesis_pipeline"
THESIS_CSV = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

# CORS (allow your web app to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://rppgweb.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz", include_in_schema=False)
def healthz():
    # Helpful for debugging Render path issues
    return {
        "ok": True,
        "version": APP_VERSION,
        "thesis_root": str(THESIS_ROOT),
        "thesis_csv_exists": THESIS_CSV.exists(),
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

@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    subject_id: str = Form(...),
    condition: str = Form(...),
    modality: str = Form(...),
    method: str = Form("thesis_precomputed"),
    min_valid_pct: float = Form(50.0),
    save: int = Form(0),
):
    # Quick fail-fast if thesis_precomputed is requested but CSV isn't present
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

            # ✅ Pass thesis_root explicitly (Render-safe)
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

        # cleanup upload if save==0
        if int(save) == 0:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": 0, "error": str(e)})
