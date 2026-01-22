from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import json
import sys
import uuid
import pathlib
import subprocess
from typing import Optional

APP_VERSION = "0.1.1"

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs" / "server_results"
SCRIPT_PATH = BASE_DIR / "realworld_demo_rppg_single.py"
THESIS_ROOT = BASE_DIR / "thesis_pipeline"


# ✅ Force correct thesis_pipeline location inside rppg_server_pack
THESIS_ROOT = BASE_DIR / "thesis_pipeline"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="rPPG Server", version=APP_VERSION)

# CORS (allow your web app to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://rppgweb.onrender.com"]
    allow_credentials=False,  # safer with allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    # nice home page instead of Not Found
    return RedirectResponse(url="/docs")

@app.get("/healthz", include_in_schema=False)
def healthz():
    return {
        "ok": True,
        "version": APP_VERSION,
        "base_dir": str(BASE_DIR),
        "script_path": str(SCRIPT_PATH),
        "thesis_root": str(THESIS_ROOT),
        "thesis_root_exists": THESIS_ROOT.exists(),
        "thesis_csv_exists": (THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv").exists(),
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

def _validate_choice(name: str, value: str, allowed: set[str]) -> str:
    v = (value or "").strip().lower()
    if v not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid {name}='{value}'. Allowed: {sorted(allowed)}")
    return v

@app.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    subject_id: str = Form(...),
    condition: str = Form(...),  # rest/exercise/breath
    modality: str = Form(...),   # face/palm
    method: str = Form("thesis_precomputed"),  # thesis_precomputed/fft
    min_valid_pct: float = Form(50.0),
    save: int = Form(0),
    timeout_sec: int = Form(120),  # Render-safe default timeout
):
    # Basic validations
    subject_id = (subject_id or "").strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id is required")

    condition = _validate_choice("condition", condition, {"rest", "exercise", "breath"})
    modality = _validate_choice("modality", modality, {"face", "palm"})
    method = _validate_choice("method", method, {"thesis_precomputed", "fft"})

    # Ensure thesis files exist if precomputed requested
    if method == "thesis_precomputed":
        csv_path = THESIS_ROOT / "08_hilbert" / "hilbert_hr_summary_all.csv"
        if not csv_path.exists():
            return JSONResponse(
                status_code=500,
                content={
                    "ok": 0,
                    "error": "Missing precomputed thesis CSV on server",
                    "expected_csv": str(csv_path),
                    "thesis_root": str(THESIS_ROOT),
                    "base_dir": str(BASE_DIR),
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

        # ✅ Run pipeline script, FORCE thesis_root path (no ../ ever)
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--video", str(dst_path),
            "--subject", str(subject_id),
            "--condition", str(condition),
            "--modality", str(modality),
            "--outdir", str(OUTPUT_DIR),
            "--method", str(method),
            "--min_valid_pct", str(float(min_valid_pct)),
            "--save", str(int(save)),
            "--thesis_root", str(THESIS_ROOT),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout_sec))
        except subprocess.TimeoutExpired as te:
            return JSONResponse(
                status_code=504,
                content={
                    "ok": 0,
                    "error": "Pipeline timeout",
                    "timeout_sec": int(timeout_sec),
                    "cmd": cmd,
                    "stdout": (te.stdout or "")[-4000:],
                    "stderr": (te.stderr or "")[-4000:],
                },
            )

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

        # add server context (helps debugging)
        data["_server"] = {
            "version": APP_VERSION,
            "base_dir": str(BASE_DIR),
            "script_path": str(SCRIPT_PATH),
            "thesis_root": str(THESIS_ROOT),
            "upload_saved_as": str(dst_path.name),
        }

        # cleanup upload if save==0
        if int(save) == 0:
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=data)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": 0,
                "error": str(e),
                "base_dir": str(BASE_DIR),
                "script_path": str(SCRIPT_PATH),
                "thesis_root": str(THESIS_ROOT),
            },
        )
