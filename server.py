import os
import sys
import time
import uuid
import csv
import pathlib
import subprocess
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

APP_VERSION = "0.3.0"

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

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
WEBUI_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONDITIONS = {"rest", "breath", "exercise"}
ALLOWED_MODALITIES = {"face", "palm"}

# NEW: include ceemdan_hilbert
ALLOWED_METHODS = {"thesis_precomputed", "fft", "ceemdan_hilbert"}


def _no_cache_headers() -> Dict[str, str]:
    return {"Cache-Control": "no-store, max-age=0"}


def _safe_suffix(filename: str) -> str:
    suffix = pathlib.Path(filename).suffix.lower()
    if suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return suffix
    return ".mp4"


def _render_git_commit() -> Optional[str]:
    return (
        os.environ.get("RENDER_GIT_COMMIT")
        or os.environ.get("GIT_COMMIT")
        or os.environ.get("COMMIT_SHA")
    )


def _parse_metrics_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    if not stdout:
        return None
    for line in stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("METRICS_JSON="):
            try:
                return json.loads(line.split("=", 1)[1])
            except Exception:
                return None
    return None


app = FastAPI(title="rPPG Server", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML), media_type="text/html", headers=_no_cache_headers())
    return RedirectResponse(url="/docs")


@app.get("/ui", include_in_schema=False)
def ui():
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML), media_type="text/html", headers=_no_cache_headers())
    return RedirectResponse(url="/docs")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {
        "ok": True,
        "version": APP_VERSION,
        "git_commit": _render_git_commit(),
        "cwd": str(pathlib.Path.cwd()),
        "__file__": str(pathlib.Path(__file__).resolve()),
        "ui_index": str(INDEX_HTML),
        "ui_exists": INDEX_HTML.exists(),
        "script_path": str(SCRIPT_PATH),
        "script_exists": SCRIPT_PATH.exists(),
        "allowed_methods": sorted(ALLOWED_METHODS),
        "thesis_hr_csv_exists": THESIS_HR_CSV.exists(),
        "thesis_spo2_dir_exists": THESIS_SPO2_DIR.exists(),
    }


@app.get("/downloads/{name}", include_in_schema=False)
def download_file(name: str):
    path = (DOWNLOAD_DIR / name).resolve()
    if not str(path).startswith(str(DOWNLOAD_DIR.resolve())):
        return JSONResponse(status_code=400, content={"ok": 0, "error": "bad_path"})
    if not path.exists():
        return JSONResponse(status_code=404, content={"ok": 0, "error": "file_not_found"})
    return FileResponse(str(path), media_type="text/csv", filename=name)


@app.post("/upload_video")
async def upload_video(
    request: Request,
    file: Optional[UploadFile] = File(None),
    subject_id: str = Form("S01"),
    condition: str = Form("rest"),
    modality: str = Form("face"),
    method: str = Form("thesis_precomputed"),
    min_valid_pct: float = Form(50.0),
    save: int = Form(0),
):
    subject_id = (subject_id or "S01").strip() or "S01"
    condition = (condition or "rest").strip().lower()
    modality = (modality or "face").strip().lower()
    method = (method or "thesis_precomputed").strip().lower()

    if condition not in ALLOWED_CONDITIONS:
        return JSONResponse(status_code=400, content={"ok": 0, "error": f"bad_condition: {condition}"})
    if modality not in ALLOWED_MODALITIES:
        return JSONResponse(status_code=400, content={"ok": 0, "error": f"bad_modality: {modality}"})
    if method not in ALLOWED_METHODS:
        return JSONResponse(status_code=400, content={"ok": 0, "error": f"bad_method: {method}", "allowed": sorted(ALLOWED_METHODS)})

    stem = f"{subject_id}_{condition}_{modality}"
    base = str(request.base_url).rstrip("/")

    # If upload-based method, we require file
    if method in {"fft", "ceemdan_hilbert"} and file is None:
        return JSONResponse(status_code=400, content={"ok": 0, "error": "no_file_uploaded"})

    # Save upload (if provided)
    dst_path = None
    if file is not None:
        suffix = _safe_suffix(file.filename or "upload.mp4")
        safe_name = f"{stem}_{uuid.uuid4().hex[:12]}{suffix}"
        dst_path = UPLOAD_DIR / safe_name
        with dst_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    run_id = uuid.uuid4().hex[:10]
    run_dir = OUTPUT_DIR / f"{stem}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # For thesis_precomputed we can pass any path; script will ignore video
    video_arg = str(dst_path) if dst_path is not None else str(run_dir / "dummy.mp4")

    timeout_sec = 120  # Render Free: keep reasonable

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--video", video_arg,
        "--subject", subject_id,
        "--condition", condition,
        "--modality", modality,
        "--outdir", str(run_dir),
        "--min_valid_pct", str(float(min_valid_pct)),
        "--method", method,
        "--save", "0",   # server does not depend on CSV
        "--max_sec", "15",
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"ok": 0, "error": "pipeline_timeout", "timeout_sec": timeout_sec})

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

    metrics = _parse_metrics_from_stdout(proc.stdout)
    if not metrics:
        return JSONResponse(
            status_code=500,
            content={
                "ok": 0,
                "error": "metrics_json_missing",
                "stdout": proc.stdout[-3000:],
                "stderr": proc.stderr[-3000:],
            },
        )

    # Return only the values you care about (plus details)
    out = {
        "ok": 1,
        "version": APP_VERSION,
        "method": method,
        "stem": metrics.get("stem", stem),
        "hr_bpm": metrics.get("hr_bpm"),
        "trusted": metrics.get("trusted", 0),
        "valid_pct_masked": metrics.get("valid_pct_masked"),
        "iqr_bpm": metrics.get("iqr_bpm"),
        "selected_imf": metrics.get("selected_imf"),
        "spo2_trend": metrics.get("spo2_trend"),
        "spo2_ratio": metrics.get("spo2_ratio"),
        "spo2_quality": metrics.get("spo2_quality"),
        "runtime_sec": round(time.time() - t0, 3),
        "details": {
            "run_dir": str(run_dir),
            "hr_method": metrics.get("hr_method"),
            "spo2_method": metrics.get("spo2_method"),
            "warning": metrics.get("warning"),
            "spo2_error": metrics.get("spo2_error"),
        }
    }

    # optional: delete upload after processing
    if dst_path is not None:
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(content=out)
