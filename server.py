from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import csv
import json
import re
import sys
import uuid
import pathlib
import subprocess
from datetime import datetime

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: lock to your rppg-web domain
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
    }

def _norm_key(k: str) -> str:
    k = (k or "").strip()
    k = re.sub(r"[^0-9a-zA-Z]+", "_", k)
    return k.strip("_").lower()

def _normalize_dict(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[_norm_key(str(k))] = v
    return out

def _parse_last_json_line(stdout_text: str) -> dict:
    lines = [ln.strip() for ln in (stdout_text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return {"ok": 0, "error": "no_json_line_found", "stdout_tail": lines[-30:]}

def _find_row_in_thesis_csv(stem: str) -> dict | None:
    if not THESIS_CSV.exists():
        return None

    stem = (stem or "").strip()

    with THESIS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None

        fields_norm = {_norm_key(c): c for c in reader.fieldnames}

        # Try to locate a "stem" column (or similar)
        key_candidates = ["stem", "video_stem", "clip_stem", "name", "video", "id"]
        key_col = None
        for cand in key_candidates:
            if cand in fields_norm:
                key_col = fields_norm[cand]
                break

        # HR column candidates
        hr_candidates = ["hr_bpm", "hr", "hrvalue", "hr_val", "hrbpm"]
        hr_col = None
        for cand in hr_candidates:
            if cand in fields_norm:
                hr_col = fields_norm[cand]
                break

        # Optional extra metrics
        trusted_col = fields_norm.get("trusted")
        valid_col = fields_norm.get("valid_pct_masked") or fields_norm.get("valid_pct")
        iqr_col = fields_norm.get("iqr_bpm")
        imf_col = fields_norm.get("selected_imf") or fields_norm.get("imf") or fields_norm.get("selected_imf_idx")

        for row in reader:
            # match by stem column if present
            if key_col and (row.get(key_col, "").strip() == stem):
                out = _normalize_dict(row)
                out["_hr_col"] = hr_col
                out["_trusted_col"] = trusted_col
                out["_valid_col"] = valid_col
                out["_iqr_col"] = iqr_col
                out["_imf_col"] = imf_col
                return out

        # fallback: try match by filename-like columns if stem col missing
        if not key_col:
            for row in csv.DictReader(THESIS_CSV.open("r", encoding="utf-8", newline="")):
                out = _normalize_dict(row)
                for v in out.values():
                    if isinstance(v, str) and stem in v:
                        out["_hr_col"] = hr_col
                        out["_trusted_col"] = trusted_col
                        out["_valid_col"] = valid_col
                        out["_iqr_col"] = iqr_col
                        out["_imf_col"] = imf_col
                        return out

    return None

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def _write_summary_csv(payload: dict, out_path: pathlib.Path):
    keys = [
        "timestamp", "subject", "condition", "modality", "stem",
        "hr_method", "hr_bpm", "trusted", "min_valid_pct",
        "valid_pct_masked", "iqr_bpm", "selected_imf",
    ]
    row = {k: payload.get(k) for k in keys}
    row["timestamp"] = datetime.utcnow().isoformat()

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(row)

@app.get("/downloads/{filename}", include_in_schema=False)
def download_file(filename: str):
    # basic safety
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="bad_filename")

    fpath = DOWNLOAD_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(str(fpath), filename=filename, media_type="text/csv")

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
    # normalize these
    subject_id = (subject_id or "S01").strip()
    condition = (condition or "rest").strip().lower()
    modality = (modality or "face").strip().lower()
    method = (method or "thesis_precomputed").strip()

    stem = f"{subject_id}_{condition}_{modality}"

    # Save upload to disk (always, because FastAPI needs a real file path for the script)
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        ext = ".mp4"

    safe_name = f"{stem}_{uuid.uuid4().hex}{ext}"
    dst_path = UPLOAD_DIR / safe_name

    try:
        with dst_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # ---------- CASE A: thesis_precomputed (lookup only; skip script) ----------
        if method == "thesis_precomputed":
            row = _find_row_in_thesis_csv(stem)
            if row is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "ok": 0,
                        "error": "not_found_in_thesis_csv",
                        "stem": stem,
                        "looked_for": str(THESIS_CSV),
                    },
                )

            # extract values robustly
            hr_val = _safe_float(row.get("hr_bpm"), None)
            if hr_val is None:
                hr_val = _safe_float(row.get("hr"), None)
            trusted_val = _safe_float(row.get("trusted"), 1.0)
            valid_val = _safe_float(row.get("valid_pct_masked"), _safe_float(row.get("valid_pct"), 100.0))
            iqr_val = _safe_float(row.get("iqr_bpm"), None)
            imf_val = _safe_float(row.get("selected_imf"), None)

            payload = {
                "ok": 1,
                "subject": subject_id,
                "condition": condition,
                "modality": modality,
                "file": str(dst_path),
                "stem": stem,
                "hr_method": "thesis_precomputed",
                "hr_bpm": hr_val,
                "trusted": trusted_val,
                "min_valid_pct": float(min_valid_pct),
                "valid_pct_masked": valid_val,
                "iqr_bpm": iqr_val,
                "selected_imf": int(imf_val) if imf_val is not None else None,
                "saved": {"summary_metrics_csv": None, "download_url": None},
            }

        # ---------- CASE B: run pipeline script ----------
        else:
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
                        "stdout": (proc.stdout or "")[-4000:],
                        "stderr": (proc.stderr or "")[-4000:],
                    },
                )

            raw = _parse_last_json_line(proc.stdout)
            payload = _normalize_dict(raw)

            # enforce standard keys that your frontend expects
            payload.setdefault("ok", 1)
            payload.setdefault("subject", subject_id)
            payload.setdefault("condition", condition)
            payload.setdefault("modality", modality)
            payload.setdefault("stem", stem)
            payload.setdefault("hr_method", method)
            payload.setdefault("file", str(dst_path))
            payload.setdefault("min_valid_pct", float(min_valid_pct))

            # reliability rule (only if trusted == 0)
            trusted_val = _safe_float(payload.get("trusted"), 0.0)
            if trusted_val == 0.0:
                payload["hr_bpm"] = None

            payload["saved"] = payload.get("saved") or {"summary_metrics_csv": None, "download_url": None}

        # ---------- Saving summary CSV + download URL ----------
        if int(save) == 1:
            csv_name = f"{stem}_{uuid.uuid4().hex}.csv"
            csv_path = DOWNLOAD_DIR / csv_name
            _write_summary_csv(payload, csv_path)

            base = str(request.base_url).rstrip("/")  # e.g. https://rppg-server-pack.onrender.com
            payload["saved"] = {
                "summary_metrics_csv": str(csv_path),
                "download_url": f"{base}/downloads/{csv_name}",
            }
        else:
            # cleanup upload if save==0
            try:
                dst_path.unlink(missing_ok=True)
            except Exception:
                pass

        return JSONResponse(content=payload)

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": 0, "error": str(e)})
