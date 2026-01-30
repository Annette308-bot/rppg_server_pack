import os
import sys
import time
import uuid
import csv
import json
import pathlib
import subprocess
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

APP_VERSION = "0.2.2"  # bump when you change behavior

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

# ✅ 3 modes supported
ALLOWED_METHODS = {"thesis_precomputed", "fft", "ceemdan_hilbert"}


def _no_cache_headers() -> Dict[str, str]:
    return {"Cache-Control": "no-store, max-age=0"}


def _render_git_commit() -> Optional[str]:
    return (
        os.environ.get("RENDER_GIT_COMMIT")
        or os.environ.get("GIT_COMMIT")
        or os.environ.get("COMMIT_SHA")
    )


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "none" or s.lower() == "null":
            return None
        return float(s)
    except Exception:
        return None


def _median(vals: List[float]) -> Optional[float]:
    vals = sorted([v for v in vals if v is not None])
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float(vals[mid - 1] + vals[mid]) / 2.0


def _safe_suffix(filename: str) -> str:
    suffix = pathlib.Path(filename).suffix.lower()
    if suffix in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return suffix
    return ".mp4"


def _pick_hr_column(fieldnames: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    lowered = [c.lower() for c in fieldnames]
    candidates = ["hr_bpm", "hr", "bpm", "median_bpm", "median_hr", "median"]
    for cand in candidates:
        for i, col in enumerate(lowered):
            if col == cand:
                return fieldnames[i]
    for i, col in enumerate(lowered):
        if "hr" in col or "bpm" in col:
            return fieldnames[i]
    return None


def _pick_spo2_column(fieldnames: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    lowered = [c.lower() for c in fieldnames]
    candidates = [
        "spo2_pct", "spo2_percent", "spo2", "spo2_value", "spo2_est",
        "spo2_trend", "spo2_index"
    ]
    for cand in candidates:
        for i, col in enumerate(lowered):
            if col == cand:
                return fieldnames[i]
    for i, col in enumerate(lowered):
        if "spo2" in col:
            return fieldnames[i]
    return None


# -----------------------------
# THESIS LOOKUPS (precomputed)
# -----------------------------
def thesis_hr_lookup(stem: str) -> Dict[str, Any]:
    """
    Reads HR from thesis HR summary CSV.
    Expected cols include: stem, median_bpm, iqr_bpm, valid_pct_masked, selected_imf
    """
    if not THESIS_HR_CSV.exists():
        return {"hr_bpm": None, "valid_pct_masked": None, "selected_imf": None, "error": "thesis_hr_csv_missing"}

    with THESIS_HR_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row_stem = (row.get("stem") or "").strip()
            if row_stem != stem:
                continue

            hr = (
                _to_float(row.get("hr_bpm"))
                or _to_float(row.get("median_bpm"))
                or _to_float(row.get("Median_Bpm"))
                or _to_float(row.get("MEDIAN_BPM"))
            )
            valid_pct = (
                _to_float(row.get("valid_pct_masked"))
                or _to_float(row.get("valid_pct"))
                or _to_float(row.get("valid_pct_mask"))
            )
            selected_imf = row.get("selected_imf") or row.get("selected_imf_idx") or row.get("imf_selected")

            iqr = _to_float(row.get("iqr_bpm")) or _to_float(row.get("IQR_BPM")) or _to_float(row.get("iqr"))

            return {
                "hr_bpm": hr,
                "valid_pct_masked": valid_pct,
                "selected_imf": selected_imf,
                "iqr_bpm": iqr,
            }

    return {"hr_bpm": None, "valid_pct_masked": None, "selected_imf": None, "error": "stem_not_found_in_thesis_hr_csv"}


def thesis_spo2_lookup(stem: str) -> Dict[str, Any]:
    """
    Reads thesis_pipeline/06_spo2/{stem}_spo2_trend.csv and returns median of spo2 column.
    """
    path = THESIS_SPO2_DIR / f"{stem}_spo2_trend.csv"
    if not path.exists():
        return {"spo2_pct": None, "n": 0, "error": "spo2_trend_missing"}

    values: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        spo2_col = _pick_spo2_column(r.fieldnames or [])
        if spo2_col is None:
            return {"spo2_pct": None, "n": 0, "error": "spo2_column_not_found", "columns": r.fieldnames}

        for row in r:
            v = _to_float(row.get(spo2_col))
            if v is not None:
                values.append(v)

    if not values:
        return {"spo2_pct": None, "n": 0, "error": "no_numeric_spo2_values"}

    med = _median(values)
    mean = float(sum(values) / len(values))
    return {
        "spo2_pct": med if med is not None else mean,
        "n": len(values),
        "min_pct": float(min(values)),
        "max_pct": float(max(values)),
        "file": str(path),
    }


# -----------------------------
# REAL PIPELINE OUTPUT PARSING
# -----------------------------
def _parse_metrics_from_stdout(stdout: str) -> Optional[dict]:
    """
    Preferred: lines like METRICS_JSON={...}
    Fallback: last JSON-looking line
    """
    if not stdout:
        return None

    lines = stdout.splitlines("\n")

    # 1) Preferred format
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("METRICS_JSON="):
            payload = line.split("=", 1)[1].strip()
            try:
                return json.loads(payload)
            except Exception:
                return None

    # 2) Fallback: try last JSON-like line
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue

    return None


def _find_summary_csv(run_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Try common places first, otherwise search for recent *summary*.csv.
    """
    candidates = [
        run_dir / "summary_metrics.csv",
        run_dir / "analysis" / "summary_metrics.csv",
        run_dir / "summary.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    found = list(run_dir.glob("**/*summary*.csv"))
    if not found:
        found = list(run_dir.glob("**/*.csv"))
    if not found:
        return None

    found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return found[0]


def _parse_summary_csv(path: pathlib.Path) -> Dict[str, Any]:
    """
    Reads first row of summary csv and tries to extract HR and SpO2.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        hr_col = _pick_hr_column(r.fieldnames or [])
        spo2_col = _pick_spo2_column(r.fieldnames or [])

        for row in r:
            hr = _to_float(row.get(hr_col)) if hr_col else None
            spo2 = _to_float(row.get(spo2_col)) if spo2_col else None
            valid_pct = (
                _to_float(row.get("valid_pct_masked"))
                or _to_float(row.get("valid_pct"))
                or _to_float(row.get("valid_pct_mask"))
            )
            return {
                "hr_bpm": hr,
                "spo2_pct": spo2,
                "valid_pct_masked": valid_pct,
                "hr_col": hr_col,
                "spo2_col": spo2_col,
                "file": str(path),
            }

    return {"hr_bpm": None, "spo2_pct": None, "valid_pct_masked": None, "file": str(path), "error": "empty_summary_csv"}


def _write_download_csv(base_url: str, stem: str, method: str, subject: str, condition: str, modality: str,
                       hr_bpm, trusted, spo2_pct, spo2_trusted) -> Dict[str, str]:
    csv_name = f"{stem}_{method}_{uuid.uuid4().hex[:8]}.csv"
    csv_path = DOWNLOAD_DIR / csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "condition", "modality", "method", "stem", "hr_bpm", "trusted", "spo2_pct", "spo2_trusted"])
        w.writerow([subject, condition, modality, method, stem, hr_bpm, trusted, spo2_pct, spo2_trusted])

    return {"summary_metrics_csv": str(csv_path), "download_url": f"{base_url}/downloads/{csv_name}"}


# -----------------------------
# FASTAPI APP
# -----------------------------
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


@app.get("/downloads/{name}", include_in_schema=False)
def download_file(name: str):
    path = (DOWNLOAD_DIR / name).resolve()
    if not str(path).startswith(str(DOWNLOAD_DIR.resolve())):
        return JSONResponse(status_code=400, content={"ok": 0, "error": "bad_path"})
    if not path.exists():
        return JSONResponse(status_code=404, content={"ok": 0, "error": "file_not_found"})
    return FileResponse(str(path), media_type="text/csv", filename=name)


@app.get("/healthz", include_in_schema=False)
def healthz():
    spo2_files = []
    if THESIS_SPO2_DIR.exists():
        spo2_files = sorted([p.name for p in THESIS_SPO2_DIR.glob("*_spo2_trend.csv")])

    return {
        "ok": True,
        "version": APP_VERSION,
        "git_commit": _render_git_commit(),
        "cwd": str(pathlib.Path.cwd()),
        "__file__": str(pathlib.Path(__file__).resolve()),
        "ui_index": str(INDEX_HTML),
        "ui_exists": INDEX_HTML.exists(),
        "thesis_hr_csv": str(THESIS_HR_CSV),
        "thesis_hr_csv_exists": THESIS_HR_CSV.exists(),
        "thesis_spo2_dir": str(THESIS_SPO2_DIR),
        "thesis_spo2_dir_exists": THESIS_SPO2_DIR.exists(),
        "spo2_files_count": len(spo2_files),
        "spo2_files_sample": spo2_files[:10],
        "allowed_methods": sorted(ALLOWED_METHODS),
    }


@app.post("/upload_video")
async def upload_video(
    request: Request,
    file: Optional[UploadFile] = File(None),  # optional for thesis_precomputed
    subject_id: str = Form("S01"),
    condition: str = Form("rest"),
    modality: str = Form("face"),
    method: str = Form("thesis_precomputed"),
    min_valid_pct: float = Form(50.0),
    save: int = Form(1),
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

    out: Dict[str, Any] = {
        "ok": 1,
        "version": APP_VERSION,
        "method": method,
        "stem": stem,
        "subject": subject_id,
        "condition": condition,
        "modality": modality,
        "min_valid_pct": float(min_valid_pct),
    }

    # -----------------------------
    # METHOD 1: THESIS (precomputed)
    # -----------------------------
    if method == "thesis_precomputed":
        hr = thesis_hr_lookup(stem)
        sp = thesis_spo2_lookup(stem)

        hr_bpm = hr.get("hr_bpm")
        valid_pct = hr.get("valid_pct_masked")
        trusted = 1 if (valid_pct is not None and valid_pct >= float(min_valid_pct)) else 0

        spo2_pct = sp.get("spo2_pct")
        spo2_trusted = 1 if (spo2_pct is not None) else 0

        out.update(
            {
                "hr_bpm": hr_bpm if trusted else None,
                "valid_pct_masked": valid_pct,
                "trusted": trusted,
                "hr_method": "thesis_csv",
                "spo2_pct": spo2_pct,
                "spo2_trusted": spo2_trusted,
                "spo2_method": "thesis_trend_csv",
                "details": {
                    "selected_imf": hr.get("selected_imf"),
                    "iqr_bpm": hr.get("iqr_bpm"),
                    "spo2_n": sp.get("n"),
                    "spo2_min": sp.get("min_pct"),
                    "spo2_max": sp.get("max_pct"),
                    "spo2_file": sp.get("file"),
                    "hr_error": hr.get("error"),
                    "spo2_error": sp.get("error"),
                },
            }
        )

        out["saved"] = _write_download_csv(
            base, stem, "thesis_precomputed", subject_id, condition, modality,
            out.get("hr_bpm"), trusted, spo2_pct, spo2_trusted
        )
        return JSONResponse(content=out)

    # -----------------------------
    # METHOD 2/3: RUN ON UPLOAD
    # -----------------------------
    if file is None:
        return JSONResponse(status_code=400, content={"ok": 0, "error": "no_file_uploaded_for_upload_method"})

    suffix = _safe_suffix(file.filename or "upload.mp4")
    safe_name = f"{stem}_{uuid.uuid4().hex[:12]}{suffix}"
    dst_path = UPLOAD_DIR / safe_name

    # Save upload
    with dst_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    run_id = uuid.uuid4().hex[:10]
    run_dir = OUTPUT_DIR / f"{stem}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    timeout_sec = 240  # Free tier can be slow; increase if needed

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--video", str(dst_path),
        "--subject", subject_id,
        "--condition", condition,
        "--modality", modality,
        "--outdir", str(run_dir),
        "--min_valid_pct", str(float(min_valid_pct)),
        "--method", method,     # ✅ fft OR ceemdan_hilbert
        "--save", "1",          # ✅ produce summary_metrics.csv (optional)
    ]

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

    # ✅ FIRST: parse METRICS_JSON from stdout
    metrics = _parse_metrics_from_stdout(proc.stdout)

    # ✅ SECOND: optional fallback to summary CSV
    summary_csv = _find_summary_csv(run_dir)
    parsed = _parse_summary_csv(summary_csv) if (summary_csv and summary_csv.exists()) else None

    # If neither exists, fail
    if metrics is None and parsed is None:
        return JSONResponse(
            status_code=500,
            content={
                "ok": 0,
                "error": "no_metrics_found",
                "run_dir": str(run_dir),
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            },
        )

    # Prefer metrics from stdout; fallback to parsed CSV
    hr_bpm = (metrics or {}).get("hr_bpm") if metrics else parsed.get("hr_bpm")
    spo2_pct = (metrics or {}).get("spo2_pct") if metrics else parsed.get("spo2_pct")

    # trusted flags
    trusted = 0
    spo2_trusted = 0
    valid_pct = None

    if metrics:
        trusted = 1 if metrics.get("trusted") else 0
        spo2_trusted = 1 if metrics.get("spo2_trusted") else 0
        valid_pct = metrics.get("valid_pct_masked")
    else:
        valid_pct = parsed.get("valid_pct_masked")
        trusted = 1 if (valid_pct is not None and valid_pct >= float(min_valid_pct) and hr_bpm is not None) else 0
        spo2_trusted = 1 if (spo2_pct is not None) else 0

    out.update(
        {
            "hr_bpm": hr_bpm if trusted else None,
            "valid_pct_masked": valid_pct,
            "trusted": trusted,
            "hr_method": (metrics or {}).get("hr_method") or f"upload_{method}",
            "spo2_pct": spo2_pct,
            "spo2_trusted": spo2_trusted,
            "spo2_method": (metrics or {}).get("spo2_method") or "upload_spo2",
            "details": {
                "runtime_sec": round(time.time() - t0, 2),
                "run_dir": str(run_dir),
                "summary_csv": str(summary_csv) if summary_csv else None,
                "stdout_tail": proc.stdout[-1200:],
                "stderr_tail": proc.stderr[-1200:],
                "metrics_from_stdout": True if metrics else False,
                "parsed_from_csv": True if parsed else False,
                "hr_col": (parsed or {}).get("hr_col") if parsed else None,
                "spo2_col": (parsed or {}).get("spo2_col") if parsed else None,
                "hr_error": (metrics or {}).get("hr_error") if metrics else None,
            },
        }
    )

    out["saved"] = _write_download_csv(
        base, stem, method, subject_id, condition, modality,
        out.get("hr_bpm"), trusted, spo2_pct, spo2_trusted
    )

    # optionally delete upload if save == 0
    if int(save) == 0:
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(content=out)
