# rppg_server_pack/realworld_demo_rppg_single.py
r"""
Single-Video rPPG Analyzer — HR-only (FastAPI/CLI friendly)

Main goal:
- Make the web app return HR values that MATCH your thesis CEEMDAN/Hilbert pipeline,
  by reading the precomputed thesis summary CSV.

Default method:
- thesis_precomputed  -> reads:
  ../thesis_pipeline/08_hilbert/hilbert_hr_summary_all.csv

Optional:
- fft               -> quick FFT on detrended green ROI (legacy baseline)
- ceemdan_hilbert   -> recompute Hilbert HR from selected IMF + stored IMFs in thesis_pipeline/07_ceemdan
                       (useful if you want recompute / debug / extend to new clips that have IMFs)

Examples (Windows PowerShell/CMD):
  python realworld_demo_rppg_single.py ^
    --video "../Data/raw/my_phone/S01/face/S01_rest_face.mp4" ^
    --subject S01 ^
    --condition rest ^
    --modality face ^
    --outdir "outputs/server_results" ^
    --method thesis_precomputed ^
    --min_valid_pct 50 ^
    --save 0

Output:
- Prints ONE JSON line to stdout (so server.py can json.loads it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _is_finite_number(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _json_safe(obj: Any) -> Any:
    """Convert NaN/Inf to None recursively so JSON is clean (standard)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float, np.integer, int)):
        try:
            x = float(obj)
            if not np.isfinite(x):
                return None
            return x
        except Exception:
            return None
    return obj


def _stem(subject: str, condition: str, modality: str) -> str:
    return f"{subject}_{condition}_{modality}"


# ----------------------------
# Method 1: thesis_precomputed
# ----------------------------
def load_thesis_precomputed_hr(
    thesis_root: Path,
    subject: str,
    condition: str,
    modality: str,
) -> Dict[str, Any]:
    """
    Reads thesis_pipeline/08_hilbert/hilbert_hr_summary_all.csv
    and returns the row for stem = S01_rest_face etc.
    """
    summary_path = thesis_root / "08_hilbert" / "hilbert_hr_summary_all.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing precomputed thesis summary: {summary_path}")

    df = pd.read_csv(summary_path)

    s = _stem(subject, condition, modality)
    hit = df[df["stem"].astype(str) == s]
    if hit.empty:
        raise KeyError(f"Stem '{s}' not found in {summary_path}")

    row = hit.iloc[0].to_dict()

    # Standardized fields we care about
    return {
        "stem": s,
        "selected_imf": int(row.get("selected_imf")) if _is_finite_number(row.get("selected_imf")) else None,
        "median_bpm": float(row.get("median_bpm")) if _is_finite_number(row.get("median_bpm")) else None,
        "iqr_bpm": float(row.get("iqr_bpm")) if _is_finite_number(row.get("iqr_bpm")) else None,
        "valid_pct_masked": float(row.get("valid_pct_masked")) if _is_finite_number(row.get("valid_pct_masked")) else None,
    }


# ----------------------------
# Method 2: FFT baseline (legacy)
# ----------------------------
def _fft_hr_from_video(video_path: Path, roi_frac: float = 0.33) -> Dict[str, Any]:
    """
    Minimal FFT baseline (kept only as an option).
    Requires OpenCV installed.
    """
    import cv2

    EXPECTED_FPS = 30.0
    MIN_SECONDS_FOR_HR = 5.0

    def center_square_roi(frame, frac=0.33):
        h, w = frame.shape[:2]
        s = int(min(h, w) * frac)
        y1 = h // 2 - s // 2
        y2 = y1 + s
        x1 = w // 2 - s // 2
        x2 = x1 + s
        return frame[y1:y2, x1:x2]

    def moving_average(x, k):
        if k <= 1:
            return x.copy()
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[k:] - c[:-k]) / float(k)

    def detrend_signal(x, fs, win_sec=3.0):
        k = int(max(1, round(fs * win_sec)))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        xpad = np.pad(x, (pad, pad), mode="edge")
        trend = moving_average(xpad, k)
        if len(trend) < len(x):
            trend = np.pad(trend, (0, len(x) - len(trend)), mode="edge")
        return x - trend[: len(x)]

    def spectral_peak_bpm(x, fs, fmin=0.7, fmax=3.0):
        n = len(x)
        if n < int(fs * MIN_SECONDS_FOR_HR):
            return np.nan
        x = x - np.mean(x)
        X = np.fft.rfft(x * np.hanning(n))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        band = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(band):
            return np.nan
        idx = np.argmax(np.abs(X[band]))
        f_peak = freqs[band][idx]
        return f_peak * 60.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "error": "open_failed"}

    fs = cap.get(cv2.CAP_PROP_FPS)
    if not fs or fs <= 0:
        fs = EXPECTED_FPS

    greens = []
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        roi = center_square_roi(frame, frac=roi_frac)
        _, g, _ = cv2.split(roi)
        greens.append(float(np.mean(g)))

    cap.release()

    greens = np.asarray(greens, dtype=float)
    if len(greens) < fs * MIN_SECONDS_FOR_HR:
        return {"ok": False, "error": "too_short", "frames": int(len(greens)), "fs": float(fs)}

    g_det = detrend_signal(greens, fs, win_sec=3.0)
    hr_bpm = spectral_peak_bpm(g_det, fs, fmin=0.7, fmax=3.0)

    return {
        "ok": True,
        "fs": float(fs),
        "frames": int(frames),
        "hr_bpm": float(hr_bpm) if np.isfinite(hr_bpm) else None,
    }


# ----------------------------
# Method 3: ceemdan_hilbert recompute (from stored IMFs + selected IMF)
# ----------------------------
def _ceemdan_hilbert_from_thesis_imfs(
    thesis_root: Path,
    subject: str,
    condition: str,
    modality: str,
    fs: float = 30.0,
    band_lo_bpm: float = 40.0,
    band_hi_bpm: float = 180.0,
    smooth_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Recomputes instantaneous HR from selected IMF using Hilbert,
    using files from thesis_pipeline/07_ceemdan:
      <stem>_imfs.npy or <stem>_imfs.csv
      <stem>_cardiac_imf_selected.csv
    """
    try:
        from scipy.signal import hilbert
    except Exception as e:
        raise RuntimeError("scipy is required for ceemdan_hilbert method (scipy.signal.hilbert).") from e

    ceemdan_dir = thesis_root / "07_ceemdan"

    stem = _stem(subject, condition, modality)

    # selection CSV
    sel_path = ceemdan_dir / f"{stem}_cardiac_imf_selected.csv"
    if not sel_path.exists():
        raise FileNotFoundError(f"Missing selection CSV: {sel_path}")

    sel_df = pd.read_csv(sel_path)
    if "selected_cardiac_imf" not in sel_df.columns:
        raise KeyError(f"selection CSV missing 'selected_cardiac_imf': {sel_path}")
    selected_imf = int(sel_df["selected_cardiac_imf"].iloc[0])

    # IMFs
    csv_path = ceemdan_dir / f"{stem}_imfs.csv"
    npy_path = ceemdan_dir / f"{stem}_imfs.npy"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        imf_cols = [c for c in df.columns if c.lower().startswith("imf")]
        if not imf_cols:
            raise RuntimeError(f"No IMF columns found in {csv_path}")
        imfs = df[imf_cols].to_numpy().T  # (K,N)
    elif npy_path.exists():
        arr = np.asarray(np.load(npy_path))
        if arr.ndim != 2:
            raise RuntimeError(f"IMF array not 2D in {npy_path} (shape={arr.shape})")
        imfs = arr if arr.shape[0] <= arr.shape[1] else arr.T
    else:
        raise FileNotFoundError(f"Missing IMFs for {stem} in {ceemdan_dir} (need _imfs.csv or _imfs.npy)")

    K, N = imfs.shape
    if not (1 <= selected_imf <= K):
        raise ValueError(f"{stem}: selected IMF={selected_imf} but only K={K} IMFs exist")

    x = np.asarray(imfs[selected_imf - 1, :], dtype=float)
    t = np.arange(N) / float(fs)

    # Hilbert analytic -> instantaneous bpm
    z = hilbert(x)
    phase = np.unwrap(np.angle(z))
    dphi = np.diff(phase)
    inst_hz = (fs / (2.0 * np.pi)) * dphi
    inst_hz = np.concatenate([inst_hz[:1], inst_hz])
    inst_bpm = inst_hz * 60.0

    # rolling median smoothing
    w = int(round(smooth_sec * fs))
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    s = pd.Series(inst_bpm)
    bpm_smooth = s.rolling(window=w, center=True, min_periods=max(3, w // 4)).median().to_numpy()

    # mask outside band
    bpm_masked = bpm_smooth.copy()
    bpm_masked[(bpm_masked < band_lo_bpm) | (bpm_masked > band_hi_bpm)] = np.nan

    valid_pct = 100.0 * float(np.isfinite(bpm_masked).sum()) / float(N) if N else 0.0
    hr_series = pd.Series(bpm_masked).dropna()
    if len(hr_series) < 5:
        hr_series = pd.Series(bpm_smooth).dropna()

    median_bpm = float(hr_series.median()) if len(hr_series) else None
    q1 = float(hr_series.quantile(0.25)) if len(hr_series) else None
    q3 = float(hr_series.quantile(0.75)) if len(hr_series) else None
    iqr = float(q3 - q1) if (q1 is not None and q3 is not None) else None

    return {
        "stem": stem,
        "selected_imf": selected_imf,
        "median_bpm": median_bpm,
        "iqr_bpm": iqr,
        "valid_pct_masked": valid_pct,
        # (If you ever want time-series later, it's here)
        # "time_s": t, "inst_bpm_raw": inst_bpm, "inst_bpm_smooth_masked": bpm_masked
    }


# ----------------------------
# Optional saving (HR-only)
# ----------------------------
def _upsert_summary_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    """
    Upsert by (subject, condition, modality).
    HR-only summary file.
    """
    cols = [
        "subject", "condition", "modality", "file",
        "hr_method", "hr_bpm", "trusted",
        "selected_imf", "iqr_bpm", "valid_pct_masked",
    ]

    df_new = pd.DataFrame([{c: row.get(c, None) for c in cols}])

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        mask = (
            (df["subject"].astype(str) == str(row["subject"])) &
            (df["condition"].astype(str) == str(row["condition"])) &
            (df["modality"].astype(str) == str(row["modality"]))
        )
        df = df.loc[~mask]
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str, help="Path to a single MP4")
    ap.add_argument("--subject", required=True, type=str, help="Subject ID, e.g., S01")
    ap.add_argument("--condition", required=True, type=str, choices=["rest", "exercise", "breath"])
    ap.add_argument("--modality", required=True, type=str, choices=["face", "palm"])
    ap.add_argument("--outdir", required=True, type=str, help="Output directory (used only if --save 1)")

    ap.add_argument("--method", type=str, default="thesis_precomputed",
                    choices=["thesis_precomputed", "ceemdan_hilbert", "fft"],
                    help="HR method to use")

    ap.add_argument("--thesis_root", type=str, default="../thesis_pipeline",
                    help="Path to thesis_pipeline (relative is OK)")

    ap.add_argument("--min_valid_pct", type=float, default=50.0,
                    help="Reliability threshold: minimum valid_pct_masked to mark trusted=True")

    ap.add_argument("--save", type=int, default=0, choices=[0, 1],
                    help="1=save HR-only summary CSV into outdir, 0=print JSON only")

    return ap.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    outdir = Path(args.outdir)
    thesis_root = Path(args.thesis_root)

    if not video_path.exists():
        payload = {"ok": 0, "error": "file_not_found", "file": str(video_path)}
        print(json.dumps(_json_safe(payload), ensure_ascii=False))
        return

    subject = args.subject
    condition = args.condition
    modality = args.modality

    hr_method = args.method

    # Compute HR (HR-only)
    try:
        if hr_method == "thesis_precomputed":
            r = load_thesis_precomputed_hr(thesis_root, subject, condition, modality)
        elif hr_method == "ceemdan_hilbert":
            r = _ceemdan_hilbert_from_thesis_imfs(thesis_root, subject, condition, modality, fs=30.0)
        else:  # fft
            r = _fft_hr_from_video(video_path)
            if not r.get("ok"):
                raise RuntimeError(r.get("error", "fft_failed"))
            r = {
                "stem": _stem(subject, condition, modality),
                "selected_imf": None,
                "median_bpm": r.get("hr_bpm"),
                "iqr_bpm": None,
                "valid_pct_masked": None,
            }
    except Exception as e:
        payload = {
            "ok": 0,
            "error": str(e),
            "hr_method": hr_method,
            "stem": _stem(subject, condition, modality),
            "file": str(video_path),
        }
        print(json.dumps(_json_safe(payload), ensure_ascii=False))
        return

    hr_bpm = r.get("median_bpm")
    valid_pct = r.get("valid_pct_masked")

    trusted = bool(
        (hr_bpm is not None) and np.isfinite(float(hr_bpm)) and
        (valid_pct is not None) and np.isfinite(float(valid_pct)) and
        (float(valid_pct) >= float(args.min_valid_pct))
    )

    payload = {
        "ok": 1,
        "subject": subject,
        "condition": condition,
        "modality": modality,
        "file": str(video_path),
        "stem": r.get("stem"),
        "hr_method": hr_method,
        "hr_bpm": hr_bpm,
        "trusted": trusted,
        "min_valid_pct": float(args.min_valid_pct),
        "valid_pct_masked": valid_pct,
        "iqr_bpm": r.get("iqr_bpm"),
        "selected_imf": r.get("selected_imf"),
    }

    # Optional save (HR-only summary)
    if args.save == 1:
        summary_csv = outdir / "summary_metrics_hr_only.csv"
        row = {
            "subject": subject,
            "condition": condition,
            "modality": modality,
            "file": str(video_path),
            "hr_method": hr_method,
            "hr_bpm": hr_bpm,
            "trusted": trusted,
            "selected_imf": r.get("selected_imf"),
            "iqr_bpm": r.get("iqr_bpm"),
            "valid_pct_masked": valid_pct,
        }
        _upsert_summary_csv(summary_csv, row)
        payload["saved"] = {"summary_metrics_hr_only_csv": str(summary_csv.resolve())}

    print(json.dumps(_json_safe(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
