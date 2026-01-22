# rppg_server_pack/realworld_demo_rppg_single.py
r"""
Single-Video rPPG Analyzer (Web-friendly) — HR only + reliability-aware
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def _as_float_or_none(x) -> float | None:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _stem(subject: str, condition: str, modality: str) -> str:
    return f"{subject}_{condition}_{modality}"


def load_thesis_row(thesis_root: Path, stem: str) -> dict:
    csv_path = thesis_root / "08_hilbert" / "hilbert_hr_summary_all.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing thesis summary CSV.\n"
            f"Looked for: {csv_path}\n"
            f"thesis_root resolved to: {thesis_root}"
        )

    df = pd.read_csv(csv_path)
    if "stem" not in df.columns:
        raise ValueError(f"'stem' column not found in {csv_path.name}. Columns={list(df.columns)}")

    row = df.loc[df["stem"].astype(str) == stem]
    if row.empty:
        raise KeyError(f"Stem '{stem}' not found in {csv_path}")

    r = row.iloc[0].to_dict()
    return {
        "stem": stem,
        "selected_imf": _as_float_or_none(r.get("selected_imf")),
        "hr_hilbert_median_bpm": _as_float_or_none(r.get("median_bpm")),
        "hr_hilbert_iqr_bpm": _as_float_or_none(r.get("iqr_bpm")),
        "valid_pct_masked": _as_float_or_none(r.get("valid_pct_masked")),
    }


EXPECTED_FPS = 30.0
ROI_FRAC = 0.33
MIN_SECONDS_FOR_HR = 5


def center_square_roi(frame, frac: float = ROI_FRAC):
    h, w = frame.shape[:2]
    s = int(min(h, w) * float(frac))
    y1 = h // 2 - s // 2
    y2 = y1 + s
    x1 = w // 2 - s // 2
    x2 = x1 + s
    return frame[y1:y2, x1:x2]


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / float(k)


def detrend_signal(x: np.ndarray, fs: float, win_sec: float = 3.0) -> np.ndarray:
    k = int(max(1, round(fs * win_sec)))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    trend = moving_average(xpad, k)
    if len(trend) < len(x):
        trend = np.pad(trend, (0, len(x) - len(trend)), mode="edge")
    return x - trend[: len(x)]


def spectral_peak_bpm(x: np.ndarray, fs: float, fmin: float = 0.7, fmax: float = 3.0) -> float | None:
    n = len(x)
    if n < int(fs * MIN_SECONDS_FOR_HR):
        return None
    x = x - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None
    idx = int(np.argmax(np.abs(X[band])))
    f_peak = float(freqs[band][idx])
    bpm = f_peak * 60.0
    return bpm if np.isfinite(bpm) else None


def fft_hr_from_video(video_path: Path) -> dict:
    if cv2 is None:
        return {"ok": 0.0, "error": "cv2_missing_for_fft"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": 0.0, "error": "open_failed"}

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
        roi = center_square_roi(frame, frac=ROI_FRAC)
        b, g, r = cv2.split(roi)
        greens.append(float(np.mean(g)))

    cap.release()

    greens = np.asarray(greens, dtype=float)
    if len(greens) < int(fs * MIN_SECONDS_FOR_HR):
        return {"ok": 0.0, "error": "too_short", "frames": float(len(greens)), "fs": float(fs)}

    g_det = detrend_signal(greens, float(fs), win_sec=3.0)
    hr_fft_bpm = spectral_peak_bpm(g_det, float(fs), fmin=0.7, fmax=3.0)

    return {"ok": 1.0, "fs": float(fs), "frames": float(frames), "hr_fft_bpm": _as_float_or_none(hr_fft_bpm)}


def upsert_summary(outdir: Path, row: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "summary_metrics.csv"

    keep_cols = [
        "subject", "condition", "modality", "hr_method",
        "hr_bpm", "trusted", "min_valid_pct",
        "valid_pct_masked", "iqr_bpm", "selected_imf",
        "file", "stem",
    ]
    clean_row = {k: row.get(k) for k in keep_cols}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        mask = (
            (df.get("subject") == clean_row["subject"])
            & (df.get("condition") == clean_row["condition"])
            & (df.get("modality") == clean_row["modality"])
            & (df.get("hr_method") == clean_row["hr_method"])
        )
        df = df.loc[~mask].copy()
        df = pd.concat([df, pd.DataFrame([clean_row])], ignore_index=True)
    else:
        df = pd.DataFrame([clean_row])

    df.to_csv(csv_path, index=False)
    return csv_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--subject", required=True, type=str)
    ap.add_argument("--condition", required=True, type=str, choices=["rest", "exercise", "breath"])
    ap.add_argument("--modality", required=True, type=str, choices=["face", "palm"])
    ap.add_argument("--outdir", required=True, type=str)

    ap.add_argument("--method", type=str, default="thesis_precomputed",
                    choices=["thesis_precomputed", "fft"])

    ap.add_argument("--min_valid_pct", type=float, default=50.0)
    ap.add_argument("--save", type=int, default=0, choices=[0, 1])

    # ✅ REQUIRED CHANGE
    default_thesis_root = (Path(__file__).resolve().parent / "thesis_pipeline")
    ap.add_argument("--thesis_root", type=str, default=str(default_thesis_root))

    return ap.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    video_path = Path(args.video)
    outdir = Path(args.outdir)

    thesis_root = Path(args.thesis_root)
    # ✅ Render-safe resolution
    if not thesis_root.is_absolute():
        thesis_root = (script_dir / thesis_root).resolve()
    else:
        thesis_root = thesis_root.resolve()

    # ✅ auto-correct if user passes project root
    direct_csv = thesis_root / "08_hilbert" / "hilbert_hr_summary_all.csv"
    nested_csv = thesis_root / "thesis_pipeline" / "08_hilbert" / "hilbert_hr_summary_all.csv"
    if (not direct_csv.exists()) and nested_csv.exists():
        thesis_root = (thesis_root / "thesis_pipeline").resolve()

    stem = _stem(args.subject, args.condition, args.modality)

    try:
        if args.method == "thesis_precomputed":
            trow = load_thesis_row(thesis_root, stem)

            hr_med = trow["hr_hilbert_median_bpm"]
            iqr = trow["hr_hilbert_iqr_bpm"]
            valid_pct = trow["valid_pct_masked"]
            selected_imf = trow["selected_imf"]

            trusted = 1.0 if (valid_pct is not None and valid_pct >= float(args.min_valid_pct)) else 0.0
            hr_bpm = hr_med if trusted == 1.0 else None

            out = {
                "ok": 1.0,
                "subject": args.subject,
                "condition": args.condition,
                "modality": args.modality,
                "file": str(video_path),
                "stem": stem,
                "hr_method": "thesis_precomputed",
                "hr_bpm": hr_bpm,
                "trusted": trusted,
                "min_valid_pct": float(args.min_valid_pct),
                "valid_pct_masked": valid_pct,
                "iqr_bpm": iqr,
                "selected_imf": selected_imf,
            }

        else:
            r = fft_hr_from_video(video_path)
            if float(r.get("ok", 0.0)) != 1.0:
                out = {
                    "ok": 0.0,
                    "error": r.get("error", "fft_failed"),
                    "subject": args.subject,
                    "condition": args.condition,
                    "modality": args.modality,
                    "file": str(video_path),
                    "stem": stem,
                    "hr_method": "fft",
                    "hr_bpm": None,
                    "trusted": 0.0,
                }
            else:
                hr_fft = r.get("hr_fft_bpm")
                trusted = 1.0 if hr_fft is not None else 0.0
                hr_bpm = hr_fft if trusted == 1.0 else None

                out = {
                    "ok": 1.0,
                    "subject": args.subject,
                    "condition": args.condition,
                    "modality": args.modality,
                    "file": str(video_path),
                    "stem": stem,
                    "hr_method": "fft",
                    "hr_bpm": hr_bpm,
                    "trusted": trusted,
                    "min_valid_pct": None,
                    "valid_pct_masked": None,
                    "iqr_bpm": None,
                    "selected_imf": None,
                }

        if int(args.save) == 1:
            saved_csv = upsert_summary(outdir, out)
            out["saved"] = {"summary_metrics_csv": str(saved_csv.resolve())}
        else:
            out["saved"] = {"summary_metrics_csv": None}

        print(json.dumps(out, ensure_ascii=False))

    except Exception as e:
        err = {
            "ok": 0.0,
            "error": str(e),
            "subject": args.subject,
            "condition": args.condition,
            "modality": args.modality,
            "file": str(video_path),
            "stem": stem,
            "hr_method": args.method,
            "hr_bpm": None,
            "trusted": 0.0,
        }
        print(json.dumps(err, ensure_ascii=False))


if __name__ == "__main__":
    main()
