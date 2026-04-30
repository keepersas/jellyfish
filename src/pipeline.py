"""
Feature-extraction pipeline for YOLO-tracked Aurelia aurita.

Reads a track.csv produced by the per-frame YOLO detector and produces a
per-window feature table (window_features.csv) that is consumed by bews.py.

Input columns (track.csv):
    frame, time_s, area, conf, center_x, center_y, x1, y1, x2, y2

Output columns (window_features.csv):
    t_mid, Fp, ipi_cv, amp_rel, spec_ent,
    speed_mean, speed_cv, tortuosity, v_vert, immobility,
    conf_mean, dropout, bell_diam_px

Pipeline stages:
    1. Linear-interpolate detection gaps up to 1 s; flag windows with >20 % dropout
    2. Detrend bell-area signal with 15-s Savitzky–Golay low-pass; band-pass 0.1–2 Hz
    3. Detect contraction events as prominent negative peaks in the band-passed signal
    4. Aggregate per 60-s window with 50 % overlap

Quick usage:
    from pipeline import run
    features = run('track.csv')
    features.to_csv('window_features.csv', index=False)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks, welch


# ----- Window settings -----
WINDOW_S = 60.0     # window length (seconds)
STRIDE_S = 30.0     # window stride (seconds), giving 50 % overlap

# ----- Pulsation band-pass settings -----
BASELINE_S = 15.0   # Savitzky–Golay low-pass length for drift removal
BAND_LOW_HZ = 0.1
BAND_HIGH_HZ = 2.0
BAND_ORDER = 3
PEAK_MIN_SPACING_S = 0.4   # >= 1 / max plausible Fp

# ----- Detection-gap handling -----
MAX_GAP_S = 1.0     # interpolate detection gaps up to this length
DROPOUT_LIMIT = 0.20  # windows with more than this fraction of missing detections excluded


def _load_track(track_csv: str | Path) -> tuple[pd.DataFrame, float]:
    """Load track.csv and return (dataframe, fps)."""
    df = pd.read_csv(track_csv)
    fps = 1.0 / np.median(np.diff(df['time_s']))
    return df, fps


def _interpolate_gaps(df: pd.DataFrame, fps: float):
    """Linear-interpolate area and centroid through short detection gaps.

    Returns area_filled, cx, cy, conf, gap_mask (bool array of still-missing frames).
    """
    max_gap = int(round(MAX_GAP_S * fps))
    area_interp = df['area'].interpolate(limit=max_gap, limit_area='inside')
    gap_mask = area_interp.isna()
    area = area_interp.bfill().ffill().values
    cx = df['center_x'].interpolate(limit=max_gap).bfill().ffill().values
    cy = df['center_y'].interpolate(limit=max_gap).bfill().ffill().values
    conf = df['conf'].fillna(0.0).values
    return area, cx, cy, conf, gap_mask.values


def _bandpass_area(area: np.ndarray, fps: float):
    """Detrend with Savitzky–Golay low-pass and band-pass to the Fp range.

    Returns the band-passed area signal and the indices of detected
    contraction events (negative peaks of the band-passed signal).
    """
    win_baseline = int(round(BASELINE_S * fps)) | 1   # odd window
    baseline = savgol_filter(area, win_baseline, polyorder=2)
    area_ac = area - baseline

    b, a = butter(BAND_ORDER, [BAND_LOW_HZ, BAND_HIGH_HZ], fs=fps, btype='band')
    area_bp = filtfilt(b, a, area_ac)

    distance = max(1, int(PEAK_MIN_SPACING_S * fps))
    prominence = np.std(area_bp) * 0.5
    peaks, _ = find_peaks(-area_bp, distance=distance, prominence=prominence)
    return area_bp, peaks


def _window_features(i0: int, i1: int, *,
                     t: np.ndarray, area_bp: np.ndarray, area: np.ndarray,
                     cx: np.ndarray, cy: np.ndarray, conf: np.ndarray,
                     gap_mask: np.ndarray, fps: float) -> dict:
    """Compute one feature dict for samples [i0:i1)."""
    seg_t = t[i0:i1]
    seg_area_bp = area_bp[i0:i1]
    seg_area = area[i0:i1]
    seg_cx = cx[i0:i1]
    seg_cy = cy[i0:i1]
    seg_conf = conf[i0:i1]
    seg_gap = gap_mask[i0:i1]

    # ----- Pulsation features -----
    distance = max(1, int(PEAK_MIN_SPACING_S * fps))
    prominence = np.std(seg_area_bp) * 0.5 + 1e-9
    pk, _ = find_peaks(-seg_area_bp, distance=distance, prominence=prominence)

    if len(pk) >= 3:
        ipi = np.diff(pk) / fps
        Fp = 1.0 / np.median(ipi)
        ipi_cv = np.std(ipi) / max(np.mean(ipi), 1e-9)
        amp = (np.percentile(seg_area, 95) - np.percentile(seg_area, 5)) \
              / max(np.percentile(seg_area, 95), 1e-9)
        f, P = welch(seg_area_bp, fs=fps, nperseg=min(len(seg_area_bp), 512))
        Pn = P / (P.sum() + 1e-12)
        spec_ent = -(Pn * np.log(Pn + 1e-12)).sum() / np.log(len(Pn))
    else:
        Fp = ipi_cv = amp = spec_ent = np.nan

    # ----- Kinematic features -----
    dx = np.diff(seg_cx)
    dy = np.diff(seg_cy)
    dt = np.diff(seg_t)
    dt[dt == 0] = 1.0 / fps
    speed = np.sqrt(dx ** 2 + dy ** 2) / dt
    spd_mean = float(np.nanmean(speed))
    spd_cv = float(np.nanstd(speed) / (spd_mean + 1e-9))

    path = float(np.nansum(np.sqrt(dx ** 2 + dy ** 2)))
    disp = float(np.hypot(seg_cx[-1] - seg_cx[0], seg_cy[-1] - seg_cy[0]))
    tort = path / (disp + 1e-9)

    v_vert = float(np.nanmean(dy / dt))

    bell_diam_px = float(np.sqrt(np.nanmean(seg_area) * 4.0 / np.pi))
    spd_thr = bell_diam_px / 5.0
    immob = float(np.nanmean(speed < spd_thr))

    # ----- Quality features -----
    valid = ~seg_gap
    conf_mean = float(np.nanmean(seg_conf[valid])) if valid.any() else np.nan
    dropout = float(np.mean(seg_gap))

    return dict(
        t_mid=(seg_t[0] + seg_t[-1]) / 2.0,
        Fp=Fp, ipi_cv=ipi_cv, amp_rel=amp, spec_ent=spec_ent,
        speed_mean=spd_mean, speed_cv=spd_cv, tortuosity=tort,
        v_vert=v_vert, immobility=immob,
        conf_mean=conf_mean, dropout=dropout,
        bell_diam_px=bell_diam_px,
    )


def run(track_csv: str | Path) -> pd.DataFrame:
    """Top-level pipeline: track.csv → per-window feature DataFrame.

    Drops any window whose detection-dropout fraction exceeds DROPOUT_LIMIT
    (= 20 % by default).
    """
    df, fps = _load_track(track_csv)
    area, cx, cy, conf, gap_mask = _interpolate_gaps(df, fps)
    area_bp, _ = _bandpass_area(area, fps)
    t = df['time_s'].values

    win_len = int(WINDOW_S * fps)
    stride = int(STRIDE_S * fps)
    feats = []
    for i0 in range(0, len(df) - win_len, stride):
        f = _window_features(
            i0, i0 + win_len,
            t=t, area_bp=area_bp, area=area, cx=cx, cy=cy,
            conf=conf, gap_mask=gap_mask, fps=fps,
        )
        if f['dropout'] <= DROPOUT_LIMIT:
            feats.append(f)
    return pd.DataFrame(feats)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract window features from a YOLO track.csv')
    parser.add_argument('track_csv', help='Input track.csv path')
    parser.add_argument('--output', '-o', default=None, help='Output CSV path')
    args = parser.parse_args()
    out = args.output or f'window_features_{Path(args.track_csv).stem}.csv'
    features = run(args.track_csv)
    features.to_csv(out, index=False)
    print(f'wrote {len(features)} feature windows to {out}')
