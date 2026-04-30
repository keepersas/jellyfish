"""
BEWS: unsupervised pollution detection for Aurelia aurita movement features.

Two-layer detector:
  Layer 1 (fast): mean |z| > T_z over k consecutive windows  — step changes, outliers
  Layer 2 (drift): one-sided CUSUM on mean|z|       — sustained small distributional shifts

Per-animal calibration: μ, σ from a clean-water baseline (≥15 min) define z-scores;
global thresholds T_z, target, h are calibrated once on pooled clean-window scores
from training animals and frozen for deployment.

This is the simplified production architecture. An earlier prototype included an
IsolationForest novelty layer; empirical analysis on five individuals showed it
produced zero unique true positives because the toxicological response is monotonic
along the dominant axis already captured by mean|z|, so it was removed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ['Fp', 'ipi_cv', 'amp_rel', 'spec_ent',
            'speed_mean', 'speed_cv', 'tortuosity', 'immobility']


class AnimalBaseline:
    """Per-animal clean-water baseline: feature-wise μ and σ."""

    def __init__(self):
        self.mu: pd.Series | None = None
        self.sd: pd.Series | None = None
        self.features = FEATURES

    def fit(self, clean_windows: pd.DataFrame) -> "AnimalBaseline":
        X = clean_windows[self.features]
        self.mu = X.mean()
        self.sd = X.std().replace(0, 1e-9)
        return self

    def score(self, windows: pd.DataFrame) -> pd.DataFrame:
        """Return z-scored features and their mean absolute value per window."""
        X = windows[self.features]
        Z = (X - self.mu) / self.sd
        out = pd.DataFrame(index=windows.index)
        for f in self.features:
            out[f + '_z'] = Z[f]
        out['mean_abs_z'] = Z.abs().mean(axis=1)
        return out


def _cusum_positive(x: np.ndarray, target: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided positive CUSUM with reset on alarm.

    S_i = max(0, S_{i-1} + (x_i - target));  alarm when S_i > h.
    """
    S = np.zeros_like(x, dtype=float)
    alarm = np.zeros_like(x, dtype=bool)
    s = 0.0
    for i, v in enumerate(x):
        s = max(0.0, s + (v - target))
        S[i] = s
        if s > h:
            alarm[i] = True
            s = 0.0     # reset after alarm to allow re-detection
    return S, alarm


class BEWSDetector:
    """
    Two-layer streaming alarm logic.

      Layer 1 (fast):  mean|z| > T_z for k consecutive windows
                       — catches step changes and outlier windows
      Layer 2 (drift): CUSUM on (mean|z| - target) > h
                       — catches sustained small shifts that no single window flags

    Final alarm = Layer 1 alarm OR Layer 2 alarm.
    """

    def __init__(self,
                 threshold_mean_abs_z: float,
                 cusum_target: float,
                 cusum_h: float,
                 k_consecutive: int = 3):
        self.t_z = threshold_mean_abs_z
        self.cusum_target = cusum_target
        self.cusum_h = cusum_h
        self.k = k_consecutive

    def apply(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Apply both layers and return scored with alarm columns appended."""
        out = scored.copy()
        out['above_z'] = out['mean_abs_z'] > self.t_z

        # Layer 1: k-consecutive rule on above_z
        flag = out['above_z'].values
        fast_alarm = np.zeros_like(flag, dtype=bool)
        run = 0
        for i, v in enumerate(flag):
            run = run + 1 if v else 0
            if run >= self.k:
                fast_alarm[i] = True

        # Layer 2: CUSUM on mean|z|
        S, slow_alarm = _cusum_positive(
            out['mean_abs_z'].values.astype(float),
            target=self.cusum_target, h=self.cusum_h,
        )
        out['cusum_S'] = S
        out['alarm_fast'] = fast_alarm
        out['alarm_slow'] = slow_alarm
        out['alarm'] = fast_alarm | slow_alarm
        return out


def calibrate_thresholds(train_scores_list,
                         target_fpr_per_window: float = 0.01,
                         cusum_safety_factor: float = 1.3):
    """
    Pool clean-window scores from training animals and pick thresholds.

    Layer 1 threshold T_z: percentile cut on pooled clean mean|z|; target_fpr_per_window
        is the desired single-window false-positive rate.
    Layer 2 (CUSUM): target = median pooled clean mean|z|;
        h = cusum_safety_factor × max CUSUM observed on any training animal's clean stream
        — guarantees no training animal produces a clean false alarm and leaves headroom.
    """
    pooled = []
    for s in train_scores_list:
        pooled.extend(s['mean_abs_z'].dropna().values)
    q = 100 * (1 - target_fpr_per_window)
    t_z = float(np.percentile(pooled, q))
    target = float(np.median(pooled))

    max_clean_cusum = 0.0
    for s in train_scores_list:
        x = s['mean_abs_z'].dropna().values.astype(float)
        sc = 0.0
        m = 0.0
        for v in x:
            sc = max(0.0, sc + (v - target))
            m = max(m, sc)
        max_clean_cusum = max(max_clean_cusum, m)
    cusum_h = cusum_safety_factor * max_clean_cusum

    return {
        'threshold_mean_abs_z':  t_z,
        'cusum_target':          target,
        'cusum_h':               cusum_h,
        'cusum_max_clean':       max_clean_cusum,
        'calibration_n':         len(pooled),
        'target_fpr_per_window': target_fpr_per_window,
    }


def load_animal(window_csv_paths: dict, cond_map: dict, artefact_map: dict,
                inj_abs_s: float | None = None,
                video_starts_s: dict | None = None) -> pd.DataFrame:
    """
    window_csv_paths: {video_id: csv_path}
    cond_map:         {video_id: 'clean' | 'diesel'}
    artefact_map:     {video_id: [row_indices]} — windows with camera motion etc.
    video_starts_s:   {video_id: absolute start time in seconds} — optional
    inj_abs_s:        absolute injection time in seconds — optional
    Returns: merged DataFrame with video, cond, artefact flag, abs_t (if starts
             provided), t_inj_s (if injection time provided).
    """
    frames = []
    for vid, p in window_csv_paths.items():
        d = pd.read_csv(p)
        d['video'] = vid
        d['cond'] = cond_map.get(vid, 'unknown')
        d['artefact'] = False
        if vid in artefact_map:
            for i in artefact_map[vid]:
                if i < len(d):
                    d.loc[d.index[i], 'artefact'] = True
        if video_starts_s and vid in video_starts_s:
            d['abs_t'] = d['t_mid'] + video_starts_s[vid]
            if inj_abs_s is not None:
                d['t_inj_s'] = d['abs_t'] - inj_abs_s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
