"""Smoke tests for the BEWS detector. Run with: pytest tests/"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd
import pytest

from bews import (AnimalBaseline, BEWSDetector, calibrate_thresholds,
                  FEATURES, _cusum_positive)


def _synthetic_clean(n: int, seed: int = 0) -> pd.DataFrame:
    """Generate n synthetic clean-water feature windows."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})


def _synthetic_polluted(n: int, shift: float = 3.0, seed: int = 1) -> pd.DataFrame:
    """Generate n synthetic polluted windows shifted by `shift` standard deviations."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.normal(shift, 1, n) for f in FEATURES})


# ----------------------------------------------------------------------
# AnimalBaseline
# ----------------------------------------------------------------------

class TestAnimalBaseline:

    def test_fit_stores_mu_and_sigma(self):
        bl = AnimalBaseline().fit(_synthetic_clean(100))
        assert bl.mu is not None and bl.sd is not None
        assert set(bl.mu.index) == set(FEATURES)
        # Synthetic clean has mean ~0 and std ~1
        assert all(abs(bl.mu) < 0.3)
        assert all(abs(bl.sd - 1) < 0.2)

    def test_score_clean_against_own_baseline_yields_small_z(self):
        clean = _synthetic_clean(200)
        bl = AnimalBaseline().fit(clean)
        out = bl.score(clean)
        assert 'mean_abs_z' in out.columns
        # Clean windows scored against their own baseline should have mean|z| ~ 0.8
        # (expected value of |N(0,1)| is sqrt(2/pi) ≈ 0.798)
        assert 0.5 < out['mean_abs_z'].mean() < 1.0

    def test_score_polluted_yields_large_z(self):
        clean = _synthetic_clean(200)
        polluted = _synthetic_polluted(50, shift=3.0)
        bl = AnimalBaseline().fit(clean)
        clean_z = bl.score(clean)['mean_abs_z']
        poll_z = bl.score(polluted)['mean_abs_z']
        assert poll_z.mean() > clean_z.mean() + 1.5

    def test_zero_variance_feature_is_handled(self):
        """A constant feature must not produce NaN or inf z-scores."""
        clean = _synthetic_clean(100)
        clean['Fp'] = 0.5  # constant
        bl = AnimalBaseline().fit(clean)
        new = _synthetic_clean(10)
        out = bl.score(new)
        assert np.isfinite(out['mean_abs_z']).all()


# ----------------------------------------------------------------------
# BEWSDetector
# ----------------------------------------------------------------------

class TestBEWSDetector:

    def test_clean_stream_produces_no_alarms(self):
        """A long clean stream should produce zero alarms below the threshold."""
        clean = _synthetic_clean(500)
        bl = AnimalBaseline().fit(clean)
        scored = bl.score(clean)
        det = BEWSDetector(threshold_mean_abs_z=2.0, cusum_target=0.8,
                           cusum_h=20.0, k_consecutive=3)
        out = det.apply(scored)
        assert not out['alarm'].any()

    def test_step_change_triggers_layer_1(self):
        """A sustained step change in mean|z| should trigger Layer 1."""
        clean = _synthetic_clean(200)
        polluted = _synthetic_polluted(20, shift=4.0)   # very anomalous
        bl = AnimalBaseline().fit(clean)
        scored_polluted = bl.score(polluted)
        det = BEWSDetector(threshold_mean_abs_z=2.0, cusum_target=0.8,
                           cusum_h=20.0, k_consecutive=3)
        out = det.apply(scored_polluted)
        assert out['alarm_fast'].any()
        # First alarm should occur within the first ~k+2 windows
        first_alarm = out['alarm_fast'].idxmax()
        assert first_alarm < 6

    def test_drift_triggers_layer_2_cusum(self):
        """A small sustained shift, invisible to Layer 1, should trigger CUSUM."""
        clean = _synthetic_clean(200)
        # Shift small enough that no single window exceeds T_z=3.0
        drifting = _synthetic_polluted(60, shift=1.0)
        bl = AnimalBaseline().fit(clean)
        scored_drift = bl.score(drifting)
        det = BEWSDetector(threshold_mean_abs_z=3.0,   # high — Layer 1 won't fire
                           cusum_target=0.8, cusum_h=3.0, k_consecutive=3)
        out = det.apply(scored_drift)
        assert out['alarm_slow'].any(), 'CUSUM layer should fire on sustained drift'
        assert not out['alarm_fast'].any(), 'Layer 1 should remain silent on small shift'

    def test_k_consecutive_suppresses_isolated_outliers(self):
        """A single above-threshold window must not raise an alarm."""
        bl = AnimalBaseline().fit(_synthetic_clean(200))
        # Build a scored frame with one outlier in the middle
        clean_score = bl.score(_synthetic_clean(50))
        outlier = pd.DataFrame({c: [10.0 if c == 'mean_abs_z' else 0.0]
                                for c in clean_score.columns})
        scored = pd.concat([clean_score, outlier, clean_score], ignore_index=True)
        det = BEWSDetector(threshold_mean_abs_z=2.0, cusum_target=0.8,
                           cusum_h=50.0, k_consecutive=3)
        out = det.apply(scored)
        assert not out['alarm_fast'].any(), \
            'Single outlier must not produce a Layer 1 alarm with k=3'


# ----------------------------------------------------------------------
# CUSUM internals
# ----------------------------------------------------------------------

class TestCUSUM:

    def test_cusum_resets_to_zero_on_below_target(self):
        x = np.array([0.0] * 10, dtype=float)
        S, alarm = _cusum_positive(x, target=1.0, h=5.0)
        assert (S == 0).all()
        assert not alarm.any()

    def test_cusum_accumulates_above_target(self):
        x = np.array([2.0] * 10, dtype=float)   # +1 above target each step
        S, alarm = _cusum_positive(x, target=1.0, h=4.5)
        # Should reach S=4 then alarm at S=5 (i=4), then reset
        assert alarm[4]
        assert S[3] == 4.0

    def test_cusum_resets_after_alarm(self):
        x = np.array([5.0] * 4 + [5.0] * 4, dtype=float)
        S, alarm = _cusum_positive(x, target=0.0, h=4.5)
        # Should alarm twice (at i=0 with S=5, and again later after reset)
        assert alarm.sum() >= 2


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

class TestCalibration:

    def test_calibrate_returns_required_keys(self):
        clean1 = _synthetic_clean(100, seed=0)
        clean2 = _synthetic_clean(100, seed=1)
        bl1 = AnimalBaseline().fit(clean1)
        bl2 = AnimalBaseline().fit(clean2)
        scored = [bl1.score(clean1), bl2.score(clean2)]
        cal = calibrate_thresholds(scored, target_fpr_per_window=0.01)
        for k in ('threshold_mean_abs_z', 'cusum_target',
                  'cusum_h', 'cusum_max_clean', 'calibration_n'):
            assert k in cal
        assert cal['threshold_mean_abs_z'] > cal['cusum_target']
        assert cal['cusum_h'] >= cal['cusum_max_clean']

    def test_calibrate_thresholds_then_polluted_triggers_alarm(self):
        """End-to-end: calibrate on clean → apply to polluted → at least one alarm."""
        clean1 = _synthetic_clean(200, seed=0)
        clean2 = _synthetic_clean(200, seed=1)
        clean3 = _synthetic_clean(200, seed=2)

        # Each animal gets its own baseline
        bls = [AnimalBaseline().fit(c) for c in (clean1, clean2, clean3)]
        scored_train = [bl.score(c) for bl, c in zip(bls, (clean1, clean2, clean3))]
        cal = calibrate_thresholds(scored_train, target_fpr_per_window=0.01)

        det = BEWSDetector(
            threshold_mean_abs_z=cal['threshold_mean_abs_z'],
            cusum_target=cal['cusum_target'],
            cusum_h=cal['cusum_h'],
            k_consecutive=3,
        )
        # Apply to a clearly polluted stream from animal 0's perspective
        polluted = _synthetic_polluted(50, shift=2.5, seed=42)
        out = det.apply(bls[0].score(polluted))
        assert out['alarm'].any(), 'Detector must fire on a clearly polluted stream'

    def test_calibrate_thresholds_no_alarm_on_training_clean(self):
        """The safety factor guarantees no clean false alarms on the *training* animals."""
        clean1 = _synthetic_clean(200, seed=0)
        clean2 = _synthetic_clean(200, seed=1)
        clean3 = _synthetic_clean(200, seed=2)

        bls = [AnimalBaseline().fit(c) for c in (clean1, clean2, clean3)]
        scored_train = [bl.score(c) for bl, c in zip(bls, (clean1, clean2, clean3))]
        cal = calibrate_thresholds(scored_train, target_fpr_per_window=0.01,
                                   cusum_safety_factor=1.3)

        det = BEWSDetector(
            threshold_mean_abs_z=cal['threshold_mean_abs_z'],
            cusum_target=cal['cusum_target'],
            cusum_h=cal['cusum_h'],
            k_consecutive=3,
        )
        # No training animal should produce an alarm on its own clean stream
        for bl, clean in zip(bls, (clean1, clean2, clean3)):
            out = det.apply(bl.score(clean))
            assert not out['alarm'].any(), \
                'Calibration must guarantee no false alarms on training animals'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
