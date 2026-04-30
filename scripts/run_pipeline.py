#!/usr/bin/env python3
"""
Run the feature-extraction pipeline on a YOLO track.csv file.

Usage:
    python scripts/run_pipeline.py path/to/track.csv [--output window_features.csv]

The output CSV is the standard per-window feature table consumed by bews.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make src/ importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd  # noqa: E402
from pipeline import run as run_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('track_csv', type=Path, help='Input track.csv from YOLO detector')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output CSV (default: window_features_<stem>.csv next to input)')
    args = parser.parse_args()

    if not args.track_csv.exists():
        parser.error(f'input file not found: {args.track_csv}')

    out_path = args.output or args.track_csv.parent / f'window_features_{args.track_csv.stem}.csv'
    features = run_pipeline(args.track_csv)
    features.to_csv(out_path, index=False)
    print(f'wrote {len(features)} feature windows to {out_path}')


if __name__ == '__main__':
    main()
