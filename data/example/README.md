# Example data

These two files are bundled with the repository so the `notebooks/quickstart.ipynb`
demonstration can run without external downloads.

## Files

- **`track.csv`** — raw YOLO output for one recording (PA170007), 22 minutes at ~30 fps.
  Columns: `frame, time_s, area, conf, center_x, center_y, x1, y1, x2, y2`.
  This is the format `pipeline.py` expects as input.

- **`window_features_PA300032.csv`** — the per-window feature table for one diesel-phase
  recording (ExpID 5, the held-out test animal). 23 windows.
  This is the format `bews.py` expects as input.

## What's missing

For the **full multi-animal validation** (`notebooks/bews_demo.ipynb`) you also need:
- `window_features_<VideoFile>.csv` for ExpIDs 2, 3, 4, 5 (about 16 files)
- The experimental database `experiments.xlsx` with columns
  `ExpID, VideoFile, Time, PollutantType, Date`

These are not included in the repository because of size and because they are research data
specific to this study. Contact the maintainers if you need them, or substitute your own data
in the same format.

## Running on your own data

The pipeline is not species- or species-pose-specific in any deep way; it requires a single
moving target with a stable bounding box per frame. To adapt:

1. Train a YOLO detector on your subject and export per-frame detections to `track.csv` in
   the format above.
2. Run `python scripts/run_pipeline.py path/to/track.csv` to get a `window_features.csv`.
3. Repeat for clean-water and exposure phases of each individual; populate `EXPERIMENTS` in
   the demo notebook.
