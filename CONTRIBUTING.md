# Contributing

Thanks for considering a contribution to **Jellyfish**.

## Development setup

```bash
git clone https://github.com/<your-username>/jellyfish.git
cd jellyfish
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## Running tests

```bash
pytest tests/ -v
```

The detector module ships with 14 unit tests covering baseline fitting, the three CUSUM
behaviours (reset on below-target, accumulation, post-alarm reset), threshold calibration,
and the k-consecutive rule. Any change to `src/bews.py` should keep all tests green and ideally
add new tests for new behaviour.

## Coding conventions

- Python ≥ 3.10. Use type hints in module-level functions and class methods.
- Stick to NumPy + SciPy + pandas in `src/bews.py`. The whole point of the simplified architecture
  is that it has no machine-learning dependencies; please don't add any without strong justification.
- Keep functions short and side-effect-free where possible. The detector itself should be pure
  (no I/O, no globals).
- Docstrings: describe what each public function does in plain English, with a one-line summary
  followed by parameter/return semantics.

## What kinds of contribution are welcome

- New behavioural features in `src/pipeline.py` that improve detection on novel pollutants
- Validation runs on additional individuals or different toxicants
- Notebook examples for new use cases (different species, different camera setups)
- Bug fixes and documentation improvements

## What requires discussion before contributing

- Adding machine-learning models or scikit-learn dependencies. Empirical evidence on this dataset
  is that classical statistics outperform ML for this task; we'd want a clear demonstration of
  benefit on held-out data before reverting that decision.
- Changing the detector API (`AnimalBaseline`, `BEWSDetector`, `calibrate_thresholds`). These are
  the deployment contract; please open an issue first.

## Reporting bugs and issues

Please include:
- The version of `jellyfish-bews` (or commit hash)
- Python version and OS
- A minimal reproducer (ideally on the bundled example data)
- The full traceback if there is one
