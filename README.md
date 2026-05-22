# MOF Automatic ML — v0.1

A literature-trained machine-learning advisor that suggests synthesis conditions
(solvent cluster, reaction temperature, synthesis method) for new metal-organic
frameworks given a metal source and linker.

This repository contains the **shipped v0.1 model**: training script, inference
app, acceptance harness, persisted model bundle, and the data files needed at
inference time.

## What v0.1 is

Three XGBoost heads, calibrated with Platt scaling on a held-out validation
fold, evaluated on a scaffold-split test fold:

| Head             | Top-1  | Coverage @ 90 % precision | n_test |
|------------------|--------|---------------------------|--------|
| Solvent cluster  | 0.450  | 0.104                     | 442    |
| Temperature      | within ±10 °C: 0.306, ±20 °C: 0.511, MAE 31.67 °C | — | 409 |
| Method           | 0.749  | 0.717                     | 649    |

Features: numeric pipeline + Morgan-2 fingerprints over resolved linkers
(2 203 columns). Solvent + method heads are PU-style classifiers with
calibrated abstention; temperature is two-stage (RT vs heated → regression
on heated subset). A Morgan k-NN retrieval index is bundled for the
"nearest literature precedents" panel in the UI.

See [docs/v0.1_README.md](docs/v0.1_README.md) for the full technical
README, known limitations, and methodology notes.

For chemists evaluating the tool, see
[docs/v0.1_chemist_guide.md](docs/v0.1_chemist_guide.md) — a one-page guide
with trust / hedge / ignore rules and three worked examples.

For the project close-out (final numbers, what shipped, what we cannot do,
deferred work), see [docs/phase_3_closeout.md](docs/phase_3_closeout.md).

## Repository layout

```
MOF-automatic-ML-v1/
├── README.md                        ← you are here
├── requirements.txt
├── scripts/
│   ├── build_v01_checkpoint.py      ← trains all heads, dumps joblib bundle
│   ├── advisor_v01_app.py           ← Streamlit advisor UI (loads bundle)
│   ├── canonical_tests.py           ← 5-MOF acceptance test harness
│   ├── xgb_baseline.py              ← feature-matrix builder (import dep)
│   └── build_chem_features.py       ← chem features (import dep)
├── data/cleaned/
│   ├── v01_checkpoint.joblib        ← shipped model bundle (~0.9 MB)
│   ├── recipes.parquet              ← required for feature-schema seed
│   ├── chem_features.parquet
│   ├── split_assignments.parquet
│   └── solvent_target.parquet
└── docs/
    ├── v0.1_README.md
    ├── v0.1_chemist_guide.md
    └── phase_3_closeout.md
```

## Quickstart

```bash
# 1. install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# 2. run the 5-MOF acceptance harness (no GUI; ~6 s)
python3 scripts/canonical_tests.py
#   expected: 4 PASS / 1 XFAIL (MIL-101 temperature)

# 3. launch the Streamlit advisor
python3 -m streamlit run scripts/advisor_v01_app.py
#   open http://localhost:8501 in a browser
```

The first launch reads the joblib bundle in ~1 s (vs ~30 s if it had to
retrain). If `data/cleaned/v01_checkpoint.joblib` is missing, the app falls
back to in-process training of the solvent + temperature heads (no method
head in fallback).

## Rebuilding the model

```bash
python3 scripts/build_v01_checkpoint.py
```

Reads the parquets under `data/cleaned/`, retrains all three heads on the
scaffold-train fold, calibrates on val, evaluates on test, and overwrites
`data/cleaned/v01_checkpoint.joblib`. Final metrics are written to stdout.

## What this tool will NOT do

- Predict per-trial probability of synthesis success (training data is
  positive-only literature reports — no failures).
- Predict reaction time (current `time_h` head is a stub).
- Handle modulators outside the top-30 vocabulary.
- Recommend conditions for chemistries with no nearest-neighbour support in
  the corpus (the UI flags these as out-of-distribution).

See [docs/v0.1_README.md](docs/v0.1_README.md) "Known limitations" for the
full list.

## License

Research code, released as-is. No warranty. Underlying corpus is derived
from a published MOF synthesis-conditions dataset; see project docs for
attribution.
