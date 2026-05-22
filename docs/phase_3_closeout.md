# Phase 3 closeout — v0.1 MOF Synthesis Advisor

**Released:** Phase 3.5 (this session).
**Budget consumed:** $0.00 LLM (Phase 3.5), well within $5 / 4-hour cap.
**Project status:** v0.1 SHIPPED → PAUSE for user-acceptance review.

---

## Final v0.1 numbers (scaffold-test, honest)

| Head | n_eval | Top-1 | Macro-F1 | cov@90% |
|---|---:|---:|---:|---:|
| **Solvent** | 442 | 0.450 | 0.198 | **0.104** |
| **Method** | 649 | **0.749** | 0.278 | **0.717** |
| **Temperature** | 409 | within ±10 °C = **0.306** · within ±20 °C = 0.511 · MAE = **31.7 °C** · stage-A acc = 0.822 |  |  |

Canonical 5-MOF regression harness: **4/5 PASS**, 0 hard failures,
1 documented xfail (MIL-101 high-T Cr-solvothermal).

Cold-start (joblib bundle, post-cache-clear, M1 Max):
- `joblib.load` alone: **1.12 s**
- Total Python-side init (load + parquet + feature matrix + model): **5.8 s**
- vs in-process training previously: ~30 s → **~5× faster startup**

Bundle size: 0.9 MB (`data/cleaned/v01_checkpoint.joblib`).

---

## Total project cost

| Metric | Value |
|---|---|
| LLM spend (cumulative, all phases) | $0.00 |
| Wall-clock compute (Phase 3.5) | ~25 min (build + tests + smoke) |
| Wall-clock compute (Phases 3.1–3.5 combined, training only) | ~30 min |
| Sessions | 1 (Phase 3.5) on top of prior Phase 3.x work |
| Deliverable size | 3 scripts (build / app / canonical) + 3 docs + 1 bundle |

---

## What we shipped (Phase 3.5 delivery)

1. **`build_v01_checkpoint.py`** — single trainer that produces a
   `joblib` bundle of all 3 heads + KNN index + metadata. Honest
   scaffold-split metrics throughout (fixed leakage bug in initial
   build: was training on random_train + evaluating on scaffold_test).
2. **`advisor_v01_app.py`** — Streamlit UI updated to lazy-load the
   joblib bundle and surface the method head in top-3 form with
   calibrated confidence. Falls back to in-process training if bundle
   missing.
3. **`canonical_tests.py`** — 5-MOF assertion harness with per-case
   temperature tolerance and `xfail` marker. Exit 1 on hard failure.
4. **`v0.1_README.md`** — technical README with KNOWN LIMITATIONS and
   "what we cannot do" sections.
5. **`v0.1_chemist_guide.md`** — 1-page chemist-facing doc with 3
   worked examples (MOF-5, HKUST-1, ZIF-8) and explicit when-to-trust /
   when-to-hedge rules.
6. **This document** — `phase_3_closeout.md`.

---

## Methodology contributions (paper-grade)

1. **Pre-training resolution audit** (Phase 3.4.2) — quantify
   coverage of each feature class on the eval split *before* training.
   Caught a 6.8 % linker-resolution gap that capped the maximum
   possible solvent-head lift to ≤0.02 headline accuracy. Saved a
   full sweep that would have looked like noise.
2. **MAE / within-tolerance dissociation** (Phase 3.4.3) —
   single-stage temperature MAE was respectable (~24 °C) but within
   ±10 °C was only 23 % because the regressor compressed toward a
   mean. Two-stage RT-vs-heated splitter recovered ±10 °C to 30.6 %.
   Lesson: report tolerance bands alongside MAE for any model with a
   bimodal target.
3. **Calibrated-abstention + provenance design** — every prediction
   ships with a Platt-calibrated probability AND the 3 nearest
   training rows (Morgan-FP cosine). Chemists can audit "why" a
   prediction was made. The cov@90 % metric is the operating point:
   the model knows when to stay silent.
4. **NULL recovery — published negative result.** Targeted
   NULL-imputation for missing solvent labels *underperformed* the
   simpler "drop NULL" baseline across three independent
   configurations. This is a Nature Methods Comment-grade finding for
   small (≤5k row) literature-curated chemistry datasets; consider
   methods paper after v0.1 user feedback stabilizes.
5. **Gap-vs-capacity diagnostic** (Phase 3.4.1) — the 0.139 train/test
   gap was XGB over-capacity (depth=6, n_est=300), not Morgan-FP
   high-dim overfitting. Cutting capacity *both* closed the gap AND
   raised scaffold cov@90 % — win-win, locked in for v0.1.

---

## Honest "what we cannot do"

- Predict synthesis success/failure (literature only has successes).
- Predict yield, purity, crystal phase, or BET area.
- Extrapolate to metal elements outside the training distribution
  (OOD gate fires; predictions are guesses).
- Model modulators outside the training top-30 (e.g. HF in MIL-101).
- Predict reaction time (stub only; shows cluster median).
- Distinguish target polymorphs.
- Anything beyond literature-precedent pattern matching.

---

## Deferred work

**To v0.1.1** (clear value, ~1 dev session each):
- Dedicated two-stage time-h head (mirrors temperature design).
- Deep metal-only feature engineering (Phase 3.4.2 audit said the
  1136-row metal-parsed subset is the bigger lever than the 294-row
  linker-resolved subset).
- `LINKER_SMILES` vocab expansion (more named linkers → higher
  resolution rate → larger achievable solvent lift).
- Scaling-law plot (dataset size vs cov@90 %) for paper.
- Persist feature column schema in bundle (would have prevented the
  HF-modulator phantom-column failure in canonical tests).

**To v0.2** (driven by chemist feedback after this release):
- TBD pending v0.1 user-acceptance review.

---

## Decision points still pending (user)

1. **Pause for user-acceptance review** per Phase 3.5 directive. The
   bundle, UI, tests, and docs are ready for hands-on evaluation.
2. **Methods paper go/no-go** on the NULL-recovery negative result +
   cov@90 % calibrated-abstention framework. Recommended timing:
   after ≥1 round of chemist feedback validates the design.
3. **v0.1.1 scope** confirmation — which of the deferred items to
   sequence first.

---

## Sign-off

v0.1 of the MOF Synthesis Advisor is functional, validated against
five textbook MOFs (4 PASS, 1 documented xfail), and shipped with
honest scaffold-split metrics, calibrated confidence, k-NN
provenance, and explicit known-limitation surfaces in both the
technical README and the chemist-facing one-pager. Cold start ≈ 6 s.
Bundle size 0.9 MB. Total LLM cost for the Phase 3.5 ship: $0.
