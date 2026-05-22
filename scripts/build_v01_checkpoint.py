#!/usr/bin/env python3
"""Phase 3.5 — Train + persist v0.1 model checkpoint as joblib bundle.

Trains all three heads on the full train split using the v0.1 shipping
config and dumps a single joblib file the Streamlit app loads in ~1 s
instead of retraining for ~30 s on every restart.

Heads bundled:
  1. solvent classifier  (CalibratedClassifierCV over XGBClassifier)
  2. temperature stage-A (XGBClassifier, RT vs heated)
  3. temperature stage-B (XGBRegressor on heated subset)
  4. method classifier   (CalibratedClassifierCV over XGBClassifier)
     -- target = method_llm from Step 3.1.5c (10 classes incl. unknown)

Also bundles:
  - solv_inv, meth_inv  (int → label dicts)
  - knn_M_norm, knn_has_linker  (Morgan k-NN index)
  - X_train_shape, n_classes_*, MODEL_CONFIG, build_metadata

Usage:
  python3 -u Claude_auto_MOF_script/scripts/build_v01_checkpoint.py

Output:
  data/cleaned/v01_checkpoint.joblib  (single bundle)
  other/v01_checkpoint_metrics.md     (scaffold cov@90% & top-1 per head)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned"
OTHER = ROOT / "other"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xgb_baseline import (  # noqa: E402
    build_feature_matrix, precision_coverage_curve, coverage_at_precision,
)

OUT_BUNDLE = DATA / "v01_checkpoint.joblib"
OUT_REPORT = OTHER / "v01_checkpoint_metrics.md"

MODEL_CONFIG = {
    "version": "v3.4.1-shipping",
    "depth": 4,
    "n_est": 150,
    "lr": 0.1,
    "feature_pipeline": "base+chem (2203 cols)",
    "rt_threshold_C": 30.0,
    "temp_range": (0.0, 300.0),
}

NULL = "NULL"


def build_X(df: pd.DataFrame, chem: pd.DataFrame):
    X_base, _ = build_feature_matrix(df)
    if sp.issparse(X_base):
        X_base = X_base.toarray()
    X_base = X_base.astype(np.float32)
    morgan_cols = [c for c in chem.columns if c.startswith("morgan_")]
    nm_cols = [c for c in chem.columns if not c.startswith("morgan_")]
    M = chem[morgan_cols].to_numpy(dtype=np.float32)
    NM = chem[nm_cols].to_numpy(dtype=np.float32)
    return np.hstack([X_base, NM, M]).astype(np.float32)


def fit_calibrated(X_tr, y_tr_i, X_val, y_val_i, n_class: int):
    base = XGBClassifier(
        n_estimators=MODEL_CONFIG["n_est"], max_depth=MODEL_CONFIG["depth"],
        learning_rate=MODEL_CONFIG["lr"], tree_method="hist",
        n_jobs=-1, verbosity=0,
        objective="multi:softprob", num_class=n_class,
    )
    base.fit(X_tr, y_tr_i)
    if len(y_val_i) >= 20 and len(set(y_val_i.tolist())) >= 2:
        try:
            cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
            cal.fit(X_val, y_val_i)
            return cal
        except Exception as e:
            print(f"  [warn] calibration failed: {e!r} — using base")
    return base


def eval_classifier(model, le_inv, X_te, y_te) -> dict:
    classes_known = set(le_inv.values())
    mask = np.array([y in classes_known for y in y_te])
    if mask.sum() == 0:
        return {"n_eval": 0}
    X_te_k, y_te_k = X_te[mask], y_te[mask]
    proba = model.predict_proba(X_te_k)
    pred_i = np.argmax(proba, axis=1)
    pred = np.array([le_inv[i] for i in pred_i])
    conf = proba.max(axis=1)
    correct1 = (pred == y_te_k).astype(int)
    curve = precision_coverage_curve(correct1, conf)
    cov90 = (None if curve.empty
             else round(coverage_at_precision(curve, 0.90), 4))
    return {
        "n_eval": int(mask.sum()),
        "top1": round(float(correct1.mean()), 4),
        "macro_f1": round(float(
            f1_score(y_te_k, pred, average="macro", zero_division=0)), 4),
        "cov@90%": cov90,
    }


def main():
    t0 = time.time()
    print("Loading data…")
    df = pd.read_parquet(DATA / "recipes.parquet").reset_index(drop=True)
    chem = pd.read_parquet(DATA / "chem_features.parquet").reset_index(drop=True)
    assert "method_llm" in df.columns, "recipes.parquet missing method_llm"
    splits = (pd.read_parquet(DATA / "split_assignments.parquet")
              .set_index("row_idx").reindex(range(len(df))).reset_index())

    X = build_X(df, chem)
    print(f"X shape: {X.shape}")

    metrics: dict = {}
    bundle: dict = {
        "model_config": MODEL_CONFIG,
        "X_train_shape": tuple(int(s) for s in X.shape),
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- SOLVENT (scaffold-train → scaffold-test; ships scaffold model for honest metrics) ----
    print("\n[1/4] solvent classifier")
    y = df["solvent_cluster"].to_numpy()
    sps_scaf = splits["split_scaffold"].to_numpy()

    tr = (sps_scaf == "train") & (y != NULL)
    val = (sps_scaf == "val") & (y != NULL)
    te = (sps_scaf == "test") & (y != NULL)

    le = {c: i for i, c in enumerate(sorted(set(y[tr])))}
    inv = {i: c for c, i in le.items()}
    y_tr_i = np.array([le[v] for v in y[tr]])
    y_val_known = np.array([v in le for v in y[val]])
    y_val_i = np.array([le[v] for v in y[val][y_val_known]])

    solv_model = fit_calibrated(X[tr], y_tr_i, X[val][y_val_known], y_val_i,
                                 n_class=len(le))
    bundle["solvent_model"] = solv_model
    bundle["solvent_inv"] = inv
    metrics["solvent"] = eval_classifier(solv_model, inv, X[te], y[te])
    print(f"  scaffold: {metrics['solvent']}")

    # ---- TEMPERATURE (two-stage, eval on scaffold) ----
    print("\n[2/4] temperature two-stage")
    yt = df["temperature_C"].to_numpy()
    lo, hi = MODEL_CONFIG["temp_range"]
    valid = (~np.isnan(yt)) & (yt >= lo) & (yt <= hi)
    tr_t = (sps_scaf == "train") & valid
    te_t = (sps_scaf == "test") & valid

    is_heated = (yt[tr_t] > MODEL_CONFIG["rt_threshold_C"]).astype(int)
    clf = XGBClassifier(
        n_estimators=MODEL_CONFIG["n_est"], max_depth=MODEL_CONFIG["depth"],
        learning_rate=MODEL_CONFIG["lr"], tree_method="hist",
        n_jobs=-1, verbosity=0, objective="binary:logistic",
    )
    clf.fit(X[tr_t], is_heated)
    reg = XGBRegressor(
        n_estimators=MODEL_CONFIG["n_est"], max_depth=MODEL_CONFIG["depth"],
        learning_rate=MODEL_CONFIG["lr"], tree_method="hist",
        n_jobs=-1, verbosity=0, objective="reg:squarederror",
    )
    reg.fit(X[tr_t][is_heated == 1], yt[tr_t][is_heated == 1])
    bundle["temp_stage_a"] = clf
    bundle["temp_stage_b"] = reg

    p_heated_te = clf.predict_proba(X[te_t])[:, 1]
    pred_t = np.where(p_heated_te > 0.5, reg.predict(X[te_t]), 25.0)
    y_te_t = yt[te_t]
    within10 = float(np.mean(np.abs(pred_t - y_te_t) <= 10.0))
    within20 = float(np.mean(np.abs(pred_t - y_te_t) <= 20.0))
    mae = float(np.mean(np.abs(pred_t - y_te_t)))
    stage_a_acc = float(((p_heated_te > 0.5) ==
                         (y_te_t > MODEL_CONFIG["rt_threshold_C"])).mean())
    metrics["temperature"] = {
        "n_eval": int(te_t.sum()),
        "within_pm10C": round(within10, 4),
        "within_pm20C": round(within20, 4),
        "MAE": round(mae, 2),
        "stage_a_acc": round(stage_a_acc, 4),
    }
    print(f"  scaffold: {metrics['temperature']}")

    # ---- METHOD (eval on scaffold) ----
    print("\n[3/4] method classifier")
    ym = df["method_llm"].fillna("unknown").to_numpy()
    tr_m = (sps_scaf == "train")
    val_m = (sps_scaf == "val")
    te_m = (sps_scaf == "test")

    le_m = {c: i for i, c in enumerate(sorted(set(ym[tr_m])))}
    inv_m = {i: c for c, i in le_m.items()}
    y_tr_mi = np.array([le_m[v] for v in ym[tr_m]])
    y_val_known_m = np.array([v in le_m for v in ym[val_m]])
    y_val_mi = np.array([le_m[v] for v in ym[val_m][y_val_known_m]])

    meth_model = fit_calibrated(X[tr_m], y_tr_mi,
                                X[val_m][y_val_known_m], y_val_mi,
                                n_class=len(le_m))
    bundle["method_model"] = meth_model
    bundle["method_inv"] = inv_m
    metrics["method"] = eval_classifier(meth_model, inv_m, X[te_m], ym[te_m])
    print(f"  scaffold: {metrics['method']}")

    # ---- KNN INDEX ----
    print("\n[4/4] KNN index over Morgan-2048")
    morgan_cols = [c for c in chem.columns if c.startswith("morgan_")]
    M = chem[morgan_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    bundle["knn_M_norm"] = (M / norms[:, None]).astype(np.float32)
    bundle["knn_has_linker"] = (M.sum(axis=1) > 0)

    bundle["metrics_scaffold"] = metrics

    print(f"\nDumping bundle → {OUT_BUNDLE}")
    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT_BUNDLE, compress=3)
    size_mb = OUT_BUNDLE.stat().st_size / 1024**2
    print(f"  size: {size_mb:.1f} MB")

    # Report
    OTHER.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v0.1 model checkpoint metrics",
        "",
        f"Built: {bundle['build_timestamp']}",
        f"Bundle: `{OUT_BUNDLE.relative_to(ROOT.parent)}` ({size_mb:.1f} MB)",
        f"X shape: {bundle['X_train_shape']}",
        f"Config: depth={MODEL_CONFIG['depth']}, n_est={MODEL_CONFIG['n_est']},"
        f" feature_pipeline={MODEL_CONFIG['feature_pipeline']}",
        "",
        "## Scaffold-split metrics",
        "",
        "| head | n_eval | metric | value |",
        "|---|---:|---|---:|",
    ]
    s = metrics["solvent"]
    lines += [
        f"| solvent | {s['n_eval']} | top-1 | {s['top1']} |",
        f"| solvent | {s['n_eval']} | macro-F1 | {s['macro_f1']} |",
        f"| solvent | {s['n_eval']} | cov@90% | {s['cov@90%']} |",
    ]
    t = metrics["temperature"]
    lines += [
        f"| temperature | {t['n_eval']} | within ±10°C | {t['within_pm10C']} |",
        f"| temperature | {t['n_eval']} | within ±20°C | {t['within_pm20C']} |",
        f"| temperature | {t['n_eval']} | MAE | {t['MAE']} |",
        f"| temperature | {t['n_eval']} | stage-A acc | {t['stage_a_acc']} |",
    ]
    m = metrics["method"]
    lines += [
        f"| method | {m['n_eval']} | top-1 | {m['top1']} |",
        f"| method | {m['n_eval']} | macro-F1 | {m['macro_f1']} |",
        f"| method | {m['n_eval']} | cov@90% | {m['cov@90%']} |",
    ]
    lines += ["", f"Build wall-clock: {time.time() - t0:.1f}s"]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_REPORT}")
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
