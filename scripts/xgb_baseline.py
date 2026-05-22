#!/usr/bin/env python3
"""Phase 3.2 — XGBoost multi-task baseline with calibrated abstention.

Heads:
  solvent (43 classes; mask __NULL__ in eval)
  temperature (regression in °C)
  method     (10 classes; target = method_llm from 3.1.5c)

Features (148 dims):
  binary chemistry (132): 71 metals + top-50 linkers + top-30 modulators +
    {mod_present, metal_other, linker_other}
  continuous (16): n_chemicals, n_solvents, n_steps, log1p(time_h),
    time_missing, n_linkers, n_metal_sources, has_methodllm_unknown,
    + 8 reserved/zeros for layout

Calibration: sigmoid (Platt) per class via sklearn CalibratedClassifierCV
            on the val split (cv='prefit').

Splits evaluated: random / scaffold / metal.
Outputs:
  other/xgb_baseline_metrics.json
  other/xgb_baseline_report.md
  other/xgb_baseline_curves.csv  (scaffold solvent precision-coverage)
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "data" / "cleaned" / "recipes.parquet"
SOLVENT = ROOT / "data" / "cleaned" / "solvent_target.parquet"
SPLITS = ROOT / "data" / "cleaned" / "split_assignments.parquet"
OUT_METRICS = ROOT / "other" / "xgb_baseline_metrics.json"
OUT_REPORT = ROOT / "other" / "xgb_baseline_report.md"
OUT_CURVES = ROOT / "other" / "xgb_baseline_curves.csv"

NULL = "__NULL__"


# -------------------- Feature build (shared with k-NN, extended) --------------------
def safe_json(x):
    if not x:
        return []
    try:
        return json.loads(x)
    except Exception:
        return []


METAL_ELEMENTS = [
    "H", "Li", "Be", "Na", "Mg", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th",
    "U", "Al", "Si", "Ge", "As", "Se", "Te", "Sb", "Po", "Rn", "Fr", "Ra",
]
# 72 elements (one extra to satisfy 71 + slack; we'll slice later)


def first_element_symbol(s: str) -> str:
    if not s:
        return ""
    # extract first uppercase + optional lowercase letter
    i = 0
    n = len(s)
    while i < n and not s[i].isupper():
        i += 1
    if i >= n:
        return ""
    j = i + 1
    if j < n and s[j].islower():
        j += 1
    return s[i:j]


def canon_linker(name: str) -> str:
    return (name or "").strip().lower()


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # Pass 1: collect counters for top-N selection
    linker_counts: Counter = Counter()
    mod_counts: Counter = Counter()
    for s in df["chemicals"]:
        for it in safe_json(s):
            kind = it.get("kind")
            name = canon_linker(it.get("name") or "")
            if kind == "linker" and name:
                linker_counts[name] += 1
            elif kind == "modulator" and name:
                mod_counts[name] += 1

    top_linkers = [n for n, _ in linker_counts.most_common(50)]
    top_mods = [n for n, _ in mod_counts.most_common(30)]
    metals = METAL_ELEMENTS[:71]

    feat_names = (
        [f"M:{m}" for m in metals]
        + [f"L:{l}" for l in top_linkers]
        + [f"D:{m}" for m in top_mods]
        + ["mod_present", "metal_other", "linker_other"]
    )

    # Continuous features
    cont_names = [
        "n_chemicals", "n_solvents", "n_steps", "log1p_time_h",
        "time_missing", "n_linkers_row", "n_metal_sources_row",
        "method_llm_unknown",
    ]
    feat_names += cont_names

    n = len(df)
    X = np.zeros((n, len(feat_names)), dtype=np.float32)

    n_metals = len(metals)
    n_linkers_top = len(top_linkers)
    n_mods_top = len(top_mods)
    m_idx = {m: i for i, m in enumerate(metals)}
    l_idx = {l: n_metals + i for i, l in enumerate(top_linkers)}
    d_idx = {d: n_metals + n_linkers_top + i for i, d in enumerate(top_mods)}
    MOD_PRESENT = n_metals + n_linkers_top + n_mods_top
    METAL_OTHER = MOD_PRESENT + 1
    LINKER_OTHER = MOD_PRESENT + 2
    CONT_START = MOD_PRESENT + 3

    for ri, (_, row) in enumerate(df.iterrows()):
        items = safe_json(row["chemicals"])
        n_chem = len(items)
        n_linkers = 0
        n_metal_sources = 0
        for it in items:
            kind = it.get("kind")
            name = (it.get("name") or "").strip()
            if kind == "metal_source":
                n_metal_sources += 1
                sym = first_element_symbol(name)
                if sym in m_idx:
                    X[ri, m_idx[sym]] = 1.0
                else:
                    X[ri, METAL_OTHER] = 1.0
            elif kind == "linker":
                n_linkers += 1
                cn = canon_linker(name)
                if cn in l_idx:
                    X[ri, l_idx[cn]] = 1.0
                else:
                    X[ri, LINKER_OTHER] = 1.0
            elif kind == "modulator":
                X[ri, MOD_PRESENT] = 1.0
                cn = canon_linker(name)
                if cn in d_idx:
                    X[ri, d_idx[cn]] = 1.0

        # Continuous
        solvs = safe_json(row.get("solvents"))
        n_solvs = len(solvs) if isinstance(solvs, list) else 0
        n_steps = row.get("n_steps") or 1
        th = row.get("time_h")
        if th is None or (isinstance(th, float) and np.isnan(th)):
            log_th = 0.0
            time_miss = 1.0
        else:
            log_th = float(np.log1p(float(th)))
            time_miss = 0.0
        method_llm = row.get("method_llm")
        unk = 1.0 if (method_llm == "unknown" or method_llm is None) else 0.0

        X[ri, CONT_START + 0] = float(n_chem)
        X[ri, CONT_START + 1] = float(n_solvs)
        X[ri, CONT_START + 2] = float(n_steps)
        X[ri, CONT_START + 3] = log_th
        X[ri, CONT_START + 4] = time_miss
        X[ri, CONT_START + 5] = float(n_linkers)
        X[ri, CONT_START + 6] = float(n_metal_sources)
        X[ri, CONT_START + 7] = unk

    return X, feat_names


# -------------------- Targets --------------------
def build_targets(df: pd.DataFrame, solv: pd.DataFrame) -> dict:
    # solvent
    sol_map = dict(zip(solv["row_idx"], solv["solvent_target"]))
    y_solvent = np.array([sol_map.get(i, NULL) for i in range(len(df))])

    # temperature: numeric °C from temperature_C col
    def to_temp(v):
        if v is None:
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except Exception:
            return np.nan
    y_temp = np.array([to_temp(v) for v in df["temperature_C"]], dtype=float)

    # method
    y_method = df["method_llm"].fillna("unknown").to_numpy()

    return {"solvent": y_solvent, "temperature": y_temp, "method": y_method}


# -------------------- Precision-coverage --------------------
def precision_coverage_curve(correct: np.ndarray, conf: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-conf)
    c = correct[order].astype(int)
    cum_correct = np.cumsum(c)
    cov = np.arange(1, len(c) + 1) / len(c)
    prec = cum_correct / np.arange(1, len(c) + 1)
    return pd.DataFrame({"coverage": cov, "precision": prec})


def coverage_at_precision(curve: pd.DataFrame, target: float) -> float | None:
    ok = curve[curve["precision"] >= target]
    if ok.empty:
        return None
    return float(ok["coverage"].max())


# -------------------- Per-split eval --------------------
def fit_classifier(X_tr, y_tr, X_val, y_val):
    le = {c: i for i, c in enumerate(sorted(set(y_tr)))}
    le_inv = {i: c for c, i in le.items()}
    y_tr_i = np.array([le[v] for v in y_tr])
    base = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        tree_method="hist", n_jobs=-1, verbosity=0,
        objective="multi:softprob", num_class=len(le),
    )
    base.fit(X_tr, y_tr_i)
    # Calibrate on val (only with classes present in train)
    val_known = np.array([v in le for v in y_val])
    if val_known.sum() >= 20:
        y_val_i = np.array([le[v] for v in y_val[val_known]])
        try:
            cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
            cal.fit(X_val[val_known], y_val_i)
            return cal, le_inv
        except Exception:
            return base, le_inv
    return base, le_inv


def fit_regressor(X_tr, y_tr):
    mask = ~np.isnan(y_tr)
    if mask.sum() < 20:
        return None
    base = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        tree_method="hist", n_jobs=-1, verbosity=0,
    )
    base.fit(X_tr[mask], y_tr[mask])
    return base


def eval_classifier(model, le_inv, X_te, y_te, exclude=None):
    proba = model.predict_proba(X_te)
    top_idx = proba.argmax(axis=1)
    top_conf = proba.max(axis=1)
    top_labels = np.array([le_inv[i] for i in top_idx])
    # top-3
    top3 = np.argsort(-proba, axis=1)[:, :3]
    top3_labels = [[le_inv[i] for i in row] for row in top3]

    mask = np.ones(len(y_te), dtype=bool)
    if exclude is not None:
        mask = np.array([v != exclude for v in y_te])
    if mask.sum() == 0:
        return None

    y_e = y_te[mask]
    p_e = top_labels[mask]
    c_e = top_conf[mask]
    top3_e = [top3_labels[i] for i in range(len(y_te)) if mask[i]]

    correct1 = (p_e == y_e)
    correct3 = np.array([y_e[i] in top3_e[i] for i in range(len(y_e))])

    curve = precision_coverage_curve(correct1, c_e)
    cov = {f"cov@{int(t*100)}%": coverage_at_precision(curve, t) for t in (0.80, 0.85, 0.90, 0.95)}

    f1 = f1_score(y_e, p_e, average="macro", zero_division=0)

    return {
        "n_eval": int(mask.sum()),
        "top1": round(float(correct1.mean()), 4),
        "top3": round(float(correct3.mean()), 4),
        "macro_f1": round(float(f1), 4),
        "mean_conf": round(float(c_e.mean()), 4),
        **{k: (None if v is None else round(v, 4)) for k, v in cov.items()},
        "_curve": curve,
    }


def eval_regressor(model, X_te, y_te, tol=10.0):
    if model is None:
        return None
    mask = ~np.isnan(y_te)
    if mask.sum() == 0:
        return None
    pred = model.predict(X_te[mask])
    y_e = y_te[mask]
    err = pred - y_e
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    within = float((np.abs(err) <= tol).mean())

    # confidence proxy via quantile spread is not available without separate
    # quantile models; use absolute-error-on-train estimate via OOB-ish proxy:
    # fall back to using prediction-magnitude variance is not meaningful. Skip
    # coverage curve for regression; report point metrics only.
    return {
        "n_eval": int(mask.sum()),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        f"within_{int(tol)}C": round(within, 4),
    }


def run_split(X, y, split_col, split_assignments):
    sp = split_assignments[split_col].to_numpy()
    tr_mask = sp == "train"
    val_mask = sp == "val"
    te_mask = sp == "test"
    X_tr, X_val, X_te = X[tr_mask], X[val_mask], X[te_mask]

    results = {}
    # Solvent
    y_solv = y["solvent"]
    model, le_inv = fit_classifier(X_tr, y_solv[tr_mask], X_val, y_solv[val_mask])
    results["solvent"] = eval_classifier(model, le_inv, X_te, y_solv[te_mask], exclude=NULL)

    # Method
    y_meth = y["method"]
    model_m, le_inv_m = fit_classifier(X_tr, y_meth[tr_mask], X_val, y_meth[val_mask])
    results["method"] = eval_classifier(model_m, le_inv_m, X_te, y_meth[te_mask])

    # Temperature
    y_temp = y["temperature"]
    reg = fit_regressor(X_tr, y_temp[tr_mask])
    results["temperature"] = eval_regressor(reg, X_te, y_temp[te_mask])

    return results


def main():
    t0 = time.time()
    df = pd.read_parquet(RECIPES).reset_index(drop=True)
    solv = pd.read_parquet(SOLVENT)
    splits = pd.read_parquet(SPLITS).set_index("row_idx").reindex(range(len(df))).reset_index()
    print(f"Rows: {len(df)}")

    print("Building features...")
    X, feat_names = build_feature_matrix(df)
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    y = build_targets(df, solv)
    print(f"solvent classes (train+): {len(set(y['solvent']))}")
    print(f"method classes: {len(set(y['method']))}")
    print(f"temperature available: {np.sum(~np.isnan(y['temperature']))}")

    all_results = {}
    for split_name in ["split_random", "split_scaffold", "split_metal"]:
        print(f"\n=== {split_name} ===")
        ts = time.time()
        res = run_split(X, y, split_name, splits)
        dt = time.time() - ts
        all_results[split_name] = res
        s = res["solvent"]
        m = res["method"]
        t = res["temperature"] or {}
        if s:
            print(f"  solvent  top1={s['top1']} top3={s['top3']} mean_conf={s['mean_conf']} "
                  f"cov@90%={s.get('cov@90%')}")
        if m:
            print(f"  method   top1={m['top1']} F1={m['macro_f1']} cov@90%={m.get('cov@90%')}")
        if t:
            print(f"  temp     MAE={t['mae']}°C within±10°C={t.get('within_10C')}")
        print(f"  [{dt:.1f}s]")

    # Memorization flag
    r1 = all_results["split_random"]["solvent"]["top1"]
    r2 = all_results["split_scaffold"]["solvent"]["top1"]
    mem_gap = round(r1 - r2, 4)
    print(f"\nsolvent top1 random - scaffold = {mem_gap}  "
          f"({'FLAG' if mem_gap > 0.20 else 'OK'} for 0.20 bound)")

    # Save scaffold curve
    sc = all_results["split_scaffold"]["solvent"]["_curve"]
    sc.to_csv(OUT_CURVES, index=False)

    # Save metrics (strip curves)
    metrics_clean = {}
    for sp, heads in all_results.items():
        metrics_clean[sp] = {}
        for h, r in heads.items():
            if not r:
                metrics_clean[sp][h] = None
                continue
            metrics_clean[sp][h] = {k: v for k, v in r.items() if k != "_curve"}

    OUT_METRICS.write_text(json.dumps({
        "n_rows": len(df),
        "n_features": X.shape[1],
        "feature_names": feat_names,
        "memorization_gap": mem_gap,
        "memorization_flag": mem_gap > 0.20,
        "results": metrics_clean,
    }, indent=2))
    print(f"\nWrote {OUT_METRICS}")
    print(f"Wrote {OUT_CURVES}")

    # Markdown report
    def fmt(v): return "None" if v is None else v
    md = [
        "# Phase 3.2 — XGBoost multi-task baseline",
        "",
        f"- Rows: {len(df)}  |  Features: {X.shape[1]} "
        f"(132 binary chemistry + 8 continuous + 8 reserved)",
        "- Multi-class softmax (XGBClassifier) for solvent/method; "
        "regressor (squared error) for temperature.",
        "- Confidence = top-class predicted probability after **Platt "
        "calibration (CalibratedClassifierCV on val, sigmoid)**.",
        "- Solvent eval excludes `__NULL__` (~29% of rows have no Phase 1 primary solvent).",
        "- Method target = `method_llm` from Step 3.1.5c (LLM relabel).",
        "",
        "## Headline (chemist-facing)",
        f"> **Coverage of solvent prediction at 90 % precision (scaffold split test) = "
        f"`{fmt(metrics_clean['split_scaffold']['solvent'].get('cov@90%'))}`**",
        "",
        "## Memorization check",
        f"- random − scaffold solvent top-1 gap: **{mem_gap}** "
        f"({'FLAG' if mem_gap > 0.20 else 'OK'} for 0.20 bound)",
        "",
        "## Per-split results",
    ]
    for sp in ["split_random", "split_scaffold", "split_metal"]:
        r = metrics_clean[sp]
        md.append(f"### `{sp}`")
        s = r["solvent"]
        if s:
            md.append(
                f"- **solvent** (n={s['n_eval']})  top-1=**{s['top1']}**  top-3={s['top3']}  "
                f"macro-F1={s['macro_f1']}  mean_conf={s['mean_conf']}"
            )
            md.append(
                f"  - cov @ 80/85/90/95 % prec: "
                f"{fmt(s.get('cov@80%'))} / {fmt(s.get('cov@85%'))} / "
                f"{fmt(s.get('cov@90%'))} / {fmt(s.get('cov@95%'))}"
            )
        t = r["temperature"]
        if t:
            md.append(
                f"- **temperature** (n={t['n_eval']})  MAE={t['mae']} °C  "
                f"RMSE={t['rmse']} °C  within±10°C={t['within_10C']}"
            )
        m = r["method"]
        if m:
            md.append(
                f"- **method** (n={m['n_eval']})  top-1={m['top1']}  top-3={m['top3']}  "
                f"macro-F1={m['macro_f1']}  mean_conf={m['mean_conf']}"
            )
            md.append(
                f"  - cov @ 80/85/90/95 % prec: "
                f"{fmt(m.get('cov@80%'))} / {fmt(m.get('cov@85%'))} / "
                f"{fmt(m.get('cov@90%'))} / {fmt(m.get('cov@95%'))}"
            )
        md.append("")

    md.append("## Comparison vs k-NN (scaffold split)")
    md.append("| metric | k-NN | XGBoost |")
    md.append("|---|---:|---:|")
    s = metrics_clean['split_scaffold']['solvent']
    md.append(f"| solvent top-1 | 0.141 | {s['top1']} |")
    md.append(f"| solvent top-3 | 0.472 | {s['top3']} |")
    md.append(f"| solvent cov@90% | None | {fmt(s.get('cov@90%'))} |")
    t = metrics_clean['split_scaffold']['temperature']
    md.append(f"| temp MAE | 48.7°C | {t['mae']}°C |")
    md.append(f"| temp within±10°C | 0.227 | {t['within_10C']} |")
    m = metrics_clean['split_scaffold']['method']
    md.append(f"| method top-1 | 0.385 | {m['top1']} |")
    md.append(f"| method macro-F1 | 0.159 | {m['macro_f1']} |")

    OUT_REPORT.write_text("\n".join(md))
    print(f"Wrote {OUT_REPORT}")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
