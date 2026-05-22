#!/usr/bin/env python3
"""Phase 4 — v0.1 MOF synthesis advisor (Streamlit demo, SCAFFOLD).

Goal: Working end-to-end demo with PLACEHOLDER model = v3.4.1 config
(depth=4, n_est=150, base+chem). Swap to a better checkpoint by changing
`MODEL_CONFIG` at top.

Run:
  cd /Users/mac/Documents/MOF
  python3 -m streamlit run Claude_auto_MOF_script/scripts/advisor_v01_app.py

Features:
  - Free-text input: metal source, linker(s), modulator (comma-separated)
  - Top-3 solvent-cluster predictions with calibrated confidence bars
  - Temperature recommendation (two-stage head from Phase 3.4.3)
  - k-NN provenance: 3 nearest training recipes (cosine over Morgan FP)
  - OOD-metal gate: warn if metal not in METAL_PT training distribution

NOT YET WIRED (intentionally stubbed for v0.1):
  - Time-h prediction (stub: shows training median for predicted cluster)
  - Method-head full reasoning (shows just "solvothermal-default")
  - Reaction yield / phase purity
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xgb_baseline import build_feature_matrix  # noqa: E402
from build_chem_features import (  # noqa: E402
    LINKER_SMILES, resolve_linker, METAL_PT,
)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator  # sklearn ≥1.6
except Exception:
    FrozenEstimator = None
from xgboost import XGBClassifier, XGBRegressor

CHECKPOINT = ROOT / "data" / "cleaned" / "v01_checkpoint.joblib"

# ---------------------------------------------------------------------
# MODEL CONFIG — swap pointer to upgrade later
# ---------------------------------------------------------------------
MODEL_CONFIG = {
    "version": "v0.1-shipping",
    "depth": 4,
    "n_est": 150,
    "feature_pipeline": "base+chem (2203 cols)",
    "scaffold_cov90_solvent": 0.104,  # Phase 3.5 honest scaffold
    "scaffold_cov90_method": 0.717,   # Phase 3.5
    "temp_two_stage": True,            # Phase 3.4.3
    "rt_threshold_C": 30.0,
}

NULL = "NULL"


# ---------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading v0.1 checkpoint…")
def load_checkpoint():
    if not CHECKPOINT.exists():
        return None
    return joblib.load(CHECKPOINT)


@st.cache_resource(show_spinner="Loading training data and chem features…")
def load_data():
    df = pd.read_parquet(DATA / "recipes.parquet").reset_index(drop=True)
    chem = pd.read_parquet(DATA / "chem_features.parquet").reset_index(drop=True)
    splits = (pd.read_parquet(DATA / "split_assignments.parquet")
              .set_index("row_idx").reindex(range(len(df))).reset_index())
    return df, chem, splits


@st.cache_resource(show_spinner="Building feature matrix…")
def load_X(_df, _chem):
    X_base, _ = build_feature_matrix(_df)
    if sp.issparse(X_base):
        X_base = X_base.toarray()
    X_base = X_base.astype(np.float32)
    morgan_cols = [c for c in _chem.columns if c.startswith("morgan_")]
    nm_cols = [c for c in _chem.columns if not c.startswith("morgan_")]
    M = _chem[morgan_cols].to_numpy(dtype=np.float32)
    NM = _chem[nm_cols].to_numpy(dtype=np.float32)
    X = np.hstack([X_base, NM, M]).astype(np.float32)
    return X, X_base.shape[1], NM.shape[1], M.shape[1]


@st.cache_resource(show_spinner=f"Training solvent classifier "
                   f"(depth={MODEL_CONFIG['depth']}, n_est={MODEL_CONFIG['n_est']})…")
def train_solvent_model(_X, _df, _splits):
    y = _df["solvent_cluster"].to_numpy()
    sps = _splits["split_random"].to_numpy()
    tr = (sps == "train") & (y != NULL)
    val = (sps == "val") & (y != NULL)
    y_tr = y[tr]
    le = {c: i for i, c in enumerate(sorted(set(y_tr)))}
    inv = {i: c for c, i in le.items()}
    y_tr_i = np.array([le[v] for v in y_tr])
    base = XGBClassifier(
        n_estimators=MODEL_CONFIG["n_est"], max_depth=MODEL_CONFIG["depth"],
        learning_rate=0.1, tree_method="hist", n_jobs=-1, verbosity=0,
        objective="multi:softprob", num_class=len(le),
    )
    base.fit(_X[tr], y_tr_i)
    y_val = y[val]
    val_known = np.array([v in le for v in y_val])
    try:
        if FrozenEstimator is not None:
            cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        else:
            cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
        cal.fit(_X[val][val_known], np.array([le[v] for v in y_val[val_known]]))
        model = cal
    except Exception:
        model = base
    return model, inv


@st.cache_resource(show_spinner="Training two-stage temperature head…")
def train_temp_model(_X, _df, _splits):
    y = _df["temperature_C"].to_numpy()
    sps = _splits["split_random"].to_numpy()
    valid = (~np.isnan(y)) & (y >= 0) & (y <= 300)
    tr = (sps == "train") & valid
    y_tr = y[tr]
    X_tr = _X[tr]
    is_heated = (y_tr > MODEL_CONFIG["rt_threshold_C"]).astype(int)
    clf = XGBClassifier(
        n_estimators=MODEL_CONFIG["n_est"], max_depth=MODEL_CONFIG["depth"],
        learning_rate=0.1, tree_method="hist", n_jobs=-1, verbosity=0,
        objective="binary:logistic",
    )
    clf.fit(X_tr, is_heated)
    reg = XGBRegressor(
        n_estimators=MODEL_CONFIG["n_est"], max_depth=MODEL_CONFIG["depth"],
        learning_rate=0.1, tree_method="hist", n_jobs=-1, verbosity=0,
        objective="reg:squarederror",
    )
    reg.fit(X_tr[is_heated == 1], y_tr[is_heated == 1])
    return clf, reg


@st.cache_resource(show_spinner="Precomputing k-NN index…")
def build_knn_index(_chem):
    morgan_cols = [c for c in _chem.columns if c.startswith("morgan_")]
    M = _chem[morgan_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    M_norm = M / norms[:, None]
    return M_norm, (M.sum(axis=1) > 0)


# ---------------------------------------------------------------------
# Input → feature vector
# ---------------------------------------------------------------------
def smiles_from_linker_input(s: str) -> str | None:
    """Try resolve_linker first, then treat as raw SMILES."""
    sm, _ = resolve_linker(s.strip())
    if sm is not None:
        return sm
    # try raw SMILES
    try:
        mol = Chem.MolFromSmiles(s.strip())
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def build_input_row(metal: str, linkers: list[str], modulator: str,
                    n_metal: float, n_link: float):
    """Build a synthetic 1-row recipes dataframe + chem features."""
    # Build a fake row matching recipes.parquet's expected columns.
    # We'll fill chemicals with a JSON list mirroring how recipes builds.
    chems = []
    if metal:
        chems.append({"kind": "metal_source", "name": metal,
                      "amount_value": n_metal, "amount_unit": "mmol"})
    for ln in linkers:
        if ln:
            chems.append({"kind": "linker", "name": ln,
                          "amount_value": n_link, "amount_unit": "mmol"})
    if modulator:
        chems.append({"kind": "modulator", "name": modulator,
                      "amount_value": 1.0, "amount_unit": "mmol"})

    # Use one real row as template, then overwrite chemicals + parsed cols
    df = pd.read_parquet(DATA / "recipes.parquet").iloc[[0]].copy()
    df["chemicals"] = json.dumps(chems)
    df["linkers"] = [linkers]
    df["modulators"] = [[modulator] if modulator else []]
    # parsed metal symbol → metal_sources JSON
    m_match = re.match(r"\s*([A-Z][a-z]?)", metal or "")
    metal_elem = m_match.group(1) if m_match else None
    df["metal_sources"] = json.dumps([metal] if metal else [])
    df["solvent_cluster"] = NULL  # unknown (this is what we predict)
    return df.reset_index(drop=True), metal_elem


def build_input_chem(linkers: list[str], metal_elem: str | None):
    """Build a 1-row chem_features matching chem_features.parquet schema."""
    chem_template = pd.read_parquet(DATA / "chem_features.parquet").iloc[[0]].copy()
    chem_template.iloc[0, :] = 0.0  # zero everything

    # Morgan: average of resolved linkers
    morgan_cols = [c for c in chem_template.columns if c.startswith("morgan_")]
    fps = []
    n_resolved = 0
    for ln in linkers:
        sm = smiles_from_linker_input(ln)
        if sm is None:
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        n_resolved += 1
    if fps:
        avg = np.mean(np.stack(fps), axis=0).astype(np.float32)
        chem_template.loc[chem_template.index[0], morgan_cols] = avg

    chem_template.loc[chem_template.index[0], "n_linkers"] = len(linkers)
    chem_template.loc[chem_template.index[0], "n_linkers_resolved"] = n_resolved
    chem_template.loc[chem_template.index[0], "n_linkers_unresolved"] = len(linkers) - n_resolved

    if metal_elem and metal_elem in METAL_PT:
        per, grp, en, r_ion, d_e = METAL_PT[metal_elem]
        chem_template.loc[chem_template.index[0], "metal_period"] = per
        chem_template.loc[chem_template.index[0], "metal_group"] = grp
        chem_template.loc[chem_template.index[0], "metal_en"] = en
        chem_template.loc[chem_template.index[0], "metal_ionic_radius"] = r_ion
        chem_template.loc[chem_template.index[0], "metal_d_electrons"] = d_e
        chem_template.loc[chem_template.index[0], "metal_present"] = 1

    return chem_template, n_resolved


def vectorize_input(df_in, chem_in, df_train):
    """Build feature vector matching X_train layout.

    build_feature_matrix derives the top-50-linker / top-30-modulator vocab
    from the input df, so a 1-row df gives the wrong column count. Solution:
    concatenate the input row to df_train, vectorize together, then take the
    last row. Training vocab dominates → schema matches.
    """
    df_combined = pd.concat([df_train, df_in], ignore_index=True)
    X_base, _ = build_feature_matrix(df_combined)
    if sp.issparse(X_base):
        X_base = X_base.toarray()
    X_base_in = X_base[-1:].astype(np.float32)
    morgan_cols = [c for c in chem_in.columns if c.startswith("morgan_")]
    nm_cols = [c for c in chem_in.columns if not c.startswith("morgan_")]
    M = chem_in[morgan_cols].to_numpy(dtype=np.float32)
    NM = chem_in[nm_cols].to_numpy(dtype=np.float32)
    return np.hstack([X_base_in, NM, M]).astype(np.float32)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="MOF Synthesis Advisor v0.1", layout="wide")
    st.title("MOF Synthesis Advisor — v0.1 (scaffold)")
    st.caption(
        f"Model: {MODEL_CONFIG['version']}  ·  "
        f"depth={MODEL_CONFIG['depth']}, n_est={MODEL_CONFIG['n_est']}  ·  "
        f"scaffold cov@90% solvent={MODEL_CONFIG['scaffold_cov90_solvent']}, "
        f"method={MODEL_CONFIG['scaffold_cov90_method']}"
    )

    df, chem, splits = load_data()
    X_train, n_base, n_nm, n_morgan = load_X(df, chem)

    ckpt = load_checkpoint()
    if ckpt is not None:
        solv_model = ckpt["solvent_model"]
        solv_inv = ckpt["solvent_inv"]
        temp_clf = ckpt["temp_stage_a"]
        temp_reg = ckpt["temp_stage_b"]
        meth_model = ckpt["method_model"]
        meth_inv = ckpt["method_inv"]
        knn_M_norm = ckpt["knn_M_norm"]
        knn_has_linker = ckpt["knn_has_linker"]
        st.sidebar.success(
            f"✓ Loaded checkpoint built {ckpt['build_timestamp']}"
        )
    else:
        st.sidebar.warning("No v01_checkpoint.joblib — training in-process…")
        solv_model, solv_inv = train_solvent_model(X_train, df, splits)
        temp_clf, temp_reg = train_temp_model(X_train, df, splits)
        meth_model, meth_inv = None, None
        knn_M_norm, knn_has_linker = build_knn_index(chem)

    with st.sidebar:
        st.header("Input recipe")
        metal = st.text_input("Metal source", value="Zn(NO3)2·6H2O",
                              help="e.g. Zn(NO3)2·6H2O, ZrCl4, CuSO4")
        linkers_raw = st.text_area(
            "Linker(s) — comma-separated",
            value="H2BDC",
            help="Linker name (H2BDC, H3BTC, H2BPDC, …) or raw SMILES",
        )
        modulator = st.text_input("Modulator (optional)", value="")
        n_metal = st.number_input("Metal amount (mmol)", 0.01, 100.0, 1.0)
        n_link = st.number_input("Linker amount (mmol)", 0.01, 100.0, 1.0)
        go = st.button("Predict", type="primary")

    if not go:
        st.info("Enter a recipe in the sidebar and click **Predict**.")
        st.markdown(
            "### v0.1 readiness checklist\n"
            "- ✅ Solvent classifier head (v0.1 shipping)\n"
            "- ✅ Two-stage temperature head (Phase 3.4.3)\n"
            "- ✅ Method classifier head (10 classes, Phase 3.5)\n"
            "- ✅ k-NN provenance over Morgan-FP\n"
            "- ✅ OOD-metal gate\n"
            "- ⏸ Time-h prediction (stub)\n"
        )
        return

    linkers = [s.strip() for s in linkers_raw.split(",") if s.strip()]

    # OOD gate
    m_match = re.match(r"\s*([A-Z][a-z]?)", metal)
    metal_elem = m_match.group(1) if m_match else None
    if metal_elem and metal_elem not in METAL_PT:
        st.warning(
            f"⚠️ Metal element **{metal_elem}** is not in the training "
            f"distribution ({len(METAL_PT)} known). Predictions are out-of-distribution."
        )

    # Linker resolution feedback
    resolved_linkers, unresolved_linkers = [], []
    for ln in linkers:
        sm = smiles_from_linker_input(ln)
        if sm:
            resolved_linkers.append((ln, sm))
        else:
            unresolved_linkers.append(ln)
    if unresolved_linkers:
        st.warning(
            f"⚠️ Could not resolve {len(unresolved_linkers)} linker(s) to SMILES: "
            f"{', '.join(unresolved_linkers)}. Predictions will rely on metal/recipe "
            f"context only for those entries."
        )

    # Build feature vector
    df_in, _ = build_input_row(metal, linkers, modulator, n_metal, n_link)
    chem_in, n_resolved = build_input_chem(linkers, metal_elem)
    try:
        X_in = vectorize_input(df_in, chem_in, df)
    except Exception as e:
        st.error(f"Failed to build feature vector: {e}")
        return

    if X_in.shape[1] != X_train.shape[1]:
        st.error(
            f"Feature shape mismatch: input={X_in.shape[1]} vs train={X_train.shape[1]}. "
            "Categorical encoder saw a value it hadn't trained on — falling back gracefully "
            "is not yet wired in v0.1."
        )
        return

    # ---------- predict ----------
    proba = solv_model.predict_proba(X_in)[0]
    order = np.argsort(-proba)[:5]
    top_labels = [solv_inv[i] for i in order]
    top_confs = [float(proba[i]) for i in order]

    p_heated = float(temp_clf.predict_proba(X_in)[0, 1])
    if p_heated > 0.5:
        t_pred = float(temp_reg.predict(X_in)[0])
        t_method = f"solvothermal (P(heated)={p_heated:.2f})"
    else:
        t_pred = 25.0
        t_method = f"room-temp / slow-diffusion (P(heated)={p_heated:.2f})"

    # ---------- display ----------
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Top-5 solvent cluster predictions")
        bar_df = pd.DataFrame({"cluster": top_labels, "confidence": top_confs})
        st.bar_chart(bar_df.set_index("cluster"), height=260)
        st.dataframe(
            bar_df.assign(confidence=lambda d: d.confidence.round(4)),
            hide_index=True, use_container_width=True,
        )

        st.subheader("Temperature recommendation")
        st.metric("Predicted reaction temperature", f"{t_pred:.0f} °C",
                  help=f"Two-stage head: {t_method}")
        st.caption(
            "Within-±10°C accuracy (scaffold test) = "
            "**0.306** for two-stage vs 0.232 single-stage."
        )

        st.subheader("Time (stub)")
        clust_median = df[(df.solvent_cluster == top_labels[0]) &
                          (df.time_h.notna())]["time_h"].median()
        st.write(f"Training median `time_h` for `{top_labels[0]}`: "
                 f"**{clust_median:.1f} h**  *(stub; no dedicated time model)*")

        st.subheader("Synthesis method (top-3)")
        if meth_model is None:
            st.info("Method head not in checkpoint — defaulting to `solvothermal`.")
        else:
            mproba = meth_model.predict_proba(X_in)[0]
            morder = np.argsort(-mproba)[:3]
            m_df = pd.DataFrame({
                "method": [meth_inv[i] for i in morder],
                "confidence": [round(float(mproba[i]), 4) for i in morder],
            })
            st.dataframe(m_df, hide_index=True, use_container_width=True)
            st.caption(
                f"Method head: scaffold top-1 = 0.749, cov@90% = "
                f"{MODEL_CONFIG['scaffold_cov90_method']}."
            )

    with c2:
        st.subheader("k-NN provenance")
        st.caption("Nearest 3 training recipes by Morgan-FP cosine similarity.")
        # input fingerprint (need same shape as knn_M_norm)
        morgan_cols = [c for c in chem_in.columns if c.startswith("morgan_")]
        q = chem_in[morgan_cols].to_numpy(dtype=np.float32)[0]
        qn = np.linalg.norm(q)
        if qn == 0 or not knn_has_linker.any():
            st.info("No resolvable linker → cannot compute Morgan-similarity "
                    "neighbors. Showing top-cluster training rows instead.")
            cand = df[df.solvent_cluster == top_labels[0]].head(3)
        else:
            q_norm = q / qn
            # only score against rows that themselves have a Morgan signal
            sims = knn_M_norm @ q_norm
            sims[~knn_has_linker] = -1.0
            top_k = np.argsort(-sims)[:3]
            cand = df.iloc[top_k].copy()
            cand["similarity"] = sims[top_k].round(3)

        show_cols = [c for c in
                     ["similarity", "solvent_cluster", "temperature_C",
                      "time_h", "linkers", "metal_sources"]
                     if c in cand.columns]
        st.dataframe(cand[show_cols], hide_index=True, use_container_width=True)

        with st.expander("Debug — input feature stats"):
            st.write({
                "X_in shape": X_in.shape,
                "X_train shape": X_train.shape,
                "n_linkers_resolved": int(n_resolved),
                "metal_elem": metal_elem,
                "metal_in_training": metal_elem in METAL_PT if metal_elem else None,
                "morgan_nonzero_bits": int((q > 0).sum()),
            })


if __name__ == "__main__":
    main()
