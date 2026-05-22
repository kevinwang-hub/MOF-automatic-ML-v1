#!/usr/bin/env python3
"""Phase 3.5 — Canonical 5-MOF regression harness.

Asserts the v0.1 joblib checkpoint produces sane top-k predictions for
five textbook MOFs. Loose tolerances: predictions for unseen-in-detail
recipes are inherently noisy, so we check the truth label appears in
top-3 (cluster + method) and |temp_pred − canonical_temp| ≤ 30 °C.

Exit 1 on any failure. Designed for CI / pre-release smoke.

Usage:
  python3 Claude_auto_MOF_script/scripts/canonical_tests.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

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

CHECKPOINT = DATA / "v01_checkpoint.joblib"

# (loose) canonical syntheses
CASES = [
    {
        "name": "MOF-5",
        "metal": "Zn(NO3)2·6H2O",
        "linkers": ["H2BDC"],
        "modulator": "",
        "temp_C": 100.0, "temp_tol_C": 30.0,
        "method_in_top3": {"solvothermal"},
        "solvent_in_top3": {"polar_aprotic_amide", "mixed_organic"},
    },
    {
        "name": "HKUST-1",
        "metal": "Cu(NO3)2·3H2O",
        "linkers": ["H3BTC"],
        "modulator": "",
        "temp_C": 85.0, "temp_tol_C": 30.0,
        "method_in_top3": {"solvothermal"},
        "solvent_in_top3": {"polar_aprotic_amide", "mixed_organic", "mixed_aqueous"},
    },
    {
        "name": "UiO-66",
        "metal": "ZrCl4",
        "linkers": ["H2BDC"],
        "modulator": "",
        "temp_C": 120.0, "temp_tol_C": 30.0,
        "method_in_top3": {"solvothermal"},
        "solvent_in_top3": {"polar_aprotic_amide"},
    },
    {
        "name": "ZIF-8",
        "metal": "Zn(NO3)2·6H2O",
        "linkers": ["2-methylimidazole"],
        "modulator": "",
        "temp_C": 60.0, "temp_tol_C": 45.0,  # literature spans RT–100°C
        "method_in_top3": {"solvothermal", "room_temperature"},
        "solvent_in_top3": {"polar_aprotic_amide", "methanol",
                             "mixed_organic", "mixed_aqueous"},
    },
    {
        "name": "MIL-101(Cr)",
        "metal": "CrCl3·6H2O",
        "linkers": ["H2BDC"],
        "modulator": "",
        "temp_C": 220.0, "temp_tol_C": 30.0,
        "method_in_top3": {"solvothermal"},
        "solvent_in_top3": {"aqueous_only", "polar_aprotic_amide"},
        "xfail": True,  # high-T Cr-solvothermal sparse in training; documented gap
    },
]


# ---------- input vectorization (lifted from advisor_v01_app.main) ----------
def smiles_from_linker(s: str) -> str | None:
    sm, _ = resolve_linker(s.strip())
    if sm is not None:
        return sm
    try:
        mol = Chem.MolFromSmiles(s.strip())
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def build_input_row(df_train_first, metal, linkers, modulator):
    chems = [{"kind": "metal_source", "name": metal,
              "amount_value": 1.0, "amount_unit": "mmol"}]
    for ln in linkers:
        chems.append({"kind": "linker", "name": ln,
                      "amount_value": 1.0, "amount_unit": "mmol"})
    if modulator:
        chems.append({"kind": "modulator", "name": modulator,
                      "amount_value": 1.0, "amount_unit": "mmol"})
    df = df_train_first.copy()
    df["chemicals"] = json.dumps(chems)
    df["linkers"] = [linkers]
    df["modulators"] = [[modulator] if modulator else []]
    df["metal_sources"] = json.dumps([metal])
    df["solvent_cluster"] = "NULL"
    return df.reset_index(drop=True)


def build_input_chem(chem_template, linkers, metal_elem):
    chem = chem_template.iloc[[0]].copy()
    chem.iloc[0, :] = 0.0
    morgan_cols = [c for c in chem.columns if c.startswith("morgan_")]
    fps, n_res = [], 0
    for ln in linkers:
        sm = smiles_from_linker(ln)
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
        n_res += 1
    if fps:
        avg = np.mean(np.stack(fps), axis=0).astype(np.float32)
        chem.loc[chem.index[0], morgan_cols] = avg
    chem.loc[chem.index[0], "n_linkers"] = len(linkers)
    chem.loc[chem.index[0], "n_linkers_resolved"] = n_res
    if metal_elem and metal_elem in METAL_PT:
        per, grp, en, r, d_e = METAL_PT[metal_elem]
        for k, v in [("metal_period", per), ("metal_group", grp),
                     ("metal_en", en), ("metal_ionic_radius", r),
                     ("metal_d_electrons", d_e), ("metal_present", 1)]:
            chem.loc[chem.index[0], k] = v
    return chem


def vectorize_input(df_in, chem_in, df_train):
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


def predict_case(ckpt, df_train, chem_template, case):
    metal = case["metal"]
    m_match = re.match(r"\s*([A-Z][a-z]?)", metal)
    metal_elem = m_match.group(1) if m_match else None
    df_in = build_input_row(df_train.iloc[[0]], metal,
                            case["linkers"], case["modulator"])
    chem_in = build_input_chem(chem_template, case["linkers"], metal_elem)
    X_in = vectorize_input(df_in, chem_in, df_train)

    solv_inv = ckpt["solvent_inv"]
    meth_inv = ckpt["method_inv"]
    s_proba = ckpt["solvent_model"].predict_proba(X_in)[0]
    s_top3 = [solv_inv[i] for i in np.argsort(-s_proba)[:3]]
    m_proba = ckpt["method_model"].predict_proba(X_in)[0]
    m_top3 = [meth_inv[i] for i in np.argsort(-m_proba)[:3]]

    p_heat = float(ckpt["temp_stage_a"].predict_proba(X_in)[0, 1])
    if p_heat > 0.5:
        t_pred = float(ckpt["temp_stage_b"].predict(X_in)[0])
    else:
        t_pred = 25.0
    return s_top3, m_top3, t_pred, p_heat


def main():
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    print(f"Loading {CHECKPOINT.name}…")
    ckpt = joblib.load(CHECKPOINT)
    print(f"  built {ckpt['build_timestamp']}, "
          f"X_train shape {ckpt['X_train_shape']}")
    df = pd.read_parquet(DATA / "recipes.parquet").reset_index(drop=True)
    chem = pd.read_parquet(DATA / "chem_features.parquet").reset_index(drop=True)

    rows, n_fail, n_xfail = [], 0, 0
    for c in CASES:
        s_top3, m_top3, t_pred, p_heat = predict_case(ckpt, df, chem, c)
        solv_ok = bool(set(s_top3) & c["solvent_in_top3"])
        meth_ok = bool(set(m_top3) & c["method_in_top3"])
        temp_ok = abs(t_pred - c["temp_C"]) <= c["temp_tol_C"]
        passed = solv_ok and meth_ok and temp_ok
        is_xfail = c.get("xfail", False)
        if not passed:
            if is_xfail:
                n_xfail += 1
            else:
                n_fail += 1
        rows.append({
            "name": c["name"] + (" [xfail]" if is_xfail else ""),
            "solvent_top1": s_top3[0],
            "solvent_ok": "✓" if solv_ok else "✗",
            "method_top1": m_top3[0],
            "method_ok": "✓" if meth_ok else "✗",
            "temp_pred_C": round(t_pred, 1),
            "temp_target_C": c["temp_C"],
            "temp_tol_C": c["temp_tol_C"],
            "temp_ok": "✓" if temp_ok else "✗",
            "passed": ("PASS" if passed else
                       ("XFAIL" if is_xfail else "FAIL")),
        })
    rep = pd.DataFrame(rows)
    print("\n" + rep.to_string(index=False))
    n_pass = len(CASES) - n_fail - n_xfail
    print(f"\n{n_pass}/{len(CASES)} passed, {n_fail} hard failures, "
          f"{n_xfail} expected failures.")
    out = ROOT / "other" / "canonical_tests_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Canonical 5-MOF test report\n\n"
        + rep.to_markdown(index=False) + "\n\n"
        f"**{n_pass}/{len(CASES)} passed**, {n_fail} hard failures, "
        f"{n_xfail} expected failures.\n"
        "\nTolerances: top-3 cluster match, top-3 method match, "
        "per-case temperature tolerance.\n"
    )
    print(f"Wrote {out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
