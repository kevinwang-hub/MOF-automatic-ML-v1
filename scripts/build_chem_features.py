"""
Build chemistry-aware features for XGBoost (Phase 3.3.1, Option A).

Produces `data/cleaned/chem_features.parquet` aligned with `recipes.parquet`
(by row order). Columns:

  - morgan_*               : 2048-bit Morgan fingerprint, radius=2, averaged
                             over resolvable linker SMILES on the row.
                             0-vector if no resolvable linkers.
  - n_linkers              : number of linker entries on the row.
  - n_linkers_resolved     : number resolved to SMILES via LINKER_SMILES dict.
  - n_linkers_unresolved   : n_linkers - n_linkers_resolved.
  - has_low_info_linker    : 1 if at least one unresolved linker (e.g. "H4L").
  - linker_n_carboxylate   : avg #(C(=O)O) over resolved linkers.
  - linker_n_aromatic_ring : avg num aromatic rings over resolved linkers.
  - linker_n_n_donor       : avg #N (excluding nitro) over resolved linkers.
  - linker_logp_avg        : avg Crippen logP over resolved linkers.
  - linker_mw_avg          : avg MolWt over resolved linkers.
  - metal_period           : period of first metal in metal_sources (0 if none)
  - metal_group            : group number (0 if none)
  - metal_en               : Pauling electronegativity (0 if none)
  - metal_ionic_radius     : Shannon ionic radius (pm, 0 if none)
  - metal_d_electrons      : d-electron count for the common ox. state (0 if none)
  - metal_present          : 1 if a metal was parsed
  - has_carboxylic_cosolvent : co-solvent override (AcOH/HCOOH/TFA/propionic)
  - has_aromatic_solvent     : co-solvent override (benzene/toluene/xylene/pyridine)

The Morgan bits are written as columns morgan_0000..morgan_2047 (sparse).
"""
from __future__ import annotations
import json, re, sys, ast
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "cleaned"
OTHER = ROOT / "other"

# ----------------------------------------------------------------------
# Linker SMILES dictionary — top abbreviations / common names.
# Covers the top ~50% of occurrences from the linker frequency audit.
# Keys are lowercased; matching is case-insensitive and strips whitespace.
# Generic placeholders (H4L, H3L1, etc.) intentionally NOT included → they
# fall through to the "low-info-linker" bucket.
# ----------------------------------------------------------------------
LINKER_SMILES: dict[str, str] = {
    # mono-aromatic dicarboxylates
    "h2bdc": "OC(=O)c1ccc(C(=O)O)cc1",
    "bdc": "OC(=O)c1ccc(C(=O)O)cc1",
    "1,4-benzenedicarboxylic acid": "OC(=O)c1ccc(C(=O)O)cc1",
    "terephthalic acid": "OC(=O)c1ccc(C(=O)O)cc1",
    "h2bdc-nh2": "Nc1cc(C(=O)O)ccc1C(=O)O",
    "2-aminoterephthalic acid": "Nc1cc(C(=O)O)ccc1C(=O)O",
    "nh2-bdc": "Nc1cc(C(=O)O)ccc1C(=O)O",
    "h2bdc-br": "Brc1cc(C(=O)O)ccc1C(=O)O",
    "h2bdc-no2": "[O-][N+](=O)c1cc(C(=O)O)ccc1C(=O)O",
    "h2bdc-oh": "Oc1cc(C(=O)O)ccc1C(=O)O",
    # tricarboxylates
    "h3btc": "OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "1,3,5-benzenetricarboxylic acid": "OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "trimesic acid": "OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "btc": "OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1",
    "h3btb": "OC(=O)c1ccc(-c2cc(-c3ccc(C(=O)O)cc3)cc(-c3ccc(C(=O)O)cc3)c2)cc1",
    # biphenyl
    "h2bpdc": "OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
    "bpdc": "OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
    "biphenyl-4,4'-dicarboxylic acid": "OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
    # naphthalene
    "h2ndc": "OC(=O)c1ccc2cc(C(=O)O)ccc2c1",
    "ndc": "OC(=O)c1ccc2cc(C(=O)O)ccc2c1",
    "2,6-naphthalenedicarboxylic acid": "OC(=O)c1ccc2cc(C(=O)O)ccc2c1",
    # carbazole
    "h2cdc": "O=C(O)c1ccc2c(c1)[nH]c1cc(C(=O)O)ccc12",
    # imidazole / azole linkers
    "2-methylimidazole": "Cc1ncc[nH]1",
    "imidazole": "c1cnc[nH]1",
    "mim": "Cc1ncc[nH]1",
    "him": "c1cnc[nH]1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "bim": "c1ccc2[nH]cnc2c1",
    "pyrazine": "c1cnccn1",
    "4,4'-bipyridine": "c1cc(-c2ccncc2)ncc1",
    "bipy": "c1cc(-c2ccncc2)ncc1",
    "4,4-bipyridine": "c1cc(-c2ccncc2)ncc1",
    "h2bipy-dc": "OC(=O)c1cnc(-c2ccnc(C(=O)O)c2)cc1",
    # other common
    "h4abtc": "OC(=O)c1ccc(/N=N/c2ccc(C(=O)O)c(C(=O)O)c2)cc1C(=O)O",
    "azobenzenetetracarboxylic acid": "OC(=O)c1ccc(/N=N/c2ccc(C(=O)O)c(C(=O)O)c2)cc1C(=O)O",
    "h2dobdc": "OC(=O)c1cc(O)c(O)cc1C(=O)O",
    "dobdc": "OC(=O)c1cc(O)c(O)cc1C(=O)O",
    "h4dobpdc": "Oc1cc(C(=O)O)cc(-c2cc(O)c(O)cc2C(=O)O)c1",
    "h2pzdc": "OC(=O)c1nncc1C(=O)O",  # pyrazine-2,3-dicarboxylic acid (approx)
    "h2sdc": "OC(=O)c1ccc(/C=C/c2ccc(C(=O)O)cc2)cc1",  # stilbene-4,4'-dicarboxylic
    "sdc": "OC(=O)c1ccc(/C=C/c2ccc(C(=O)O)cc2)cc1",
    "h2tcpp": "OC(=O)c1ccc(-c2c3ccc(-c4ccc(C(=O)O)cc4)[nH]3)cc1",  # truncated rep
    "fumaric acid": "OC(=O)/C=C/C(=O)O",
    "h2fum": "OC(=O)/C=C/C(=O)O",
    "oxalic acid": "OC(=O)C(=O)O",
    "h2ox": "OC(=O)C(=O)O",
    "isophthalic acid": "OC(=O)c1cccc(C(=O)O)c1",
    "h2ipa": "OC(=O)c1cccc(C(=O)O)c1",
    # imidazoledicarboxylic
    "h3imdc": "OC(=O)c1[nH]c(C(=O)O)nc1",
    "4,5-imidazoledicarboxylic acid": "OC(=O)c1[nH]cnc1C(=O)O",
    # adenine / nucleobases
    "adenine": "c1nc2[nH]cnc2c(N)n1",
    "had": "c1nc2[nH]cnc2c(N)n1",
    # extra additions from unresolved audit (Phase 3.3.1 iteration)
    "biphenyl-3,3',5,5'-tetracarboxylic acid": "OC(=O)c1cc(C(=O)O)cc(-c2cc(C(=O)O)cc(C(=O)O)c2)c1",
    "h4bptc": "OC(=O)c1cc(C(=O)O)cc(-c2cc(C(=O)O)cc(C(=O)O)c2)c1",
    "h4dobdc": "OC(=O)c1cc(O)c(O)cc1C(=O)O",
    "2,5-dihydroxyterephthalic acid": "OC(=O)c1cc(O)c(O)cc1C(=O)O",
    "h4(m-dobdc)": "Oc1cc(C(=O)O)cc(O)c1C(=O)O",
    "h2pyc": "OC(=O)c1ccncc1",  # pyridine-4-carboxylic
    "isonicotinic acid": "OC(=O)c1ccncc1",
    "h2tdc": "OC(=O)c1cc(C(=O)O)cs1",  # thiophene-2,5-dicarboxylic
    "thiophene-2,5-dicarboxylic acid": "OC(=O)c1cc(C(=O)O)cs1",
    "h4tcpe": "OC(=O)c1ccc(/C(=C(/c2ccc(C(=O)O)cc2)\\c2ccc(C(=O)O)cc2)c2ccc(C(=O)O)cc2)cc1",
    "h2tpp": "c1cc(-c2[nH]c(-c3ccc(-c4nc(-c5ccc(-c6[nH]c2cc6)cc5)cc4)cc3)cc1)cc1",  # porphyrin (approx)
    "oxalic acid dihydrate": "OC(=O)C(=O)O.O.O",
    "h2c2o4": "OC(=O)C(=O)O",
    "h2c2o4·2h2o": "OC(=O)C(=O)O.O.O",
    "squaric acid": "OC1=C(O)C(=O)C1=O",
    "glutaric acid": "OC(=O)CCCC(=O)O",
    "adipic acid": "OC(=O)CCCCC(=O)O",
    "succinic acid": "OC(=O)CCC(=O)O",
    "malonic acid": "OC(=O)CC(=O)O",
    "maleic acid": "OC(=O)/C=C\\C(=O)O",
    "4,4'-azopyridine": "c1cc(/N=N/c2ccncc2)ncc1",
    "pyrazole": "c1cc[nH]n1",
    "1,2,4-triazole": "c1nnc[nH]1",
    "1,2,3-triazole": "c1cn[nH]n1",
    "2-methyl-1h-imidazole": "Cc1ncc[nH]1",
    "h3btct": "OC(=O)c1cc(C(=O)O)cc(C(=O)O)c1",  # alias trimesate
    "h2pzdc": "OC(=O)c1cnncc1C(=O)O",
    "thiocyanate": "[S-]C#N",
    "h2adp": "OC(=O)CCCCC(=O)O",
}

_GENERIC_LINKER_RE = re.compile(r"^h\d*l\d*[a-z]?$", re.I)
_PAREN_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*$")
_QUANT_TAIL_RE = re.compile(r"\s*\([^)]*\b(?:mg|g|mmol|mol|ml)\b[^)]*\)\s*$", re.I)


def _normalize_linker_key(name: str) -> str:
    s = name.strip().lower()
    # unicode apostrophes / dashes
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def resolve_linker(name: str) -> tuple[str | None, bool]:
    """Return (smiles or None, is_generic_placeholder)."""
    if not isinstance(name, str):
        return None, False
    key = _normalize_linker_key(name)
    if not key:
        return None, False
    # direct
    if key in LINKER_SMILES:
        return LINKER_SMILES[key], False
    # strip quantity parenthetical (e.g. "H6cpb (16 mg, 0.020 mmol)")
    k2 = _QUANT_TAIL_RE.sub("", key)
    if k2 != key and k2 in LINKER_SMILES:
        return LINKER_SMILES[k2], False
    # strip abbreviation parenthetical (e.g. "4,5-imidazoledicarboxylic acid (H3ImDC)")
    k3 = _PAREN_TAIL_RE.sub("", key).strip()
    if k3 != key and k3 in LINKER_SMILES:
        return LINKER_SMILES[k3], False
    # try the inner abbreviation if it's the parenthetical
    m = re.search(r"\(([^)]+)\)\s*$", key)
    if m:
        inner = m.group(1).strip().lower()
        if inner in LINKER_SMILES:
            return LINKER_SMILES[inner], False
    # try stripping trailing " acid"
    if key.endswith(" acid") and key[:-5] in LINKER_SMILES:
        return LINKER_SMILES[key[:-5]], False
    # placeholder pattern
    if _GENERIC_LINKER_RE.match(key):
        return None, True
    return None, False


# ----------------------------------------------------------------------
# Metal periodic-table lookup. (period, group, electronegativity_Pauling,
# Shannon ionic radius for typical ox.state in pm, d-electron count)
# Sources: standard periodic tables; ionic radius for most common CN (6).
# d-electron count is for the common synthesis oxidation state.
# ----------------------------------------------------------------------
METAL_PT: dict[str, tuple[int, int, float, float, int]] = {
    # alkali / alkaline earth
    "Li": (2, 1, 0.98, 76, 0),
    "Na": (3, 1, 0.93, 102, 0),
    "K": (4, 1, 0.82, 138, 0),
    "Mg": (3, 2, 1.31, 72, 0),
    "Ca": (4, 2, 1.00, 100, 0),
    "Sr": (5, 2, 0.95, 118, 0),
    "Ba": (6, 2, 0.89, 135, 0),
    # 3d transition (common ox state in MOFs)
    "Sc": (4, 3, 1.36, 75, 0),  # +3
    "Ti": (4, 4, 1.54, 61, 0),  # +4
    "V": (4, 5, 1.63, 64, 0),  # +4
    "Cr": (4, 6, 1.66, 62, 3),  # +3
    "Mn": (4, 7, 1.55, 83, 5),  # +2
    "Fe": (4, 8, 1.83, 65, 5),  # +3 (high spin)
    "Co": (4, 9, 1.88, 75, 7),  # +2
    "Ni": (4, 10, 1.91, 69, 8),  # +2
    "Cu": (4, 11, 1.90, 73, 9),  # +2
    "Zn": (4, 12, 1.65, 74, 10),  # +2
    # 4d
    "Y": (5, 3, 1.22, 90, 0),  # +3
    "Zr": (5, 4, 1.33, 72, 0),  # +4
    "Mo": (5, 6, 2.16, 73, 0),  # +4-6
    "Ru": (5, 8, 2.20, 70, 5),  # +3
    "Rh": (5, 9, 2.28, 67, 6),  # +3
    "Pd": (5, 10, 2.20, 86, 8),  # +2
    "Ag": (5, 11, 1.93, 115, 10),  # +1
    "Cd": (5, 12, 1.69, 95, 10),  # +2
    "In": (5, 13, 1.78, 80, 10),  # +3
    "Sn": (5, 14, 1.96, 83, 10),  # +2
    # 5d
    "La": (6, 3, 1.10, 103, 0),
    "Hf": (6, 4, 1.30, 71, 0),  # +4
    "W": (6, 6, 2.36, 60, 0),
    "Pt": (6, 10, 2.28, 80, 8),
    "Au": (6, 11, 2.54, 137, 10),
    "Hg": (6, 12, 2.00, 102, 10),
    "Pb": (6, 14, 2.33, 119, 10),
    "Bi": (6, 15, 2.02, 103, 10),
    # lanthanides (treat group as 3, d-electrons 0; ionic radius +3 CN8)
    "Ce": (6, 3, 1.12, 114, 0),
    "Pr": (6, 3, 1.13, 113, 0),
    "Nd": (6, 3, 1.14, 112, 0),
    "Sm": (6, 3, 1.17, 109, 0),
    "Eu": (6, 3, 1.20, 108, 0),
    "Gd": (6, 3, 1.20, 107, 0),
    "Tb": (6, 3, 1.20, 106, 0),
    "Dy": (6, 3, 1.22, 105, 0),
    "Ho": (6, 3, 1.23, 104, 0),
    "Er": (6, 3, 1.24, 103, 0),
    "Tm": (6, 3, 1.25, 102, 0),
    "Yb": (6, 3, 1.10, 101, 0),
    "Lu": (6, 3, 1.27, 100, 0),
    # p-block
    "Al": (3, 13, 1.61, 54, 0),
    "Ga": (4, 13, 1.81, 62, 10),
    "Ge": (4, 14, 2.01, 73, 10),
}

_METAL_ELEM_RE = re.compile(r"^([A-Z][a-z]?)")


def parse_first_metal_element(metal_sources_str: str) -> str | None:
    """metal_sources_str is JSON-stringified list like '[\"Zn2+\",\"Cd2+\"]'."""
    if not isinstance(metal_sources_str, str) or metal_sources_str in ("[]", ""):
        return None
    try:
        items = json.loads(metal_sources_str)
    except Exception:
        try:
            items = ast.literal_eval(metal_sources_str)
        except Exception:
            return None
    if not items:
        return None
    first = str(items[0]).strip()
    m = _METAL_ELEM_RE.match(first)
    if m and m.group(1) in METAL_PT:
        return m.group(1)
    return None


# ----------------------------------------------------------------------
# Co-solvent override features (per Phase 3.3 §3 chemistry call).
# ----------------------------------------------------------------------
CARBOX_NAMES = {
    "acetic acid", "aceticacid", "acoh",
    "formic acid", "formicacid", "hcooh", "hco2h",
    "propionic acid", "propionicacid",
    "trifluoroacetic acid", "tfa",
    "butyric acid",
    "benzoic acid",
    "2-fluorobenzoic acid",
}
AROMATIC_NAMES = {
    "benzene", "toluene", "xylene", "o-xylene", "m-xylene", "p-xylene",
    "mesitylene", "nitrobenzene", "chlorobenzene", "fluorobenzene",
    "pyridine", "anisole",
}


def parse_solvent_canonicals(solvents_str: str) -> list[str]:
    if not isinstance(solvents_str, str) or solvents_str in ("[]", ""):
        return []
    try:
        items = json.loads(solvents_str)
    except Exception:
        return []
    out = []
    for it in items:
        c = it.get("canonical", "") if isinstance(it, dict) else ""
        if c:
            out.append(c.strip().lower())
    return out


# ----------------------------------------------------------------------
# RDKit descriptors per-mol
# ----------------------------------------------------------------------
def mol_descriptors(mol):
    n_cooh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[OH]")))
    n_arom_rings = Chem.GetSSSR(mol)  # ssr count; aromatic check below
    arom_count = sum(1 for r in mol.GetRingInfo().AtomRings()
                     if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    # N donors: aromatic or sp3 N not in nitro
    n_n = 0
    for a in mol.GetAtoms():
        if a.GetSymbol() == "N" and not any(
            (b.GetOtherAtom(a).GetSymbol() == "O" and b.GetBondTypeAsDouble() == 2.0)
            for b in a.GetBonds()
        ):
            n_n += 1
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    return n_cooh, arom_count, n_n, logp, mw


def main():
    r = pd.read_parquet(DATA / "recipes.parquet")
    n = len(r)
    print(f"Rows: {n}")

    NBITS = 2048
    morgan_acc = np.zeros((n, NBITS), dtype=np.float32)
    n_linkers = np.zeros(n, dtype=np.int16)
    n_resolved = np.zeros(n, dtype=np.int16)
    n_unresolved = np.zeros(n, dtype=np.int16)
    has_low_info = np.zeros(n, dtype=np.int8)
    n_cooh_avg = np.zeros(n, dtype=np.float32)
    n_arom_avg = np.zeros(n, dtype=np.float32)
    n_n_avg = np.zeros(n, dtype=np.float32)
    logp_avg = np.zeros(n, dtype=np.float32)
    mw_avg = np.zeros(n, dtype=np.float32)

    metal_period = np.zeros(n, dtype=np.int8)
    metal_group = np.zeros(n, dtype=np.int8)
    metal_en = np.zeros(n, dtype=np.float32)
    metal_ir = np.zeros(n, dtype=np.float32)
    metal_de = np.zeros(n, dtype=np.int8)
    metal_present = np.zeros(n, dtype=np.int8)

    has_carbox = np.zeros(n, dtype=np.int8)
    has_arom_solv = np.zeros(n, dtype=np.int8)

    # cache parsed mols
    _mol_cache: dict[str, object] = {}

    n_unique_unresolved: dict[str, int] = {}

    for i, row in r.reset_index(drop=True).iterrows():
        # ---- linker features ----
        linkers = list(row.linkers) if row.linkers is not None else []
        n_linkers[i] = len(linkers)
        fps = []
        descrs = []
        for name in linkers:
            sm, is_gen = resolve_linker(name)
            if sm is None:
                n_unresolved[i] += 1
                if is_gen:
                    has_low_info[i] = 1
                else:
                    n_unique_unresolved[str(name)] = n_unique_unresolved.get(str(name), 0) + 1
                continue
            mol = _mol_cache.get(sm)
            if mol is None:
                mol = Chem.MolFromSmiles(sm)
                _mol_cache[sm] = mol
            if mol is None:
                n_unresolved[i] += 1
                continue
            n_resolved[i] += 1
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=NBITS)
            arr = np.zeros(NBITS, dtype=np.float32)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            descrs.append(mol_descriptors(mol))

        if fps:
            morgan_acc[i] = np.mean(fps, axis=0)
            d = np.array(descrs, dtype=np.float32)
            n_cooh_avg[i], n_arom_avg[i], n_n_avg[i], logp_avg[i], mw_avg[i] = d.mean(axis=0)

        # ---- metal features ----
        elem = parse_first_metal_element(row.metal_sources)
        if elem is not None:
            p, g, en, ir, de = METAL_PT[elem]
            metal_period[i] = p
            metal_group[i] = g
            metal_en[i] = en
            metal_ir[i] = ir
            metal_de[i] = de
            metal_present[i] = 1

        # ---- co-solvent override features ----
        cans = parse_solvent_canonicals(row.solvents)
        cans_lower = set(cans)
        if cans_lower & CARBOX_NAMES:
            has_carbox[i] = 1
        if cans_lower & AROMATIC_NAMES:
            has_arom_solv[i] = 1

    # ---- assemble ----
    print(f"linker resolution: {n_resolved.sum()} resolved / {n_linkers.sum()} total occurrences")
    print(f"rows with ≥1 resolved linker: {(n_resolved>0).sum()}")
    print(f"rows with low_info_linker flag: {has_low_info.sum()}")
    print(f"rows with parsed metal: {metal_present.sum()}")
    print(f"rows with carboxylic cosolvent: {has_carbox.sum()}")
    print(f"rows with aromatic cosolvent: {has_arom_solv.sum()}")
    print()
    if n_unique_unresolved:
        top = sorted(n_unique_unresolved.items(), key=lambda kv: -kv[1])[:20]
        print("Top unresolved non-placeholder linkers (consider adding):")
        for nm, c in top:
            print(f"   {c:3d}  {nm}")

    # write parquet
    cols = {}
    cols["n_linkers"] = n_linkers
    cols["n_linkers_resolved"] = n_resolved
    cols["n_linkers_unresolved"] = n_unresolved
    cols["has_low_info_linker"] = has_low_info
    cols["linker_n_carboxylate"] = n_cooh_avg
    cols["linker_n_aromatic_ring"] = n_arom_avg
    cols["linker_n_n_donor"] = n_n_avg
    cols["linker_logp_avg"] = logp_avg
    cols["linker_mw_avg"] = mw_avg
    cols["metal_period"] = metal_period
    cols["metal_group"] = metal_group
    cols["metal_en"] = metal_en
    cols["metal_ionic_radius"] = metal_ir
    cols["metal_d_electrons"] = metal_de
    cols["metal_present"] = metal_present
    cols["has_carboxylic_cosolvent"] = has_carbox
    cols["has_aromatic_solvent"] = has_arom_solv
    for b in range(NBITS):
        cols[f"morgan_{b:04d}"] = morgan_acc[:, b]
    out = pd.DataFrame(cols)
    out_path = DATA / "chem_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  shape={out.shape}")

    # audit
    aud = OTHER / "chem_features_audit.md"
    nonzero_morgan = (morgan_acc.sum(axis=1) > 0).sum()
    with open(aud, "w") as f:
        f.write("# chem_features audit\n\n")
        f.write(f"- total rows: {n}\n")
        f.write(f"- rows with ≥1 resolved linker (nonzero Morgan): {nonzero_morgan}  ({nonzero_morgan/n:.1%})\n")
        f.write(f"- rows with parsed metal: {int(metal_present.sum())}  ({metal_present.sum()/n:.1%})\n")
        f.write(f"- rows with low_info_linker flag: {int(has_low_info.sum())}\n")
        f.write(f"- rows with has_carboxylic_cosolvent: {int(has_carbox.sum())}\n")
        f.write(f"- rows with has_aromatic_solvent: {int(has_arom_solv.sum())}\n\n")
        f.write("## Top unresolved non-placeholder linkers\n\n")
        if n_unique_unresolved:
            for nm, c in sorted(n_unique_unresolved.items(), key=lambda kv: -kv[1])[:40]:
                f.write(f"- {c}×  `{nm}`\n")
        else:
            f.write("(none)\n")
    print(f"wrote {aud}")


if __name__ == "__main__":
    main()
