from __future__ import annotations

import pandas as pd


# All descriptor columns produced by compute_descriptors()
DESCRIPTOR_COLUMNS = [
    "MW", "LogP", "TPSA",
    "HBA", "HBD", "HeavyAtomCount", "RotatableBonds",
    "RingCount", "AromaticRings",
]


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add 9 standard RDKit molecular descriptors as new columns to a DataFrame.

    Computes: MW, LogP, TPSA, HBA, HBD, HeavyAtomCount, RotatableBonds,
    RingCount, AromaticRings.

    Requires a `canonical_smiles` column. Returns df unchanged if the column
    is absent. Invalid SMILES produce NaN for that row — they never raise.
    RDKit is imported inside this function so the module is safe to import
    even when RDKit is not installed.

    For any descriptor beyond these nine, users can ask the sidebar LLM chat
    box to compute it — the edit-DataFrame exec namespace exposes RDKit's
    `Chem`, `Descriptors`, and `rdMolDescriptors` modules.
    """
    if "canonical_smiles" not in df.columns:
        return df

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for descriptor computation. "
            "Install it with: pip install rdkit>=2023.9.1"
        ) from exc

    df = df.copy()

    # Parse all SMILES once — MolFromSmiles returns None for invalid SMILES
    mols = [
        Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        for smi in df["canonical_smiles"]
    ]

    def _apply(func, mol):
        """Return func(mol), or NaN when mol is None or an exception occurs."""
        if mol is None:
            return float("nan")
        try:
            return func(mol)
        except Exception:
            return float("nan")

    _DESCRIPTOR_FUNCS = {
        "MW":             Descriptors.MolWt,
        "LogP":           Descriptors.MolLogP,
        "TPSA":           Descriptors.TPSA,
        "HBA":            rdMolDescriptors.CalcNumHBA,
        "HBD":            rdMolDescriptors.CalcNumHBD,
        "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms,
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
        "RingCount":      rdMolDescriptors.CalcNumRings,
        "AromaticRings":  rdMolDescriptors.CalcNumAromaticRings,
    }
    for col, func in _DESCRIPTOR_FUNCS.items():
        df[col] = [_apply(func, m) for m in mols]

    return df


def compute_descriptor_correlations(
    df: pd.DataFrame,
    descriptor_cols: list[str],
    target_col: str = "pIC50",
) -> pd.DataFrame:
    """Correlate each descriptor column against the target (default pIC50).

    For every numeric descriptor, computes the Pearson r and Spearman rho
    against the target using the rows where both values are present. Pearson
    captures linear association; Spearman captures any monotonic trend (robust
    to outliers and non-linearity).

    Returns a DataFrame sorted by |Pearson r| descending with columns:
        descriptor · pearson_r · pearson_p · spearman_rho · n
    Descriptors with fewer than 3 valid paired values, or with zero variance,
    are skipped. Returns an empty DataFrame if the target column is absent.
    """
    import numpy as np

    empty = pd.DataFrame(
        columns=["descriptor", "pearson_r", "pearson_p", "spearman_rho", "n"]
    )
    if target_col not in df.columns:
        return empty

    from scipy.stats import pearsonr, spearmanr

    rows = []
    for col in descriptor_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        pair = df[[col, target_col]].dropna()
        if len(pair) < 3:
            continue
        x = pair[col].to_numpy(dtype=float)
        y = pair[target_col].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        try:
            pr, pp = pearsonr(x, y)
            sr, _ = spearmanr(x, y)
        except Exception:
            continue
        rows.append({
            "descriptor":   col,
            "pearson_r":    round(float(pr), 3),
            "pearson_p":    float(pp),
            "spearman_rho": round(float(sr), 3),
            "n":            int(len(pair)),
        })

    if not rows:
        return empty

    out = pd.DataFrame(rows)
    out = (
        out.assign(_abs=out["pearson_r"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )
    return out
