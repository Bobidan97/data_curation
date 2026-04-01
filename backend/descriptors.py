from __future__ import annotations

import pandas as pd


# All descriptor columns produced by compute_descriptors()
DESCRIPTOR_COLUMNS = [
    "MW", "LogP", "TPSA",
    "HBA", "HBD", "HeavyAtomCount", "RotatableBonds",
    "RingCount", "AromaticRings",
]


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add RDKit molecular descriptors as new columns to a DataFrame.

    Computes 10 standard descriptors in one pass:MW, LogP, TPSA, HBA, HBD, Ro5_violations,
    HeavyAtomCount, RotatableBonds, RingCount, AromaticRings

    Requires a `canonical_smiles` column. Returns df unchanged if the column
    is absent. Invalid SMILES produce NaN for that row — they never raise.
    RDKit is imported inside this function so the module is safe to import
    even when RDKit is not installed.
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

    df["MW"]   = [_apply(Descriptors.MolWt, m) for m in mols]
    df["LogP"] = [_apply(Descriptors.MolLogP, m) for m in mols]
    df["TPSA"] = [_apply(Descriptors.TPSA, m) for m in mols]
    df["HBA"] = [_apply(rdMolDescriptors.CalcNumHBA, m) for m in mols]
    df["HBD"] = [_apply(rdMolDescriptors.CalcNumHBD, m) for m in mols]
    df["HeavyAtomCount"] = [_apply(rdMolDescriptors.CalcNumHeavyAtoms, m) for m in mols]
    df["RotatableBonds"] = [_apply(rdMolDescriptors.CalcNumRotatableBonds, m) for m in mols]
    df["RingCount"]      = [_apply(rdMolDescriptors.CalcNumRings, m) for m in mols]
    df["AromaticRings"]  = [_apply(rdMolDescriptors.CalcNumAromaticRings, m) for m in mols]

    return df
