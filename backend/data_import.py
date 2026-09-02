"""data_import.py — Import a user-supplied compound list into the app's
working-dataset shape, bypassing ChEMBL entity resolution entirely.

Used by the Phase 1 "upload your own compounds" path: the user provides a CSV
with a SMILES column (and optionally a name/ID column), and this module
validates + reshapes it into the same column layout ChEMBL fetches produce
(canonical_smiles, molecule_chembl_id), so every downstream phase — curation,
descriptors, chemical space, modelling — works unmodified.
"""
from __future__ import annotations

import pandas as pd

# Column-name aliases used for auto-detection (case-insensitive, exact match
# on the stripped header). Falls back to manual selection when nothing matches.
_SMILES_ALIASES = {"smiles", "smile", "canonical_smiles", "structure", "smiles_string"}
_NAME_ALIASES = {"name", "compound_name", "molecule_name", "id", "title", "compound"}


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Guess (smiles_col, name_col) from a DataFrame's headers.

    Returns None for either slot when no confident match is found — the
    caller should fall back to manual column selection.
    """
    smiles_col: str | None = None
    name_col: str | None = None
    for col in df.columns:
        key = str(col).strip().lower()
        if smiles_col is None and key in _SMILES_ALIASES:
            smiles_col = col
        elif name_col is None and key in _NAME_ALIASES:
            name_col = col
    return smiles_col, name_col


def prepare_compound_import(
    df: pd.DataFrame,
    smiles_col: str,
    name_col: str | None = None,
) -> dict:
    """Validate and reshape an uploaded compound table into raw_df shape.

    Every row's SMILES is parsed with RDKit and re-canonicalised; unparseable
    rows are dropped and reported separately rather than silently discarded.

    Args:
        df:         The uploaded table (as read by pandas).
        smiles_col: Column holding SMILES strings.
        name_col:   Optional column holding a compound name/ID. When absent,
                    rows are labelled "Compound 1", "Compound 2", …

    Returns:
        dict with:
            df         — DataFrame[canonical_smiles, name, molecule_chembl_id]
                         for the valid rows only. molecule_chembl_id reuses
                         the name/label so existing display code (which looks
                         for that column) works without modification — it is
                         NOT a real ChEMBL identifier.
            n_input    — rows in the original upload
            n_valid    — rows that parsed successfully
            n_invalid  — rows dropped for unparseable SMILES
            invalid_df — the dropped rows (original columns + a 'reason')

    Raises:
        ValueError: if smiles_col is not a column in df.
    """
    from rdkit import Chem

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in the uploaded file.")

    work = df.copy()
    n_input = len(work)

    def _canonicalise(smi) -> str | None:
        if not isinstance(smi, str) or not smi.strip():
            return None
        mol = Chem.MolFromSmiles(smi.strip())
        return Chem.MolToSmiles(mol) if mol is not None else None

    work["_canonical"] = work[smiles_col].map(_canonicalise)
    invalid_mask = work["_canonical"].isna()

    invalid_df = work[invalid_mask].drop(columns=["_canonical"]).copy()
    if not invalid_df.empty:
        invalid_df.insert(0, "reason", "Unparseable SMILES")

    valid = work[~invalid_mask].reset_index(drop=True)

    if name_col and name_col in valid.columns:
        labels = valid[name_col].astype(str).tolist()
    else:
        labels = [f"Compound {i + 1}" for i in range(len(valid))]

    out = pd.DataFrame({
        "canonical_smiles":   valid["_canonical"].tolist(),
        "name":               labels,
        "molecule_chembl_id": labels,
    })

    return {
        "df":         out,
        "n_input":    n_input,
        "n_valid":    len(out),
        "n_invalid":  int(invalid_mask.sum()),
        "invalid_df": invalid_df,
    }
