import pandas as pd
import numpy as np
from dataclasses import dataclass

DROP_COLUMNS = [
    "standard_upper_value",
    "standard_text_value",
    "relation",
    "upper_value",
    "units",
    "value",
    "activity_comment",
    "activity_properties",
    "activity_id",
    "record_id",
    "qudt_units",
    "uo_units",
    "ligand_efficiency",
    "assay_variant_accession",
    "assay_variant_mutation",
    "action_type",
    "text_value",
    "toid",
    "data_validity_description"]


@dataclass
class CurationConfig:
    """Controls which optional curation steps are applied in curate_dataframe()."""
    exact_relation_only: bool = True          # keep only standard_relation == "="
    standardised_only: bool = True             # keep only standard_flag == 1
    remove_validity_issues: bool = True        # drop rows with data_validity_comment
    compute_pic50: bool = True                 # compute pIC50 column
    remove_structural_duplicates: bool = True  # re-canonicalise SMILES, then deduplicate


def curate_dataframe(
    df: pd.DataFrame,
    config: CurationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Curate a raw ChEMBL activity DataFrame into an ML-ready dataset.

    Hard-coded steps (always applied):
      1. Drop unwanted columns
      2. Drop rows missing canonical_smiles or standard_value
      3. Cast standard_value to numeric

    Configurable steps (controlled by CurationConfig):
      4. Keep only exact measurements (standard_relation == "=")
      5. Keep only standardised rows (standard_flag == 1)
      6. Remove rows with data validity issues
      7. Compute pIC50 from pchembl_value, falling back to -log10(IC50 in M) for nM rows
      8. Re-canonicalise SMILES with RDKit and remove structural duplicates

    Returns:
        (df_curated, df_dropped) — the curated DataFrame and a DataFrame of every
        dropped row with a 'reason' column inserted at position 0.
    """
    if config is None:
        config = CurationConfig()

    _dropped: list[pd.DataFrame] = []

    def _capture(rows: pd.DataFrame, reason: str) -> None:
        """Record dropped rows with the reason label in the first column."""
        if not rows.empty:
            chunk = rows.copy()
            chunk.insert(0, "reason", reason)
            _dropped.append(chunk)

    # step 1: drop unwanted columns (only those present in df)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop).copy()

    # step 2: drop rows missing SMILES or standard_value (always required for ML)
    required_cols = [c for c in ["canonical_smiles", "standard_value"] if c in df.columns]
    if required_cols:
        mask = df[required_cols].isna().any(axis=1)
        _capture(df[mask], "Missing SMILES or standard_value")
        df = df[~mask]

    # step 3: cast standard_value to numeric, drop any new NaNs
    if "standard_value" in df.columns:
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        mask = df["standard_value"].isna()
        _capture(df[mask], "Non-numeric standard_value")
        df = df[~mask]

    # step 4: exact measurements only
    if config.exact_relation_only and "standard_relation" in df.columns:
        mask = df["standard_relation"] != "="
        _capture(df[mask], 'Non-exact measurement (standard_relation ≠ "=")')
        df = df[~mask]

    # step 5: standardised units only
    if config.standardised_only and "standard_flag" in df.columns:
        mask = df["standard_flag"] != 1
        _capture(df[mask], "Not standardised (standard_flag ≠ 1)")
        df = df[~mask]

    # step 6: remove rows flagged with data validity issues
    if config.remove_validity_issues and "data_validity_comment" in df.columns:
        mask = df["data_validity_comment"].notna()
        _capture(df[mask], "Data validity issue")
        df = df[~mask]
        df = df.drop(columns=["data_validity_comment"])
    elif "data_validity_comment" in df.columns:
        df = df.drop(columns=["data_validity_comment"])

    # step 7: compute pIC50
    if config.compute_pic50:
        df["pIC50"] = pd.to_numeric(df.get("pchembl_value"), errors="coerce")
        if "standard_units" in df.columns:
            n_m_null = df["pIC50"].isna() & (df["standard_units"] == "nM")
            df.loc[n_m_null, "pIC50"] = -np.log10(df.loc[n_m_null, "standard_value"] * 1e-9)

    if "pchembl_value" in df.columns:
        df = df.drop(columns=["pchembl_value"])

    # step 8: re-canonicalise SMILES with RDKit, then remove structural duplicates
    if config.remove_structural_duplicates and "canonical_smiles" in df.columns:
        try:
            from rdkit import Chem

            def _to_canonical(smi: str):
                if not isinstance(smi, str):
                    return None
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol) if mol is not None else None

            df["canonical_smiles"] = df["canonical_smiles"].map(_to_canonical)

            mask_invalid = df["canonical_smiles"].isna()
            _capture(df[mask_invalid], "Unparseable SMILES")
            df = df[~mask_invalid]

            mask_dup = df.duplicated(subset=["canonical_smiles"], keep="first")
            _capture(df[mask_dup], "Structural duplicate")
            df = df[~mask_dup]

        except ImportError:
            pass  # skip silently if RDKit is not installed

    # Build the dropped-rows DataFrame
    if _dropped:
        df_dropped = pd.concat(_dropped, ignore_index=True)
    else:
        df_dropped = pd.DataFrame(columns=["reason"])

    return df.reset_index(drop=True), df_dropped


def find_structural_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify groups of molecules that are structurally identical.

    Re-canonicalises canonical_smiles with RDKit and groups rows that share the
    same canonical form.  Only groups with ≥ 2 members are returned.

    Returns a DataFrame with one row per duplicate group:
      canonical_smiles — the RDKit canonical form shared by the group
      original_smiles  — all original SMILES strings in the group (newline-separated)
      count            — total molecules in the group
      would_remove     — count − 1 (rows that deduplication would drop, keeping first)

    Returns an empty DataFrame if no duplicates are found or RDKit is unavailable.
    """
    if "canonical_smiles" not in df.columns:
        return pd.DataFrame()
    try:
        from rdkit import Chem
    except ImportError:
        return pd.DataFrame()

    def _to_canonical(smi):
        if not isinstance(smi, str):
            return None
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol is not None else None

    tmp = df[["canonical_smiles"]].copy()
    tmp["_canon"] = tmp["canonical_smiles"].map(_to_canonical)
    tmp = tmp.dropna(subset=["_canon"])

    grouped = (
        tmp.groupby("_canon")["canonical_smiles"]
        .agg(list)
        .reset_index()
        .rename(columns={"_canon": "canonical_smiles", "canonical_smiles": "_variants"})
    )
    grouped["count"] = grouped["_variants"].map(len)
    duplicates = grouped[grouped["count"] >= 2].copy()

    if duplicates.empty:
        return pd.DataFrame()

    duplicates["original_smiles"] = duplicates["_variants"].map(lambda v: "\n".join(v))
    duplicates["would_remove"] = duplicates["count"] - 1
    return (
        duplicates[["canonical_smiles", "original_smiles", "count", "would_remove"]]
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def get_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return all individual rows that belong to structural duplicate groups.

    Re-canonicalises canonical_smiles, identifies groups with ≥ 2 members,
    and returns the matching rows with two extra columns added:
      _canonical — the RDKit canonical SMILES (group key)
      _group     — human-readable label e.g. "Group 1", "Group 2" (largest group first)

    Returns an empty DataFrame if no duplicates, no SMILES column, or RDKit unavailable.
    Used to supply per-molecule activity data for the duplicate visualisation plot.
    """
    if "canonical_smiles" not in df.columns:
        return pd.DataFrame()
    try:
        from rdkit import Chem
    except ImportError:
        return pd.DataFrame()

    def _to_canonical(smi):
        if not isinstance(smi, str):
            return None
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol is not None else None

    df_work = df.copy()
    df_work["_canonical"] = df_work["canonical_smiles"].map(_to_canonical)

    counts = df_work["_canonical"].dropna().value_counts()
    dup_canons = counts[counts >= 2].index

    rows = df_work[df_work["_canonical"].isin(dup_canons)].copy()
    if rows.empty:
        return pd.DataFrame()

    # Label groups largest-first so Group 1 is the most duplicated molecule
    ordered = counts[counts >= 2].index.tolist()
    canon_to_group = {c: f"Group {i + 1}" for i, c in enumerate(ordered)}
    rows["_group"] = rows["_canonical"].map(canon_to_group)

    return rows.sort_values("_group").reset_index(drop=True)
