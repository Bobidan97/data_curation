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
    exact_relation_only: bool = True    # keep only standard_relation == "="
    standardised_only: bool = True       # keep only standard_flag == 1
    remove_validity_issues: bool = True  # drop rows with data_validity_comment
    compute_pic50: bool = True           # compute pIC50 column


def curate_dataframe(df: pd.DataFrame, config: CurationConfig | None = None) -> pd.DataFrame:
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
    """
    if config is None:
        config = CurationConfig()

    # step 1: drop unwanted columns (only those present in df)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop).copy()

    # step 2: drop rows missing SMILES or standard_value (always required for ML)
    required_cols = [c for c in ["canonical_smiles", "standard_value"] if c in df.columns]
    df = df.dropna(subset=required_cols)

    # step 3: cast standard_value to numeric, drop any new NaNs
    if "standard_value" in df.columns:
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value"])

    # step 4: exact measurements only
    if config.exact_relation_only and "standard_relation" in df.columns:
        df = df[df["standard_relation"] == "="]

    # step 5: standardised units only
    if config.standardised_only and "standard_flag" in df.columns:
        df = df[df["standard_flag"] == 1]

    # step 6: remove rows flagged with data validity issues
    if config.remove_validity_issues and "data_validity_comment" in df.columns:
        df = df[df["data_validity_comment"].isna()]
        df = df.drop(columns=["data_validity_comment"])

    # step 7: compute pIC50
    if config.compute_pic50:
        df["pIC50"] = pd.to_numeric(df.get("pchembl_value"), errors="coerce")
        if "standard_units" in df.columns:
            n_m_null = df["pIC50"].isna() & (df["standard_units"] == "nM")
            df.loc[n_m_null, "pIC50"] = -np.log10(df.loc[n_m_null, "standard_value"] * 1e-9)

    if "pchembl_value" in df.columns:
        df = df.drop(columns=["pchembl_value"])

    return df.reset_index(drop=True)
