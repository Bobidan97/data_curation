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
    strip_salts: bool = True                   # keep only the largest organic fragment
    flag_structural_alerts: bool = False       # PAINS/Brenk flags as a column (no removal)


# ── Salt stripping ────────────────────────────────────────────────────────────
# RDKit's LargestFragmentChooser picks the heavy-atom fragment (the parent drug)
# and discards counter-ions, solvents and other components. Built once and reused.

_LARGEST_FRAGMENT_CHOOSER = None


def _get_largest_fragment_chooser():
    """Return a cached LargestFragmentChooser instance (built lazily)."""
    global _LARGEST_FRAGMENT_CHOOSER
    if _LARGEST_FRAGMENT_CHOOSER is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()
    return _LARGEST_FRAGMENT_CHOOSER


def _to_canonical(smi: str, strip_salts: bool = True) -> str | None:
    """Re-canonicalise a SMILES string with RDKit; returns None for invalid input.

    When ``strip_salts`` is True (default), the largest connected fragment is
    chosen first via ``rdMolStandardize.LargestFragmentChooser`` so that salts,
    counter-ions and co-crystallised solvents are discarded.
    """
    if not isinstance(smi, str):
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if strip_salts:
        try:
            mol = _get_largest_fragment_chooser().choose(mol)
        except Exception:
            # If the standardiser barfs on an exotic input, fall back to the
            # original mol — better to keep a possibly-salty form than to drop
            # the row entirely.
            pass
    return Chem.MolToSmiles(mol)


# ── Structural-alert catalogues (PAINS, Brenk, NIH, ZINC) ─────────────────────

def flag_structural_alerts(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> pd.DataFrame:
    """Add a ``structural_alerts`` column listing matched PAINS/Brenk/NIH/ZINC entries.

    The column is a comma-separated string of matched alert descriptions, or an
    empty string when nothing matches. Rows are NOT removed — surface, don't
    auto-drop. Returns a *copy* of df with the new column appended.

    Silently no-ops (returns ``df`` unchanged) when RDKit is not installed.
    """
    if smiles_col not in df.columns:
        return df
    try:
        from rdkit import Chem
    except ImportError:
        return df
    from utils.rdkit_utils import alert_catalog
    catalog = alert_catalog()
    if catalog is None:
        return df

    results: list[str] = []
    for smi in df[smiles_col]:
        if not isinstance(smi, str):
            results.append("")
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append("")
                continue
            matches = catalog.GetMatches(mol)
            results.append(", ".join(m.GetDescription() for m in matches) if matches else "")
        except Exception:
            results.append("")

    out = df.copy()
    out["structural_alerts"] = results
    return out


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

    All per-row property checks (steps 2–6) are evaluated against the same
    snapshot of the data, so a row that fails multiple checks accumulates ALL
    applicable reasons in its dropped-row entry (separated by "; ").
    Structural deduplication (step 8) is evaluated separately because it is
    cross-row — only the duplicate of a survivor is flagged.

    Returns:
        (df_curated, df_dropped) — the curated DataFrame and a DataFrame of every
        dropped row with a 'reason' column inserted at position 0. The reason
        column is a "; "-separated string of every reason that applied.
    """
    if config is None:
        config = CurationConfig()

    # step 1: drop unwanted columns (only those present in df)
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop).copy()

    # ── Per-row independent checks (steps 2–6): evaluate ALL simultaneously ───
    # Each row accumulates every reason that applies, then we drop in one pass.
    reason_masks: dict[str, pd.Series] = {}

    if "canonical_smiles" in df.columns:
        reason_masks["Missing canonical_smiles"] = df["canonical_smiles"].isna()

    if "standard_value" in df.columns:
        reason_masks["Missing standard_value"] = df["standard_value"].isna()
        _numeric = pd.to_numeric(df["standard_value"], errors="coerce")
        # Non-numeric only flags rows that were non-null but couldn't be coerced
        # (mutually exclusive with "Missing standard_value")
        reason_masks["Non-numeric standard_value"] = (
            df["standard_value"].notna() & _numeric.isna()
        )

    if config.exact_relation_only and "standard_relation" in df.columns:
        reason_masks['Non-exact measurement (standard_relation ≠ "=")'] = (
            df["standard_relation"] != "="
        )

    if config.standardised_only and "standard_flag" in df.columns:
        reason_masks["Not standardised (standard_flag ≠ 1)"] = df["standard_flag"] != 1

    if config.remove_validity_issues and "data_validity_comment" in df.columns:
        reason_masks["Data validity issue"] = df["data_validity_comment"].notna()

    # Build per-row "; "-joined reason strings for rows that failed any check
    if reason_masks:
        combined_mask = pd.Series(False, index=df.index)
        for m in reason_masks.values():
            combined_mask |= m

        def _reasons_for(idx) -> str:
            return "; ".join(name for name, mask in reason_masks.items() if mask.loc[idx])

        df_dropped_phase1 = df.loc[combined_mask].copy()
        df_dropped_phase1.insert(
            0, "reason", df_dropped_phase1.index.map(_reasons_for)
        )
        df = df.loc[~combined_mask].copy()
    else:
        df_dropped_phase1 = pd.DataFrame(columns=["reason"])

    # step 3 (post-drop): standard_value is now safe to coerce to numeric
    if "standard_value" in df.columns:
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    if "data_validity_comment" in df.columns:
        df = df.drop(columns=["data_validity_comment"])

    # step 7: compute pIC50
    if config.compute_pic50:
        df["pIC50"] = pd.to_numeric(df.get("pchembl_value"), errors="coerce")
        if "standard_units" in df.columns:
            n_m_null = df["pIC50"].isna() & (df["standard_units"] == "nM")
            df.loc[n_m_null, "pIC50"] = -np.log10(df.loc[n_m_null, "standard_value"] * 1e-9)

    if "pchembl_value" in df.columns:
        df = df.drop(columns=["pchembl_value"])

    # step 8: re-canonicalise SMILES with RDKit (optionally strip salts first),
    # then remove structural duplicates. Cross-row — kept separate from the
    # per-row reason accumulation.
    _phase2_drops: list[pd.DataFrame] = []
    if config.remove_structural_duplicates and "canonical_smiles" in df.columns:
        try:
            df["canonical_smiles"] = df["canonical_smiles"].map(
                lambda s: _to_canonical(s, strip_salts=config.strip_salts)
            )

            mask_invalid = df["canonical_smiles"].isna()
            if mask_invalid.any():
                invalid_rows = df[mask_invalid].copy()
                invalid_rows.insert(0, "reason", "Unparseable SMILES")
                _phase2_drops.append(invalid_rows)
                df = df[~mask_invalid]

            mask_dup = df.duplicated(subset=["canonical_smiles"], keep="first")
            if mask_dup.any():
                dup_rows = df[mask_dup].copy()
                dup_rows.insert(0, "reason", "Structural duplicate")
                _phase2_drops.append(dup_rows)
                df = df[~mask_dup]

        except ImportError:
            pass  # skip silently if RDKit is not installed

    # step 9 (optional): annotate surviving rows with structural-alert matches
    # (does NOT drop anything — adds a column for downstream filtering / review).
    if config.flag_structural_alerts and "canonical_smiles" in df.columns:
        try:
            df = flag_structural_alerts(df)
        except Exception:
            pass

    # Combine all dropped rows
    _all_drops = []
    if not df_dropped_phase1.empty:
        _all_drops.append(df_dropped_phase1)
    _all_drops.extend(_phase2_drops)

    if _all_drops:
        df_dropped = pd.concat(_all_drops, ignore_index=True)
    else:
        df_dropped = pd.DataFrame(columns=["reason"])

    return df.reset_index(drop=True), df_dropped


def summarise_drops(df_dropped: pd.DataFrame) -> dict:
    """Summarise a dropped-rows DataFrame produced by :func:`curate_dataframe`.

    Returns a dict with four keys:
        total          — int: total rows dropped
        multi_reason   — int: rows that failed more than one check
        by_reason      — DataFrame[Reason, Rows]: count per individual reason.
                         Rows with multiple reasons are counted in EACH of
                         their reasons (so the column sum may exceed `total`).
        by_combination — DataFrame[Reasons, Rows]: count per exact reason
                         combination. Each row is counted exactly once.
    """
    if df_dropped.empty or "reason" not in df_dropped.columns:
        return {
            "total":          0,
            "multi_reason":   0,
            "by_reason":      pd.DataFrame(columns=["Reason", "Rows"]),
            "by_combination": pd.DataFrame(columns=["Reasons", "Rows"]),
        }

    total = len(df_dropped)
    reason_lists = df_dropped["reason"].fillna("").str.split("; ")
    multi_reason = int((reason_lists.str.len() > 1).sum())

    # by_reason: every individual reason, multi-reason rows counted in each
    flat = [r for sublist in reason_lists for r in sublist if r]
    by_reason = (
        pd.Series(flat, dtype=object)
        .value_counts()
        .rename_axis("Reason")
        .reset_index(name="Rows")
    )

    # by_combination: unique reason set, each row counted once
    by_combination = (
        df_dropped["reason"]
        .value_counts()
        .rename_axis("Reasons")
        .reset_index(name="Rows")
    )

    return {
        "total":          total,
        "multi_reason":   multi_reason,
        "by_reason":      by_reason,
        "by_combination": by_combination,
    }


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
        from rdkit import Chem  # noqa: F401 — validates RDKit is available
    except ImportError:
        return pd.DataFrame()

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
        from rdkit import Chem  # noqa: F401 — validates RDKit is available
    except ImportError:
        return pd.DataFrame()

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
