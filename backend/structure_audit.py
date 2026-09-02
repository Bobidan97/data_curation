"""structure_audit.py — Detect structural errors / oddities in a SMILES column.

Flags molecules that are likely problematic for downstream cheminformatics —
unparseable structures, disconnected fragments (unstripped salts/mixtures),
inorganic or exotic-element species, radicals, isotopes, and net-charged
species. These aren't always errors, but they warrant a look before modelling.

The audit is additive and non-destructive: it appends a ``validity_issues``
column (a comma-separated list of flags, empty string when clean) so the user
can review and optionally filter.
"""
from __future__ import annotations

import pandas as pd

# Elements considered "normal" for small-molecule drug discovery. Anything else
# (metals, boron-beyond-basics, etc.) is flagged as an unusual atom.
ORGANIC_ELEMENTS: frozenset[str] = frozenset({
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
})

# Human-readable order for summaries
ISSUE_TYPES: tuple[str, ...] = (
    "Unparseable SMILES",
    "Disconnected (multiple fragments)",
    "No carbon atoms",
    "Unusual element",
    "Radical electrons",
    "Isotope label",
    "Net formal charge",
)


def _audit_one(smiles: str) -> list[str]:
    """Return the list of validity issues for a single SMILES (empty if clean)."""
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return ["Unparseable SMILES"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["Unparseable SMILES"]

    issues: list[str] = []

    # Disconnected — more than one fragment (salt, counter-ion, mixture)
    if len(Chem.GetMolFrags(mol)) > 1:
        issues.append("Disconnected (multiple fragments)")

    atoms = list(mol.GetAtoms())

    # No carbon (inorganic / non-drug-like)
    if not any(a.GetSymbol() == "C" for a in atoms):
        issues.append("No carbon atoms")

    # Unusual elements — collect the offending symbols for the message
    exotic = sorted({
        a.GetSymbol() for a in atoms if a.GetSymbol() not in ORGANIC_ELEMENTS
    })
    if exotic:
        issues.append("Unusual element (" + ", ".join(exotic) + ")")

    # Radical electrons
    if any(a.GetNumRadicalElectrons() > 0 for a in atoms):
        issues.append("Radical electrons")

    # Isotope labels
    if any(a.GetIsotope() != 0 for a in atoms):
        issues.append("Isotope label")

    # Net formal charge (a non-zero total suggests an un-neutralised species)
    net_charge = sum(a.GetFormalCharge() for a in atoms)
    if net_charge != 0:
        issues.append(f"Net formal charge ({net_charge:+d})")

    return issues


def audit_structures(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> pd.DataFrame:
    """Append a ``validity_issues`` column flagging structural problems.

    Returns a copy of df. Each entry is a comma-separated list of issue labels
    (empty string when the molecule is clean). Returns df unchanged if the
    SMILES column is absent or RDKit is unavailable.
    """
    if smiles_col not in df.columns:
        return df
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return df

    out = df.copy()
    out["validity_issues"] = [
        ", ".join(_audit_one(s)) for s in out[smiles_col]
    ]
    return out


def summarise_audit(audited_df: pd.DataFrame) -> dict:
    """Summarise an audited DataFrame (must contain ``validity_issues``).

    Returns:
        dict with:
            n_total   — rows audited
            n_flagged — rows with at least one issue
            n_clean   — rows with no issues
            by_issue  — DataFrame[Issue, Rows] counting each issue category
                        (a row with multiple issues is counted in each).
    """
    empty_summary = {
        "n_total": 0, "n_flagged": 0, "n_clean": 0,
        "by_issue": pd.DataFrame(columns=["Issue", "Rows"]),
    }
    if "validity_issues" not in audited_df.columns:
        return empty_summary

    col = audited_df["validity_issues"].fillna("")
    n_total = len(col)
    flagged_mask = col.str.len() > 0
    n_flagged = int(flagged_mask.sum())

    # Count by canonical issue category. The per-row string may contain
    # parameterised labels ("Unusual element (Na)", "Net formal charge (+1)"),
    # so we bucket by the ISSUE_TYPES prefix.
    counts: dict[str, int] = {t: 0 for t in ISSUE_TYPES}
    for entry in col[flagged_mask]:
        for part in entry.split(", "):
            for t in ISSUE_TYPES:
                # match on the stem before any parenthesised detail
                stem = t.split(" (")[0]
                if part == t or part.startswith(stem):
                    counts[t] += 1
                    break

    # Explicit columns= so an all-clean audit (no counts > 0) still produces a
    # DataFrame with the Issue/Rows schema instead of zero columns — otherwise
    # sort_values("Rows") raises KeyError on a genuinely clean dataset.
    by_issue = (
        pd.DataFrame(
            [{"Issue": t, "Rows": n} for t, n in counts.items() if n > 0],
            columns=["Issue", "Rows"],
        )
        .sort_values("Rows", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "n_total":   n_total,
        "n_flagged": n_flagged,
        "n_clean":   n_total - n_flagged,
        "by_issue":  by_issue,
    }
