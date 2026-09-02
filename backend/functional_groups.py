"""functional_groups.py — Model-free SAR: which functional groups track activity.

Screens each molecule for a curated library of common medicinal-chemistry
functional groups (SMARTS), then splits pIC50 by presence/absence of each group
to estimate its **impact on activity** — the mean pIC50 difference between
molecules that have the group and those that don't, with a non-parametric
significance test.

This is a fast, model-independent read on structure–activity relationships:
"molecules with a sulfonamide are, on average, 0.8 pIC50 units more potent
(p = 0.01)".
"""
from __future__ import annotations

import pandas as pd

# Curated functional-group SMARTS library. Kept deliberately readable — each
# entry is (display name, SMARTS pattern). Order is not significant.
FUNCTIONAL_GROUPS: dict[str, str] = {
    "Carboxylic acid":    "C(=O)[OX2H1]",
    "Carboxylate anion":  "C(=O)[O-]",
    "Ester":              "C(=O)O[#6]",
    "Amide":              "C(=O)[NX3]",
    "Primary amine":      "[NX3;H2;!$(NC=O);!$(N=*)]",
    "Secondary amine":    "[NX3;H1;!$(NC=O);!$(N=*)]",
    "Tertiary amine":     "[NX3;H0;!$(NC=O);!$(N=*);!$([N+])]",
    "Aniline":            "[NX3;H2]c",
    "Hydroxyl":           "[OX2H]",
    "Phenol":             "c[OX2H]",
    "Ether":              "[OD2]([#6])[#6]",
    "Ketone":             "[#6][CX3](=O)[#6]",
    "Aldehyde":           "[CX3H1](=O)",
    "Nitrile":            "C#N",
    "Nitro":              "[NX3+](=O)[O-]",
    "Sulfonamide":        "S(=O)(=O)[NX3]",
    "Sulfone":            "[#6]S(=O)(=O)[#6]",
    "Trifluoromethyl":    "C(F)(F)F",
    "Fluorine":           "[F]",
    "Chlorine":           "[Cl]",
    "Bromo/iodo":         "[Br,I]",
    "Pyridine":           "c1ccncc1",
    "Aromatic N (any)":   "[n]",
    "Urea":               "[NX3][CX3](=O)[NX3]",
    "Guanidine":          "[NX3][CX3](=[NX2])[NX3]",
    "Piperazine":         "C1CNCCN1",
    "Morpholine":         "C1COCCN1",
}


def compute_functional_group_impact(
    df: pd.DataFrame,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Estimate each functional group's impact on pIC50.

    For every group in FUNCTIONAL_GROUPS, molecules are split into those that
    contain the substructure and those that don't. The impact is the difference
    in mean pIC50 (present − absent); a Mann–Whitney U test gives a p-value.

    Only groups where BOTH the present and absent sets have at least
    ``min_group_size`` molecules are reported (otherwise the comparison is
    meaningless).

    Args:
        df:             DataFrame with 'canonical_smiles' and 'pIC50'.
        min_group_size: Minimum molecules required in each of the present/absent
                        sets for a group to be reported.

    Returns:
        DataFrame sorted by |impact| descending with columns:
            group · smarts · n_present · n_absent · mean_present ·
            mean_absent · impact · p_value
        Empty DataFrame if inputs are missing or nothing qualifies.
    """
    empty = pd.DataFrame(columns=[
        "group", "smarts", "n_present", "n_absent",
        "mean_present", "mean_absent", "impact", "p_value",
    ])
    for col in ("canonical_smiles", "pIC50"):
        if col not in df.columns:
            return empty

    try:
        from rdkit import Chem
    except ImportError:
        return empty
    from scipy.stats import mannwhitneyu

    work = df[["canonical_smiles", "pIC50"]].dropna().reset_index(drop=True)
    if len(work) < 2 * min_group_size:
        return empty

    mols = [Chem.MolFromSmiles(s) for s in work["canonical_smiles"]]
    pic50 = work["pIC50"].to_numpy(dtype=float)

    rows = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        present_mask = [
            (m is not None and m.HasSubstructMatch(patt)) for m in mols
        ]
        present = pic50[present_mask]
        absent = pic50[[not p for p in present_mask]]
        if len(present) < min_group_size or len(absent) < min_group_size:
            continue

        mean_p = float(present.mean())
        mean_a = float(absent.mean())
        try:
            _, p_val = mannwhitneyu(present, absent, alternative="two-sided")
        except ValueError:
            p_val = float("nan")

        rows.append({
            "group":        name,
            "smarts":       smarts,
            "n_present":    int(len(present)),
            "n_absent":     int(len(absent)),
            "mean_present": round(mean_p, 2),
            "mean_absent":  round(mean_a, 2),
            "impact":       round(mean_p - mean_a, 2),
            "p_value":      float(p_val),
        })

    if not rows:
        return empty

    out = pd.DataFrame(rows)
    out = (
        out.assign(_abs=out["impact"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )
    return out


def example_with_group(df: pd.DataFrame, smarts: str) -> str | None:
    """Return one canonical SMILES from df that contains the given SMARTS group."""
    if "canonical_smiles" not in df.columns:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None
    for s in df["canonical_smiles"]:
        if not isinstance(s, str):
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None and m.HasSubstructMatch(patt):
            return s
    return None
