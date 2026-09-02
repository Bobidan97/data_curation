"""smarts.py — SMARTS-based substructure tools.

Five capabilities, one backend module:

    1. ``validate_smarts`` / ``filter_by_smarts`` / ``preview_smarts_filter``
       — substructure search and filtering for the working dataset.

    2. ``decompose_rgroups`` — Bemis–Murcko style R-group decomposition.
       Given a core SMARTS, splits every matching molecule into core + R1/R2/...

    3. ``smarts_match_atoms`` — atom indices that match a SMARTS on a given
       molecule. Drives the structure-highlight overlay used across the app.

    4. ``morgan_bit_to_smarts`` — convert a Morgan fingerprint bit on a
       specific molecule to its canonical SMARTS pattern. Used by the SHAP
       Explainability tab to print the substructure each top bit represents.

    5. Strictness flags exposed via ``COMPARE_OPTIONS`` for the cliff-pair MCS
       refinement (see :func:`chemical_space.compute_cliff_pair_mcs`).

All functions degrade gracefully (return empty / None / unchanged) when RDKit
is absent or the input is malformed.
"""
from __future__ import annotations

import pandas as pd


COMPARE_OPTIONS: tuple[str, ...] = ("any", "elements", "isotopes")


# ── Validation & basic filtering ──────────────────────────────────────────────

def validate_smarts(smarts: str) -> tuple[bool, str | None]:
    """Return ``(is_valid, error_message)`` for a SMARTS string."""
    if not isinstance(smarts, str) or not smarts.strip():
        return False, "Empty SMARTS pattern."
    try:
        from rdkit import Chem
    except ImportError:
        return False, "RDKit is not installed."
    try:
        mol = Chem.MolFromSmarts(smarts.strip())
    except Exception as exc:
        return False, f"Parse error: {exc}"
    if mol is None:
        return False, "RDKit could not parse this SMARTS pattern."
    return True, None


def _match_mask(df: pd.DataFrame, smarts: str, smiles_col: str) -> pd.Series:
    """Internal helper: bool Series, True where SMILES contains the SMARTS."""
    from rdkit import Chem

    pat = Chem.MolFromSmarts(smarts.strip())
    if pat is None or smiles_col not in df.columns:
        return pd.Series(False, index=df.index)

    def _hit(smi):
        if not isinstance(smi, str):
            return False
        try:
            m = Chem.MolFromSmiles(smi)
            return bool(m and m.HasSubstructMatch(pat))
        except Exception:
            return False

    return df[smiles_col].map(_hit)


def preview_smarts_filter(
    df: pd.DataFrame,
    smarts: str,
    smiles_col: str = "canonical_smiles",
) -> dict:
    """Quick stats on a SMARTS match — no DataFrame copy."""
    ok, _ = validate_smarts(smarts)
    if not ok or smiles_col not in df.columns:
        return {"n_match": 0, "n_total": len(df), "pct_match": 0.0}
    n_match = int(_match_mask(df, smarts, smiles_col).sum())
    n_total = len(df)
    return {
        "n_match":   n_match,
        "n_total":   n_total,
        "pct_match": round(n_match / n_total * 100, 1) if n_total else 0.0,
    }


def filter_by_smarts(
    df: pd.DataFrame,
    smarts: str,
    keep_matches: bool = True,
    smiles_col: str = "canonical_smiles",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df by SMARTS hit. Returns ``(kept, dropped)``.

    Args:
        keep_matches: True ⇒ keep rows where the SMARTS matches (substructure
                      search). False ⇒ drop rows where it matches (anti-filter).
    """
    ok, _ = validate_smarts(smarts)
    if not ok or smiles_col not in df.columns:
        return df.reset_index(drop=True), pd.DataFrame(columns=df.columns)

    mask = _match_mask(df, smarts, smiles_col)
    kept_mask    = mask if keep_matches else ~mask
    dropped_mask = ~kept_mask
    return (
        df.loc[kept_mask].reset_index(drop=True),
        df.loc[dropped_mask].reset_index(drop=True),
    )


# ── Atom-level matching (drives the highlight overlay) ────────────────────────

def smarts_match_atoms(smiles: str, smarts: str) -> list[int]:
    """Return atom indices on ``smiles`` that match ``smarts``.

    Returns the **union** of all matches — useful for highlighting every
    occurrence of a pattern on a molecule.
    """
    ok, _ = validate_smarts(smarts)
    if not ok or not isinstance(smiles, str):
        return []
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        pat = Chem.MolFromSmarts(smarts.strip())
        if mol is None or pat is None:
            return []
        all_matches = mol.GetSubstructMatches(pat)
        atoms: set[int] = set()
        for match in all_matches:
            atoms.update(match)
        return sorted(atoms)
    except Exception:
        return []


# ── R-group decomposition ─────────────────────────────────────────────────────

def decompose_rgroups(
    df: pd.DataFrame,
    core_smarts: str,
    smiles_col: str = "canonical_smiles",
) -> dict:
    """R-group decomposition against a user-supplied core SMARTS.

    Returns a dict with:
        rgroups_df:   DataFrame with ``Core``, ``R1``, ``R2``, … columns
                      plus carried-over metadata
                      (molecule_chembl_id, pIC50, canonical_smiles).
        n_decomposed: int — how many rows matched the core and decomposed
        n_unmatched:  int — molecules that didn't match the core
        r_columns:    list[str] — R-group column names produced
                      (e.g. ["R1", "R2"]). Empty if decomposition failed.
        core_smarts:  the input pattern (echoed for the UI header)
    """
    ok, err = validate_smarts(core_smarts)
    if not ok:
        raise ValueError(f"Invalid core SMARTS: {err}")
    if smiles_col not in df.columns:
        raise ValueError(f"DataFrame missing '{smiles_col}'.")

    from rdkit import Chem
    from rdkit.Chem import rdRGroupDecomposition

    core_mol = Chem.MolFromSmarts(core_smarts.strip())

    mols: list = []
    keep_rows: list[int] = []
    for idx, smi in enumerate(df[smiles_col]):
        if not isinstance(smi, str):
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None or not m.HasSubstructMatch(core_mol):
            continue
        mols.append(m)
        keep_rows.append(idx)

    if not mols:
        return {
            "rgroups_df":   pd.DataFrame(),
            "n_decomposed": 0,
            "n_unmatched":  len(df),
            "r_columns":    [],
            "core_smarts":  core_smarts.strip(),
        }

    try:
        decomp, unmatched_indices = rdRGroupDecomposition.RGroupDecompose(
            [core_mol], mols, asSmiles=True
        )
    except Exception as exc:
        raise ValueError(f"R-group decomposition failed: {exc}") from exc

    if not decomp:
        return {
            "rgroups_df":   pd.DataFrame(),
            "n_decomposed": 0,
            "n_unmatched":  len(df),
            "r_columns":    [],
            "core_smarts":  core_smarts.strip(),
        }

    # Keep only the rows whose mols were successfully decomposed
    successful_rows = [
        keep_rows[i] for i in range(len(mols)) if i not in set(unmatched_indices)
    ]
    rgroups_df = pd.DataFrame(decomp)

    # R-group columns — they look like "R1", "R2", etc. — preserve the order
    r_cols = [c for c in rgroups_df.columns if c.startswith("R")]
    r_cols.sort(key=lambda c: int(c[1:]) if c[1:].isdigit() else 999)

    # Carry through useful metadata
    meta_cols = [
        c for c in ("molecule_chembl_id", "pIC50", "canonical_smiles")
        if c in df.columns
    ]
    meta = df.loc[successful_rows, meta_cols].reset_index(drop=True)

    out = pd.concat([meta, rgroups_df[["Core"] + r_cols].reset_index(drop=True)], axis=1)

    return {
        "rgroups_df":   out,
        "n_decomposed": len(out),
        "n_unmatched":  len(df) - len(out),
        "r_columns":    r_cols,
        "core_smarts":  core_smarts.strip(),
    }


# ── Morgan bit → canonical SMARTS pattern ─────────────────────────────────────

def morgan_bit_to_smarts(smiles: str, bit_id: int, radius: int = 2) -> str | None:
    """Convert a Morgan fingerprint bit (as it appears on ``smiles``) to SMARTS.

    Picks the first atom-environment that lit the bit, extracts the bond
    neighbourhood at the bit's effective radius, and canonicalises it as
    SMARTS. Returns None if the bit isn't present on this molecule.
    """
    if not isinstance(smiles, str):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bi: dict = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=2048, bitInfo=bi
    )
    centres = bi.get(int(bit_id), [])
    if not centres:
        return None

    centre_atom, env_radius = centres[0]
    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, centre_atom)
    if not env_bonds:
        # Radius-0 case — single atom
        atom = mol.GetAtomWithIdx(centre_atom)
        return f"[#{atom.GetAtomicNum()}]"

    atom_map: dict = {}
    try:
        submol = Chem.PathToSubmol(mol, env_bonds, atomMap=atom_map)
        if submol is None or submol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmarts(submol)
    except Exception:
        return None
