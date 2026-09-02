"""desirability.py — Multi-criteria desirability ranking for compound triage.

Combines several drug-discovery-relevant metrics into a single rankable score,
the kind of output a medicinal chemist would actually use for triage:

    - **Predicted pIC50**       (only when a trained model is supplied)
    - **QED**                   (Bickerton drug-likeness, 0-1)
    - **Lipinski violations**   (0-4 penalty count from Ro5)
    - **Structural alerts**     (PAINS / Brenk / NIH / ZINC matches from RDKit)
    - **Heavy atom count**      (sanity — too small / too large compounds penalised)

Each component is normalised to [0, 1] (higher = more desirable) and combined as
a weighted sum. Reasonable defaults are provided; the caller can override
weights or skip components by setting the weight to 0.

The function never raises — invalid SMILES yield ``None`` scores that fall to
the bottom of the ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Default weights used when the caller doesn't override
DEFAULT_WEIGHTS: dict[str, float] = {
    "predicted_pIC50":   0.40,   # bioactivity is the dominant signal when available
    "QED":               0.30,
    "lipinski":          0.15,
    "structural_alerts": 0.10,
    "size":              0.05,
}


def _safe_mol(smi):
    from rdkit import Chem
    if not isinstance(smi, str):
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _qed(mol) -> float | None:
    if mol is None:
        return None
    try:
        from rdkit.Chem import QED
        return float(QED.qed(mol))
    except Exception:
        return None


def _lipinski_violations(mol) -> int | None:
    """Count of Lipinski's Rule-of-5 violations (0-4). Lower is better."""
    if mol is None:
        return None
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors
        viols = 0
        if Descriptors.MolWt(mol)             > 500: viols += 1
        if Descriptors.MolLogP(mol)           >   5: viols += 1
        if rdMolDescriptors.CalcNumHBA(mol)   >  10: viols += 1
        if rdMolDescriptors.CalcNumHBD(mol)   >   5: viols += 1
        return int(viols)
    except Exception:
        return None


def _heavy_atom_size_score(mol) -> float | None:
    """Triangular score peaked around 20-40 heavy atoms (typical drug-like range)."""
    if mol is None:
        return None
    try:
        n = mol.GetNumHeavyAtoms()
    except Exception:
        return None
    if   n < 10:           return float(max(0, (n - 5) / 5))            # ramp up from 5 to 10
    elif n <= 40:          return 1.0                                    # full credit 10-40
    elif n <= 70:          return float(max(0.0, (70 - n) / 30))         # ramp down 40-70
    return 0.0


def _alert_count(mol, catalog) -> int | None:
    if mol is None or catalog is None:
        return None
    try:
        return int(len(catalog.GetMatches(mol)))
    except Exception:
        return None


def _get_filter_catalog():
    """Cached PAINS/Brenk/NIH/ZINC FilterCatalog (shared via utils.rdkit_utils)."""
    from utils.rdkit_utils import alert_catalog
    return alert_catalog()


def compute_desirability(
    df: pd.DataFrame,
    predicted_pic50: pd.Series | None = None,
    weights: dict[str, float] | None = None,
    pic50_range: tuple[float, float] = (4.0, 9.0),
) -> pd.DataFrame:
    """Score each row by a weighted combination of drug-discovery criteria.

    Args:
        df:              DataFrame with at least a ``canonical_smiles`` column.
        predicted_pic50: Optional pre-aligned Series of predicted pIC50 values
                         (e.g. from a trained model). When None, the
                         ``predicted_pIC50`` component is skipped and its
                         weight redistributed across the remaining components.
        weights:         Override the default component weights. Set any weight
                         to 0 to skip that component.
        pic50_range:     Min/max used to normalise predicted_pIC50 to [0, 1].

    Returns:
        DataFrame with one row per input molecule and these columns added:
            QED · lipinski_violations · n_structural_alerts · heavy_atoms
            (predicted_pIC50 — when supplied)
            component_score_* — per-component normalised [0,1] values
            desirability      — weighted aggregate, [0,1] (higher = better)
            desirability_pct  — desirability × 100, 1 decimal
        The DataFrame is sorted by desirability descending.
    """
    if "canonical_smiles" not in df.columns:
        raise ValueError("DataFrame must contain a 'canonical_smiles' column.")

    weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
    # If the caller didn't supply predictions, drop that component cleanly
    if predicted_pic50 is None:
        weights["predicted_pIC50"] = 0.0

    catalog = _get_filter_catalog()

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    if predicted_pic50 is not None:
        # Align by position to the input df (caller's responsibility)
        out["predicted_pIC50"] = pd.Series(predicted_pic50).reset_index(drop=True).values

    qed_vals:      list = []
    lip_vals:      list = []
    alert_vals:    list = []
    size_vals:     list = []
    heavy_atoms:   list = []

    for smi in out["canonical_smiles"]:
        mol = _safe_mol(smi)
        qed_vals.append(_qed(mol))
        lip_vals.append(_lipinski_violations(mol))
        alert_vals.append(_alert_count(mol, catalog))
        size_vals.append(_heavy_atom_size_score(mol))
        heavy_atoms.append(int(mol.GetNumHeavyAtoms()) if mol is not None else None)

    out["QED"]                  = qed_vals
    out["lipinski_violations"]  = lip_vals
    out["n_structural_alerts"]  = alert_vals
    out["heavy_atoms"]          = heavy_atoms

    # ── Normalise each component to [0, 1] (higher = more desirable) ──────────
    # predicted_pIC50 — linear scale within configured range
    if predicted_pic50 is not None:
        lo, hi = pic50_range
        rng = max(hi - lo, 1e-6)
        out["component_score_predicted_pIC50"] = (
            (out["predicted_pIC50"] - lo) / rng
        ).clip(0, 1)

    # QED is already 0-1
    out["component_score_QED"] = out["QED"].clip(0, 1)

    # Lipinski: 0 violations → 1.0, every violation -0.25
    out["component_score_lipinski"] = (
        (4 - out["lipinski_violations"].fillna(4)).clip(0, 4) / 4.0
    )

    # Alerts: 0 alerts → 1.0, 5+ alerts → 0.0
    out["component_score_structural_alerts"] = (
        (5 - out["n_structural_alerts"].fillna(5)).clip(0, 5) / 5.0
    )

    # Size already normalised
    out["component_score_size"] = pd.Series(size_vals).fillna(0).clip(0, 1)

    # ── Weighted sum (renormalise weights to active ones) ─────────────────────
    active_weights = {k: w for k, w in weights.items() if w > 0}
    total_w = sum(active_weights.values())
    if total_w <= 0:
        out["desirability"] = 0.0
    else:
        score = pd.Series(0.0, index=out.index)
        for component, w in active_weights.items():
            col = f"component_score_{component}"
            if col in out.columns:
                score = score + (w / total_w) * out[col].fillna(0)
        out["desirability"] = score.clip(0, 1)

    out["desirability_pct"] = (out["desirability"] * 100).round(1)

    return out.sort_values("desirability", ascending=False).reset_index(drop=True)


def reasons_for_top(row: pd.Series, top_k: int = 3) -> list[str]:
    """Build a short list of natural-language 'reasons' for why this row scored high.

    Picks the K highest-scoring components and turns each into a chip-style label
    like 'QED 0.78', 'predicted pIC50 7.4', '0 PAINS alerts'.
    """
    bits = []
    if "predicted_pIC50" in row.index and pd.notna(row.get("predicted_pIC50")):
        bits.append(("predicted pIC50",   row["predicted_pIC50"],   row.get("component_score_predicted_pIC50", 0)))
    if "QED" in row.index and pd.notna(row.get("QED")):
        bits.append(("QED",               row["QED"],               row.get("component_score_QED", 0)))
    if "lipinski_violations" in row.index and pd.notna(row.get("lipinski_violations")):
        bits.append(("Lipinski viol.",    int(row["lipinski_violations"]), row.get("component_score_lipinski", 0)))
    if "n_structural_alerts" in row.index and pd.notna(row.get("n_structural_alerts")):
        bits.append(("alerts",            int(row["n_structural_alerts"]), row.get("component_score_structural_alerts", 0)))
    if "heavy_atoms" in row.index and pd.notna(row.get("heavy_atoms")):
        bits.append(("heavy atoms",       int(row["heavy_atoms"]),  row.get("component_score_size", 0)))

    bits.sort(key=lambda x: x[2], reverse=True)
    out = []
    for label, val, _score in bits[:top_k]:
        if isinstance(val, float):
            out.append(f"{label} {val:.2f}")
        else:
            out.append(f"{label} {val}")
    return out
