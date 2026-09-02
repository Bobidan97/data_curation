"""explainability.py — SHAP attribution for fingerprint-based models.

Provides two levels of explanation for a trained pIC50 regressor:

    1. **Global** — which Morgan bits contribute most across the test set
       (mean |SHAP|), with per-bit substructure visualisation.

    2. **Local** — for a specific molecule, which bits drove the prediction up
       or down, plus the atoms on that molecule responsible for each bit.

Implementation notes:
    * Tree-based models (RandomForest, GradientBoosting) use the fast
      ``shap.TreeExplainer`` — exact, no sampling, takes seconds even for
      thousands of test rows.
    * Ridge wrapped in a StandardScaler+Pipeline uses ``shap.LinearExplainer``;
      we unwrap the pipeline and explain on the scaled features.
    * Morgan ``bitInfo`` from RDKit is the link between fingerprint bits and
      atom neighbourhoods — same machinery used elsewhere in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.fingerprints import MORGAN_RADIUS, MORGAN_N_BITS


# ── SHAP value computation ────────────────────────────────────────────────────

@dataclass
class ShapResult:
    """Container for SHAP values on a test set."""
    shap_values:        np.ndarray         # (n_samples, n_features)
    feature_importance: np.ndarray         # (n_features,) — mean |shap| over samples
    base_value:         float              # model's expected output
    sample_indices:     np.ndarray         # row indices in the test set
    explainer_type:     str                # "tree" | "linear"


def compute_shap_values(model, X_train: np.ndarray, X_test: np.ndarray) -> ShapResult:
    """Compute SHAP values for ``X_test`` using the most appropriate explainer.

    For RandomForest / GradientBoosting → TreeExplainer (exact, fast).
    For Ridge inside a Pipeline → LinearExplainer on the scaled features.

    Returns ShapResult with everything the UI needs to rank bits & build
    per-molecule force plots.
    """
    import shap

    # Detect model family by introspecting the estimator chain
    inner = model
    pre_transform = None
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # Last step is the regressor; earlier steps are preprocessors
            inner = model.steps[-1][1]
            if len(model.steps) > 1:
                pre_transform = model[:-1]
    except ImportError:
        pass

    # Choose explainer
    is_tree = inner.__class__.__name__ in {
        "RandomForestRegressor", "GradientBoostingRegressor"
    }

    if is_tree:
        explainer  = shap.TreeExplainer(inner)
        X_to_explain = X_test if pre_transform is None else pre_transform.transform(X_test)
        sv = explainer.shap_values(X_to_explain, check_additivity=False)
        base = float(np.asarray(explainer.expected_value).flatten()[0])
        explainer_type = "tree"
    else:
        # Linear / kernel fallback
        X_bg_scaled = X_train if pre_transform is None else pre_transform.transform(X_train)
        X_te_scaled = X_test  if pre_transform is None else pre_transform.transform(X_test)
        # Use a small subsample as background to keep things snappy
        bg = X_bg_scaled if len(X_bg_scaled) <= 200 else shap.utils.sample(X_bg_scaled, 200, random_state=42)
        explainer = shap.LinearExplainer(inner, bg)
        sv   = explainer.shap_values(X_te_scaled)
        base = float(np.asarray(explainer.expected_value).flatten()[0])
        explainer_type = "linear"

    sv = np.asarray(sv)
    feature_importance = np.mean(np.abs(sv), axis=0)

    return ShapResult(
        shap_values=sv,
        feature_importance=feature_importance,
        base_value=base,
        sample_indices=np.arange(len(sv)),
        explainer_type=explainer_type,
    )


# ── Bit-to-substructure mapping ───────────────────────────────────────────────

def get_morgan_bit_info(smiles: str, n_bits: int = MORGAN_N_BITS, radius: int = MORGAN_RADIUS) -> dict:
    """Return RDKit's ``bitInfo`` mapping for a single SMILES string.

    Each key is a fingerprint bit ID; each value is a list of
    ``(center_atom_idx, radius)`` tuples for every time the bit was triggered
    by an atom neighbourhood in this molecule. Returns an empty dict on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {}

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {}

    bitInfo: dict = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bitInfo
    )
    return bitInfo


def atoms_for_bit(smiles: str, bit_id: int, radius: int = MORGAN_RADIUS) -> list[int]:
    """Return atom indices on ``smiles`` that triggered the given Morgan bit.

    Includes the atoms in the neighbourhood (radius-bounded ball) around each
    centre that lit the bit. Returns an empty list if the bit isn't present.
    ``radius`` must match the fingerprint the model was trained on (2 = ECFP4,
    3 = ECFP6).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return []

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return []

    bi = get_morgan_bit_info(smiles, radius=radius)
    centres = bi.get(bit_id, [])
    if not centres:
        return []

    atoms: set[int] = set()
    for centre_atom, radius in centres:
        # Find all atoms within `radius` bonds of the centre
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, centre_atom)
        atom_set = {centre_atom}
        for bond_idx in env:
            bond = mol.GetBondWithIdx(bond_idx)
            atom_set.add(bond.GetBeginAtomIdx())
            atom_set.add(bond.GetEndAtomIdx())
        atoms.update(atom_set)
    return sorted(atoms)


def top_bits_for_prediction(
    shap_row: np.ndarray,
    fingerprint_row: np.ndarray,
    top_n: int = 6,
) -> pd.DataFrame:
    """Rank the bits that contributed most to one prediction.

    Args:
        shap_row:        SHAP values for one molecule, shape (n_bits,).
        fingerprint_row: The molecule's fingerprint (0/1), shape (n_bits,).
        top_n:           Number of top contributors to return.

    Returns:
        DataFrame ordered by |shap| descending:
            bit_id · shap_value · present · direction
        ('direction' is +1 for "raised prediction" and -1 for "lowered it".)
    """
    abs_shap   = np.abs(shap_row)
    top_idx    = np.argsort(-abs_shap)[:top_n]
    rows = []
    for bit in top_idx:
        if abs_shap[bit] == 0:
            continue
        rows.append({
            "bit_id":      int(bit),
            "shap_value":  float(shap_row[bit]),
            "present":     bool(fingerprint_row[bit]),
            "direction":   "↑ raised" if shap_row[bit] > 0 else "↓ lowered",
        })
    return pd.DataFrame(rows)


def global_top_bits(shap_result: ShapResult, top_n: int = 12) -> pd.DataFrame:
    """Rank bits by mean |SHAP| across the whole test set."""
    importances = shap_result.feature_importance
    top_idx = np.argsort(-importances)[:top_n]
    return pd.DataFrame({
        "bit_id":     top_idx.astype(int),
        "mean_abs_shap": importances[top_idx].round(4),
        "mean_signed_shap": shap_result.shap_values[:, top_idx].mean(axis=0).round(4),
    })
