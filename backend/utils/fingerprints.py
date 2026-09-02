"""fingerprints.py — Shared molecular fingerprint utilities.

Single implementation of the SMILES → uint8 bit-array pattern used by
chemical_space.py (MACCS keys / t-SNE) and model.py (Morgan ECFP4 / ML).

Uses GetOnBits() rather than DataStructs.ConvertToNumpyArray to avoid the
NumPy 2.x C-API incompatibility introduced by Boost.Python.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Module constants ──────────────────────────────────────────────────────────

MORGAN_RADIUS: int = 2     # ECFP4
MORGAN_N_BITS: int = 2048

# Named fingerprint methods offered in the modelling UI. Each maps to the
# keyword arguments passed to compute_fp_array. ``is_morgan`` / ``radius`` are
# consumed by the SHAP bit→substructure mapping (only meaningful for Morgan).
FINGERPRINT_METHODS: dict[str, dict] = {
    "ECFP4 (Morgan r2)": {"fp_type": "morgan",   "radius": 2, "n_bits": 2048},
    "ECFP6 (Morgan r3)": {"fp_type": "morgan",   "radius": 3, "n_bits": 2048},
    "MACCS keys":        {"fp_type": "maccs"},
    "Atom Pair":         {"fp_type": "atompair", "n_bits": 2048},
}
DEFAULT_FINGERPRINT: str = "ECFP4 (Morgan r2)"


# ── Public API ────────────────────────────────────────────────────────────────

def compute_fp_array(
    smiles_series: pd.Series,
    fp_type: str = "morgan",
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_N_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a binary fingerprint matrix from a Series of SMILES strings.

    Args:
        smiles_series: 1-D Series of SMILES strings (may contain None / NaN).
        fp_type:       "morgan" (ECFP, default), "maccs", or "atompair".
        radius:        Morgan circular radius — ignored for MACCS / atompair.
        n_bits:        Bit-vector length — ignored for MACCS (always 167 bits).

    Returns:
        fp_array   — np.ndarray of shape (n_valid, n_bits), dtype uint8.
                     Rows correspond only to entries where SMILES parsed
                     successfully (use valid_mask to align with the input).
        valid_mask — bool np.ndarray of shape (len(smiles_series),);
                     True where RDKit parsed the SMILES without error.

    Raises:
        ValueError:  if fp_type is not recognised.
        ImportError: if RDKit is not installed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

    if fp_type not in ("morgan", "maccs", "atompair"):
        raise ValueError(
            f"fp_type must be 'morgan', 'maccs' or 'atompair', got {fp_type!r}"
        )

    # ── Parse SMILES ──────────────────────────────────────────────────────────
    mols = [
        Chem.MolFromSmiles(s) if isinstance(s, str) else None
        for s in smiles_series
    ]
    valid_mask = np.array([m is not None for m in mols], dtype=bool)
    valid_mols = [m for m, v in zip(mols, valid_mask) if v]

    # ── Generate fingerprints ─────────────────────────────────────────────────
    if fp_type == "maccs":
        fps = [MACCSkeys.GenMACCSKeys(m) for m in valid_mols]
        n_bits_actual = 167   # MACCS keys are always 167 bits
    elif fp_type == "atompair":
        fps = [
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=n_bits)
            for m in valid_mols
        ]
        n_bits_actual = n_bits
    else:  # morgan / ECFP
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
            for m in valid_mols
        ]
        n_bits_actual = n_bits

    # ── Build uint8 bit array via GetOnBits() ─────────────────────────────────
    fp_array = np.zeros((len(valid_mols), n_bits_actual), dtype=np.uint8)
    for i, fp in enumerate(fps):
        for bit in fp.GetOnBits():
            fp_array[i, bit] = 1

    return fp_array, valid_mask


def compute_fp_array_named(
    smiles_series: pd.Series, method: str = DEFAULT_FINGERPRINT
) -> tuple[np.ndarray, np.ndarray]:
    """compute_fp_array dispatched by a friendly method name (see FINGERPRINT_METHODS)."""
    cfg = FINGERPRINT_METHODS.get(method, FINGERPRINT_METHODS[DEFAULT_FINGERPRINT])
    return compute_fp_array(smiles_series, **cfg)


def method_is_morgan(method: str) -> bool:
    """True if the named method is a Morgan/ECFP fingerprint (SHAP-mappable)."""
    return FINGERPRINT_METHODS.get(method, {}).get("fp_type") == "morgan"


def method_radius(method: str) -> int:
    """Morgan radius for the named method (defaults to 2; meaningless for non-Morgan)."""
    return FINGERPRINT_METHODS.get(method, {}).get("radius", MORGAN_RADIUS)
