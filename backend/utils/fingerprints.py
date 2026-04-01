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
        fp_type:       "morgan" (ECFP4, default) or "maccs".
        radius:        Morgan circular radius — ignored for MACCS.
        n_bits:        Morgan bit-vector length — ignored for MACCS
                       (MACCS keys are always 167 bits).

    Returns:
        fp_array   — np.ndarray of shape (n_valid, n_bits), dtype uint8.
                     Rows correspond only to entries where SMILES parsed
                     successfully (use valid_mask to align with the input).
        valid_mask — bool np.ndarray of shape (len(smiles_series),);
                     True where RDKit parsed the SMILES without error.

    Raises:
        ValueError:  if fp_type is not "morgan" or "maccs".
        ImportError: if RDKit is not installed.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys

    if fp_type not in ("morgan", "maccs"):
        raise ValueError(
            f"fp_type must be 'morgan' or 'maccs', got {fp_type!r}"
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
        n_bits_actual = fps[0].GetNumBits()   # always 167 for MACCS keys
    else:  # morgan / ECFP4
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
