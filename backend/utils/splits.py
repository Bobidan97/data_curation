"""splits.py — Train/test splitting strategies for QSAR datasets.

Three methods are available via :func:`split_train_test`:
    "random"   — sklearn-style uniform random split.
    "scaffold" — Bemis–Murcko scaffold split. Whole scaffolds go to either
                 train or test; largest scaffolds end up in train, so the
                 test set is composed of the smaller / rarer scaffolds.
                 Mirrors the MoleculeNet convention.
    "butina"   — Butina clustering on Morgan fingerprints. Whole clusters are
                 assigned to either train or test; largest clusters go to
                 train. The cluster cutoff is a Tanimoto **distance** threshold
                 (e.g. 0.35 ≡ Tanimoto similarity > 0.65 → same cluster).

All splitters return aligned integer index arrays ``(idx_train, idx_test)`` that
index into the input SMILES sequence. They assume the caller has already
removed invalid/unparseable SMILES (scaffold/butina degrade gracefully if not).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SUPPORTED_METHODS: tuple[str, ...] = ("random", "scaffold", "butina")

DEFAULT_BUTINA_CUTOFF: float = 0.35   # Tanimoto distance — similarity > 0.65 → same cluster


# ── Public API ────────────────────────────────────────────────────────────────

def split_train_test(
    smiles_list: list[str] | pd.Series,
    test_size: float = 0.2,
    method: str = "random",
    random_state: int = 42,
    cluster_cutoff: float = DEFAULT_BUTINA_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(idx_train, idx_test)`` for the chosen split strategy.

    Args:
        smiles_list:   SMILES strings, one per row.
        test_size:     Fraction of rows held out for testing (0–1).
        method:        One of :data:`SUPPORTED_METHODS`.
        random_state:  Seed (used by 'random'; ignored by 'scaffold' / 'butina').
        cluster_cutoff: Tanimoto distance threshold for Butina (default 0.35).

    Returns:
        Two ``np.ndarray`` of integer positions into ``smiles_list``.

    Raises:
        ValueError:  unknown method.
        ImportError: RDKit not installed (scaffold / butina only).
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"method must be one of {SUPPORTED_METHODS}. Got: {method!r}"
        )

    smiles = list(smiles_list)
    n = len(smiles)
    if n == 0:
        empty = np.array([], dtype=int)
        return empty, empty

    if method == "random":
        return _random_split(n, test_size, random_state)
    if method == "scaffold":
        return _scaffold_split(smiles, test_size)
    return _butina_split(smiles, test_size, cluster_cutoff)


# ── Implementations ───────────────────────────────────────────────────────────

def _random_split(n: int, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    train, test = train_test_split(idx, test_size=test_size, random_state=random_state)
    return train, test


def _scaffold_split(smiles_list: list[str], test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Bemis–Murcko scaffold split.

    Groups molecules by canonical scaffold SMILES, sorts groups largest-first,
    then fills train until adding another group would overflow the train
    target; remaining groups go to test. Largest scaffold series end up in
    train — the test set is composed of smaller, rarer scaffolds.
    """
    from utils.rdkit_utils import murcko_scaffold

    n = len(smiles_list)
    groups: dict[str, list[int]] = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        # Empty-string key groups all unparseable molecules together
        scaf = murcko_scaffold(smi) or ""
        groups[scaf].append(i)

    # Largest groups first (MoleculeNet convention)
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    train_cutoff = n * (1.0 - test_size)
    train_idx: list[int] = []
    test_idx:  list[int] = []
    for group in sorted_groups:
        if len(train_idx) + len(group) > train_cutoff:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def _butina_split(
    smiles_list: list[str],
    test_size: float,
    cluster_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Butina cluster split on Morgan fingerprints.

    Tanimoto **distance** between every pair of fingerprints is fed to RDKit's
    Butina clustering. Compounds within ``cluster_cutoff`` distance share a
    cluster (e.g. 0.35 ≡ similarity > 0.65). Whole clusters are assigned to
    either train or test; largest clusters go to train.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    n = len(smiles_list)

    # Compute fingerprints; remember which positions had valid SMILES
    valid_pos:  list[int]    = []
    valid_fps:  list[object] = []
    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid_pos.append(i)
        valid_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    if not valid_fps:
        # No valid fingerprints — fall back to a random split
        return _random_split(n, test_size, 42)

    # Flat lower-triangle distance vector for Butina
    n_valid = len(valid_fps)
    dists: list[float] = []
    for i in range(1, n_valid):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(
        dists, n_valid, cluster_cutoff, isDistData=True
    )

    # Map cluster positions back to original SMILES indices, then sort largest-first
    cluster_groups = [
        [valid_pos[pos] for pos in cluster]
        for cluster in clusters
    ]
    cluster_groups.sort(key=len, reverse=True)

    # Any invalid SMILES (shouldn't happen in normal use) drop straight into train
    valid_set = set(valid_pos)
    invalid_idx = [i for i in range(n) if i not in valid_set]

    train_cutoff = n * (1.0 - test_size)
    train_idx: list[int] = list(invalid_idx)
    test_idx:  list[int] = []
    for group in cluster_groups:
        if len(train_idx) + len(group) > train_cutoff:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)
