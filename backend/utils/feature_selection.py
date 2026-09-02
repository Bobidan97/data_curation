"""feature_selection.py — minimum-Redundancy Maximum-Relevance (mRMR) selection.

Selects a compact subset of fingerprint bits that are individually informative
about the target (high relevance) while being non-redundant with each other
(low mutual correlation). Reduces a 2048-bit fingerprint to the top-K bits,
which can improve linear models and speed up training/SHAP.

Relevance uses the F-statistic (``f_regression``); redundancy uses the mean
absolute Pearson correlation with already-selected features (the classic FCD /
"F-test Correlation Difference" variant of mRMR).
"""
from __future__ import annotations

import numpy as np


def mrmr_select(X: np.ndarray, y: np.ndarray, n_features: int = 100) -> list[int]:
    """Return the indices of the top ``n_features`` bits by mRMR.

    Args:
        X:          (n_samples, n_feats) feature matrix (binary or float).
        y:          (n_samples,) regression target.
        n_features: number of features to keep.

    Returns:
        Sorted list of selected column indices into X. Zero-variance columns are
        never selected. If fewer informative columns exist than requested, all
        of them are returned.
    """
    from sklearn.feature_selection import f_regression

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_feats = X.shape[1]
    n_features = int(min(n_features, n_feats))

    # Relevance (F-statistic); guard against zero-variance / NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        F, _ = f_regression(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    # Restrict to columns with non-zero variance (constant bits carry no signal)
    var = X.var(axis=0)
    valid = np.where(var > 0)[0]
    if valid.size == 0:
        return list(range(min(n_features, n_feats)))
    if valid.size <= n_features:
        return sorted(int(i) for i in valid)

    # Standardise valid columns for correlation-based redundancy
    Xv = X[:, valid]
    Xz = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + 1e-9)
    n_samples = Xz.shape[0]
    rel = F[valid]

    selected_local: list[int] = []
    red_sum = np.zeros(valid.size)      # running Σ |corr| with the selected set

    for k in range(n_features):
        if not selected_local:
            best = int(np.argmax(rel))
        else:
            score = rel - red_sum / len(selected_local)
            score[selected_local] = -np.inf          # don't reselect
            best = int(np.argmax(score))
        selected_local.append(best)
        # accumulate |corr(best, every valid feature)|
        corr = np.abs(Xz.T @ Xz[:, best] / n_samples)
        red_sum += corr

    return sorted(int(valid[i]) for i in selected_local)
