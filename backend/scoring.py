"""scoring.py — Score user-supplied compounds against a trained model.

Given a list of SMILES and a completed model-result dict (as returned by
``model.train_bioactivity_model`` / ``_tuned``), this module:

    1. Parses/validates the SMILES and computes Morgan fingerprints.
    2. Predicts pIC50 with the stored estimator.
    3. Runs an **applicability-domain (AD)** check — the maximum Tanimoto
       similarity of each input to any training-set compound. Low similarity
       ⇒ the model is extrapolating ⇒ the prediction is less trustworthy.

The AD threshold defaults to Tanimoto 0.30 (a common QSAR convention): inputs
whose nearest training neighbour is below this are flagged "outside domain".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.fingerprints import compute_fp_array_named, DEFAULT_FINGERPRINT

# Default applicability-domain similarity threshold (max Tanimoto to training set)
DEFAULT_AD_THRESHOLD: float = 0.30


def _bulk_max_tanimoto(
    query_fps: np.ndarray, train_fps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Max Tanimoto similarity of each query row to any training row.

    Both inputs are uint8/bool bit matrices of shape (n, 2048). Tanimoto for
    binary vectors a, b = |a∩b| / (|a| + |b| − |a∩b|). Computed vectorised:
        intersection = query · trainᵀ
        |a|, |b|     = row sums
    Returns ``(max_sim, argmax_idx)`` — the nearest-neighbour similarity and the
    index of that neighbour in ``train_fps``, both 1-D of length len(query_fps).
    """
    q = query_fps.astype(np.float32)
    t = train_fps.astype(np.float32)

    inter = q @ t.T                      # (n_query, n_train) intersection counts
    q_sum = q.sum(axis=1, keepdims=True)  # (n_query, 1)
    t_sum = t.sum(axis=1, keepdims=True).T  # (1, n_train)
    union = q_sum + t_sum - inter
    # Avoid divide-by-zero for all-zero fingerprints
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.max(axis=1), sim.argmax(axis=1)


def score_compounds(
    smiles_list: list[str],
    model_result: dict,
    ad_threshold: float = DEFAULT_AD_THRESHOLD,
) -> pd.DataFrame:
    """Score a list of SMILES against a trained model.

    Args:
        smiles_list:   Input SMILES strings (one per compound).
        model_result:  A completed result dict from train_bioactivity_model /
                       _tuned. Must contain ``model`` and ``X_train``.
        ad_threshold:  Min nearest-neighbour Tanimoto for a compound to count
                       as inside the applicability domain.

    Returns:
        DataFrame with one row per *valid* input SMILES and columns:
            input_smiles · predicted_pIC50 · nn_tanimoto · in_domain
        plus, when the model result carries training references:
            nn_smiles · nn_pIC50 · nn_id
        (the nearest training-set neighbour's structure, measured activity, and
        ChEMBL ID — for the user to eyeball what the prediction is anchored to).
        Invalid SMILES are dropped (use len difference to report how many).

    Raises:
        ValueError: if the model result lacks the data needed for scoring,
                    or if no input SMILES are parseable.
    """
    model = model_result.get("model")
    X_train = model_result.get("X_train")
    if model is None or X_train is None:
        raise ValueError(
            "This model result does not contain the data needed for scoring "
            "(model + training fingerprints). Re-train the model and try again."
        )

    clean = [s.strip() for s in smiles_list if isinstance(s, str) and s.strip()]
    if not clean:
        raise ValueError("No SMILES provided.")

    fp_method = model_result.get("fingerprint", DEFAULT_FINGERPRINT)
    fp_array, valid_mask = compute_fp_array_named(pd.Series(clean), fp_method)
    valid_smiles = [s for s, ok in zip(clean, valid_mask) if ok]
    if fp_array.shape[0] == 0:
        raise ValueError("None of the provided SMILES could be parsed by RDKit.")

    # If the model was trained on an mRMR-reduced feature set, subset new
    # compounds to the same columns for PREDICTION. The applicability-domain
    # check, however, uses the FULL fingerprint (similarity is a chemical-space
    # question, independent of which bits the model kept).
    _sel = model_result.get("selected_features")
    fp_full = fp_array
    fp_model = fp_array[:, _sel] if _sel is not None else fp_array

    preds = model.predict(fp_model)

    X_train_ad = model_result.get("X_train_full")
    if X_train_ad is None:
        X_train_ad = X_train   # X_train is already full when no selection
    nn_sim, nn_idx = _bulk_max_tanimoto(fp_full, np.asarray(X_train_ad))

    out = pd.DataFrame({
        "input_smiles":    valid_smiles,
        "predicted_pIC50": np.round(preds, 3),
        "nn_tanimoto":     np.round(nn_sim, 3),
        "in_domain":       nn_sim >= ad_threshold,
    })

    # Attach nearest-neighbour references when the model result carries them
    train_smiles = model_result.get("train_smiles")
    if train_smiles is not None:
        n_train = len(train_smiles)
        safe_idx = [int(i) if 0 <= int(i) < n_train else 0 for i in nn_idx]
        out["nn_smiles"] = [train_smiles[i] for i in safe_idx]

        train_pic50 = model_result.get("train_pic50")
        if train_pic50 is not None and len(train_pic50) == n_train:
            out["nn_pIC50"] = [round(float(train_pic50[i]), 3) for i in safe_idx]

        train_ids = model_result.get("train_ids")
        if train_ids is not None and len(train_ids) == n_train:
            out["nn_id"] = [train_ids[i] for i in safe_idx]

    return out


def training_pic50_distribution(model_result: dict) -> dict:
    """Summary stats of the training+test pIC50 used to fit the model.

    Pulls actual pIC50 values out of the predictions table (test set) — used to
    contextualise where a newly-scored compound falls. Returns a dict with
    min / q1 / median / q3 / max plus the raw test actuals for plotting.
    """
    preds = model_result.get("predictions_df")
    if preds is None or "actual_pIC50" not in preds.columns:
        return {}
    vals = preds["actual_pIC50"].dropna()
    if vals.empty:
        return {}
    return {
        "min":    float(vals.min()),
        "q1":     float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "q3":     float(vals.quantile(0.75)),
        "max":    float(vals.max()),
        "values": vals.tolist(),
    }
