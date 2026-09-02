"""error_analysis.py — Diagnose what drives model prediction errors.

Given a completed model-result dict (from model.train_bioactivity_model /
_tuned), analyse the held-out test-set errors to surface *trends*: are the
worst-predicted compounds structurally unusual (outside the applicability
domain), do they share physicochemical properties, is the model simply worse at
the activity extremes, and do the errors concentrate in one chemotype?

Reuses the applicability-domain machinery from scoring.py and the descriptor
tooling from descriptors.py so the analysis stays consistent with the rest of
the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _abs_error_correlation(x: np.ndarray, abs_err: np.ndarray) -> dict | None:
    """Pearson correlation between a variable and |error|, or None if degenerate."""
    from scipy.stats import pearsonr

    mask = ~(np.isnan(x) | np.isnan(abs_err))
    if int(mask.sum()) < 3 or np.std(x[mask]) == 0 or np.std(abs_err[mask]) == 0:
        return None
    r, p = pearsonr(x[mask], abs_err[mask])
    return {"r": round(float(r), 3), "p": float(p)}


def _build_summary(ad_corr, actual_corr, descriptor_corr, scaffolds, n_high) -> str:
    """One-paragraph natural-language read of the strongest error trend."""
    bits: list[str] = []

    if ad_corr and ad_corr["p"] < 0.05 and ad_corr["r"] <= -0.30:
        bits.append(
            f"⚠️ Errors **grow as compounds become less similar to the training "
            f"set** (r = {ad_corr['r']:+.2f}) — the model is least reliable "
            f"outside its applicability domain."
        )
    elif ad_corr and ad_corr["r"] <= -0.15:
        bits.append(
            f"There is a weak tendency for less-similar compounds to be "
            f"predicted worse (r = {ad_corr['r']:+.2f})."
        )

    if actual_corr and actual_corr["p"] < 0.05 and abs(actual_corr["r"]) >= 0.30:
        direction = "more active" if actual_corr["r"] > 0 else "less active"
        bits.append(
            f"The model struggles most on **{direction} compounds** "
            f"(|error| vs actual pIC50 r = {actual_corr['r']:+.2f}) — a sign of "
            f"regression toward the mean at the activity extremes."
        )

    if descriptor_corr is not None and not descriptor_corr.empty:
        top = descriptor_corr.iloc[0]
        if abs(top["pearson_r"]) >= 0.30 and top["pearson_p"] < 0.05:
            direction = "higher" if top["pearson_r"] > 0 else "lower"
            bits.append(
                f"Larger errors are associated with **{direction} "
                f"{top['descriptor']}** (r = {top['pearson_r']:+.2f})."
            )

    if scaffolds is not None and not scaffolds.empty:
        top_scaf = scaffolds.iloc[0]
        if int(top_scaf["count"]) >= max(2, n_high // 3):
            bits.append(
                f"A single scaffold accounts for {int(top_scaf['count'])} of the "
                f"{n_high} worst predictions — errors are partly concentrated in "
                f"one chemotype."
            )

    if not bits:
        return (
            "No strong error trend detected — the model's mistakes look evenly "
            "spread across chemical space, similarity, and activity. That usually "
            "means errors are noise-driven rather than systematic."
        )
    return " ".join(bits)


def analyze_prediction_errors(
    model_result: dict,
    high_error_quantile: float = 0.75,
    max_worst: int = 6,
) -> dict:
    """Analyse the test-set prediction errors of a trained model.

    Args:
        model_result:       Result dict from train_bioactivity_model / _tuned.
        high_error_quantile: |error| quantile above which a row is "high error".
        max_worst:          How many worst-predicted structures to surface.

    Returns:
        dict (empty if there are no predictions) with:
            error_df             — predictions + abs_error + error_group
                                   (+ nn_tanimoto and descriptor columns when
                                   available)
            has_ad               — bool: applicability-domain distance computed
            ad_corr              — {r, p} of nn_tanimoto vs |error| (or None)
            actual_corr          — {r, p} of actual pIC50 vs |error| (or None)
            descriptor_corr      — DataFrame: |error| vs each descriptor
            group_comparison     — DataFrame: high vs low error group descriptor means
            high_error_scaffolds — DataFrame: dominant scaffolds among high-error rows
            threshold            — |error| cut defining the high-error group
            n_high               — number of high-error rows
            worst_df             — the worst-predicted rows (max_worst)
            summary              — natural-language read of the strongest trend
    """
    preds = model_result.get("predictions_df")
    if preds is None or preds.empty:
        return {}

    err_df = preds.copy()
    err_df["abs_error"] = err_df["error"].abs()

    # ── Applicability domain: nearest-neighbour Tanimoto vs |error| ──────────
    has_ad = False
    ad_corr = None
    # Prefer full fingerprints for AD when mRMR reduced the model features.
    X_train = model_result.get("X_train_full")
    if X_train is None:
        X_train = model_result.get("X_train")
    X_test = model_result.get("X_test_full")
    if X_test is None:
        X_test = model_result.get("X_test")
    test_smiles = model_result.get("test_smiles")
    if (
        X_train is not None and X_test is not None and test_smiles is not None
        and "canonical_smiles" in err_df.columns
    ):
        from scoring import _bulk_max_tanimoto
        nn_sim, _ = _bulk_max_tanimoto(np.asarray(X_test), np.asarray(X_train))
        smi_to_nn = dict(zip(test_smiles, nn_sim))
        err_df["nn_tanimoto"] = err_df["canonical_smiles"].map(smi_to_nn)
        has_ad = bool(err_df["nn_tanimoto"].notna().any())
        if has_ad:
            ad_corr = _abs_error_correlation(
                err_df["nn_tanimoto"].to_numpy(dtype=float),
                err_df["abs_error"].to_numpy(dtype=float),
            )

    # ── Error vs actual pIC50 (do the extremes fail?) ────────────────────────
    actual_corr = _abs_error_correlation(
        err_df["actual_pIC50"].to_numpy(dtype=float),
        err_df["abs_error"].to_numpy(dtype=float),
    )

    # ── High / low error grouping ────────────────────────────────────────────
    threshold = float(err_df["abs_error"].quantile(high_error_quantile))
    err_df["error_group"] = np.where(err_df["abs_error"] >= threshold, "High", "Low")
    n_high = int((err_df["error_group"] == "High").sum())

    # ── Descriptors of the test compounds ────────────────────────────────────
    descriptor_corr = pd.DataFrame()
    group_comparison = pd.DataFrame()
    high_error_scaffolds = pd.DataFrame()

    if "canonical_smiles" in err_df.columns:
        from descriptors import (
            compute_descriptors, DESCRIPTOR_COLUMNS, compute_descriptor_correlations,
        )
        desc = compute_descriptors(err_df[["canonical_smiles"]].copy())
        present = [c for c in DESCRIPTOR_COLUMNS if c in desc.columns]
        for c in present:
            err_df[c] = desc[c].values

        if present:
            # |error| vs each descriptor
            _tmp = err_df.rename(columns={"abs_error": "_ae"})
            descriptor_corr = compute_descriptor_correlations(_tmp, present, target_col="_ae")

            # High vs low group descriptor means
            rows = []
            for c in present:
                hi = err_df.loc[err_df["error_group"] == "High", c].mean()
                lo = err_df.loc[err_df["error_group"] == "Low", c].mean()
                if pd.notna(hi) and pd.notna(lo):
                    rows.append({
                        "descriptor":      c,
                        "high_error_mean": round(float(hi), 2),
                        "low_error_mean":  round(float(lo), 2),
                        "delta":           round(float(hi - lo), 2),
                    })
            group_comparison = pd.DataFrame(rows)

        # Dominant scaffolds among the high-error compounds
        from collections import Counter
        from utils.rdkit_utils import murcko_scaffold
        hi_smiles = err_df.loc[err_df["error_group"] == "High", "canonical_smiles"].tolist()
        scafs = [s for s in (murcko_scaffold(x) for x in hi_smiles) if s]
        if scafs:
            counts = Counter(scafs)
            high_error_scaffolds = pd.DataFrame(
                [{"scaffold_smiles": s, "count": n} for s, n in counts.most_common(5)]
            )

    worst_df = err_df.nlargest(min(max_worst, len(err_df)), "abs_error").reset_index(drop=True)

    return {
        "error_df":             err_df,
        "has_ad":               has_ad,
        "ad_corr":              ad_corr,
        "actual_corr":          actual_corr,
        "descriptor_corr":      descriptor_corr,
        "group_comparison":     group_comparison,
        "high_error_scaffolds": high_error_scaffolds,
        "threshold":            threshold,
        "n_high":               n_high,
        "worst_df":             worst_df,
        "summary":              _build_summary(
            ad_corr, actual_corr, descriptor_corr, high_error_scaffolds, n_high
        ),
    }
