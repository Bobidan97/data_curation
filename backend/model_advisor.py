"""model_advisor.py — Recommend QSAR modelling conditions for a dataset.

Given the working dataset, this inspects a handful of properties that actually
drive how a bioactivity model should be set up — how many compounds there are,
how wide the activity range is, how chemically diverse the set is, and how
balanced active/inactive is - and turns them into a concrete recommendation:
which model, which fingerprint, how to split train/test, whether to reduce
features, and whether hyperparameter tuning is worth the time.

The recommendations are heuristics grounded in standard QSAR practice, not
guarantees. Every recommendation carries a short "why" so the user can judge it
against their own knowledge of the data, and the panel is advisory - the user
is still free to pick any settings.

Design notes / thresholds:
    * "active" uses the same pIC50 >= 5.0 cutoff as the rest of the app.
    * Dataset-size buckets (very small / small / moderate / large) come from the
      rough points at which tree ensembles stop over-fitting and where
      hyperparameter tuning starts to generalise rather than chase CV noise.
    * Scaffold diversity = unique Bemis-Murcko scaffolds / compounds. A high
      value means many singletons (random splits leak analogues into the test
      set and flatter the score); a very low value means a congeneric series
      (a scaffold split is then too coarse - Butina clustering is gentler).

Everything degrades gracefully: missing RDKit only disables the scaffold-based
reasoning (the split defaults to a safe choice with a note).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── UI-facing labels (must match the selectbox options in the frontend) ───────
MODEL_RF = "Random Forest"
MODEL_RIDGE = "Ridge Regression"
MODEL_ENET = "Elastic Net"
MODEL_GB = "Gradient Boosting"

FP_ECFP4 = "ECFP4 (Morgan r2)"
FP_MACCS = "MACCS keys"

SPLIT_LABELS = {
    "random":   "Random",
    "scaffold": "Scaffold (Bemis-Murcko)",
    "butina":   "Butina clustering",
}
FS_LABELS = {"none": "None", "mrmr": "mRMR (top-K bits)"}

ACTIVE_THRESHOLD = 5.0

# Fingerprints that live in a high-dimensional bit space (feature selection can
# help linear models here; MACCS is already compact so it does not).
_HIGH_DIM_FPS = frozenset({FP_ECFP4, "ECFP6 (Morgan r3)", "Atom Pair"})


def _dataset_stats(
    df: pd.DataFrame,
    smiles_col: str,
    pic50_col: str,
) -> dict | None:
    """Compute the properties that drive the recommendation.

    Returns None if there is no modellable data (no valid SMILES + pIC50 pairs).
    """
    if smiles_col not in df.columns or pic50_col not in df.columns:
        return None

    work = df[[smiles_col, pic50_col]].copy()
    work[pic50_col] = pd.to_numeric(work[pic50_col], errors="coerce")
    work = work[
        work[smiles_col].apply(lambda s: isinstance(s, str) and bool(s.strip()))
        & work[pic50_col].notna()
    ]
    n = len(work)
    if n == 0:
        return None

    pic = work[pic50_col].to_numpy(dtype=float)
    pic50_min, pic50_max = float(pic.min()), float(pic.max())
    pic50_span = pic50_max - pic50_min
    pic50_std = float(pic.std(ddof=0))
    active_fraction = float((pic >= ACTIVE_THRESHOLD).mean())

    # Scaffold diversity (RDKit optional). Cap the work to keep this snappy.
    n_scaffolds: int | None = None
    scaffold_diversity: float | None = None
    try:
        from utils.rdkit_utils import murcko_scaffold
    except Exception:
        murcko_scaffold = None
    if murcko_scaffold is not None:
        sample = work[smiles_col].head(5000)
        scaffolds = {
            s for s in (murcko_scaffold(smi) for smi in sample) if s
        }
        if scaffolds:
            n_scaffolds = len(scaffolds)
            # Diversity relative to the number of molecules actually scaffolded.
            scaffold_diversity = round(n_scaffolds / len(sample), 3)

    return {
        "n_compounds": n,
        "pic50_min": round(pic50_min, 2),
        "pic50_max": round(pic50_max, 2),
        "pic50_span": round(pic50_span, 2),
        "pic50_std": round(pic50_std, 2),
        "n_scaffolds": n_scaffolds,
        "scaffold_diversity": scaffold_diversity,
        "active_fraction": round(active_fraction, 3),
    }


def recommend_model_conditions(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    pic50_col: str = "pIC50",
) -> dict:
    """Recommend QSAR modelling conditions for ``df``.

    Returns a dict:
        ok            - False when there is nothing to model (with ``message``)
        stats         - the dataset properties the advice is based on
        recommendation - the suggested settings (see keys below)
        reasons       - list of {setting, value, why} explaining each choice
        warnings      - list of caveats the user should know before trusting it
    """
    stats = _dataset_stats(df, smiles_col, pic50_col)
    if stats is None:
        return {
            "ok": False,
            "message": (
                "No modellable data yet - need a SMILES column and a pIC50 "
                "column with numeric values. Compute pIC50 during curation first."
            ),
        }

    n = stats["n_compounds"]
    span = stats["pic50_span"]
    std = stats["pic50_std"]
    diversity = stats["scaffold_diversity"]
    active_frac = stats["active_fraction"]

    reasons: list[dict] = []
    warnings: list[str] = []

    # Size bucket - the single biggest driver.
    if n < 40:
        size_bucket = "very small"
    elif n < 150:
        size_bucket = "small"
    elif n < 500:
        size_bucket = "moderate"
    else:
        size_bucket = "large"

    congeneric = diversity is not None and diversity < 0.15
    diverse = diversity is not None and diversity >= 0.50

    # ── Model ─────────────────────────────────────────────────────────────────
    if size_bucket == "very small":
        model = MODEL_RIDGE
        reasons.append({
            "setting": "Model", "value": model,
            "why": (
                f"Only {n} compounds - tree ensembles over-fit sets this small. "
                "A regularised linear model has the lowest variance here."
            ),
        })
    elif congeneric:
        model = MODEL_ENET
        reasons.append({
            "setting": "Model", "value": model,
            "why": (
                "Low scaffold diversity points to a congeneric series, where SAR "
                "tends to be smooth - a regularised linear model (Elastic Net) "
                "fits that well and stays interpretable."
            ),
        })
    elif size_bucket == "large":
        model = MODEL_GB
        reasons.append({
            "setting": "Model", "value": model,
            "why": (
                f"{n} compounds is enough for gradient boosting to exploit the "
                "extra data; pair it with tuning below for the best accuracy."
            ),
        })
    else:
        model = MODEL_RF
        reasons.append({
            "setting": "Model", "value": model,
            "why": (
                "Random Forest is the robust default for small-to-moderate, "
                "diverse sets - strong out of the box with little tuning."
            ),
        })

    # ── Fingerprint ───────────────────────────────────────────────────────────
    if size_bucket == "very small":
        fingerprint = FP_MACCS
        reasons.append({
            "setting": "Fingerprint", "value": fingerprint,
            "why": (
                "166 fixed keys are far lower-dimensional than ECFP bits, which "
                "curbs over-fitting on a tiny set (trade-off: no per-bit SHAP "
                "structure mapping)."
            ),
        })
    else:
        fingerprint = FP_ECFP4
        reasons.append({
            "setting": "Fingerprint", "value": fingerprint,
            "why": (
                "ECFP4 is the QSAR standard and supports full SHAP "
                "bit→substructure attribution."
            ),
        })

    # ── Train/test split ──────────────────────────────────────────────────────
    if n < 40:
        split = "random"
        reasons.append({
            "setting": "Split", "value": SPLIT_LABELS[split],
            "why": (
                "Too few compounds to hold out whole scaffold groups - a random "
                "split at least keeps a usable test set (but expect an optimistic "
                "score)."
            ),
        })
    elif diversity is None:
        split = "random"
        reasons.append({
            "setting": "Split", "value": SPLIT_LABELS[split],
            "why": (
                "Couldn't assess scaffold diversity (RDKit unavailable), so a "
                "structure-aware split isn't possible - defaulting to random."
            ),
        })
    elif congeneric:
        split = "butina"
        reasons.append({
            "setting": "Split", "value": SPLIT_LABELS[split],
            "why": (
                "In a congeneric series a scaffold split is too coarse (nearly "
                "everything shares one scaffold); Butina clustering separates "
                "near-duplicates more gently."
            ),
        })
    else:
        split = "scaffold"
        why = (
            "Holding out whole Bemis-Murcko scaffolds gives an honest estimate of "
            "performance on genuinely new chemotypes; a random split would leak "
            "close analogues into the test set."
        )
        if diverse:
            why = "High scaffold diversity - " + why[0].lower() + why[1:]
        reasons.append({
            "setting": "Split", "value": SPLIT_LABELS[split], "why": why,
        })

    # ── Feature selection ─────────────────────────────────────────────────────
    linear = model in (MODEL_RIDGE, MODEL_ENET)
    high_dim = fingerprint in _HIGH_DIM_FPS
    if linear and high_dim:
        feature_selection = "mrmr"
        # Keep K well under the sample count to avoid a p >> n linear fit.
        n_features = int(np.clip(round(n / 2), 20, 200))
        reasons.append({
            "setting": "Feature selection", "value": f"mRMR · K={n_features}",
            "why": (
                "A linear model on thousands of sparse bits over-fits; mRMR keeps "
                f"the ~{n_features} most informative, least-redundant bits "
                "(fit on the training fold only)."
            ),
        })
    else:
        feature_selection = "none"
        n_features = 100
        reasons.append({
            "setting": "Feature selection", "value": "None",
            "why": (
                "Tree ensembles handle high-dimensional fingerprints natively, so "
                "the full bit vector is fine."
                if not linear else
                "The chosen fingerprint is already compact - no reduction needed."
            ),
        })

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    if n >= 500:
        use_optuna, optuna_trials = True, 50
    elif n >= 150:
        use_optuna, optuna_trials = True, 30
    else:
        use_optuna, optuna_trials = False, 0
    reasons.append({
        "setting": "Tuning", "value": (
            f"Optuna · {optuna_trials} trials (Detailed tab)" if use_optuna
            else "Quick defaults (skip Optuna)"
        ),
        "why": (
            "Enough data for cross-validated tuning to generalise rather than "
            "chase CV noise."
            if use_optuna else
            "On a set this small, tuning tends to over-fit the CV folds - the "
            "Quick defaults are a safer baseline."
        ),
    })

    # ── Validation geometry ───────────────────────────────────────────────────
    test_size = 0.20
    cv_folds = 3 if n < 50 else 5

    # ── Warnings ──────────────────────────────────────────────────────────────
    if n < 40:
        warnings.append(
            f"Very small dataset ({n} compounds): any model will have high "
            "variance and the test score is unreliable - treat results as "
            "indicative, not validated."
        )
    if span < 2.0 or std < 0.5:
        warnings.append(
            f"Narrow activity range (pIC50 spans {span:g} log units, "
            f"σ={std:g}): there is little signal to regress - consider adding "
            "more potent and/or inactive compounds."
        )
    if active_frac < 0.10 or active_frac > 0.90:
        warnings.append(
            f"Imbalanced activities ({active_frac * 100:.0f}% active at "
            f"pIC50 ≥ {ACTIVE_THRESHOLD:g}): the model sees mostly one end of "
            "the scale."
        )
    if congeneric and split != "scaffold":
        warnings.append(
            "Congeneric series: even a clustered split can be optimistic because "
            "compounds are structurally close - external validation is worthwhile."
        )

    recommendation = {
        "model": model,
        "fingerprint": fingerprint,
        "split_method": split,
        "split_label": SPLIT_LABELS[split],
        "feature_selection": feature_selection,
        "feature_selection_label": FS_LABELS[feature_selection],
        "n_features": n_features,
        "use_optuna": use_optuna,
        "optuna_trials": optuna_trials,
        "test_size": test_size,
        "cv_folds": cv_folds,
    }

    return {
        "ok": True,
        "stats": stats,
        "recommendation": recommendation,
        "reasons": reasons,
        "warnings": warnings,
    }
