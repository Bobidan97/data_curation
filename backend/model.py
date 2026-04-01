"""model.py — Bioactivity regression models using ECFP4 (Morgan) fingerprints.

Provides:
    train_bioactivity_model — train RF / Ridge / GB on pIC50 from SMILES

Fingerprint computation is delegated to fingerprints.compute_fp_array.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from utils.fingerprints import compute_fp_array

# ── Module constants ──────────────────────────────────────────────────────────

MIN_MOLECULES: int = 10   # minimum rows required to attempt model training


# ── Model training ────────────────────────────────────────────────────────────

_SUPPORTED_MODELS = frozenset(
    {"Random Forest", "Ridge Regression", "Gradient Boosting"}
)


def train_bioactivity_model(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    n_estimators: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Train a regression model to predict pIC50 from ECFP4 fingerprints.

    Args:
        df:            DataFrame with 'canonical_smiles' and 'pIC50' columns.
        model_type:    One of "Random Forest", "Ridge Regression",
                       "Gradient Boosting".
        test_size:     Fraction of data held out for the test set (0.10–0.40).
        n_estimators:  Number of trees; used by RF and GB only (ignored for Ridge).
        cv_folds:      Number of cross-validation folds on the training set.
        random_state:  Random seed for reproducibility.

    Returns:
        dict with keys:
            model_type, n_estimators, test_size, cv_folds,
            n_train, n_test, n_valid_smiles, n_invalid_smiles,
            test_r2, test_rmse, test_mae,
            cv_r2_mean, cv_r2_std, cv_scores,
            feature_importances   (np.ndarray shape (2048,) or None for Ridge),
            predictions_df        (pd.DataFrame — test rows),
            model                 (fitted sklearn estimator),
            timestamp             (ISO string).

    Raises:
        ValueError: missing columns, too few valid molecules, unsupported model
                    type, or test split produces zero samples.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ── Validate inputs ───────────────────────────────────────────────────────
    for col in ("canonical_smiles", "pIC50"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")

    if model_type not in _SUPPORTED_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(_SUPPORTED_MODELS)}. "
            f"Got: {model_type!r}"
        )

    df_work = df.dropna(subset=["pIC50"]).reset_index(drop=True)
    if len(df_work) < MIN_MOLECULES:
        raise ValueError(
            f"Need at least {MIN_MOLECULES} rows with valid pIC50 values "
            f"(got {len(df_work)})."
        )

    # ── Compute fingerprints ──────────────────────────────────────────────────
    fp_array, valid_mask = compute_fp_array(df_work["canonical_smiles"], fp_type="morgan")
    n_invalid = int((~valid_mask).sum())
    df_valid = df_work[valid_mask].reset_index(drop=True)
    y = df_valid["pIC50"].values.astype(float)

    # ── Train / test split ────────────────────────────────────────────────────
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        fp_array, y, indices,
        test_size=test_size,
        random_state=random_state,
    )

    if len(X_test) == 0:
        raise ValueError(
            "Test split produced zero samples. "
            "Increase test_size or provide more data."
        )

    # ── Build model ───────────────────────────────────────────────────────────
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    else:  # Ridge Regression
        # StandardScaler is required: Ridge penalises by coefficient magnitude,
        # so features on different scales would be treated unequally.
        # Wrapping in a Pipeline ensures the scaler is bundled with the model
        # so the pickled artefact is self-contained.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])

    model.fit(X_train, y_train)

    # ── Cross-validation (on training set) ───────────────────────────────────
    # Clamp cv to len(X_train) to avoid "n_splits > n_samples" ValueError
    effective_cv = min(cv_folds, len(X_train))
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=effective_cv,
        scoring="r2",
    )

    # ── Test-set metrics ──────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    test_r2   = float(r2_score(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_mae  = float(mean_absolute_error(y_test, y_pred))

    # ── Feature importances ───────────────────────────────────────────────────
    feature_importances: np.ndarray | None = getattr(
        model, "feature_importances_", None
    )
    if feature_importances is not None:
        feature_importances = feature_importances.copy()

    # ── Predictions table ─────────────────────────────────────────────────────
    pred_data: dict[str, list] = {
        "actual_pIC50":    y_test.tolist(),
        "predicted_pIC50": y_pred.tolist(),
        "error":           (y_pred - y_test).tolist(),
    }
    if "molecule_chembl_id" in df_valid.columns:
        pred_data["molecule_chembl_id"] = (
            df_valid["molecule_chembl_id"].iloc[idx_test].tolist()
        )
    if "canonical_smiles" in df_valid.columns:
        pred_data["canonical_smiles"] = (
            df_valid["canonical_smiles"].iloc[idx_test].tolist()
        )

    predictions_df = pd.DataFrame(pred_data)

    # Reorder columns: identifiers first
    leading = [c for c in ("molecule_chembl_id", "canonical_smiles") if c in predictions_df.columns]
    trailing = ["actual_pIC50", "predicted_pIC50", "error"]
    predictions_df = predictions_df[leading + trailing]

    # Sort by absolute error descending (worst predictions first — most actionable)
    predictions_df = (
        predictions_df
        .assign(_abs_err=predictions_df["error"].abs())
        .sort_values("_abs_err", ascending=False)
        .drop(columns=["_abs_err"])
        .reset_index(drop=True)
    )

    # ── Return ────────────────────────────────────────────────────────────────
    return {
        "model_type":          model_type,
        "n_estimators":        n_estimators if model_type != "Ridge Regression" else None,
        "test_size":           test_size,
        "cv_folds":            effective_cv,
        "n_train":             int(len(X_train)),
        "n_test":              int(len(X_test)),
        "n_valid_smiles":      int(valid_mask.sum()),
        "n_invalid_smiles":    n_invalid,
        "test_r2":             test_r2,
        "test_rmse":           test_rmse,
        "test_mae":            test_mae,
        "cv_r2_mean":          float(cv_scores.mean()),
        "cv_r2_std":           float(cv_scores.std()),
        "cv_scores":           cv_scores,
        "feature_importances": feature_importances,
        "predictions_df":      predictions_df,
        "model":               model,
        "timestamp":           datetime.datetime.now().isoformat(timespec="seconds"),
    }
