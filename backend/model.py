"""model.py — Bioactivity regression models using ECFP4 (Morgan) fingerprints.

Provides:
    train_bioactivity_model — train RF / Ridge / GB on pIC50 from SMILES

Fingerprint computation is delegated to fingerprints.compute_fp_array.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from utils.fingerprints import (
    compute_fp_array_named, method_is_morgan, method_radius, DEFAULT_FINGERPRINT,
)
from utils.splits import split_train_test, DEFAULT_BUTINA_CUTOFF

# ── Module constants ──────────────────────────────────────────────────────────

MIN_MOLECULES: int = 10   # minimum rows required to attempt model training


# ── Model training ────────────────────────────────────────────────────────────

_SUPPORTED_MODELS = frozenset(
    {"Random Forest", "Ridge Regression", "Elastic Net", "Gradient Boosting"}
)


def train_bioactivity_model(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    n_estimators: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    split_method: str = "random",
    cluster_cutoff: float = DEFAULT_BUTINA_CUTOFF,
    fingerprint: str = DEFAULT_FINGERPRINT,
    feature_selection: str = "none",
    n_features: int = 100,
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
    from sklearn.model_selection import cross_val_score
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
    fp_array, valid_mask = compute_fp_array_named(df_work["canonical_smiles"], fingerprint)
    n_invalid = int((~valid_mask).sum())
    df_valid = df_work[valid_mask].reset_index(drop=True)
    y = df_valid["pIC50"].values.astype(float)

    # ── Train / test split (dispatched by method) ─────────────────────────────
    idx_train, idx_test = split_train_test(
        df_valid["canonical_smiles"],
        test_size=test_size,
        method=split_method,
        random_state=random_state,
        cluster_cutoff=cluster_cutoff,
    )
    X_train = fp_array[idx_train]
    X_test  = fp_array[idx_test]
    y_train = y[idx_train]
    y_test  = y[idx_test]

    if len(X_test) == 0:
        raise ValueError(
            "Test split produced zero samples. "
            "Increase test_size, choose a different split method, or provide more data."
        )

    # ── Optional mRMR feature selection (fit on train only) ───────────────────
    # Keep the FULL fingerprints too: the applicability-domain (AD) check should
    # use full-fingerprint similarity, not the model's reduced bit subset.
    selected_features = None
    X_train_full = X_test_full = None
    if feature_selection == "mrmr":
        from utils.feature_selection import mrmr_select
        selected_features = mrmr_select(X_train, y_train, n_features)
        X_train_full, X_test_full = X_train, X_test
        X_train = X_train_full[:, selected_features]
        X_test  = X_test_full[:, selected_features]

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
    elif model_type == "Elastic Net":
        # Elastic Net blends L1 (sparsity) and L2 (shrinkage) penalties.
        # StandardScaler is required (penalty is scale-sensitive) and bundled in
        # the pipeline so the pickled artefact is self-contained.
        from sklearn.linear_model import ElasticNet
        model = Pipeline([
            ("scaler",     StandardScaler()),
            ("elasticnet", ElasticNet(random_state=random_state, max_iter=5000)),
        ])
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
        "mode":                "quick",
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
        "best_params":         None,
        "tuning_cv_results":   None,
        "split_method":        split_method,
        "cluster_cutoff":      cluster_cutoff if split_method == "butina" else None,
        # Test-set features needed for SHAP explainability
        "X_train":             X_train,
        "X_test":              X_test,
        "test_smiles":         df_valid["canonical_smiles"].iloc[idx_test].tolist(),
        # Training-set references for applicability-domain nearest-neighbour lookup
        "train_smiles":        df_valid["canonical_smiles"].iloc[idx_train].tolist(),
        "train_pic50":         [float(v) for v in y_train],
        "train_ids":           (
            df_valid["molecule_chembl_id"].iloc[idx_train].astype(str).tolist()
            if "molecule_chembl_id" in df_valid.columns else None
        ),
        # Fingerprint metadata (drives Score New re-encoding + SHAP bit mapping)
        "fingerprint":         fingerprint,
        "fp_is_morgan":        method_is_morgan(fingerprint),
        "fp_radius":           method_radius(fingerprint),
        "selected_features":   selected_features,
        "X_train_full":        X_train_full,   # full fps for AD (None if no selection)
        "X_test_full":         X_test_full,
    }


def train_bioactivity_model_tuned(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    cv_folds: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
    split_method: str = "random",
    cluster_cutoff: float = DEFAULT_BUTINA_CUTOFF,
    fingerprint: str = DEFAULT_FINGERPRINT,
    feature_selection: str = "none",
    n_features: int = 100,
) -> dict:
    tuned_ = """Train a regression model with **Optuna**-driven hyperparameter optimisation.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler — smarter than
    random search because it learns from earlier trials. The best estimator is
    refit on the full training set and evaluated on the held-out test set.

    Args:
        df:          DataFrame with 'canonical_smiles' and 'pIC50' columns.
        model_type:  One of "Random Forest", "Ridge Regression", "Gradient Boosting".
        test_size:   Fraction held out for evaluation (0.10–0.40).
        cv_folds:    Inner CV folds used during the search.
        n_iter:      Number of Optuna trials (hyperparameter combinations evaluated).
                     Larger ⇒ better search but slower.
        random_state: Reproducibility seed for both the sampler and the model.

    Returns:
        Same keys as train_bioactivity_model plus:
            best_params        — dict of best hyperparameters found
            tuning_cv_results  — DataFrame: top-10 trials (params, mean / std CV R², rank)
            mode               — "tuned"
    """
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Silence Optuna's per-trial chatter — the UI surfaces results separately.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Validate ──────────────────────────────────────────────────────────────
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

    # ── Fingerprints ──────────────────────────────────────────────────────────
    fp_array, valid_mask = compute_fp_array_named(df_work["canonical_smiles"], fingerprint)
    n_invalid = int((~valid_mask).sum())
    df_valid = df_work[valid_mask].reset_index(drop=True)
    y = df_valid["pIC50"].values.astype(float)

    # ── Split (dispatched by method) ──────────────────────────────────────────
    idx_train, idx_test = split_train_test(
        df_valid["canonical_smiles"],
        test_size=test_size,
        method=split_method,
        random_state=random_state,
        cluster_cutoff=cluster_cutoff,
    )
    X_train = fp_array[idx_train]
    X_test  = fp_array[idx_test]
    y_train = y[idx_train]
    y_test  = y[idx_test]
    if len(X_test) == 0:
        raise ValueError(
            "Test split produced zero samples. "
            "Increase test_size, choose a different split method, or provide more data."
        )

    # ── Optional mRMR feature selection (fit on train only) ───────────────────
    selected_features = None
    X_train_full = X_test_full = None
    if feature_selection == "mrmr":
        from utils.feature_selection import mrmr_select
        selected_features = mrmr_select(X_train, y_train, n_features)
        X_train_full, X_test_full = X_train, X_test
        X_train = X_train_full[:, selected_features]
        X_test  = X_test_full[:, selected_features]

    effective_cv = min(cv_folds, len(X_train))

    # ── Hyperparameter optimisation with Optuna ───────────────────────────────
    def _build_model(params: dict):
        """Construct a fresh estimator from a sampled parameter dict."""
        if model_type == "Random Forest":
            return RandomForestRegressor(random_state=random_state, n_jobs=1, **params)
        if model_type == "Gradient Boosting":
            return GradientBoostingRegressor(random_state=random_state, **params)
        if model_type == "Elastic Net":
            from sklearn.linear_model import ElasticNet
            return Pipeline([
                ("scaler",     StandardScaler()),
                ("elasticnet", ElasticNet(random_state=random_state, max_iter=5000, **params)),
            ])
        # Ridge — wrap in scaler pipeline (penalty is scale-sensitive)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(**params)),
        ])

    def _suggest_params(trial: "optuna.Trial") -> dict:
        """Sample one hyperparameter configuration for the chosen model."""
        if model_type == "Random Forest":
            return {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth":         trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "max_features":      trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5]
                ),
            }
        if model_type == "Gradient Boosting":
            return {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":     trial.suggest_int("max_depth", 3, 9),
                "subsample":     trial.suggest_float("subsample", 0.7, 1.0),
            }
        if model_type == "Elastic Net":
            return {
                "alpha":    trial.suggest_float("alpha", 1e-3, 1e2, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
            }
        # Ridge — single hyperparameter, broad log range
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    def _objective(trial: "optuna.Trial") -> float:
        params = _suggest_params(trial)
        candidate = _build_model(params)
        scores = cross_val_score(
            candidate, X_train, y_train,
            cv=effective_cv, scoring="r2", n_jobs=1,
        )
        # Stash the CV std so we can show it in the trials table
        trial.set_user_attr("std_cv_r2", float(scores.std()))
        return float(scores.mean())

    # Ridge has only one hyperparameter — cap trials to avoid wasted evaluations
    _n_trials = n_iter if model_type != "Ridge Regression" else min(n_iter, 15)

    sampler = TPESampler(seed=random_state)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective, n_trials=_n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    model = _build_model(best_params)
    model.fit(X_train, y_train)

    # ── Top-10 trials table ───────────────────────────────────────────────────
    _trials_df = study.trials_dataframe()
    _done = _trials_df[_trials_df["state"] == "COMPLETE"].copy()
    _done = _done.sort_values("value", ascending=False).reset_index(drop=True)
    _param_cols = [c for c in _done.columns if c.startswith("params_")]

    def _fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    _rows = []
    for rank, (_, t) in enumerate(_done.iterrows(), start=1):
        param_str = " · ".join(
            f"{c.removeprefix('params_')}={_fmt(t[c])}"
            for c in _param_cols if pd.notna(t[c])
        )
        _rows.append({
            "Rank":         rank,
            "Trial":        int(t["number"]),
            "Parameters":   param_str,
            "Mean CV R²":   round(float(t["value"]), 4),
            "Std CV R²":    round(float(t.get("user_attrs_std_cv_r2", float("nan"))), 4),
        })
    tuning_df = pd.DataFrame(_rows).head(10)

    # ── CV on best model (training set) ──────────────────────────────────────
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=effective_cv, scoring="r2",
    )

    # ── Test metrics ──────────────────────────────────────────────────────────
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
    leading = [c for c in ("molecule_chembl_id", "canonical_smiles")
               if c in predictions_df.columns]
    predictions_df = (
        predictions_df[leading + ["actual_pIC50", "predicted_pIC50", "error"]]
        .assign(_abs_err=lambda d: d["error"].abs())
        .sort_values("_abs_err", ascending=False)
        .drop(columns=["_abs_err"])
        .reset_index(drop=True)
    )

    # ── Return ────────────────────────────────────────────────────────────────
    n_est = best_params.get("n_estimators", None)
    return {
        "mode":                "tuned",
        "model_type":          model_type,
        "n_estimators":        n_est,
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
        "best_params":         best_params,
        "tuning_cv_results":   tuning_df,
        "n_iter":              _n_trials,
        "tuner":               "optuna-tpe",
        "best_cv_r2":          float(study.best_value),
        "study":               study,     # exposed for the visualisation tab
        "split_method":        split_method,
        "cluster_cutoff":      cluster_cutoff if split_method == "butina" else None,
        # Test-set features needed for SHAP explainability
        "X_train":             X_train,
        "X_test":              X_test,
        "test_smiles":         df_valid["canonical_smiles"].iloc[idx_test].tolist(),
        # Training-set references for applicability-domain nearest-neighbour lookup
        "train_smiles":        df_valid["canonical_smiles"].iloc[idx_train].tolist(),
        "train_pic50":         [float(v) for v in y_train],
        "train_ids":           (
            df_valid["molecule_chembl_id"].iloc[idx_train].astype(str).tolist()
            if "molecule_chembl_id" in df_valid.columns else None
        ),
        # Fingerprint metadata (drives Score New re-encoding + SHAP bit mapping)
        "fingerprint":         fingerprint,
        "fp_is_morgan":        method_is_morgan(fingerprint),
        "fp_radius":           method_radius(fingerprint),
        "selected_features":   selected_features,
        "X_train_full":        X_train_full,
        "X_test_full":         X_test_full,
    }
