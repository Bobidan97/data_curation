from __future__ import annotations

import numpy as np
import pandas as pd

from utils.fingerprints import compute_fp_array

# Physicochemical features used for PCA (must be computable from canonical_smiles)
PCA_FEATURES = ["MW", "LogP", "TPSA", "RotatableBonds", "HBA", "HBD"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_pca_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute any PCA_FEATURES columns that are missing from df."""
    missing = [c for c in PCA_FEATURES if c not in df.columns]
    if not missing:
        return df

    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    df = df.copy()
    mols = [
        Chem.MolFromSmiles(s) if isinstance(s, str) else None
        for s in df["canonical_smiles"]
    ]

    def _apply(func, mol):
        if mol is None:
            return float("nan")
        try:
            return func(mol)
        except Exception:
            return float("nan")

    col_map = {
        "MW":            lambda m: _apply(Descriptors.MolWt, m),
        "LogP":          lambda m: _apply(Descriptors.MolLogP, m),
        "TPSA":          lambda m: _apply(Descriptors.TPSA, m),
        "RotatableBonds":lambda m: _apply(rdMolDescriptors.CalcNumRotatableBonds, m),
        "HBA":           lambda m: _apply(rdMolDescriptors.CalcNumHBA, m),
        "HBD":           lambda m: _apply(rdMolDescriptors.CalcNumHBD, m),
    }
    for col in missing:
        df[col] = [col_map[col](m) for m in mols]

    return df


# ── PCA ────────────────────────────────────────────────────────────────────────

def run_pca(df: pd.DataFrame) -> dict:
    """PCA on physicochemical descriptors (MW, LogP, TPSA, RotatableBonds, HBA, HBD).

    Descriptors are computed inline if not already present in df.
    Returns a dict with all data needed to render the scree plot and biplot.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Ensure descriptor columns exist
    df_work = _compute_pca_descriptors(df)

    # Drop rows with NaN in any feature column
    df_work = df_work.dropna(subset=PCA_FEATURES).reset_index(drop=True)
    n_samples = len(df_work)

    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 valid molecules for PCA (got {n_samples})."
        )

    features = df_work[PCA_FEATURES].values
    n_components = min(len(PCA_FEATURES), n_samples)

    # Standardise then fit PCA
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    # Build PC score DataFrame aligned to the working df
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    pc_df = pd.DataFrame(scores, columns=pc_cols, index=df_work.index)

    # Normalise PC1 and PC2 to [-1, 1] for biplot overlay with unit-circle loadings
    def _norm(arr):
        rng = arr.max() - arr.min()
        return arr / rng if rng > 0 else arr

    pc_df["PC1_norm"] = _norm(pc_df["PC1"].values)
    pc_df["PC2_norm"] = _norm(pc_df["PC2"].values)

    # Attach metadata columns for hover labels (keep only what exists)
    for col in ["molecule_chembl_id", "pIC50", "canonical_smiles"]:
        if col in df_work.columns:
            pc_df[col] = df_work[col].values

    # Loadings: shape (n_components, n_features) → transpose to (n_features, n_components)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_FEATURES,
        columns=pc_cols,
    )

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    return {
        "pc_df": pc_df,
        "loadings": loadings,
        "explained_var": explained_var,
        "cum_var": cum_var,
        "feature_names": PCA_FEATURES,
        "n_samples": n_samples,
        "n_components": n_components,
    }


# ── t-SNE + K-means ────────────────────────────────────────────────────────────

def run_tsne_kmeans(df: pd.DataFrame, perplexity: int = 30, n_components: int = 2) -> dict:
    """t-SNE on MACCS-key fingerprints (Jaccard ≡ Tanimoto for binary vectors),
    followed by K-means clustering.

    Fingerprint: 167-bit MACCS keys as a dense bit array.
    Distance:    Jaccard (mathematically identical to Tanimoto for binary fps).

    Using bit arrays with metric='jaccard' instead of a precomputed distance
    matrix avoids building an n×n matrix in memory and sidesteps a known
    psutil/sklearn incompatibility that occurs with metric='precomputed'.

    Perplexity is automatically clamped to max(5, min(perplexity, (n-1)//3))
    to satisfy sklearn's constraint.  n_components may be 2 (default) or 3 for
    a 3-D projection.  Returns a dict with all data needed to render the
    silhouette bar chart and t-SNE scatter.
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Build MACCS-key fingerprint array; valid_mask aligns rows with df
    fp_array, valid_mask = compute_fp_array(df["canonical_smiles"], fp_type="maccs")
    df_work = df[valid_mask].reset_index(drop=True)
    n_samples = len(df_work)

    if n_samples < 4:
        raise ValueError(
            f"Need at least 4 valid molecules for t-SNE (got {n_samples})."
        )

    # Clamp perplexity: sklearn requires perplexity < n_samples
    effective_perp = max(5, min(perplexity, (n_samples - 1) // 3))

    # Jaccard metric on binary vectors is equivalent to Tanimoto on fingerprints
    tsne = TSNE(
        n_components=n_components,
        metric="jaccard",
        init="random",
        random_state=42,
        perplexity=effective_perp,
    )
    coords = tsne.fit_transform(fp_array)
    tc_cols = [f"TC{i + 1}" for i in range(n_components)]
    tsne_df = pd.DataFrame(coords, columns=tc_cols, index=df_work.index)

    # Attach metadata for hover labels
    for col in ["molecule_chembl_id", "pIC50", "canonical_smiles"]:
        if col in df_work.columns:
            tsne_df[col] = df_work[col].values

    # K-means: find optimal k by silhouette score (range 2 … min(10, n-1))
    max_k = min(10, n_samples - 1)
    if max_k < 2:
        # Too few molecules to cluster — assign everything to cluster 0
        tsne_df["Cluster"] = "0"
        return {
            "tsne_df": tsne_df,
            "silhouette_scores": {},
            "best_k": 1,
            "effective_perp": effective_perp,
            "n_samples": n_samples,
            "n_components": n_components,
        }

    silhouette_scores: dict[int, float] = {}
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=10, n_init="auto").fit_predict(
            tsne_df[tc_cols]
        )
        silhouette_scores[k] = silhouette_score(tsne_df[tc_cols], labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)

    final_labels = KMeans(n_clusters=best_k, random_state=10, n_init="auto").fit_predict(
        tsne_df[tc_cols]
    )
    tsne_df["Cluster"] = [str(c) for c in final_labels]

    return {
        "tsne_df": tsne_df,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "effective_perp": effective_perp,
        "n_samples": n_samples,
        "n_components": n_components,
    }


# ── Activity Cliffs (SALI) ─────────────────────────────────────────────────────

def compute_activity_cliffs(df: pd.DataFrame, max_molecules: int = 500) -> dict:
    """Compute pairwise Tanimoto similarity and |ΔpIC50| for activity cliff detection.

    Uses Morgan ECFP4 fingerprints as RDKit ExplicitBitVect objects and
    DataStructs.BulkTanimotoSimilarity for pairwise computation.
    DataStructs.ConvertToNumpyArray is intentionally avoided (Boost.Python
    incompatibility in this environment).

    Args:
        df:             DataFrame with 'canonical_smiles' and 'pIC50' columns.
        max_molecules:  Cap on molecules considered; if more are present, a
                        random sample of max_molecules is drawn (random_state=42).

    Returns:
        dict with keys:
            pairs_df      — DataFrame(tanimoto, delta_pic50)
            n_molecules   — int: molecules actually used
            n_pairs       — int: number of valid pairs
            truncated     — bool: True if original set was larger than max_molecules
            n_original    — int: size of df before any sampling
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    for col in ("canonical_smiles", "pIC50"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")

    df_work = df[["canonical_smiles", "pIC50"]].dropna().reset_index(drop=True)
    n_original = len(df_work)
    truncated = n_original > max_molecules
    if truncated:
        df_work = df_work.sample(n=max_molecules, random_state=42).reset_index(drop=True)

    fps: list = []
    pic50_vals: list[float] = []
    for _, row in df_work.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        pic50_vals.append(float(row["pIC50"]))

    if len(fps) < 2:
        raise ValueError(
            f"Need at least 2 molecules with valid SMILES and pIC50 (got {len(fps)})."
        )

    tanimoto_list: list[float] = []
    delta_list: list[float] = []
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j_off, sim in enumerate(sims):
            tanimoto_list.append(float(sim))
            delta_list.append(abs(pic50_vals[i] - pic50_vals[i + 1 + j_off]))

    pairs_df = pd.DataFrame({"tanimoto": tanimoto_list, "delta_pic50": delta_list})
    return {
        "pairs_df":    pairs_df,
        "n_molecules": len(fps),
        "n_pairs":     len(pairs_df),
        "truncated":   truncated,
        "n_original":  n_original,
    }
