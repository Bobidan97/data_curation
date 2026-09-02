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


# ── Interpretive insights for PCA / t-SNE results ─────────────────────────────

def compute_pca_insights(
    pca_result: dict,
    active_threshold: float = 5.0,
    outlier_z: float = 3.0,
) -> dict:
    """Compute interpretive metrics for a PCA result.

    Args:
        pca_result:       Dict returned by :func:`run_pca`.
        active_threshold: pIC50 cutoff used for the quadrant-enrichment analysis.
        outlier_z:        Standardised radial distance threshold for PCA outliers.

    Returns:
        dict with keys:
            effective_dim       — int: # PCs needed for cum. variance ≥ 90 %.
            pc12_variance       — float: combined variance % captured by PC1+PC2.
            dominant_features   — dict["PC1"|"PC2", list[(feature, signed_loading)]]
                                  Top 2 features per PC ranked by |loading|.
            pic50_correlations  — list[dict] one per PC with Pearson r, p, n.
                                  Empty if pIC50 not in pca_result["pc_df"].
            n_outliers          — int: rows beyond ``outlier_z`` σ in PC1-PC2.
            outlier_ids         — list[str] up to 10 molecule_chembl_id values
                                  (or row indices if molecule_chembl_id absent).
            quadrant_enrichment — DataFrame[quadrant, n, mean_pIC50, pct_active]
                                  | None if pIC50 missing or <10 valid values.
            best_quadrant       — str | None: quadrant with the highest % active.
    """
    pc_df         = pca_result["pc_df"]
    loadings      = pca_result["loadings"]
    explained_var = np.asarray(pca_result["explained_var"])
    cum_var       = np.asarray(pca_result["cum_var"])

    # ── 1. Effective dimensionality (90 % variance) ───────────────────────────
    crossed = np.where(cum_var >= 0.90)[0]
    effective_dim = int(crossed[0] + 1) if crossed.size else int(len(cum_var))
    pc12_var = float((explained_var[:2].sum()) * 100) if len(explained_var) >= 2 else float(explained_var[0] * 100)

    # ── 2. Dominant features per PC (top 2 by |loading|) ──────────────────────
    dominant: dict[str, list[tuple[str, float]]] = {}
    for pc in ("PC1", "PC2"):
        if pc in loadings.columns:
            ranked = loadings[pc].abs().sort_values(ascending=False).head(2).index
            dominant[pc] = [(feat, float(loadings.loc[feat, pc])) for feat in ranked]

    # ── 3. pIC50 ↔ PC correlations ────────────────────────────────────────────
    pic50_corrs: list[dict] = []
    if "pIC50" in pc_df.columns:
        from scipy.stats import pearsonr
        pic50 = pc_df["pIC50"]
        valid_mask = pic50.notna()
        if int(valid_mask.sum()) >= 3:
            for pc in [f"PC{i + 1}" for i in range(len(explained_var))]:
                if pc in pc_df.columns:
                    r, p = pearsonr(pc_df.loc[valid_mask, pc], pic50[valid_mask])
                    pic50_corrs.append({
                        "PC": pc,
                        "r": float(r),
                        "p": float(p),
                        "n": int(valid_mask.sum()),
                    })

    # ── 4. Outliers in PC1-PC2 (radial z-score) ───────────────────────────────
    pc1 = pc_df["PC1"].to_numpy()
    pc2 = pc_df["PC2"].to_numpy()
    s1  = pc1.std() or 1.0
    s2  = pc2.std() or 1.0
    radial_z = np.sqrt(((pc1 - pc1.mean()) / s1) ** 2 + ((pc2 - pc2.mean()) / s2) ** 2)
    outlier_mask = radial_z > outlier_z
    outlier_ids: list[str] = []
    if "molecule_chembl_id" in pc_df.columns:
        outlier_ids = pc_df.loc[outlier_mask, "molecule_chembl_id"].astype(str).head(10).tolist()
    else:
        outlier_ids = [str(i) for i in np.where(outlier_mask)[0][:10]]

    # ── 5. Quadrant enrichment (PC1×PC2 sign quadrants) ───────────────────────
    quadrant_df = None
    best_quad: str | None = None
    if "pIC50" in pc_df.columns and int(pc_df["pIC50"].notna().sum()) >= 10:
        v = pc_df.dropna(subset=["pIC50"]).copy()
        def _q(row):
            if row["PC1"] >= 0 and row["PC2"] >= 0: return "Q1 (+,+)"
            if row["PC1"] <  0 and row["PC2"] >= 0: return "Q2 (-,+)"
            if row["PC1"] <  0 and row["PC2"] <  0: return "Q3 (-,-)"
            return "Q4 (+,-)"
        v["quadrant"]  = v.apply(_q, axis=1)
        v["is_active"] = v["pIC50"] >= active_threshold
        quadrant_df = (
            v.groupby("quadrant")
             .agg(n=("pIC50", "size"),
                  mean_pIC50=("pIC50", "mean"),
                  pct_active=("is_active", "mean"))
             .reset_index()
        )
        quadrant_df["pct_active"]  = (quadrant_df["pct_active"] * 100).round(1)
        quadrant_df["mean_pIC50"]  = quadrant_df["mean_pIC50"].round(2)
        best_quad = str(quadrant_df.loc[quadrant_df["pct_active"].idxmax(), "quadrant"])

    return {
        "effective_dim":       effective_dim,
        "pc12_variance":       pc12_var,
        "dominant_features":   dominant,
        "pic50_correlations":  pic50_corrs,
        "n_outliers":          int(outlier_mask.sum()),
        "outlier_ids":         outlier_ids,
        "quadrant_enrichment": quadrant_df,
        "best_quadrant":       best_quad,
    }


def _scaffold_stats_for_cluster(smiles_tuple: tuple) -> dict:
    """Return per-cluster Bemis-Murcko scaffold statistics."""
    from collections import Counter
    from utils.rdkit_utils import murcko_scaffold

    scaffolds = [s for s in (murcko_scaffold(smi) for smi in smiles_tuple) if s]
    if not scaffolds:
        return {"n_unique": 0, "dominant_pct": 0.0}
    counts = Counter(scaffolds)
    _, top_count = counts.most_common(1)[0]
    return {
        "n_unique":     len(counts),
        "dominant_pct": round(top_count / len(scaffolds) * 100, 1),
    }


def compute_cluster_insights(
    tsne_result: dict,
    active_threshold: float = 5.0,
) -> dict:
    """Per-cluster statistics + 'winner' identification for a t-SNE + K-means result.

    Args:
        tsne_result:      Dict returned by :func:`run_tsne_kmeans`.
        active_threshold: pIC50 cutoff used for the "% active" column.

    Returns:
        dict with keys:
            per_cluster_df       — DataFrame summarising each cluster
                                  (Cluster, Size, Mean pIC50, Median pIC50,
                                  pIC50 range, % active, Unique scaffolds,
                                  Dominant scaffold %).
            most_potent_cluster  — cluster_id with highest Mean pIC50 (or None)
            most_diverse_cluster — cluster_id with widest pIC50 range (or None)
            hit_enriched_cluster — cluster_id with highest % active (or None)
            singleton_clusters   — list of cluster_ids with Size == 1
    """
    tsne_df = tsne_result.get("tsne_df", pd.DataFrame())
    if tsne_df.empty or "Cluster" not in tsne_df.columns:
        return {
            "per_cluster_df":       pd.DataFrame(),
            "most_potent_cluster":  None,
            "most_diverse_cluster": None,
            "hit_enriched_cluster": None,
            "singleton_clusters":   [],
        }

    has_pic50  = "pIC50" in tsne_df.columns and tsne_df["pIC50"].notna().any()
    has_smiles = "canonical_smiles" in tsne_df.columns

    rows = []
    for cluster_id, grp in tsne_df.groupby("Cluster", sort=True):
        row: dict = {"Cluster": str(cluster_id), "Size": int(len(grp))}
        if has_pic50:
            pic = grp["pIC50"].dropna()
            if len(pic) > 0:
                row["Mean pIC50"]   = round(float(pic.mean()),   2)
                row["Median pIC50"] = round(float(pic.median()), 2)
                row["pIC50 range"]  = round(float(pic.max() - pic.min()), 2)
                row["% active"]     = round(float((pic >= active_threshold).mean() * 100), 1)
            else:
                row.update({"Mean pIC50": None, "Median pIC50": None,
                            "pIC50 range": None, "% active": None})
        if has_smiles:
            scaf = _scaffold_stats_for_cluster(
                tuple(grp["canonical_smiles"].dropna().tolist())
            )
            row["Unique scaffolds"]     = scaf["n_unique"]
            row["Dominant scaffold %"]  = scaf["dominant_pct"]
        rows.append(row)

    per_cluster_df = pd.DataFrame(rows)

    most_potent  = None
    most_diverse = None
    hit_rich     = None
    if has_pic50 and not per_cluster_df.empty:
        with_pic = per_cluster_df.dropna(subset=["Mean pIC50"])
        if not with_pic.empty:
            most_potent  = str(with_pic.loc[with_pic["Mean pIC50"].idxmax(),  "Cluster"])
            most_diverse = str(with_pic.loc[with_pic["pIC50 range"].idxmax(), "Cluster"])
            hit_rich     = str(with_pic.loc[with_pic["% active"].idxmax(),    "Cluster"])

    singletons = per_cluster_df.loc[per_cluster_df["Size"] == 1, "Cluster"].astype(str).tolist()

    return {
        "per_cluster_df":       per_cluster_df,
        "most_potent_cluster":  most_potent,
        "most_diverse_cluster": most_diverse,
        "hit_enriched_cluster": hit_rich,
        "singleton_clusters":   singletons,
    }


def build_chemical_space_summary(
    pca_insights:     dict | None,
    cluster_insights: dict | None,
    active_threshold: float = 5.0,
) -> str:
    """Build a one-paragraph natural-language summary from the two insight dicts."""
    parts: list[str] = []

    if pca_insights:
        eff = pca_insights["effective_dim"]
        v12 = pca_insights["pc12_variance"]
        parts.append(
            f"Chemical space requires **{eff} effective dimension"
            f"{'s' if eff != 1 else ''}** to capture 90 % of variance; "
            f"PC1+PC2 alone explain **{v12:.0f} %**."
        )

        pc1 = pca_insights["dominant_features"].get("PC1")
        if pc1:
            feat, load = pc1[0]
            parts.append(f"PC1 is driven primarily by **{feat}** (loading {load:+.2f}).")

        corrs = pca_insights["pic50_correlations"]
        if corrs:
            strongest = max(corrs, key=lambda c: abs(c["r"]))
            if abs(strongest["r"]) >= 0.30 and strongest["p"] < 0.05:
                direction = "positively" if strongest["r"] > 0 else "negatively"
                parts.append(
                    f"pIC50 is **{direction} correlated** with {strongest['PC']} "
                    f"(r = {strongest['r']:+.2f}, p < 0.05) — a property axis is "
                    f"driving activity."
                )

        n_out = pca_insights["n_outliers"]
        if n_out > 0:
            parts.append(
                f"⚠️ **{n_out} potential outlier"
                f"{'s' if n_out != 1 else ''}** beyond 3 σ in PC1-PC2 space — "
                f"worth manual review."
            )

        if pca_insights.get("best_quadrant"):
            parts.append(
                f"Actives concentrate in **{pca_insights['best_quadrant']}** "
                f"of PC1×PC2."
            )

    if cluster_insights:
        pcdf = cluster_insights["per_cluster_df"]
        if not pcdf.empty:
            k = len(pcdf)
            parts.append(f"The dataset splits into **{k} structural cluster"
                         f"{'s' if k != 1 else ''}**.")

            hc = cluster_insights["hit_enriched_cluster"]
            if hc is not None:
                row = pcdf[pcdf["Cluster"] == hc].iloc[0]
                pct = row.get("% active")
                if pct is not None:
                    parts.append(
                        f"Cluster **{hc}** is hit-enriched "
                        f"(n = {int(row['Size'])}, {pct:.0f} % active at "
                        f"pIC50 ≥ {active_threshold:g}, mean pIC50 "
                        f"{row['Mean pIC50']:.2f})."
                    )

            sing = cluster_insights["singleton_clusters"]
            if sing:
                parts.append(
                    f"⚠️ {len(sing)} singleton cluster"
                    f"{'s' if len(sing) != 1 else ''} (size 1) — likely outliers."
                )

    if not parts:
        return "Run PCA or t-SNE above to populate chemical-space insights."
    return "\n".join(f"- {p}" for p in parts)


# ── Pharmacophore profiling (RDKit BaseFeatures) ──────────────────────────────

# Canonical pharmacophore families tracked across the app
PHARMACOPHORE_FAMILIES: tuple[str, ...] = (
    "Donor", "Acceptor", "Hydrophobe", "Aromatic", "PosIonizable", "NegIonizable"
)

_PHARM_FEATURE_FACTORY = None


def _get_pharm_factory():
    """Cached RDKit MolChemicalFeatureFactory built from BaseFeatures.fdef."""
    global _PHARM_FEATURE_FACTORY
    if _PHARM_FEATURE_FACTORY is None:
        import os
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _PHARM_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    return _PHARM_FEATURE_FACTORY


def compute_pharmacophore_profiles(
    df: pd.DataFrame,
    max_molecules: int = 500,
) -> dict:
    """Per-molecule pharmacophore feature counts using RDKit's BaseFeatures.

    For each molecule the six canonical feature families (Donor, Acceptor,
    Hydrophobe, Aromatic, PosIonizable, NegIonizable) are counted. The
    LumpedHydrophobe family — RDKit's bundling of multiple hydrophobic atoms —
    is folded into Hydrophobe so the counts stay intuitive.

    Args:
        df:            DataFrame containing a 'canonical_smiles' column.
        max_molecules: Cap on the number of molecules profiled. When the input
                       is larger, a deterministic sample (random_state=42) is
                       drawn — keeps interactive performance reasonable.

    Returns:
        dict with:
            profiles_df   — DataFrame with one row per profiled molecule and
                            one column per PHARMACOPHORE_FAMILIES entry, plus
                            any of [molecule_chembl_id, canonical_smiles, pIC50]
                            that were present in df.
            family_means  — dict {family: float} — mean count across profiled
                            molecules. Useful for the dataset-level bar chart.
            n_molecules   — int: molecules successfully profiled
            n_original    — int: rows in df before sampling
            truncated     — bool: True if a sample was drawn
    """
    if "canonical_smiles" not in df.columns:
        raise ValueError("DataFrame must contain a 'canonical_smiles' column.")
    try:
        from rdkit import Chem
        factory = _get_pharm_factory()
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for pharmacophore profiling."
        ) from exc

    from collections import Counter

    df_work    = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    n_original = len(df_work)
    truncated  = n_original > max_molecules
    if truncated:
        df_work = df_work.sample(n=max_molecules, random_state=42).reset_index(drop=True)

    has_id    = "molecule_chembl_id" in df_work.columns
    has_smi   = "canonical_smiles" in df_work.columns
    has_pic50 = "pIC50" in df_work.columns

    rows: list[dict] = []
    for _, r in df_work.iterrows():
        smi = r["canonical_smiles"]
        try:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        except Exception:
            mol = None

        if mol is None:
            counts = {f: 0 for f in PHARMACOPHORE_FAMILIES}
        else:
            raw = Counter(feat.GetFamily() for feat in factory.GetFeaturesForMol(mol))
            # RDKit reports "LumpedHydrophobe" separately — fold into Hydrophobe
            raw["Hydrophobe"] = raw.get("Hydrophobe", 0) + raw.pop("LumpedHydrophobe", 0)
            counts = {f: int(raw.get(f, 0)) for f in PHARMACOPHORE_FAMILIES}

        # Carry useful metadata through
        if has_id:    counts["molecule_chembl_id"] = r["molecule_chembl_id"]
        if has_smi:   counts["canonical_smiles"]   = smi
        if has_pic50: counts["pIC50"]              = r.get("pIC50")
        rows.append(counts)

    profiles_df  = pd.DataFrame(rows)
    family_means = {f: float(profiles_df[f].mean()) for f in PHARMACOPHORE_FAMILIES}

    return {
        "profiles_df":  profiles_df,
        "family_means": family_means,
        "n_molecules":  len(profiles_df),
        "n_original":   n_original,
        "truncated":    truncated,
    }


# ── Structure-gallery helpers (PCA + t-SNE chemistry views) ───────────────────

def compute_top_scaffolds(
    df: pd.DataFrame,
    top_n: int = 10,
    active_threshold: float = 5.0,
) -> pd.DataFrame:
    """Dataset-wide Bemis–Murcko scaffold frequency table.

    Returns a DataFrame with one row per scaffold (top N by count), columns:
        scaffold_smiles · sample_smiles · count · pct · mean_pIC50 · pct_active
    pIC50 columns are omitted if the column isn't present.
    """
    if "canonical_smiles" not in df.columns:
        return pd.DataFrame()
    from utils.rdkit_utils import murcko_scaffold

    # Gate on actual values, not just column presence — a pIC50 column that's
    # present but entirely NaN (e.g. a molecule-only upload) would otherwise
    # produce an all-None "pIC50" field below, which pandas infers as object
    # dtype (it can't tell it should be float without seeing a real number),
    # and grouped["pIC50"].mean() then raises on the non-numeric dtype.
    has_pic50 = "pIC50" in df.columns and df["pIC50"].notna().any()
    records = []
    for _, row in df.iterrows():
        smi = row.get("canonical_smiles")
        if not isinstance(smi, str):
            continue
        scaf = murcko_scaffold(smi)
        if scaf is None:
            continue
        records.append({
            "scaffold_smiles": scaf,
            "original_smiles": smi,
            "pIC50":           float(row["pIC50"]) if has_pic50 and pd.notna(row.get("pIC50")) else None,
        })

    if not records:
        return pd.DataFrame()

    work    = pd.DataFrame(records)
    n_total = len(work)
    grouped = work.groupby("scaffold_smiles")

    summary = pd.DataFrame({
        "count":         grouped.size(),
        "pct":           grouped.size() / n_total * 100,
        "sample_smiles": grouped["original_smiles"].first(),
    })

    if has_pic50:
        summary["mean_pIC50"] = grouped["pIC50"].mean()
        summary["pct_active"] = grouped["pIC50"].apply(
            lambda s: (s.dropna() >= active_threshold).mean() * 100 if s.notna().any() else None
        )

    summary = (
        summary
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index()
    )
    summary["count"] = summary["count"].astype(int)
    summary["pct"]   = summary["pct"].round(1)
    if has_pic50:
        summary["mean_pIC50"] = summary["mean_pIC50"].round(2)
        summary["pct_active"] = summary["pct_active"].round(1)
    return summary


def compute_cluster_chemistry(
    tsne_result: dict,
    max_samples: int = 3,
) -> dict:
    """Per-cluster dominant scaffold + sample molecules for a structure gallery.

    Returns:
        dict keyed by cluster_id (string) with:
            dominant_scaffold — canonical SMILES of the most common scaffold (or None)
            scaffold_pct      — % of cluster molecules sharing that scaffold
            n_unique          — # unique scaffolds in the cluster
            samples           — list[dict] of up to ``max_samples`` molecules,
                                each with smiles, molecule_chembl_id, pIC50.
                                Preference order: members of the dominant
                                scaffold, then highest-pIC50 first.
    """
    tsne_df = tsne_result.get("tsne_df", pd.DataFrame())
    if tsne_df.empty or "Cluster" not in tsne_df.columns or "canonical_smiles" not in tsne_df.columns:
        return {}

    from collections import Counter
    from utils.rdkit_utils import murcko_scaffold

    has_pic50 = "pIC50" in tsne_df.columns and tsne_df["pIC50"].notna().any()
    has_id    = "molecule_chembl_id" in tsne_df.columns
    out: dict[str, dict] = {}

    for cluster_id, grp in tsne_df.groupby("Cluster", sort=True):
        scaffolds: list[str] = []
        valid:    list[dict] = []
        for _, r in grp.iterrows():
            smi = r.get("canonical_smiles")
            if not isinstance(smi, str):
                continue
            scaf = murcko_scaffold(smi)
            if scaf is None:
                continue
            scaffolds.append(scaf)
            valid.append({
                "smiles":             smi,
                "scaffold":           scaf,
                "molecule_chembl_id": str(r["molecule_chembl_id"]) if has_id else None,
                "pIC50":              float(r["pIC50"]) if has_pic50 and pd.notna(r.get("pIC50")) else None,
            })

        if not scaffolds:
            out[str(cluster_id)] = {
                "dominant_scaffold": None,
                "scaffold_pct":      0.0,
                "n_unique":          0,
                "samples":           [],
            }
            continue

        counts                 = Counter(scaffolds)
        dom_scaf, dom_count    = counts.most_common(1)[0]
        dom_pct                = dom_count / len(scaffolds) * 100

        # Prefer molecules sharing the dominant scaffold; sort by pIC50 desc when available
        members  = [v for v in valid if v["scaffold"] == dom_scaf]
        outsiders = [v for v in valid if v["scaffold"] != dom_scaf]
        if has_pic50:
            members  = sorted(members,  key=lambda x: x["pIC50"] if x["pIC50"] is not None else -1e9, reverse=True)
            outsiders = sorted(outsiders, key=lambda x: x["pIC50"] if x["pIC50"] is not None else -1e9, reverse=True)

        samples = members[:max_samples]
        if len(samples) < max_samples:
            samples.extend(outsiders[: max_samples - len(samples)])

        out[str(cluster_id)] = {
            "dominant_scaffold": dom_scaf,
            "scaffold_pct":      round(dom_pct, 1),
            "n_unique":          len(counts),
            "samples":           samples,
        }
    return out


def get_pca_outlier_details(
    pca_result: dict,
    max_outliers: int = 8,
    outlier_z: float = 3.0,
) -> list[dict]:
    """Return the most extreme PCA outliers with full structure info.

    Each entry has: molecule_chembl_id (if present), canonical_smiles, PC1,
    PC2, radial_z, pIC50 (if present). Ordered by radial_z descending.
    """
    pc_df = pca_result["pc_df"]
    if "PC1" not in pc_df.columns or "PC2" not in pc_df.columns:
        return []

    pc1, pc2 = pc_df["PC1"].to_numpy(), pc_df["PC2"].to_numpy()
    s1 = pc1.std() or 1.0
    s2 = pc2.std() or 1.0
    radial_z = np.sqrt(((pc1 - pc1.mean()) / s1) ** 2 + ((pc2 - pc2.mean()) / s2) ** 2)
    mask = radial_z > outlier_z
    if not mask.any():
        return []

    rows = []
    for i in np.where(mask)[0]:
        entry = {
            "PC1":      float(pc_df["PC1"].iloc[i]),
            "PC2":      float(pc_df["PC2"].iloc[i]),
            "radial_z": float(radial_z[i]),
        }
        if "molecule_chembl_id" in pc_df.columns:
            entry["molecule_chembl_id"] = str(pc_df["molecule_chembl_id"].iloc[i])
        if "canonical_smiles" in pc_df.columns:
            entry["canonical_smiles"] = str(pc_df["canonical_smiles"].iloc[i])
        if "pIC50" in pc_df.columns:
            v = pc_df["pIC50"].iloc[i]
            entry["pIC50"] = float(v) if pd.notna(v) else None
        rows.append(entry)

    rows.sort(key=lambda r: r["radial_z"], reverse=True)
    return rows[:max_outliers]


def get_pc_axis_extremes(pca_result: dict) -> dict:
    """For PC1 and PC2, return the molecule at each extreme (max / min score)."""
    pc_df = pca_result["pc_df"]
    if "canonical_smiles" not in pc_df.columns:
        return {}

    has_id   = "molecule_chembl_id" in pc_df.columns
    has_pic  = "pIC50" in pc_df.columns
    out: dict[str, dict] = {}

    for pc in ("PC1", "PC2"):
        if pc not in pc_df.columns:
            continue
        for direction, idx in (("max", pc_df[pc].idxmax()), ("min", pc_df[pc].idxmin())):
            row   = pc_df.loc[idx]
            entry = {
                "pc_value":         float(row[pc]),
                "canonical_smiles": str(row["canonical_smiles"]),
            }
            if has_id:
                entry["molecule_chembl_id"] = str(row["molecule_chembl_id"])
            if has_pic:
                v = row["pIC50"]
                entry["pIC50"] = float(v) if pd.notna(v) else None
            out[f"{pc}_{direction}"] = entry
    return out


def compute_cliff_pair_mcs(
    smi_1: str,
    smi_2: str,
    atom_compare: str = "elements",
    bond_compare: str = "order_exact",
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
    timeout: int = 2,
) -> dict | None:
    """Find the Maximum Common Substructure (MCS) between two SMILES strings.

    Args:
        smi_1, smi_2:           Input SMILES strings.
        atom_compare:           ``"any"`` | ``"elements"`` | ``"isotopes"``
                                 — control how atoms are matched. "elements"
                                 (default) requires identical atomic numbers.
        bond_compare:           ``"any"`` | ``"order"`` | ``"order_exact"``
                                 — control how bonds are matched.
        complete_rings_only:    Refuse partial rings in the MCS (recommended).
        ring_matches_ring_only: Ring atoms only match ring atoms.
        timeout:                Hard timeout in seconds (MCS is NP-hard in
                                 the worst case).

    Returns a dict with everything needed to render two molecules with the
    shared scaffold greyed-out and the differing R-groups highlighted:
        mol_1, mol_2          — RDKit Mol objects
        mcs_atoms_1, mcs_atoms_2 — atom indices belonging to the MCS in each
        diff_atoms_1, diff_atoms_2 — atom indices NOT in the MCS (= R-groups)
        diff_bonds_1, diff_bonds_2 — bond indices NOT in the MCS
        mcs_smarts            — SMARTS string of the matched MCS
        mcs_size              — number of atoms in the MCS
    Returns None when either SMILES is invalid or no MCS can be found.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
    except ImportError:
        return None

    mol_1 = Chem.MolFromSmiles(smi_1) if isinstance(smi_1, str) else None
    mol_2 = Chem.MolFromSmiles(smi_2) if isinstance(smi_2, str) else None
    if mol_1 is None or mol_2 is None:
        return None

    _atom_lookup = {
        "any":       rdFMCS.AtomCompare.CompareAny,
        "elements":  rdFMCS.AtomCompare.CompareElements,
        "isotopes":  rdFMCS.AtomCompare.CompareIsotopes,
    }
    _bond_lookup = {
        "any":         rdFMCS.BondCompare.CompareAny,
        "order":       rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
    }

    try:
        result = rdFMCS.FindMCS(
            [mol_1, mol_2],
            timeout=timeout,
            completeRingsOnly=complete_rings_only,
            ringMatchesRingOnly=ring_matches_ring_only,
            atomCompare=_atom_lookup.get(atom_compare, rdFMCS.AtomCompare.CompareElements),
            bondCompare=_bond_lookup.get(bond_compare, rdFMCS.BondCompare.CompareOrderExact),
        )
    except Exception:
        return None

    if result.canceled or result.numAtoms == 0:
        return None

    mcs_mol = Chem.MolFromSmarts(result.smartsString)
    if mcs_mol is None:
        return None

    match_1 = mol_1.GetSubstructMatch(mcs_mol)
    match_2 = mol_2.GetSubstructMatch(mcs_mol)
    if not match_1 or not match_2:
        return None

    mcs_set_1 = set(match_1)
    mcs_set_2 = set(match_2)
    diff_atoms_1 = [a.GetIdx() for a in mol_1.GetAtoms() if a.GetIdx() not in mcs_set_1]
    diff_atoms_2 = [a.GetIdx() for a in mol_2.GetAtoms() if a.GetIdx() not in mcs_set_2]

    def _diff_bonds(mol, mcs_set):
        return [
            b.GetIdx() for b in mol.GetBonds()
            if b.GetBeginAtomIdx() not in mcs_set or b.GetEndAtomIdx() not in mcs_set
        ]

    return {
        "mol_1":        mol_1,
        "mol_2":        mol_2,
        "mcs_atoms_1":  list(match_1),
        "mcs_atoms_2":  list(match_2),
        "diff_atoms_1": diff_atoms_1,
        "diff_atoms_2": diff_atoms_2,
        "diff_bonds_1": _diff_bonds(mol_1, mcs_set_1),
        "diff_bonds_2": _diff_bonds(mol_2, mcs_set_2),
        "mcs_smarts":   result.smartsString,
        "mcs_size":     int(result.numAtoms),
    }


def get_top_activity_cliff_pairs(
    df: pd.DataFrame,
    sim_threshold:   float = 0.6,
    delta_threshold: float = 1.0,
    top_n:           int   = 6,
    max_molecules:   int   = 3000,
) -> pd.DataFrame:
    """Top activity-cliff pairs ranked by Tanimoto × |ΔpIC50|.

    A pair counts as a cliff when Tanimoto ≥ ``sim_threshold`` AND
    |ΔpIC50| ≥ ``delta_threshold``. Top pairs are ranked by the product
    (strongest cliffs first).

    Returns DataFrame with columns:
        id_1, smiles_1, pIC50_1, id_2, smiles_2, pIC50_2,
        tanimoto, delta_pic50, cliff_score
    (id_* columns omitted when molecule_chembl_id is missing)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    for col in ("canonical_smiles", "pIC50"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")

    has_id = "molecule_chembl_id" in df.columns
    keep   = ["canonical_smiles", "pIC50"] + (["molecule_chembl_id"] if has_id else [])
    work   = df[keep].dropna(subset=["canonical_smiles", "pIC50"]).reset_index(drop=True)
    if len(work) > max_molecules:
        work = work.sample(n=max_molecules, random_state=42).reset_index(drop=True)

    fps, rows = [], []
    for _, r in work.iterrows():
        mol = Chem.MolFromSmiles(r["canonical_smiles"])
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        rows.append(r)
    if len(fps) < 2:
        return pd.DataFrame()

    cliffs = []
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for off, sim in enumerate(sims):
            j     = i + 1 + off
            delta = abs(rows[i]["pIC50"] - rows[j]["pIC50"])
            if sim >= sim_threshold and delta >= delta_threshold:
                cliffs.append((i, j, float(sim), float(delta), float(sim * delta)))

    if not cliffs:
        return pd.DataFrame()

    cliffs.sort(key=lambda p: p[4], reverse=True)
    out_rows = []
    for i, j, sim, delta, score in cliffs[:top_n]:
        entry = {
            "smiles_1":    rows[i]["canonical_smiles"],
            "smiles_2":    rows[j]["canonical_smiles"],
            "pIC50_1":     float(rows[i]["pIC50"]),
            "pIC50_2":     float(rows[j]["pIC50"]),
            "tanimoto":    round(sim,   3),
            "delta_pic50": round(delta, 2),
            "cliff_score": round(score, 3),
        }
        if has_id:
            entry["id_1"] = str(rows[i]["molecule_chembl_id"])
            entry["id_2"] = str(rows[j]["molecule_chembl_id"])
        out_rows.append(entry)
    return pd.DataFrame(out_rows)


# ── Activity Cliffs (SALI) ─────────────────────────────────────────────────────

def compute_activity_cliffs(
    df: pd.DataFrame,
    max_molecules: int = 3000,
    min_similarity: float = 0.3,
) -> dict:
    """Compute pairwise Tanimoto similarity and |ΔpIC50| for activity cliff detection.

    Uses Morgan ECFP4 fingerprints and DataStructs.BulkTanimotoSimilarity (fast
    C-level pairwise comparison).

    Rather than random-subsampling molecules — which risks *missing* the cliffs
    that happen to fall outside the sample — this scans every pair but only
    **retains pairs with Tanimoto ≥ ``min_similarity``**. Activity cliffs live
    in the high-similarity region, so the discarded low-similarity pairs are
    uninformative. Keeping only the relevant pairs bounds memory to a small
    fraction of the O(n²) total, letting far more molecules be analysed.

    A ``max_molecules`` cap remains as a runtime safety net for the O(n²) scan
    itself; only above that is a random sample drawn.

    Args:
        df:             DataFrame with 'canonical_smiles' and 'pIC50' columns.
        max_molecules:  Molecule cap for the pairwise scan (sampled above this).
        min_similarity: Only pairs with Tanimoto ≥ this are retained/plotted.

    Returns:
        dict with keys:
            pairs_df       — DataFrame(tanimoto, delta_pic50) of RETAINED pairs
            n_molecules    — molecules actually scanned
            n_pairs        — retained pairs (≥ min_similarity)
            n_total_pairs  — total pairs scanned before the similarity filter
            min_similarity — the floor applied
            truncated      — True if the molecule set was sampled to the cap
            n_original     — molecules before any sampling
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
    n_total_pairs = 0
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        n_total_pairs += len(sims)
        for j_off, sim in enumerate(sims):
            if sim >= min_similarity:                       # keep only relevant pairs
                tanimoto_list.append(float(sim))
                delta_list.append(abs(pic50_vals[i] - pic50_vals[i + 1 + j_off]))

    pairs_df = pd.DataFrame({"tanimoto": tanimoto_list, "delta_pic50": delta_list})
    return {
        "pairs_df":       pairs_df,
        "n_molecules":    len(fps),
        "n_pairs":        len(pairs_df),
        "n_total_pairs":  n_total_pairs,
        "min_similarity": min_similarity,
        "truncated":      truncated,
        "n_original":     n_original,
    }


# ── Chirality (stereochemistry) cliffs ────────────────────────────────────────

def find_chirality_cliffs(
    df: pd.DataFrame,
    delta_threshold: float = 1.0,
    max_molecules: int = 5000,
) -> dict:
    """Find pairs of stereoisomers with a meaningful activity difference.

    A chirality cliff is the cleanest possible activity cliff: two molecules that
    are **identical except for stereochemistry** (enantiomers, diastereomers, or
    E/Z isomers) yet differ in potency — direct evidence that 3D configuration
    drives binding. Detection removes all stereo descriptors to get each
    molecule's 2D (flat) canonical form, groups by it, and pairs distinct stereo
    variants that share the same flat structure.

    Args:
        df:              DataFrame with 'canonical_smiles' and 'pIC50' columns.
        delta_threshold: |ΔpIC50| at/above which a stereo pair is flagged a cliff.
        max_molecules:   Cap; larger inputs are sampled (random_state=42).

    Returns:
        dict with:
            pairs_df    — DataFrame(flat_smiles, smiles_1, pIC50_1, smiles_2,
                          pIC50_2, delta_pic50, is_cliff [, id_1, id_2]),
                          sorted by |ΔpIC50| descending. Empty if no stereoisomer
                          pairs exist.
            n_molecules — molecules considered
            n_stereo_pairs — total stereoisomer pairs found
            n_cliffs    — pairs at/above the threshold
            n_stereo_molecules — molecules that carry any stereochemistry
            truncated / n_original — sampling info
    """
    from rdkit import Chem

    for col in ("canonical_smiles", "pIC50"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")

    has_id = "molecule_chembl_id" in df.columns
    keep = ["canonical_smiles", "pIC50"] + (["molecule_chembl_id"] if has_id else [])
    work = df[keep].dropna(subset=["canonical_smiles", "pIC50"]).reset_index(drop=True)
    n_original = len(work)
    truncated = n_original > max_molecules
    if truncated:
        work = work.sample(n=max_molecules, random_state=42).reset_index(drop=True)

    # ── Compute flat (stereo-removed) + full canonical SMILES per molecule ────
    records = []
    n_stereo_molecules = 0
    for _, row in work.iterrows():
        smi = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        full = Chem.MolToSmiles(mol)               # canonical, stereo preserved
        flat_mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(flat_mol)
        flat = Chem.MolToSmiles(flat_mol)          # canonical, stereo stripped
        if full != flat:
            n_stereo_molecules += 1
        rec = {"flat": flat, "full": full, "pIC50": float(row["pIC50"])}
        if has_id:
            rec["id"] = str(row["molecule_chembl_id"])
        records.append(rec)

    empty = pd.DataFrame(
        columns=["flat_smiles", "smiles_1", "pIC50_1", "smiles_2", "pIC50_2",
                 "delta_pic50", "is_cliff"]
    )
    if not records:
        return {
            "pairs_df": empty, "n_molecules": 0, "n_stereo_pairs": 0,
            "n_cliffs": 0, "n_stereo_molecules": 0,
            "truncated": truncated, "n_original": n_original,
        }

    rec_df = pd.DataFrame(records)

    # Aggregate pIC50 per distinct stereo variant (median across duplicates)
    agg_spec = {"pIC50": "median"}
    if has_id:
        agg_spec["id"] = "first"
    variants = rec_df.groupby(["flat", "full"]).agg(agg_spec).reset_index()

    # ── Pair up distinct stereo variants that share a flat structure ──────────
    rows = []
    for flat, grp in variants.groupby("flat"):
        if grp["full"].nunique() < 2:
            continue
        recs = grp.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                delta = abs(a["pIC50"] - b["pIC50"])
                entry = {
                    "flat_smiles":  flat,
                    "smiles_1":     a["full"],
                    "pIC50_1":      round(a["pIC50"], 3),
                    "smiles_2":     b["full"],
                    "pIC50_2":      round(b["pIC50"], 3),
                    "delta_pic50":  round(delta, 3),
                    "is_cliff":     delta >= delta_threshold,
                }
                if has_id:
                    entry["id_1"] = a["id"]
                    entry["id_2"] = b["id"]
                rows.append(entry)

    if not rows:
        return {
            "pairs_df": empty, "n_molecules": len(rec_df), "n_stereo_pairs": 0,
            "n_cliffs": 0, "n_stereo_molecules": n_stereo_molecules,
            "truncated": truncated, "n_original": n_original,
        }

    pairs_df = pd.DataFrame(rows).sort_values("delta_pic50", ascending=False).reset_index(drop=True)
    return {
        "pairs_df":           pairs_df,
        "n_molecules":        len(rec_df),
        "n_stereo_pairs":     len(pairs_df),
        "n_cliffs":           int(pairs_df["is_cliff"].sum()),
        "n_stereo_molecules": n_stereo_molecules,
        "truncated":          truncated,
        "n_original":         n_original,
    }


