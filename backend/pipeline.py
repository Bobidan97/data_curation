import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import generate_chembl_code, classify_query as _classify_query
from rag_system import RAGSystem
from utils.target_resolution import resolve_target_candidates
from utils.molecule_resolution import resolve_molecule_candidates
from descriptors import compute_descriptors as _compute_descriptors
from chemical_space import (
    run_pca as _run_pca,
    run_tsne_kmeans as _run_tsne_kmeans,
    compute_activity_cliffs as _compute_activity_cliffs,
    compute_pca_insights as _compute_pca_insights,
    compute_cluster_insights as _compute_cluster_insights,
    build_chemical_space_summary as _build_chemical_space_summary,
    compute_top_scaffolds as _compute_top_scaffolds,
    compute_cluster_chemistry as _compute_cluster_chemistry,
    get_pca_outlier_details as _get_pca_outlier_details,
    get_pc_axis_extremes as _get_pc_axis_extremes,
    get_top_activity_cliff_pairs as _get_top_activity_cliff_pairs,
    compute_pharmacophore_profiles as _compute_pharmacophore_profiles,
    compute_cliff_pair_mcs as _compute_cliff_pair_mcs,
    PHARMACOPHORE_FAMILIES,
)
from model import (
    train_bioactivity_model as _train_bioactivity_model,
    train_bioactivity_model_tuned as _train_bioactivity_model_tuned,
)
from curation import (
    find_structural_duplicates as _find_structural_duplicates,
    get_duplicate_rows as _get_duplicate_rows,
)

documents_path = Path(__file__).parent.parent / "documents"


def _schema_for_df(name: str, df: pd.DataFrame, max_samples: int = 3) -> str:
    """Return a compact schema block for a single named DataFrame."""
    lines = [
        f"{name}  ({df.shape[0]} rows x {df.shape[1]} columns):",
    ]
    for col in df.columns:
        dtype = df[col].dtype
        samples = df[col].dropna().head(max_samples).astype(str).str[:80].tolist()
        sample_str = ", ".join(
            f"'{s}'" if dtype == object else s for s in samples
        )
        lines.append(f"  {col} ({dtype}): {sample_str or 'no non-null values'}")
    return "\n".join(lines)


def build_df_schema(
    df: pd.DataFrame,
    extra_dfs: dict | None = None,
    max_samples: int = 3,
) -> str:
    """Build a schema string describing df and any extra DataFrames for an LLM prompt.

    `df` is the main working DataFrame (will be modified).
    `extra_dfs` is an optional mapping of {name: DataFrame} for read-only references.
    """
    blocks = [_schema_for_df("df  [main — write results here]", df, max_samples)]
    if extra_dfs:
        for name, frame in extra_dfs.items():
            blocks.append(_schema_for_df(f"{name}  [read-only]", frame, max_samples))
    return "\n\n".join(blocks)


def run_rag(user_query: str) -> str:
    """Run the RAG pipeline and return the retrieved context string."""
    rag = RAGSystem(directory_path=str(documents_path))
    rag.process_documents()
    result = rag.query(user_query)
    return result["content"]


def classify_query(user_query: str) -> tuple[str, str | None]:
    """Classify the query domain and extract entity name for resolution.

    Returns:
        (domain, entity_name) where domain is 'target' | 'molecule' | 'other'
        and entity_name is a human-readable name to resolve, or None.
    """
    return _classify_query(user_query)


def get_entity_candidates(domain: str, entity_name: str) -> pd.DataFrame:
    """Resolve a named entity to ChEMBL candidates based on domain."""
    if domain == "target":
        return resolve_target_candidates(entity_name)
    elif domain == "molecule":
        return resolve_molecule_candidates(entity_name)
    raise ValueError(f"No resolution defined for domain '{domain}'")


def get_generated_code(user_query: str, context: str, domain: str = "other") -> str:
    """Use the LLM to generate ChEMBL Python code from the user query and RAG context."""
    return generate_chembl_code(user_query, context, domain)


# Map of "smart" Unicode punctuation → ASCII equivalents.
# LLMs occasionally emit curly quotes (U+2018/19/1C/1D) inside generated Python,
# which raises a confusing SyntaxError. Normalise them before exec.
_SMART_PUNCTUATION = str.maketrans({
    "‘": "'",   # left single quote  '
    "’": "'",   # right single quote '
    "‚": "'",   # single low-9 quote ‚
    "‛": "'",   # single high-reversed-9 quote ‛
    "“": '"',   # left double quote  "
    "”": '"',   # right double quote "
    "„": '"',   # double low-9 quote „
    "‟": '"',   # double high-reversed-9 quote ‟
    "′": "'",   # prime ′
    "″": '"',   # double prime ″
    "–": "-",   # en dash
    "—": "-",   # em dash
    " ": " ",   # non-breaking space
})


def _sanitise_generated_code(code: str) -> str:
    """Normalise Unicode punctuation and validate Python syntax up-front.

    1. Replaces smart quotes / dashes / NBSPs with their ASCII equivalents.
    2. Calls ``compile`` (not ``ast.parse``) to surface SyntaxErrors **before**
       handing the code to ``exec``.  ``compile`` catches strictly more cases
       than ``ast.parse`` — notably the common 'keyword argument repeated'
       error that ``ast.parse`` lets through.

    Raises:
        ValueError: with a user-friendly message if the code is not parseable.
    """
    cleaned = code.translate(_SMART_PUNCTUATION)
    try:
        compile(cleaned, "<llm_generated>", "exec")
    except SyntaxError as exc:
        # exc.msg covers things like "keyword argument repeated",
        # "invalid syntax", "unterminated string literal" etc.
        location = (
            f" (line {exc.lineno}, col {exc.offset})"
            if exc.lineno is not None
            else ""
        )
        raise ValueError(
            f"Generated code is not valid Python: {exc.msg}{location}.\n\n"
            f"Code:\n{cleaned}"
        ) from exc
    return cleaned


def fetch_chembl_data(
    generated_code: str,
    domain: str = "other",
    entity_chembl_id: str | None = None,
) -> pd.DataFrame:
    """Execute the LLM-generated code with the resolved entity ID injected into scope.

    Injects `target_chembl_id` for target queries, `molecule_chembl_id` for molecule
    queries, or neither for other queries.

    The generated code is first sanitised (smart-quote normalisation + AST
    parse check) so that common LLM artefacts surface as clean ValueErrors
    rather than raw Python SyntaxErrors.

    Raises ValueError if the code does not produce a DataFrame named 'df'.
    """
    # Up-front syntax validation + smart-quote normalisation
    generated_code = _sanitise_generated_code(generated_code)

    from chembl_webresource_client.new_client import new_client  # lazy — avoids network call at import time

    # Patch QuerySet so LLM-generated .to_dataframe() calls work
    try:
        from chembl_webresource_client.query_set import QuerySet as _QuerySet
        if not hasattr(_QuerySet, "to_dataframe"):
            _QuerySet.to_dataframe = lambda self: pd.DataFrame(list(self))
    except Exception:
        pass

    namespace = {
        "new_client": new_client,
        "pd":         pd,
        "activity":               new_client.activity,
        "assay":                  new_client.assay,
        "atc_class":              new_client.atc_class,
        "cell_line":              new_client.cell_line,
        "document":               new_client.document,
        "drug":                   new_client.drug,
        "drug_indication":        new_client.drug_indication,
        "drug_warning":           new_client.drug_warning,
        "mechanism":              new_client.mechanism,
        "metabolism":             new_client.metabolism,
        "molecule":               new_client.molecule,
        "organism":               new_client.organism,
        "protein_classification": new_client.protein_classification,
        "similarity":             new_client.similarity,
        "source":                 new_client.source,
        "substructure":           new_client.substructure,
        "target":                 new_client.target,
        "target_component":       new_client.target_component,
        "tissue":                 new_client.tissue,
    }
    if entity_chembl_id is not None:
        key = "target_chembl_id" if domain == "target" else "molecule_chembl_id"
        namespace[key] = entity_chembl_id

    try:
        exec(generated_code, namespace)
    except Exception as exc:
        exc_type = type(exc).__name__
        # Network / server errors get a friendly, actionable message
        if any(kw in exc_type for kw in ("HttpApplicationError", "ConnectionError", "Timeout")):
            raise RuntimeError(
                f"ChEMBL API error ({exc_type}): the EBI server returned an error. "
                "This is a temporary server-side issue — please wait a moment and try again."
            ) from exc
        # Any other exception from the generated code
        raise RuntimeError(
            f"Generated code raised {exc_type}: {exc}\n\nCode:\n{generated_code}"
        ) from exc

    df = namespace.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Generated code did not produce a DataFrame named 'df'")

    return df


def run_chemical_space_pca(df: pd.DataFrame) -> dict:
    """PCA on physicochemical descriptors. Returns dict for scree + biplot rendering."""
    return _run_pca(df)


def run_chemical_space_tsne(df: pd.DataFrame, perplexity: int = 30, n_components: int = 2) -> dict:
    """t-SNE on MACCS Tanimoto distances + K-means clustering. Returns dict for scatter rendering."""
    return _run_tsne_kmeans(df, perplexity, n_components=n_components)


def get_structural_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate molecule groups found in df before deduplication."""
    return _find_structural_duplicates(df)


def get_duplicate_activity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return individual rows for structural duplicate groups with _group and _canonical columns."""
    return _get_duplicate_rows(df)


def _build_llm_exec_namespace(df: pd.DataFrame, extra_dfs: dict | None) -> dict:
    """Build the exec namespace shared by edit_dataframe and answer_dataframe_question.

    Provides a copy of `df`, pandas/numpy, RDKit modules (when installed) and the
    safe `_calc` per-molecule helper, plus copies of any read-only extra_dfs.
    """
    namespace = {"df": df.copy(), "pd": pd, "np": np}

    # Expose RDKit so LLM-generated code can compute descriptors / structural
    # queries on demand. Imported lazily so the pipeline still works without it.
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs

        def _calc(smi, func):
            if not isinstance(smi, str):
                return float("nan")
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return float("nan")
            try:
                return func(mol)
            except Exception:
                return float("nan")

        namespace.update({
            "Chem":             Chem,
            "Descriptors":      Descriptors,
            "rdMolDescriptors": rdMolDescriptors,
            "AllChem":          AllChem,
            "DataStructs":      DataStructs,
            "_calc":            _calc,
        })
    except ImportError:
        pass

    if extra_dfs:
        for name, frame in extra_dfs.items():
            namespace[name] = frame.copy()

    return namespace


def edit_dataframe(
    df: pd.DataFrame,
    user_instruction: str,
    extra_dfs: dict | None = None,
) -> tuple[pd.DataFrame, str]:
    """Apply a natural-language editing instruction to df via LLM-generated pandas code.

    The LLM may read from any DataFrame in `extra_dfs` but must write results to `df`.

    Args:
        df: The main working DataFrame to be modified.
        user_instruction: Free-text instruction from the user.
        extra_dfs: Optional mapping of {name: DataFrame} made available as read-only
                   references in the exec namespace (e.g. raw_df, dup_rows).

    Returns:
        (edited_df, generated_code) — the modified DataFrame and the code that was run.

    Raises:
        ValueError: if the generated code raises a runtime error, does not produce a
                    DataFrame named 'df', or would leave the DataFrame empty.
    """
    from llm import generate_dataframe_edit_code

    schema = build_df_schema(df, extra_dfs=extra_dfs)
    code = generate_dataframe_edit_code(user_instruction, schema)

    namespace = _build_llm_exec_namespace(df, extra_dfs)

    try:
        exec(code, namespace)
    except Exception as exc:
        raise ValueError(
            f"Generated code raised an error: {exc}\n\nCode:\n{code}"
        ) from exc

    result = namespace.get("df")
    if result is None or not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"Generated code did not produce a DataFrame named 'df'.\n\nCode:\n{code}"
        )
    if result.empty:
        raise ValueError(
            "This instruction would remove all rows. Not applied — "
            "try a less restrictive filter."
        )
    return result.reset_index(drop=True), code


def answer_dataframe_question(
    df: pd.DataFrame,
    user_question: str,
    extra_dfs: dict | None = None,
) -> tuple[object, str]:
    """Answer a natural-language *question* about df via read-only LLM-generated code.

    Unlike edit_dataframe, this never mutates the data — it computes an answer
    (scalar, string, Series, or small DataFrame) and returns it.

    Args:
        df:            The working DataFrame to interrogate (read-only).
        user_question: Free-text question from the user.
        extra_dfs:     Optional {name: DataFrame} read-only references.

    Returns:
        (answer, generated_code) — the computed answer object and the code run.

    Raises:
        ValueError: if the generated code errors or produces no `answer`.
    """
    from llm import generate_dataframe_query_code

    schema = build_df_schema(df, extra_dfs=extra_dfs)
    code = generate_dataframe_query_code(user_question, schema)

    namespace = _build_llm_exec_namespace(df, extra_dfs)

    try:
        exec(code, namespace)
    except Exception as exc:
        raise ValueError(
            f"Generated code raised an error: {exc}\n\nCode:\n{code}"
        ) from exc

    if "answer" not in namespace:
        raise ValueError(
            f"The generated code did not produce an `answer`.\n\nCode:\n{code}"
        )
    return namespace["answer"], code


def create_dataframe_plot(
    df: pd.DataFrame,
    user_request: str,
    extra_dfs: dict | None = None,
):
    """Build a Plotly figure from a natural-language chart request.

    Read-only: generates Plotly code that assigns a figure to ``fig`` and runs
    it against a copy of df (plus plotly.express/graph_objects in scope).

    Returns:
        (fig, generated_code) — the Plotly figure object and the code that ran.

    Raises:
        ValueError: if the code errors or produces no `fig`.
    """
    from llm import generate_plot_code
    import plotly.express as px
    import plotly.graph_objects as go

    schema = build_df_schema(df, extra_dfs=extra_dfs)
    code = generate_plot_code(user_request, schema)

    namespace = _build_llm_exec_namespace(df, extra_dfs)
    namespace.update({"px": px, "go": go})

    try:
        exec(code, namespace)
    except Exception as exc:
        raise ValueError(
            f"Generated plot code raised an error: {exc}\n\nCode:\n{code}"
        ) from exc

    fig = namespace.get("fig")
    if fig is None:
        raise ValueError(
            f"The generated code did not produce a `fig`.\n\nCode:\n{code}"
        )
    return fig, code


def run_descriptor_computation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 9 standard RDKit molecular descriptors and append them as new columns.

    Computes MW, LogP, TPSA, HBA, HBD, HeavyAtomCount, RotatableBonds,
    RingCount, and AromaticRings in one pass.

    For any descriptor beyond these nine, users can ask the sidebar LLM chat
    to compute it — RDKit's `Chem`, `Descriptors`, and `rdMolDescriptors`
    modules are exposed in the edit-DataFrame exec namespace.

    Returns df unchanged if the `canonical_smiles` column is absent.
    Raises ImportError if RDKit is not installed.
    """
    return _compute_descriptors(df)


def get_descriptor_correlations(
    df: pd.DataFrame,
    descriptor_cols: list,
    target_col: str = "pIC50",
) -> pd.DataFrame:
    """Pearson/Spearman correlation of each descriptor with the target (thin wrapper)."""
    from descriptors import compute_descriptor_correlations
    return compute_descriptor_correlations(df, descriptor_cols, target_col=target_col)


def run_model_training(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    n_estimators: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
    split_method: str = "random",
    cluster_cutoff: float = 0.35,
    fingerprint: str = "ECFP4 (Morgan r2)",
    feature_selection: str = "none",
    n_features: int = 100,
) -> dict:
    """Train a regression model (RF, Ridge, or GB) to predict pIC50 from ECFP4 fingerprints.

    Thin wrapper around model.train_bioactivity_model — see that function for the
    complete return dict specification and edge-case behaviour.

    Args:
        df:             DataFrame with 'canonical_smiles' and 'pIC50' columns.
        model_type:     One of "Random Forest", "Ridge Regression", "Gradient Boosting".
        test_size:      Fraction held out for evaluation (0.10–0.40).
        n_estimators:   Trees for RF / GB; ignored for Ridge.
        cv_folds:       Cross-validation folds on the training set.
        random_state:   Reproducibility seed.
        split_method:   "random" | "scaffold" | "butina".
        cluster_cutoff: Tanimoto distance threshold for Butina (default 0.35).

    Raises:
        ValueError: propagated from model.train_bioactivity_model for data-quality
                    or configuration errors.
    """
    return _train_bioactivity_model(
        df,
        model_type=model_type,
        test_size=test_size,
        n_estimators=n_estimators,
        cv_folds=cv_folds,
        random_state=random_state,
        split_method=split_method,
        cluster_cutoff=cluster_cutoff,
        fingerprint=fingerprint,
        feature_selection=feature_selection,
        n_features=n_features,
    )


def run_model_training_tuned(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    cv_folds: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
    split_method: str = "random",
    cluster_cutoff: float = 0.35,
    fingerprint: str = "ECFP4 (Morgan r2)",
    feature_selection: str = "none",
    n_features: int = 100,
) -> dict:
    """Train with automated hyperparameter search (RandomizedSearchCV / GridSearchCV).

    Thin wrapper around model.train_bioactivity_model_tuned — see that function
    for the complete return dict specification.
    """
    return _train_bioactivity_model_tuned(
        df,
        model_type=model_type,
        test_size=test_size,
        cv_folds=cv_folds,
        n_iter=n_iter,
        random_state=random_state,
        split_method=split_method,
        cluster_cutoff=cluster_cutoff,
        fingerprint=fingerprint,
        feature_selection=feature_selection,
        n_features=n_features,
    )


def run_activity_cliffs(
    df: pd.DataFrame,
    max_molecules: int = 3000,
    min_similarity: float = 0.3,
) -> dict:
    """Pairwise Tanimoto + |ΔpIC50| for activity cliff scatter.

    Thin wrapper around chemical_space.compute_activity_cliffs.
    See that function for the complete return dict specification.
    """
    return _compute_activity_cliffs(
        df, max_molecules=max_molecules, min_similarity=min_similarity
    )


def run_functional_group_impact(df: pd.DataFrame, min_group_size: int = 3) -> pd.DataFrame:
    """Estimate each functional group's impact on pIC50 (thin wrapper)."""
    from functional_groups import compute_functional_group_impact
    return compute_functional_group_impact(df, min_group_size=min_group_size)


def get_example_with_group(df: pd.DataFrame, smarts: str):
    """A representative dataset molecule containing a given SMARTS group (thin wrapper)."""
    from functional_groups import example_with_group
    return example_with_group(df, smarts)


def run_structure_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Append a validity_issues column flagging structural problems (thin wrapper)."""
    from structure_audit import audit_structures
    return audit_structures(df)


def get_structure_audit_summary(audited_df: pd.DataFrame) -> dict:
    """Summarise a structural audit (thin wrapper)."""
    from structure_audit import summarise_audit
    return summarise_audit(audited_df)


def run_chirality_cliffs(df: pd.DataFrame, delta_threshold: float = 1.0) -> dict:
    """Find stereoisomer pairs with divergent activity (thin wrapper)."""
    from chemical_space import find_chirality_cliffs
    return find_chirality_cliffs(df, delta_threshold=delta_threshold)


def get_pca_insights(pca_result: dict, active_threshold: float = 5.0) -> dict:
    """Interpretive metrics for a PCA result (thin wrapper)."""
    return _compute_pca_insights(pca_result, active_threshold=active_threshold)


def get_cluster_insights(tsne_result: dict, active_threshold: float = 5.0) -> dict:
    """Per-cluster statistics for a t-SNE + K-means result (thin wrapper)."""
    return _compute_cluster_insights(tsne_result, active_threshold=active_threshold)


def get_chemical_space_summary(
    pca_insights: dict | None,
    cluster_insights: dict | None,
    active_threshold: float = 5.0,
) -> str:
    """One-paragraph natural-language summary across PCA + cluster insights."""
    return _build_chemical_space_summary(pca_insights, cluster_insights, active_threshold)


def get_top_scaffolds(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Dataset-wide top-N Bemis-Murcko scaffold table (thin wrapper)."""
    return _compute_top_scaffolds(df, top_n=top_n)


def get_cluster_chemistry(tsne_result: dict, max_samples: int = 3) -> dict:
    """Per-cluster dominant scaffold + sample molecules (thin wrapper)."""
    return _compute_cluster_chemistry(tsne_result, max_samples=max_samples)


def get_pca_outliers(pca_result: dict, max_outliers: int = 8) -> list[dict]:
    """PCA outlier details with full structure info (thin wrapper)."""
    return _get_pca_outlier_details(pca_result, max_outliers=max_outliers)


def get_pca_axis_extremes(pca_result: dict) -> dict:
    """Molecules at the extremes of PC1 and PC2 (thin wrapper)."""
    return _get_pc_axis_extremes(pca_result)


def get_activity_cliff_pairs(
    df: pd.DataFrame,
    sim_threshold: float = 0.6,
    delta_threshold: float = 1.0,
    top_n: int = 6,
) -> pd.DataFrame:
    """Top activity-cliff pairs with both molecules' SMILES (thin wrapper)."""
    return _get_top_activity_cliff_pairs(
        df,
        sim_threshold=sim_threshold,
        delta_threshold=delta_threshold,
        top_n=top_n,
    )


def run_pharmacophore_profiles(df: pd.DataFrame, max_molecules: int = 500) -> dict:
    """Per-molecule pharmacophore feature counts (RDKit BaseFeatures)."""
    return _compute_pharmacophore_profiles(df, max_molecules=max_molecules)


def get_cliff_pair_mcs(
    smi_1: str,
    smi_2: str,
    atom_compare: str = "elements",
    bond_compare: str = "order_exact",
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
) -> dict | None:
    """Compute the Maximum Common Substructure between two molecules (thin wrapper)."""
    return _compute_cliff_pair_mcs(
        smi_1, smi_2,
        atom_compare=atom_compare,
        bond_compare=bond_compare,
        complete_rings_only=complete_rings_only,
        ring_matches_ring_only=ring_matches_ring_only,
    )


def run_desirability_ranking(
    df: pd.DataFrame,
    predicted_pic50: pd.Series | None = None,
    weights: dict | None = None,
    pic50_range: tuple[float, float] = (4.0, 9.0),
) -> pd.DataFrame:
    """Multi-criteria desirability ranking (thin wrapper around desirability.py)."""
    from desirability import compute_desirability
    return compute_desirability(
        df,
        predicted_pic50=predicted_pic50,
        weights=weights,
        pic50_range=pic50_range,
    )


def get_desirability_reasons(row: pd.Series, top_k: int = 3) -> list[str]:
    """Natural-language reason chips for a high-desirability row."""
    from desirability import reasons_for_top
    return reasons_for_top(row, top_k=top_k)


def compute_shap_explanation(model_result: dict):
    """Compute SHAP values for a trained-model result (thin wrapper)."""
    from explainability import compute_shap_values
    return compute_shap_values(
        model_result["model"],
        model_result["X_train"],
        model_result["X_test"],
    )


def get_global_top_bits(shap_result, top_n: int = 12) -> pd.DataFrame:
    """Bits with the largest mean |SHAP| across the test set (thin wrapper)."""
    from explainability import global_top_bits
    return global_top_bits(shap_result, top_n=top_n)


def get_top_bits_for_molecule(shap_result, fingerprint_row, mol_index: int, top_n: int = 6):
    """Top contributing bits for one molecule's prediction (thin wrapper)."""
    from explainability import top_bits_for_prediction
    return top_bits_for_prediction(
        shap_result.shap_values[mol_index],
        fingerprint_row,
        top_n=top_n,
    )


def get_atoms_for_bit(smiles: str, bit_id: int, radius: int = 2) -> list[int]:
    """Atom indices that triggered a given Morgan bit on a molecule (thin wrapper)."""
    from explainability import atoms_for_bit
    return atoms_for_bit(smiles, bit_id, radius=radius)


def extract_assay_vocabulary(
    descriptions,
    min_row_count: int = 2,
    include_unigrams: bool = True,
    include_bigrams: bool = True,
    include_trigrams: bool = False,
) -> pd.DataFrame:
    """Vocabulary table from a list of assay descriptions (thin wrapper)."""
    from assay_vocabulary import extract_vocabulary
    return extract_vocabulary(
        descriptions,
        min_row_count=min_row_count,
        include_unigrams=include_unigrams,
        include_bigrams=include_bigrams,
        include_trigrams=include_trigrams,
    )


def filter_assay_descriptions(
    df: pd.DataFrame,
    terms,
    description_col: str = "assay_description",
) -> tuple:
    """Split df into (kept, removed) by term match (thin wrapper)."""
    from assay_vocabulary import filter_by_terms
    return filter_by_terms(df, terms, description_col=description_col)


def preview_assay_filter(
    df: pd.DataFrame,
    terms,
    description_col: str = "assay_description",
) -> dict:
    """Count what filter_assay_descriptions would remove (thin wrapper)."""
    from assay_vocabulary import preview_filter_impact
    return preview_filter_impact(df, terms, description_col=description_col)


# ── SMARTS-based tools (substructure filter, R-group decomposition, …) ────────

def validate_smarts_pattern(smarts: str):
    from smarts import validate_smarts
    return validate_smarts(smarts)


def filter_dataset_by_smarts(
    df: pd.DataFrame,
    smarts: str,
    keep_matches: bool = True,
):
    from smarts import filter_by_smarts
    return filter_by_smarts(df, smarts, keep_matches=keep_matches)


def preview_smarts_match(df: pd.DataFrame, smarts: str) -> dict:
    from smarts import preview_smarts_filter
    return preview_smarts_filter(df, smarts)


def get_smarts_match_atoms(smiles: str, smarts: str) -> list:
    from smarts import smarts_match_atoms
    return smarts_match_atoms(smiles, smarts)


def run_rgroup_decomposition(df: pd.DataFrame, core_smarts: str) -> dict:
    from smarts import decompose_rgroups
    return decompose_rgroups(df, core_smarts)


def get_morgan_bit_smarts(smiles: str, bit_id: int, radius: int = 2) -> str | None:
    from smarts import morgan_bit_to_smarts
    return morgan_bit_to_smarts(smiles, bit_id, radius=radius)


def score_new_compounds(
    smiles_list: list,
    model_result: dict,
    ad_threshold: float = 0.30,
) -> pd.DataFrame:
    """Score user-supplied SMILES against a trained model (thin wrapper)."""
    from scoring import score_compounds
    return score_compounds(smiles_list, model_result, ad_threshold=ad_threshold)


def get_training_pic50_distribution(model_result: dict) -> dict:
    """Training/test pIC50 summary for contextualising scored compounds."""
    from scoring import training_pic50_distribution
    return training_pic50_distribution(model_result)


def analyze_model_errors(model_result: dict) -> dict:
    """Diagnose trends in a model's test-set prediction errors (thin wrapper)."""
    from error_analysis import analyze_prediction_errors
    return analyze_prediction_errors(model_result)


def detect_upload_columns(df: pd.DataFrame) -> tuple:
    """Guess (smiles_col, name_col) from an uploaded table's headers (thin wrapper)."""
    from data_import import detect_columns
    return detect_columns(df)


def prepare_uploaded_compounds(
    df: pd.DataFrame,
    smiles_col: str,
    name_col: str | None = None,
) -> dict:
    """Validate + reshape an uploaded compound table into raw_df shape (thin wrapper)."""
    from data_import import prepare_compound_import
    return prepare_compound_import(df, smiles_col, name_col=name_col)


