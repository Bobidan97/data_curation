import sys
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
)
from model import train_bioactivity_model as _train_bioactivity_model
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


def fetch_chembl_data(
    generated_code: str,
    domain: str = "other",
    entity_chembl_id: str | None = None,
) -> pd.DataFrame:
    """Execute the LLM-generated code with the resolved entity ID injected into scope.

    Injects `target_chembl_id` for target queries, `molecule_chembl_id` for molecule
    queries, or neither for other queries.

    Raises ValueError if the code does not produce a DataFrame named 'df'.
    """
    from chembl_webresource_client.new_client import new_client  # lazy — avoids network call at import time
    namespace = {"new_client": new_client, "pd": pd}
    if entity_chembl_id is not None:
        key = "target_chembl_id" if domain == "target" else "molecule_chembl_id"
        namespace[key] = entity_chembl_id

    exec(generated_code, namespace)

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
    import numpy as np

    schema = build_df_schema(df, extra_dfs=extra_dfs)
    code = generate_dataframe_edit_code(user_instruction, schema)

    namespace = {"df": df.copy(), "pd": pd, "np": np}
    if extra_dfs:
        for name, frame in extra_dfs.items():
            namespace[name] = frame.copy()

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


def run_descriptor_computation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RDKit molecular descriptors and append them as new columns.

    Computes MW, LogP, TPSA, HBA, HBD, Ro5_violations, HeavyAtomCount,
    RotatableBonds, RingCount, and AromaticRings in one pass.

    Returns df unchanged if the `canonical_smiles` column is absent.
    Raises ImportError if RDKit is not installed.
    """
    return _compute_descriptors(df)


def run_model_training(
    df: pd.DataFrame,
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    n_estimators: int = 100,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Train a regression model (RF, Ridge, or GB) to predict pIC50 from ECFP4 fingerprints.

    Thin wrapper around model.train_bioactivity_model — see that function for the
    complete return dict specification and edge-case behaviour.

    Args:
        df:            DataFrame with 'canonical_smiles' and 'pIC50' columns.
        model_type:    One of "Random Forest", "Ridge Regression", "Gradient Boosting".
        test_size:     Fraction held out for evaluation (0.10–0.40).
        n_estimators:  Trees for RF / GB; ignored for Ridge.
        cv_folds:      Cross-validation folds on the training set.
        random_state:  Reproducibility seed.

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
    )


def run_activity_cliffs(df: pd.DataFrame, max_molecules: int = 500) -> dict:
    """Pairwise Tanimoto + |ΔpIC50| for activity cliff scatter.

    Thin wrapper around chemical_space.compute_activity_cliffs.
    See that function for the complete return dict specification.
    """
    return _compute_activity_cliffs(df, max_molecules=max_molecules)
