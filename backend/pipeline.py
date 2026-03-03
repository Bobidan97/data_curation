import sys
import pandas as pd
from chembl_webresource_client.new_client import new_client
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import generate_chembl_code, classify_query as _classify_query
from rag_system import RAGSystem
from utils.target_resolution import resolve_target_candidates
from utils.molecule_resolution import resolve_molecule_candidates

documents_path = Path(__file__).parent.parent / "documents"


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
    namespace = {"new_client": new_client, "pd": pd}
    if entity_chembl_id is not None:
        key = "target_chembl_id" if domain == "target" else "molecule_chembl_id"
        namespace[key] = entity_chembl_id

    exec(generated_code, namespace)

    df = namespace.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Generated code did not produce a DataFrame named 'df'")

    return df
