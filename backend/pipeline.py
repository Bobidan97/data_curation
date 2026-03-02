import sys
import pandas as pd
from chembl_webresource_client.new_client import new_client
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import generate_chembl_code, extract_target_name
from rag_system import RAGSystem
from utils.target_resolution import resolve_target_candidates

documents_path = Path(__file__).parent.parent / "documents"


def run_rag(user_query: str) -> str:
    """Run the RAG pipeline and return the retrieved context string."""
    rag = RAGSystem(directory_path=str(documents_path))
    rag.process_documents()
    result = rag.query(user_query)
    return result["content"]


def get_generated_code(user_query: str, context: str) -> str:
    """Use the LLM to generate ChEMBL Python code from the user query and RAG context."""
    return generate_chembl_code(user_query, context)


def get_target_candidates(user_query: str) -> pd.DataFrame:
    """Extract the target name from the query and resolve candidates from ChEMBL."""
    target_name = extract_target_name(user_query)
    return resolve_target_candidates(target_name)


def fetch_chembl_data(generated_code: str, target_chembl_id: str) -> pd.DataFrame:
    """
    Execute the LLM-generated code with target_chembl_id injected into scope.
    Returns the resulting raw DataFrame.
    Raises ValueError if the code does not produce a DataFrame named 'df'.
    """
    namespace = {
        "target_chembl_id": target_chembl_id,
        "new_client": new_client,
        "pd": pd,
    }
    exec(generated_code, namespace)

    df = namespace.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Generated code did not produce a DataFrame named 'df'")

    return df
