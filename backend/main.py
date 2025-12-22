import sys
import pandas as pd
from llm import build_chembl_query_from_rag
from rag_system import RAGSystem
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
documents_path = Path(__file__).parent.parent / "documents"

def main():
    user_input = input("🔎 Enter your ChEMBL database query: ")

    #step 1: Get documentation from notebook (RAG)
    rag = RAGSystem(directory_path=str(documents_path))
    rag.process_documents()
    rag_result = rag.query(user_input)
    context_from_docs = rag_result['content']

    print("\n📚 Retrieved ChEMBL API usage context (aggregated):")
    print(context_from_docs)

    # Print detailed information for each retrieved chunk
    print("\n📝 Retrieved Chunks with Metadata:")
    for i, doc in enumerate(rag_result['results']):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source Notebook: {doc.metadata.get('source')}")
        print(f"Cell Index: {doc.metadata.get('cell_index')}")
        print(f"Cell Type: {doc.metadata.get('cell_type')}")
        print("Content:")
        print(doc.page_content)

    #step 2: LLM generates a ChEMBL API query plan using user input and notebook content
    pychembl_query_code = build_chembl_query_from_rag(user_input, context_from_docs)

    print("\n🧠 Generated Query Code:")
    print(pychembl_query_code)

    #step 3: Execute the interpreted ChEMBL query
    namespace = {}
    try:
        exec(pychembl_query_code, globals(), namespace)
        chembl_df = namespace.get("filtered_df", None)

        if chembl_df is None:
            chembl_df = pd.DataFrame()
        elif not isinstance(chembl_df, pd.DataFrame):
            try:
                chembl_df = pd.DataFrame.from_records(chembl_df)
            except Exception:
                chembl_df = pd.DataFrame()

        print("💊 Number of results:", len(chembl_df))
        print(chembl_df.head())
        chembl_df.to_csv("chembl_df.csv", index=False)

    except Exception as e:
        print(f"❌ Error executing generated query code: {e}")

if __name__ == "__main__":
    main()

##example Find all inhibitors for erbB2 with IC50 < 100 nM