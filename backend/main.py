import sys

from llm import build_chembl_query_from_rag
from rag_system import RAGSystem
from chembl_query import execute_chembl_query
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
documents_path = Path(__file__).parent.parent / "documents"

def main():
    user_input = input("🔎 Enter your bioactivity query: ")

    #step 1: Get documentation from notebook (RAG)
    rag = RAGSystem(directory_path=str(documents_path))
    rag.process_documents()
    rag_result = rag.query(user_input)
    context_from_docs = rag_result['content']

    print("\n📚 Retrieved ChEMBL API usage context:")
    print(context_from_docs)

    #step 2: LLM generates a ChEMBL API query plan using user input + notebook content
    chembl_query_plan = build_chembl_query_from_rag(user_input, context_from_docs)

    print("\n🧠 Generated Query Plan:")
    print(chembl_query_plan)

    #step 3: Execute the interpreted ChEMBL query
    results = execute_chembl_query(chembl_query_plan)

    print("\n💊 Results:")
    if "error" in results:
        print("❌ Error:", results["error"])
    else:
        # Automatically detect the key that contains the result list
        for key in ["activities", "molecules", "targets", "assays", "compounds"]:
            if key in results:
                records = results[key]
                df = pd.DataFrame(records)
                print(len(df))
                print(df.head())
                break
        else:
            print("⚠️ No recognized result list found in response.")

if __name__ == "__main__":
    main()
