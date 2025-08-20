import sys

from llm import build_chembl_query_from_rag
from rag_system import RAGSystem
from chembl_query import execute_chembl_query
from pathlib import Path
from chembl_webresource_client.new_client import new_client
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
documents_path = Path(__file__).parent.parent / "documents"

def main():
    user_input = input("🔎 Enter your ChEMBL database query: ")

    #step 1: Get documentation from notebook (RAG)
    rag = RAGSystem(directory_path=str(documents_path))
    rag.process_documents()
    rag_result = rag.query(user_input)
    context_from_docs = rag_result['content']

    print("\n📚 Retrieved ChEMBL API usage context:")
    print(context_from_docs)

    #step 2: LLM generates a ChEMBL API query plan using user input + notebook content
    pychembl_query_code = build_chembl_query_from_rag(user_input, context_from_docs)

    print("\n🧠 Generated Query Code:")
    print(pychembl_query_code)

    #step 3: Execute the interpreted ChEMBL query
    namespace = {}
    try:
        exec(pychembl_query_code, globals(), namespace)
        chembl_df = namespace.get("filtered_df", None)

        if chembl_df is not None:
            print("\n💊 Results:")
            print(len(chembl_df))
            print(chembl_df.head())
            chembl_df.to_csv("chembl_df.csv")
        else:
            print("⚠️ No filtered_df was produced by the generated code.")

    except Exception as e:
        print(f"❌ Error executing generated query code: {e}")

    # if "error" in results:
    #     print("❌ Error:", results["error"])
    # else:
    #     # Automatically detect the key that contains the result list
    #     for key in ["activities", "molecules", "targets", "assays", "compounds"]:
    #         if key in results:
    #             records = results[key]
    #             df = pd.DataFrame(records)
    #             df.to_csv("egfr_results.csv",index=False)
    #             print(len(df))
    #             print(df.head())
    #             break
    #     else:
    #         print("⚠️ No recognized result list found in response.")

if __name__ == "__main__":
    main()

##example Find all inhibitors for erbB1 with IC50 < 100 nM