import sys
import pandas as pd
from chembl_webresource_client.new_client import new_client
from llm import build_chembl_query_from_rag
from rag_system import RAGSystem
from utils.target_resolution import (
    resolve_target_candidates,
    prompt_user_to_select_target
)
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
documents_path = Path(__file__).parent.parent / "documents"

def main():
    user_input = input("🔎 Enter your ChEMBL database query: ")

    #step 1 get documentation from notebook (RAG)
    rag = RAGSystem(directory_path=str(documents_path))
    rag.process_documents()
    rag_result = rag.query(user_input)
    context_from_docs = rag_result['content']

    print("\n📚 Retrieved ChEMBL API usage context (aggregated):")
    print(context_from_docs)

    print("\n📝 Retrieved Chunks with Metadata:")
    for i, doc in enumerate(rag_result['results']):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source Notebook: {doc.metadata.get('source')}")
        print(f"Heading: {doc.metadata.get('heading')}")
        print(f"Level: {doc.metadata.get('level')}")
        print(f"Cell Index Range: {doc.metadata.get('cell_index_range')}")
        print(f"Content:\n{doc.page_content[:500]}...")

    #step 2 LLM generates a ChEMBL API query plan using user input and notebook content
    query_plan = build_chembl_query_from_rag(user_input, context_from_docs)

    print("\n🧠 Generated Query Plan:")
    print(query_plan)

    #step 3 resolve target interactively
    target_name = query_plan.get("target_name")
    if not target_name:
        print("❌ No target specified in query plan")
        return

    candidates_df = resolve_target_candidates(target_name)
    if len(candidates_df) > 1:
        target_chembl_id = prompt_user_to_select_target(candidates_df)

    #step 4 execute ChEMBL query deterministically
    filters = query_plan.get("filters", {})
    standard_type = filters.get("standard_type")
    standard_units = filters.get("standard_units")

    #fetch raw activities
    activity_client = new_client.activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type=standard_type,
        standard_units=standard_units
    )

    df = pd.DataFrame(list(activity_client))
    #print(df)
    #step 5 apply numeric filtering if specified
    if "value" in filters and filters["value"] is not None:
        if "standard_value" in df.columns:
            df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
            operator = filters.get("operator")
            value = filters["value"]

            if operator == "<":
                df = df.loc[df["standard_value"] < value]
            elif operator == ">":
                df = df.loc[df["standard_value"] > value]

        print("💊 Number of results:", len(df))
        print(df.head())
        df.to_csv("chembl_df.csv", index=False)

if __name__ == "__main__":
    main()
##example Find all inhibitors for erbB2 with IC50 < 100 nM
#### Fetch bioactivity for inhibitors of erbB2 with IC50 < 100 nM in mice