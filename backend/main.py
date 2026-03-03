import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    run_rag,
    classify_query,
    get_generated_code,
    get_entity_candidates,
    fetch_chembl_data,
)
from utils.target_resolution import prompt_user_to_select_target
from utils.molecule_resolution import prompt_user_to_select_molecule
from curation import curate_dataframe


def main():
    user_input = input("🔎 Enter your ChEMBL database query: ")

    # step 1: classify domain and extract entity name (if any)
    domain, entity_name = classify_query(user_input)
    print(f"📂 Domain: {domain} | Entity: {entity_name or 'none'}")

    # step 2: retrieve relevant documentation via RAG
    context_from_docs = run_rag(user_input)

    # step 3: LLM generates Python code from user query + retrieved context + domain
    generated_code = get_generated_code(user_input, context_from_docs, domain)
    print(generated_code)

    # step 4: resolve entity interactively (only when a name needs disambiguation)
    entity_chembl_id = None
    if entity_name is not None:
        candidates_df = get_entity_candidates(domain, entity_name)
        print(candidates_df)
        if domain == "target":
            entity_chembl_id = prompt_user_to_select_target(candidates_df)
        else:
            entity_chembl_id = prompt_user_to_select_molecule(candidates_df)

    # step 5: fetch raw data by executing the generated code
    df = fetch_chembl_data(generated_code, domain, entity_chembl_id)

    # step 6: curate the raw results into an ML-ready dataset
    df_curated = curate_dataframe(df)
    print(f"📊 Raw results: {len(df)} | After curation: {len(df_curated)}")
    print(df_curated.head())
    df_curated.to_csv("chembl_df.csv", index=False)


if __name__ == "__main__":
    main()
