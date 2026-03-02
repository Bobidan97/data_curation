import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    run_rag,
    get_generated_code,
    get_target_candidates,
    fetch_chembl_data,
)
from utils.target_resolution import prompt_user_to_select_target
from curation import curate_dataframe


def main():
    user_input = input("🔎 Enter your ChEMBL database query: ")

    # step 1: retrieve relevant documentation via RAG
    context_from_docs = run_rag(user_input)

    # step 2: LLM generates Python code from user query + retrieved context
    generated_code = get_generated_code(user_input, context_from_docs)
    print(generated_code)

    # step 3: resolve target interactively
    candidates_df = get_target_candidates(user_input)
    print(candidates_df)
    target_chembl_id = prompt_user_to_select_target(candidates_df)

    # step 4: fetch raw data by executing the generated code
    df = fetch_chembl_data(generated_code, target_chembl_id)

    # step 5: curate the raw results into an ML-ready dataset
    df_curated = curate_dataframe(df)
    print(f"📊 Raw results: {len(df)} | After curation: {len(df_curated)}")
    print(df_curated.head())
    df_curated.to_csv("chembl_df.csv", index=False)


if __name__ == "__main__":
    main()
