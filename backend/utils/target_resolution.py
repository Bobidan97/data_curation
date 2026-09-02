import pandas as pd


def resolve_target_candidates(query_target: str,) -> pd.DataFrame:
    from chembl_webresource_client.new_client import new_client  # lazy — avoids network call at import time
    target_client = new_client.target
    fields = ["target_chembl_id", "pref_name", "organism", "target_type"]

    by_name = pd.DataFrame(list(
        target_client.filter(pref_name__icontains=query_target).only(*fields)
    ))
    by_search = pd.DataFrame(list(
        target_client.search(query_target)
    ))[fields]

    potential_targets = (
        pd.concat([by_name, by_search], ignore_index=True)
        .drop_duplicates(subset="target_chembl_id")
        .reset_index(drop=True)
    )

    if potential_targets.empty:
        raise ValueError(f"No targets found for '{query_target}'")

    return potential_targets

def prompt_user_to_select_target(potential_targets: pd.DataFrame) -> str:
    print("\n🔬 Target(s) found matching query:\n")

    for idx, row in potential_targets.iterrows():
        print(
            f"[{idx}] {row['pref_name']} | "
            f"{row['organism']} | "
            f"{row['target_type']} | "
            f"{row['target_chembl_id']}"
        )

    while True:
        try:
            choice = int(input("\nSelect target index to confirm choice: "))
            return potential_targets.loc[choice, "target_chembl_id"]
        except (ValueError, KeyError):
            print("❌ Invalid selection. Try again.")
