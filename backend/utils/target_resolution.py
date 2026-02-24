import pandas as pd
from chembl_webresource_client.new_client import new_client


def resolve_target_candidates(query_target: str,) -> pd.DataFrame:
    target_client = new_client.target

    results = (target_client.filter(
        pref_name__icontains=query_target,)
    .only(
        "target_chembl_id",
        "pref_name",
        "organism",
        "target_type"
    ))


    potential_targets = pd.DataFrame(list(results))
    print(potential_targets)
    if potential_targets.empty:
        raise ValueError(f"No targets found for '{query_target}'")

    return potential_targets

def prompt_user_to_select_target(potential_targets: pd.DataFrame) -> str:
    print("\n🔬 Multiple targets found:\n")
    print(potential_targets.columns)
    for idx, row in potential_targets.iterrows():
        print(
            f"[{idx}] {row['pref_name']} | "
            f"{row['organism']} | "
            f"{row['target_type']} | "
            f"{row['target_chembl_id']}"
        )

    while True:
        try:
            choice = int(input("\nSelect target index: "))
            return potential_targets.loc[choice, "target_chembl_id"]
        except (ValueError, KeyError):
            print("❌ Invalid selection. Try again.")
