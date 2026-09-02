import pandas as pd


def resolve_molecule_candidates(query_mol: str) -> pd.DataFrame:
    """Name/synonym-based resolution only.
    SMILES, InChI key, and ChEMBL ID queries bypass this entirely —
    the LLM handles them directly in the generated code.
    """
    from chembl_webresource_client.new_client import new_client  # lazy — avoids network call at import time
    mol_client = new_client.molecule
    fields = ["molecule_chembl_id", "pref_name", "molecule_type", "max_phase"]

    by_name = pd.DataFrame(list(
        mol_client.filter(pref_name__icontains=query_mol).only(*fields)
    ))
    by_synonym = pd.DataFrame(list(
        mol_client.filter(molecule_synonyms__molecule_synonym__icontains=query_mol).only(*fields)
    ))

    candidates = (
        pd.concat([by_name, by_synonym], ignore_index=True)
        .drop_duplicates(subset="molecule_chembl_id")
        .reset_index(drop=True)
    )

    if candidates.empty:
        raise ValueError(f"No molecules found for '{query_mol}'")

    return candidates


def prompt_user_to_select_molecule(candidates: pd.DataFrame) -> str:
    print("\n💊 Molecule(s) found matching query:\n")

    for idx, row in candidates.iterrows():
        phase = row.get("max_phase", "N/A")
        print(
            f"[{idx}] {row['pref_name']} | "
            f"{row['molecule_type']} | "
            f"Phase {phase} | "
            f"{row['molecule_chembl_id']}"
        )

    while True:
        try:
            choice = int(input("\nSelect molecule index to confirm choice: "))
            return candidates.loc[choice, "molecule_chembl_id"]
        except (ValueError, KeyError):
            print("❌ Invalid selection. Try again.")
