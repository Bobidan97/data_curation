from chembl_webresource_client.new_client import new_client
import pandas as pd


def get_molecules_activity(target_protein, select=0):
    """Fetch IC50 bioactivity data for a given target protein."""
    target = new_client.target
    target_query = target.filter(pref_name__icontains=target_protein)
    targets = pd.DataFrame.from_dict(target_query)

    if targets.empty:
        return {"error": f"No targets found for {target_protein}"}

    if select >= len(targets):
        return {"error": f"Invalid selection index {select}. Only {len(targets)} targets found."}

    selected_target = targets.target_chembl_id.iloc[select]

    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type='IC50')
    actv_df = pd.DataFrame.from_dict(res)

    if actv_df.empty:
        return {"error": f"No IC50 data found for {target_protein}"}

    actv_df = actv_df.dropna(subset=['canonical_smiles', 'standard_value']).drop_duplicates(['canonical_smiles'])

    return actv_df[['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_units', 'standard_value']]