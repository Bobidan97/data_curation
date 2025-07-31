from chembl_webresource_client.new_client import new_client
import pandas as pd
import ast

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


def get_molecules_activity_with_filters(filter_dict, target_only=False):
    """
    Fetches activity data from ChEMBL using a filter dictionary.

    Args:
        filter_dict (dict): A dictionary of filters including at least 'target_pref_name'.
        target_only (bool): If True, only returns the target DataFrame (for debugging).

    Returns:
        pd.DataFrame or dict: Filtered activity data or error dictionary.
    """

    try:
        #extract target name
        target_name = filter_dict.pop("target_pref_name", None)
        if not target_name:
            return {"error": "Missing required filter key: 'target_pref_name'"}

        #get matching target(s)
        target_query = new_client.target.filter(pref_name__icontains=target_name)
        targets = pd.DataFrame.from_dict(target_query)

        if targets.empty:
            return {"error": f"No targets found for '{target_name}'"}

        #use first matching target (for now)
        selected_target_id = targets.target_chembl_id.iloc[0]

        if target_only:
            return targets

        #build bioactivity query
        filter_dict["target_chembl_id"] = selected_target_id
        activity_query = new_client.activity.filter(**filter_dict)
        activity_df = pd.DataFrame.from_dict(activity_query)

        if activity_df.empty:
            return {"error": f"No activity data found for '{target_name}' with given filters"}

        #clean and return relevant columns
        filtered_df = activity_df.dropna(subset=['canonical_smiles', 'standard_value'])
        filtered_df = filtered_df.drop_duplicates(subset=['canonical_smiles'])

        return filtered_df.reset_index(drop=True)

    except Exception as e:
        return {"error": f"Exception occurred while querying ChEMBL: {str(e)}"}

import requests

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

def execute_chembl_query(query_plan: dict) -> dict | list | None:
    # Determine the resource type, e.g., "activity", "molecule", "target"
    resource = query_plan.pop("resource", "activity")

    # Construct full URL
    endpoint = f"{BASE_URL}/{resource}.json"

    # Send GET request with query_plan as parameters
    try:
        response = requests.get(endpoint, params=query_plan)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}