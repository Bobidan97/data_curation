from typing import Dict, Any

from openai import OpenAI
import os
import json
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# def build_chembl_query_from_rag(user_input: str, context: str) -> dict[str, str] | Any:
#
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
#     prompt = f"""
#     You are an assistant that converts natural language questions into ChEMBL API filter parameters.
#
#     You must return a JSON dictionary with a resource key (e.g., 'activity', 'molecule', etc.)
#     and valid ChEMBL API filter parameters.”.
#     Do not wrap the output in markdown (no triple backticks).
#     Do not include any explanation, labels, or comments.
#     Only output raw JSON.
#
#     Now process this query: "{user_input}"
#     """
#
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}]
#     )
#
#     raw_response = response.choices[0].message.content.strip()
#     print("\n📤 Raw LLM Response:")
#     print(raw_response)
#     try:
#         filters = json.loads(raw_response)
#     except json.JSONDecodeError as e:
#         return {"error": f"Failed to parse JSON: {str(e)}"}
#
#     return filters

def build_chembl_query_from_rag(user_input: str, context: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    You are an assistant that converts natural language questions into valid Python code
    using the `chembl_webresource_client` library. 

    Rules:
    - Import: `from chembl_webresource_client.new_client import new_client`
    - Always import pandas as pd
    - Use `new_client.activity.filter(...)`, `new_client.molecule.filter(...)`, `new_client.target.filter(...)`, etc.
    - Always convert API results to a pandas DataFrame with `pd.DataFrame.from_dict(results)`.
    - Always check `if "standard_value" in df.columns:` before converting.
    - Cast `standard_value` with `pd.to_numeric(df["standard_value"], errors="coerce")`.
    - If the user specifies thresholds like "less than", "greater than", or a range:
      * Filter using `df.loc[(...conditions...)]`
      * Always check `standard_units` against the requested unit (e.g. "nM").
    - Use latin names when user query refers to an organism.
    - DO NOT include markdown formatting, triple backticks, or ```python fences in your output.
    - Return ONLY raw Python code (one complete block).
    - The final filtered DataFrame must be named `filtered_df`.
    - Note that below is just an example and the user query could be more complicated (i.e. more filters) or less
      complicated (i.e. a general query)
    Example:
    User: "Find all IC50 activities for erbb1 with IC50 < 100 nM"
    Output:
    results = new_client.activity.filter(target_chembl_id="CHEMBL203", standard_type="IC50")
    df = pd.DataFrame.from_dict(results)
    if "standard_value" in df.columns:
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    filtered_df = df.loc[(df['standard_units'] == "nM") & (df['standard_value'] < 100) & (df['standard_relation'] == "<")]
    
    Now process this query: "{user_input}"
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response.choices[0].message.content.strip()
    return raw_response

