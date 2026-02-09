from typing import Dict, Any

from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def build_chembl_query_from_rag(user_query: str, context: str) -> str:


    prompt = f"""
    You are a Python programmer who knows the ChEMBL API.
    You have the following relevant documentation and examples from notebooks:

    {context}

    Your Task: 
    Convert the following user query into a structured Python dictionary (query plan):

    User Query: {user_query}

    Requirements:
    - Use the ChEMBL API (or PyChEMBL) to retrieve data
    - Do NOT return executable Python code.
    - Return ONLY valid JSON.
    - Do NOT include markdown, code fences, backticks, or explanations.
    - Only output the raw JSON dictionary.
    - If the user query matches multiple targets in ChEMBL (e.g., "erbB2"), do not narrow to a single target.
    - Instead, just output the general target name as given in the query.
    - The program will fetch all candidate targets and let the user choose.
    
    Example output:
    {{
    "entity": "activity",
    "target_name": "EGFR",
    "filters": {{
        "standard_type": "IC50",
        "standard_units": "nM",
        "operator": "<",
        "value": 100
    }}
    }}
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response.choices[0].message.content.strip()

    try:
        query_plan = json.loads(raw_response)
    except json.JSONDecodeError:
        # fallback if LLM did not produce perfect JSON
        query_plan = {"error": "Failed to parse JSON from LLM response", "raw": raw_response}

    return query_plan

