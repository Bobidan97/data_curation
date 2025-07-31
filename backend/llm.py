from typing import Dict, Any

from openai import OpenAI
import os
import json
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def build_chembl_query_from_rag(user_input: str, context: str) -> dict[str, str] | Any:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    You are an assistant that converts natural language questions into ChEMBL API filter parameters.

    You must return a JSON dictionary with a resource key (e.g., 'activity', 'molecule', etc.) 
    and valid ChEMBL API filter parameters.”. 
    Do not wrap the output in markdown (no triple backticks).
    Do not include any explanation, labels, or comments.
    Only output raw JSON.

    Now process this query: "{user_input}"
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_response = response.choices[0].message.content.strip()
    print("\n📤 Raw LLM Response:")
    print(raw_response)
    try:
        filters = json.loads(raw_response)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}

    return filters

