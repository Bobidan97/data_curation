from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

def generate_chembl_code(user_query: str, context: str) -> str:

    prompt = f"""
    You are an expert Python programmer who specialises in the ChEMBL database API.
    You have the following relevant documentation and code examples retrieved from notebooks:
    
    {context}
    
    Your task:
    Write Python code that fulfils the following user query using the chembl_webresource_client library.
    
    User Query: {user_query}
    
    Requirements:
    - Use `new_client` from `chembl_webresource_client.new_client` — it is already imported and available.
    - `pd` (pandas) is already imported and available.
    - The variable `target_chembl_id` is already defined and contains the correct ChEMBL target ID — use it directly, do NOT look up the target yourself.
    - Store the final results in a pandas DataFrame named `df`.
    - Do NOT include any import statements.
    - Do NOT include any target lookup or target resolution code.
    - Return ONLY raw executable Python code — no markdown, no code fences, no backticks, no explanations.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    code = response.choices[0].message.content.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def extract_target_name(user_query: str) -> str:
    prompt = f"""Extract only the biological target name from the following query.
    Return only the target name as a short string (e.g. "erbB2", "EGFR", "BRAF").
    No explanation, no punctuation, no extra words.
    
    Query: {user_query}"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
