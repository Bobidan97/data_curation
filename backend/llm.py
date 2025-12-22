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

    Your task: Generate Python code to answer the following user query:

    User Query: {user_query}

    Requirements:
    - Use the ChEMBL API (or PyChEMBL) to retrieve data
    - Filter the data according to the user query
    - Assign the final result to a variable called 'filtered_df'
    - Include necessary imports at the top of your code
    - Do not include explanation text; output only valid Python code
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_code = response.choices[0].message.content.strip()

    if raw_code.startswith("```"):
        raw_code = "\n".join(raw_code.split("\n")[1:])
    if raw_code.endswith("```"):
        raw_code = "\n".join(raw_code.split("\n")[:-1])

    return raw_code.strip()

