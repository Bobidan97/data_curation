import openai
import os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def parse_query_with_llm(user_input):
    """Use an LLM to extract the target protein name from the user query."""

    client = openai.OpenAI(api_key=api_key)
    prompt = f"Extract the name of the target protein from this query: '{user_input}'. Only return the name."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

