from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

_ENTITY_INSTRUCTIONS = {
    "target": (
        "- The variable `target_chembl_id` is already defined and contains the resolved "
        "ChEMBL target ID — use it directly, do NOT look up the target yourself."
    ),
    "molecule": (
        "- The variable `molecule_chembl_id` is already defined and contains the resolved "
        "ChEMBL molecule ID — use it directly, do NOT look up the molecule yourself."
    ),
    "other": (
        "- No entity has been pre-resolved. Query ChEMBL data directly using `new_client`."
    ),
}


def generate_chembl_code(user_query: str, context: str, domain: str = "other") -> str:
    entity_instruction = _ENTITY_INSTRUCTIONS.get(domain, _ENTITY_INSTRUCTIONS["other"])

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
    {entity_instruction}
    - Store the final results in a pandas DataFrame named `df`.
    - Do NOT include any import statements.
    - Do NOT include any entity lookup or resolution code.
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


def classify_query(user_query: str) -> tuple[str, str | None]:
    """Returns (domain, entity_name).

    domain: 'target' | 'molecule' | 'other'
    entity_name: human-readable name to resolve (e.g. 'imatinib', 'EGFR'), or None
                 when the query uses an exact identifier (SMILES, InChI key, ChEMBL ID)
                 or has no specific entity.
    """
    prompt = f"""Classify the following ChEMBL query. Respond with exactly two plain lines — no labels, no explanations, no punctuation.

First line — the domain, one of: target, molecule, other
  - target:   query involves a specific biological protein/gene (e.g. EGFR, BRAF, erbB2)
  - molecule: any molecule/compound query — by name, synonym, SMILES, InChI key, ChEMBL ID,
              similarity, substructure, or molecular properties
  - other:    everything else — activities without a specific entity, tissues, cells,
              ATC classes, drug indications, references, sources, metabolism

Second line — the entity name to resolve, or NONE
  - Return the human-readable name if the query uses a name/synonym (e.g. "imatinib", "EGFR")
  - Return NONE if the query uses an exact identifier (SMILES, InChI key, ChEMBL ID) or has no specific entity

Query: {user_query}"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    lines = response.choices[0].message.content.strip().splitlines()

    # Strip any "Line N:" / "1." / label prefix the model may echo back
    def _clean(s: str) -> str:
        return re.sub(r'^[\w\s]*?:\s*', '', s, count=1).strip()

    raw_domain = _clean(lines[0]).lower() if lines else "other"
    # Validate domain — fall back to "other" if the model returned something unexpected
    domain = raw_domain if raw_domain in {"target", "molecule", "other"} else "other"

    raw_entity = _clean(lines[1]) if len(lines) > 1 else "NONE"
    entity_name = None if raw_entity.upper() == "NONE" else raw_entity
    return domain, entity_name
