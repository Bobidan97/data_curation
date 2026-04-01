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
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    code = response.choices[0].message.content.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def generate_dataframe_edit_code(user_instruction: str, df_schema: str) -> str:
    """Convert a natural-language DataFrame editing instruction to pandas code.

    The generated code operates on a variable named `df` (already in scope)
    and must assign the result back to `df`.  Returns raw executable Python —
    no markdown fences, no explanations.
    """
    system_prompt = (
        "You are an expert Python/pandas programmer helping a cheminformatics researcher "
        "manually curate a dataset.\n\n"
        "Rules:\n"
        "- Multiple DataFrames may be in scope (described in the schema below).\n"
        "- `df` is the MAIN working DataFrame — always write your final result back to `df`.\n"
        "- Other DataFrames (e.g. raw_df, dup_rows, df_curated) are READ-ONLY references "
        "you may use for lookups, joins, or computing filters.\n"
        "- `pd` (pandas) and `np` (numpy) are already available.\n"
        "- Do NOT include import statements or print statements.\n"
        "- Return ONLY raw executable Python — no markdown fences, no explanations.\n\n"
        "DataFrame glossary (when present in scope):\n"
        "- dup_rows: one row per raw activity measurement for every molecule that belongs "
        "to a structural duplicate group. Key columns:\n"
        "    _group           — group label, e.g. 'Group 1', 'Group 2'\n"
        "    _canonical       — canonical SMILES of the group representative; "
        "joins to df['canonical_smiles']\n"
        "    standard_value   — numeric activity value (already coerced to float)\n"
        "    molecule_chembl_id — ChEMBL ID of each molecule in the group; "
        "joins to df['molecule_chembl_id']\n"
        "  Whenever the user mentions 'duplicate groups', use dup_rows to compute "
        "group-level statistics (groupby '_group' or '_canonical'), then remove "
        "matching molecules from df via df['canonical_smiles'] or "
        "df['molecule_chembl_id'].\n"
        "  IMPORTANT: df contains ONE row per canonical SMILES (the representative "
        "kept after structural deduplication). dup_rows contains ALL raw measurements "
        "(possibly from multiple molecule_chembl_ids mapping to the same structure). "
        "For outlier removal, apply the IQR mask DIRECTLY to dup_rows — do NOT look "
        "up values via a join to df. Add the canonical SMILES to an outlier set if ANY "
        "measurement in that group is an outlier, then filter df by canonical_smiles.\n\n"
        "Examples:\n"
        "  # Simple value filter:\n"
        "  df = df[df['pIC50'] >= 5]\n\n"
        "  # Drop a column:\n"
        "  df = df.drop(columns=['assay_type'])\n\n"
        "  # Keep only IC50 rows:\n"
        "  df = df[df['standard_type'] == 'IC50'].reset_index(drop=True)\n\n"
        "  # Remove duplicate groups whose median standard_value > 100:\n"
        "  grp_med = dup_rows.groupby('_canonical')['standard_value'].median()\n"
        "  bad_smiles = grp_med[grp_med > 100].index\n"
        "  df = df[~df['canonical_smiles'].isin(bad_smiles)]\n\n"
        "  # Remove a specific duplicate group by name:\n"
        "  bad_ids = dup_rows.loc[dup_rows['_group'] == 'Group 3', 'molecule_chembl_id']\n"
        "  df = df[~df['molecule_chembl_id'].isin(bad_ids)]\n\n"
        "  # Keep only the duplicate group with the lowest median activity:\n"
        "  grp_med = dup_rows.groupby('_canonical')['standard_value'].median()\n"
        "  keep_smiles = [grp_med.idxmin()]\n"
        "  df = df[df['canonical_smiles'].isin(keep_smiles)]\n\n"
        "  # Remove molecules whose duplicate group contains any outlier measurement\n"
        "  # (outside 1.5*IQR whisker bounds, as shown in the box plot):\n"
        "  outlier_smiles = set()\n"
        "  for canonical, grp in dup_rows.groupby('_canonical'):\n"
        "      vals = grp['standard_value'].dropna()\n"
        "      if len(vals) < 2:\n"
        "          continue\n"
        "      q1, q3 = vals.quantile(0.25), vals.quantile(0.75)\n"
        "      iqr = q3 - q1\n"
        "      lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr\n"
        "      if ((grp['standard_value'] < lo) | (grp['standard_value'] > hi)).any():\n"
        "          outlier_smiles.add(canonical)\n"
        "  df = df[~df['canonical_smiles'].isin(outlier_smiles)]\n"
    )
    user_message = f"DataFrame schema:\n{df_schema}\n\nInstruction: {user_instruction}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
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
        model="gpt-5-mini",
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
