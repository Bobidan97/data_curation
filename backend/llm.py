from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

_ENTITY_INSTRUCTIONS = {
    "target": (
        "- A variable named `target_chembl_id` is ALREADY DEFINED in the runtime "
        "and holds the real, resolved ChEMBL target ID (e.g. 'CHEMBL203'). "
        "USE IT DIRECTLY by referring to the bare identifier `target_chembl_id` "
        "in your filter calls — for example: `target.filter(target_chembl_id=target_chembl_id)`.\n"
        "- DO NOT reassign `target_chembl_id` to anything (no `target_chembl_id = '...'`).\n"
        "- DO NOT use placeholder strings like 'CHEMBLXXX', 'CHEMBL_ID', or '<id>'.\n"
        "- DO NOT look up the target by name yourself — the resolution has already happened."
    ),
    "molecule": (
        "- A variable named `molecule_chembl_id` is ALREADY DEFINED in the runtime "
        "and holds the real, resolved ChEMBL molecule ID (e.g. 'CHEMBL25'). "
        "USE IT DIRECTLY by referring to the bare identifier `molecule_chembl_id` "
        "in your filter calls — for example: `activity.filter(molecule_chembl_id=molecule_chembl_id)`.\n"
        "- DO NOT reassign `molecule_chembl_id` to anything (no `molecule_chembl_id = '...'`).\n"
        "- DO NOT use placeholder strings like 'CHEMBLXXX', 'CHEMBL_ID', or '<id>'.\n"
        "- DO NOT look up the molecule by name yourself — the resolution has already happened."
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
        "- RDKit is also available (when installed): `Chem`, `Descriptors`, "
        "`rdMolDescriptors`, `AllChem`, and `DataStructs` are in scope — use them "
        "to compute molecular descriptors, fingerprints, or structural filters from "
        "the `canonical_smiles` column. Do NOT import RDKit.\n"
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
        "  df = df[~df['canonical_smiles'].isin(outlier_smiles)]\n\n"
        "RDKit descriptor examples (use `Chem`, `Descriptors`, `rdMolDescriptors` — "
        "already in scope, never import):\n"
        "  # `_calc(smi, func)` is ALREADY DEFINED in scope — do NOT redefine it.\n"
        "  # It safely parses a SMILES string and applies `func(mol)`, returning NaN\n"
        "  # on parse errors or exceptions. Use it directly in .apply() calls.\n\n"
        "  # Add QED (drug-likeness) column:\n"
        "  df['QED'] = df['canonical_smiles'].apply(lambda s: _calc(s, Descriptors.qed))\n\n"
        "  # Add fraction sp3 carbons:\n"
        "  df['FractionCSP3'] = df['canonical_smiles'].apply(\n"
        "      lambda s: _calc(s, rdMolDescriptors.CalcFractionCSP3))\n\n"
        "  # Add Bertz complexity + Labute ASA in one pass:\n"
        "  mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None\n"
        "          for s in df['canonical_smiles']]\n"
        "  df['BertzCT']   = [Descriptors.BertzCT(m)   if m else float('nan') for m in mols]\n"
        "  df['LabuteASA'] = [Descriptors.LabuteASA(m) if m else float('nan') for m in mols]\n\n"
        "  # Add molecular formula (string) and net formal charge:\n"
        "  df['MolFormula']   = df['canonical_smiles'].apply(\n"
        "      lambda s: _calc(s, rdMolDescriptors.CalcMolFormula))\n"
        "  df['FormalCharge'] = df['canonical_smiles'].apply(\n"
        "      lambda s: _calc(s, Chem.GetFormalCharge))\n\n"
        "  # Any of the 200+ names in Descriptors.descList works — e.g. 'NumAromaticRings',\n"
        "  # 'MolMR', 'Chi0v', 'SlogP_VSA1', 'fr_halogen', 'MaxPartialCharge':\n"
        "  func = dict(Descriptors.descList)['MaxPartialCharge']\n"
        "  df['MaxPartialCharge'] = df['canonical_smiles'].apply(lambda s: _calc(s, func))\n"
    )
    user_message = f"DataFrame schema:\n{df_schema}\n\nInstruction: {user_instruction}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
    )
    code = response.choices[0].message.content.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def generate_dataframe_query_code(user_question: str, df_schema: str) -> str:
    """Convert a natural-language *question* about a DataFrame into read-only pandas code.

    Unlike generate_dataframe_edit_code, this must NOT modify any DataFrame. The
    generated code computes a result and assigns it to a variable named
    ``answer`` (a scalar, string, Series, or small DataFrame). Returns raw
    executable Python — no markdown fences, no explanations.
    """
    system_prompt = (
        "You are a data analyst answering questions about a pandas DataFrame for "
        "a cheminformatics researcher.\n\n"
        "Rules:\n"
        "- `df` is the working DataFrame (described in the schema below). Treat it "
        "as READ-ONLY — never modify, filter-in-place, or reassign `df`.\n"
        "- Other DataFrames (raw_df, dup_rows, df_curated, df_with_descriptors) may "
        "be present as read-only references.\n"
        "- `pd` (pandas) and `np` (numpy) are available. RDKit (`Chem`, "
        "`Descriptors`, `rdMolDescriptors`) is available for structural questions; "
        "the safe per-molecule helper `_calc(smiles, func)` is in scope.\n"
        "- Compute the answer and assign it to a variable named `answer`.\n"
        "- `answer` may be a number, string, pandas Series, or a small DataFrame "
        "(e.g. a value_counts result or a groupby summary). Prefer a Series/DataFrame "
        "when the question implies a breakdown.\n"
        "- Do NOT print. Do NOT include imports. Return ONLY raw executable Python.\n\n"
        "Examples:\n"
        "  # 'what is the median pIC50?'\n"
        "  answer = df['pIC50'].median()\n\n"
        "  # 'how many unique targets?'\n"
        "  answer = df['target_chembl_id'].nunique()\n\n"
        "  # 'breakdown of assay types'\n"
        "  answer = df['assay_type'].value_counts()\n\n"
        "  # 'average pIC50 per standard_type, sorted'\n"
        "  answer = df.groupby('standard_type')['pIC50'].mean().sort_values(ascending=False)\n\n"
        "  # 'how many compounds have MW above 500?'\n"
        "  answer = int((df['canonical_smiles'].apply(lambda s: _calc(s, Descriptors.MolWt)) > 500).sum())\n"
    )
    user_message = f"DataFrame schema:\n{df_schema}\n\nQuestion: {user_question}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
    )
    code = response.choices[0].message.content.strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def generate_plot_code(user_request: str, df_schema: str) -> str:
    """Convert a natural-language chart request into Plotly code producing ``fig``.

    The generated code must build a Plotly figure and assign it to a variable
    named ``fig`` (read-only w.r.t. the data). Returns raw executable Python —
    no markdown fences, no explanations.
    """
    system_prompt = (
        "You are a data-visualisation assistant for a cheminformatics researcher. "
        "You write Plotly code to chart a pandas DataFrame.\n\n"
        "Rules:\n"
        "- `df` is the working DataFrame (schema below); treat it as READ-ONLY.\n"
        "- `px` (plotly.express) and `go` (plotly.graph_objects) are imported. "
        "`pd` and `np` are available.\n"
        "- Build exactly one figure and assign it to a variable named `fig`.\n"
        "- Choose a sensible chart type for the request (histogram, box, scatter, "
        "bar, violin, etc.). Add a clear title and axis labels.\n"
        "- Only reference columns that exist in the schema. If the request needs a "
        "derived quantity, compute it inline from existing columns.\n"
        "- Do NOT call fig.show(), do NOT print, do NOT import. Return ONLY raw "
        "executable Python.\n\n"
        "Examples:\n"
        "  # 'histogram of pIC50'\n"
        "  fig = px.histogram(df, x='pIC50', nbins=30, title='pIC50 distribution')\n\n"
        "  # 'pIC50 by assay type as a box plot'\n"
        "  fig = px.box(df, x='assay_type', y='pIC50', title='pIC50 by assay type')\n\n"
        "  # 'scatter of MW vs pIC50' (MW may need computing, but if a column exists use it)\n"
        "  fig = px.scatter(df, x='MW', y='pIC50', opacity=0.6, title='MW vs pIC50')\n"
    )
    user_message = f"DataFrame schema:\n{df_schema}\n\nChart request: {user_request}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
