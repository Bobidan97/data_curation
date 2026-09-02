"""
Rewrite chembl_examples.ipynb:
  1. Add numbered top-level sections matching the newer reference notebook.
  2. Fix markdown descriptions that don't match their accompanying code cells.
  3. Remove the empty code cell (cell 83).
  4. Keep all code cells unchanged.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "documents" / "chembl_examples.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ---------------------------------------------------------------------------
# Full replacement map: cell index -> new markdown source.
# Only markdown cells are listed. All code cells pass through unchanged.
# The empty code cell at index 83 is dropped.
# ---------------------------------------------------------------------------
MARKDOWN_REWRITES = {
    0: (
        "# ChEMBL Webresource Client - API Reference\n\n"
        "The `chembl_webresource_client` library provides Python access to ChEMBL data "
        "without needing SQL or REST knowledge. Results are cached locally so repeated "
        "calls are fast.\n\n"
        "**Sections:**\n"
        "1. [Compounds](#1-compounds)\n"
        "2. [Drugs](#2-drugs)\n"
        "3. [Targets](#3-targets)\n"
        "4. [Activities](#4-activities)\n"
        "5. [Assays](#5-assays)\n"
        "6. [Tissues](#6-tissues)\n"
        "7. [Cell Lines](#7-cell-lines)\n"
        "8. [References / Documents](#8-references--documents)\n"
        "9. [Sources](#9-sources)\n"
        "10. [Utils](#10-utils)\n"
        "11. [Protein Classification](#11-protein-classification)\n"
        "12. [Substructure Search](#12-substructure-search)\n"
        "13. [ATC Classification](#13-atc-classification)\n"
        "14. [Metabolism](#14-metabolism)\n"
        "15. [Target Component](#15-target-component)\n"
        "16. [Mechanism of Action](#16-mechanism-of-action)"
    ),
    1: (
        "## Available data entities\n\n"
        "List all available API resources:"
    ),
    3: (
        "## Available filters\n\n"
        "Filters follow Django QuerySet syntax with double-underscore lookups:\n\n"
        "| Filter | Meaning | Example |\n"
        "|---|---|---|\n"
        "| `exact` | exact match (default) | `max_phase=4` |\n"
        "| `iexact` | case-insensitive exact | `pref_name__iexact='aspirin'` |\n"
        "| `contains` / `icontains` | substring | `description__icontains='kinase'` |\n"
        "| `in` | value in list | `molecule_chembl_id__in=['CHEMBL25','CHEMBL192']` |\n"
        "| `gt` / `gte` | greater (or equal) | `first_approval__gte=2000` |\n"
        "| `lt` / `lte` | less (or equal) | `mw_freebase__lte=300` |\n"
        "| `startswith` / `istartswith` | prefix | `pref_name__istartswith='blood'` |\n"
        "| `endswith` / `iendswith` | suffix | `pref_name__iendswith='nib'` |\n"
        "| `isnull` | null check | `pchembl_value__isnull=False` |\n"
        "| `iregex` | regex | `description__iregex='nephrotox\|renal toxicity'` |\n\n"
        "**Nested fields** use double underscores to traverse objects: "
        "`molecule_properties__mw_freebase__lte=300` accesses `mw_freebase` "
        "inside `molecule_properties`."
    ),
    4: (
        "## `only` operator\n\n"
        "`only(['field1', 'field2'])` limits the response to specified fields — "
        "reduces payload size. The argument must be a list. "
        "Specifying a nested field name (e.g. `molecule_properties__alogp`) "
        "returns the entire parent object (`molecule_properties`); "
        "you must unpack it in code."
    ),
    5: (
        "# 1. Compounds\n\n"
        "Compounds in ChEMBL have associated bioactivity data measured against targets. "
        "Use `new_client.molecule` for all compound lookups. "
        "Retrieve by name, synonym, ChEMBL ID, InChI key, SMILES connectivity, "
        "structural similarity, or molecular property filters."
    ),
    6: (
        "## 1.1 Find a compound by preferred name\n\n"
        "Uses `pref_name__iexact` for a case-insensitive exact match on the preferred name field."
    ),
    8: (
        "## 1.2 Find a compound by synonym\n\n"
        "When a compound is better known by a trade name or synonym than its `pref_name`, "
        "filter on the nested `molecule_synonyms__molecule_synonym` field. "
        "The first `__` traverses into the `molecule_synonyms` list; "
        "the second `__` accesses the `molecule_synonym` string within it."
    ),
    10: (
        "## 1.3 Get a compound by ChEMBL ID\n\n"
        "All ChEMBL entities have a stable ChEMBL ID. "
        "Filter with `chembl_id=` (or equivalently `molecule_chembl_id=`) "
        "and use `only` to select specific fields."
    ),
    12: (
        "## 1.4 Get multiple compounds by ChEMBL ID list\n\n"
        "Use `molecule_chembl_id__in=[...]` to retrieve a batch of compounds in one call."
    ),
    14: (
        "## 1.5 Get a compound image (visualisation only)\n\n"
        "The `new_client.image` endpoint returns SVG/PNG images. "
        "This is for display purposes — it does not return structured data. "
        "Use `image.get('CHEMBL_ID')` to retrieve the image bytes."
    ),
    16: (
        "## 1.6 Find a compound by standard InChI key\n\n"
        "Filter on `molecule_structures__standard_inchi_key` to look up a compound "
        "by its InChI key. Returns the compound record including structure fields."
    ),
    18: (
        "## 1.7 Similarity search by SMILES\n\n"
        "Use `new_client.similarity` with `smiles=` and `similarity=` (integer 0–100, "
        "representing percentage) to find structurally similar compounds."
    ),
    20: (
        "## 1.8 Similarity search by ChEMBL ID\n\n"
        "Alternative to SMILES: pass `chembl_id=` to search for compounds similar "
        "to a known molecule. Returns `molecule_chembl_id`, `pref_name`, and `similarity` score."
    ),
    22: (
        "## 1.9 Find compounds with the same connectivity (SMILES)\n\n"
        "Use `molecule_structures__canonical_smiles__connectivity=` to find all compounds "
        "sharing the same core connectivity as the given SMILES, regardless of stereochemistry "
        "or salt/solvate variations."
    ),
    24: (
        "## 1.10 Get all approved drugs\n\n"
        "`max_phase=4` returns approved/marketed drugs. "
        "Use `order_by('molecule_properties__mw_freebase')` to sort by molecular weight ascending."
    ),
    26: (
        "## 1.11 Get drugs indicated for a disease\n\n"
        "Two-step pattern: query `drug_indication` filtered by `efo_term__icontains` "
        "to get matching molecule IDs, then retrieve those molecules. "
        "The `drug_indication` endpoint uses EFO (Experimental Factor Ontology) disease terms."
    ),
    28: (
        "## 1.12 Filter drugs by approval year and name stem using the `drug` endpoint\n\n"
        "The `drug` endpoint (distinct from `molecule`) contains curated drug-specific fields: "
        "`first_approval` (year), `usan_stem` / `usan_stem_definition`, `development_phase`. "
        "Multiple `.filter()` calls can be chained — each narrows the result set further."
    ),
    30: (
        "## 1.13 Get all biotherapeutic compounds\n\n"
        "Filter `biotherapeutic__isnull=False` to find compounds that have biotherapeutic "
        "data (biologics, antibodies, peptides, etc.)."
    ),
    32: (
        "## 1.14 Filter compounds by molecular weight\n\n"
        "`molecule_properties__mw_freebase__lte=300` filters on the molecular weight "
        "of the free base form (nested inside `molecule_properties`)."
    ),
    34: (
        "## 1.15 Combined molecular weight and name filters\n\n"
        "Multiple filter conditions in a single `.filter()` call act as AND. "
        "Here: MW ≤ 300 AND pref_name ends with 'nib' (kinase inhibitor suffix)."
    ),
    36: (
        "## 1.16 Filter by Lipinski Rule-of-Five violations\n\n"
        "`molecule_properties__num_ro5_violations=0` returns drug-like compounds with "
        "no Lipinski violations."
    ),
    38: (
        "# 4. Activities\n\n"
        "The `activity` endpoint provides bioactivity measurements. Key fields:\n\n"
        "- `standard_type` — measurement type: `IC50`, `Ki`, `EC50`, `Kd`, `potency`, etc.\n"
        "- `standard_value` / `standard_units` — raw value and units (often nM)\n"
        "- `pchembl_value` — standardised −log₁₀ value in molar; ≥5 (≤10 µM) is a typical hit threshold\n"
        "- `assay_type` — `B` (binding), `F` (functional), `A` (ADMET), `T` (toxicity)\n"
        "- `target_chembl_id` — the target being assayed\n"
        "- `molecule_chembl_id` — the compound tested\n\n"
        "**Note:** `target_chembl_id` only accepts exact ChEMBL IDs — "
        "it does not support `__icontains`. To filter by target name, "
        "first resolve the target ID using `new_client.target`."
    ),
    39: (
        "## 4.1 Get all IC50 activities for a named target (two-step)\n\n"
        "Resolve the target ChEMBL ID by name first (`target.filter(pref_name__iexact=...)`), "
        "then filter activities by that ID."
    ),
    41: (
        "## 4.2 Get binding activities for a target by ChEMBL ID\n\n"
        "`assay_type='B'` restricts to binding assays. "
        "Other assay types: `F` (functional), `A` (ADMET), `T` (toxicity), `P` (physicochemical)."
    ),
    43: (
        "## 4.3 Get all activities with a pChEMBL value for a compound\n\n"
        "`pchembl_value__isnull=False` filters to standardised activity records only "
        "(excludes raw qualitative or unit-inconsistent measurements)."
    ),
    45: (
        "# 5. Assays\n\n"
        "## 5.1 Find ADMET assays by description keyword\n\n"
        "`assay_type='A'` restricts to ADMET assays. "
        "`description__icontains='inhibit'` filters on the assay free-text description. "
        "Add `assay_organism=` to further restrict by species (e.g. `'Rattus norvegicus'`)."
    ),
    47: (
        "# 6. Tissues\n\n"
        "Tissues can be looked up by ontology ID (Uberon, BTO, Caloha, EFO) or by `pref_name`. "
        "Tissue records are linked to assays performed in that tissue."
    ),
    48: "## 6.1 Get tissue by BTO ID (Brenda Tissue Ontology)",
    50: "## 6.2 Get tissue by Caloha ID",
    52: "## 6.3 Get tissue by Uberon ID (cross-species anatomy ontology)",
    54: "## 6.4 Get tissue by name prefix\n\n`pref_name__istartswith=` for a case-insensitive prefix match.",
    56: (
        "# 7. Cell Lines\n\n"
        "Cell lines can be looked up by Cellosaurus ID or by `cell_description`. "
        "Cell line records are linked to assays performed in that cell line."
    ),
    57: (
        "## 7.1 Get a cell line by Cellosaurus ID\n\n"
        "Cellosaurus (https://www.cellosaurus.org) is the standard registry for cell lines."
    ),
    59: (
        "# 3. Targets\n\n"
        "Targets represent biological entities being assayed — most commonly single proteins, "
        "but also complexes, cell lines, and organisms. Key fields: "
        "`target_chembl_id`, `pref_name`, `target_type`, `organism`.\n\n"
        "Lookup patterns:\n"
        "- By gene name: `target.filter(target_synonym__icontains='EGFR')`\n"
        "- By name fragment: `target.filter(pref_name__icontains='kinase')`\n"
        "- By UniProt accession: `target.filter(target_components__accession='P00533')`\n"
        "- By protein family: use `new_client.protein_classification` to get class IDs first (see section 11)"
    ),
    60: (
        "## 3.1 Find a target by gene name\n\n"
        "`target_synonym__icontains` searches the target synonym list, "
        "which includes gene names, UniProt names, and aliases. "
        "Use `only` to return a subset of fields."
    ),
    62: (
        "# 8. References / Documents\n\n"
        "Document records map to scientific papers, patents, and datasets "
        "containing the bioactivity measurements in ChEMBL. "
        "Filter by `pubmed_id`, `year`, `doc_type` (`PUBLICATION`, `PATENT`, `DATASET`), or `journal`."
    ),
    63: (
        "## 8.1 Find documents by PubMed ID list\n\n"
        "Use `pubmed_id__in=(...)` to check which of a list of PubMed IDs exist in ChEMBL."
    ),
    65: (
        "## 8.2 Find datasets produced after a given year\n\n"
        "`doc_type='DATASET'` restricts to dataset records. "
        "`year__gte=` filters by publication year."
    ),
    67: (
        "# 9. Sources\n\n"
        "Source records describe the external databases and assay sources "
        "from which ChEMBL data was curated (e.g. BindingDB, PubChem BioAssay)."
    ),
    68: "## 9.1 Get all ChEMBL data sources\n\n`new_client.source` — no filter needed; returns all sources.",
    70: (
        "# 10. Utils\n\n"
        "Cheminformatics utilities for structure conversion, descriptor calculation, "
        "and standardisation. Uses `chembl_webresource_client.utils.utils` (not `new_client`)."
    ),
    71: "## 10.1 Convert SMILES to CTAB (molblock)\n\n`utils.smiles2ctab(smiles)` — required input for other utils functions.",
    73: "## 10.2 Compute Maximal Common Substructure (MCS)\n\n`utils.mcs(smiles_list)` — returns the MCS of a list of SMILES.",
    75: "## 10.3 Compute molecular descriptors from CTAB\n\n`utils.chemblDescriptors(ctab)` — returns a JSON string of descriptor values.",
    77: "## 10.4 Compute structural alerts\n\n`utils.structuralAlerts(ctab)` — flags known problematic substructures (PAINS, etc.).",
    79: "## 10.5 Standardise a molecule\n\n`utils.standardize(ctab)` — returns a standardised molblock (neutralise, remove salts, etc.).",
    81: "## 10.6 Get the parent molecule\n\n`utils.getParent(ctab)` — strips salts and returns the parent structure as a molblock.",
    84: (
        "# 11. Protein Classification\n\n"
        "Use `new_client.protein_classification` to retrieve the ChEMBL protein class hierarchy "
        "(up to 8 levels: `l1`..`l8`) and find targets belonging to a given family.\n\n"
        "**Important:** the resource name is `protein_classification`, NOT `protein_class`.\n\n"
        "Key fields: `protein_class_id`, `pref_name`, `protein_class_desc`, `l1`..`l6`.\n\n"
        "Typical two-step pattern to find all targets in a family:\n"
        "1. Get `protein_class_id` values for the family using `protein_classification.filter(l1__icontains=...)`\n"
        "2. Filter targets: `target.filter(target_components__protein_classifications__protein_class_id__in=[...])`"
    ),
    87: (
        "# 12. Substructure Search\n\n"
        "Use `new_client.substructure` to find all compounds containing a given SMILES substructure. "
        "Pass the substructure SMILES to the `smiles` filter. "
        "Note: these queries can be slow on the server for common fragments."
    ),
    90: (
        "# 13. ATC Classification\n\n"
        "The WHO ATC (Anatomical Therapeutic Chemical) classification system. "
        "Use `new_client.atc_class` to retrieve ATC codes and descriptions.\n\n"
        "Key fields: `level1`..`level5` (codes), `level1_description`..`level4_description`, `who_name`.\n\n"
        "You can also filter the `molecule` endpoint directly using `atc_classifications__level1=` "
        "(etc.) to get all approved drugs in a given ATC class."
    ),
    93: (
        "# 14. Metabolism\n\n"
        "Use `new_client.metabolism` to retrieve drug metabolism records.\n\n"
        "Key fields: `substrate_chembl_id` (the drug being metabolised), "
        "`metabolite_chembl_id`, `metabolite_name`, `enzyme_name`, `organism`, `target_chembl_id`.\n\n"
        "**Filter by `substrate_chembl_id`** (not `drug_chembl_id`) to get metabolites of a drug."
    ),
    96: (
        "# 15. Target Component\n\n"
        "Use `new_client.target_component` to retrieve protein sequence and annotation data "
        "for ChEMBL targets.\n\n"
        "Key fields: `accession` (UniProt ID), `component_type`, `description`, `organism`.\n\n"
        "Filter by `accession=` to find the ChEMBL target record associated with a UniProt protein."
    ),
    98: (
        "# 16. Mechanism of Action\n\n"
        "Use `new_client.mechanism` to retrieve curated drug mechanism of action records.\n\n"
        "Key fields: `action_type` (e.g. INHIBITOR, ANTAGONIST, AGONIST), "
        "`mechanism_of_action` (text description), `molecule_chembl_id`, "
        "`target_chembl_id`, `max_phase`.\n\n"
        "Filter by `target_chembl_id` (all drugs acting on a target) "
        "or `molecule_chembl_id` (mechanism for a specific drug)."
    ),
}

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

new_cells = []
for i, cell in enumerate(nb["cells"]):
    # Drop the empty code cell
    if i == 83 and cell["cell_type"] == "code" and not "".join(cell.get("source", [])).strip():
        continue
    # Rewrite markdown cells that have a replacement defined
    if cell["cell_type"] == "markdown" and i in MARKDOWN_REWRITES:
        cell = md(MARKDOWN_REWRITES[i])
    new_cells.append(cell)

nb["cells"] = new_cells
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Done. Notebook now has {len(nb['cells'])} cells.")
print(f"Rewrote {len(MARKDOWN_REWRITES)} markdown cells, removed 1 empty code cell.")
