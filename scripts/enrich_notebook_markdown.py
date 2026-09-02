"""
Replace or enrich markdown cells in documents/chembl_examples.ipynb
with the richer descriptions from ChEMBL_webresource_client_examples.ipynb.

Matching is done by section heading. Only markdown cells are touched;
code cells are left unchanged (avoids pulling in stale protein_class names).
"""
import json
from pathlib import Path

CURRENT = Path(__file__).parent.parent / "documents" / "chembl_examples.ipynb"
NEWER   = Path("C:/Users/Alex Bal/Downloads/ChEMBL_webresource_client_examples.ipynb")

# ---------------------------------------------------------------------------
# Richer markdown content taken verbatim from the newer notebook,
# with one manual correction: protein_class → protein_classification
# and removal of the table-of-contents cell (not useful in RAG context).
# ---------------------------------------------------------------------------
REPLACEMENTS = {
    # key = first line of the heading to match in the CURRENT notebook
    # value = full replacement markdown source

    "# ChEMBL webresource client examples": (
        "# ChEMBL webresource client examples\n\n"
        "The library helps to access ChEMBL data and cheminformatics tools from Python. "
        "You don't need to know how to write SQL. You don't need to know how to interact "
        "with REST APIs. You don't need to compile or install any cheminformatics frameworks. "
        "Results are cached locally so repeated calls are fast."
    ),

    "## Available filters": (
        "## Available filters\n\n"
        "The design of the client is based on Django QuerySet "
        "(https://docs.djangoproject.com/en/1.11/ref/models/querysets) "
        "and most important lookup types are supported:\n\n"
        "- `exact` - exact match\n"
        "- `iexact` - case-insensitive exact match\n"
        "- `contains` - substring match\n"
        "- `icontains` - case-insensitive substring match\n"
        "- `in` - value in a list (e.g. `molecule_chembl_id__in=['CHEMBL25', 'CHEMBL192']`)\n"
        "- `gt` / `gte` - greater than / greater than or equal to\n"
        "- `lt` / `lte` - less than / less than or equal to\n"
        "- `startswith` / `istartswith` - prefix match\n"
        "- `endswith` / `iendswith` - suffix match\n"
        "- `isnull` - filter for null/non-null values\n"
        "- `iregex` - case-insensitive regular expression match\n\n"
        "Double underscores traverse nested fields: "
        "`molecule_properties__mw_freebase__lte=300` filters the `mw_freebase` field "
        "inside the nested `molecule_properties` object."
    ),

    "## Only operator": (
        "## `only` operator\n\n"
        "`only` is a special method that limits the results to a selected set of database fields. "
        "`only` takes a **single argument**: a list of field names to include in the response.\n\n"
        "Note: the specified fields must exist in the API endpoint being queried. "
        "Using `only` with a nested field name (e.g. `molecule_properties__alogp`) "
        "returns the entire parent object (`molecule_properties`), not just the sub-field - "
        "you still need to unpack it in pandas afterward."
    ),

    "# Molecules": (
        "# Compounds\n\n"
        "Compounds in ChEMBL usually have associated bioactivity data. "
        "For example, the activity of a compound may have been measured in an experiment "
        "against a particular target, resulting in an IC50 value published in the scientific literature.\n\n"
        "Compound records may be retrieved in a number of ways: lookup by identifier "
        "(ChEMBL ID, InChI key, SMILES), by name or synonym, by similarity, "
        "by substructure, or by molecular property filters."
    ),

    "## Find a molecule by pref_name": (
        "## Find a compound by pref_name using the `molecule` endpoint\n\n"
        "Note the double underscore to filter for a case-insensitive exact match "
        "(`iexact`) within the `pref_name` database field."
    ),

    "## Find a molecule by its synonyms": (
        "## Find a compound by its synonyms\n\n"
        "- In some cases a compound may be more commonly known by a synonym than its "
        "preferred name (`pref_name`) in ChEMBL.\n"
        "- The `molecule_synonym` field is nested within `molecule_synonyms`, hence the "
        "first double underscore; the filter keyword follows after the second double underscore.\n"
        "- Use `only` to limit the returned fields."
    ),

    "## Get a single molecule by ChEMBL id": (
        "## Get a single compound by ChEMBL id\n\n"
        "All main entities in ChEMBL have a stable ChEMBL ID designed for "
        "straightforward lookup of data."
    ),

    "## Get many molecules by id": (
        "## Get many compounds using a list of ChEMBL IDs\n\n"
        "Use the double underscore followed by the `in` keyword to filter "
        "by a list of `molecule_chembl_id` values."
    ),

    "## Display a molecule image": (
        "## Display a compound image using the `image` endpoint"
    ),

    "## Get all approved drugs": (
        "## Get all approved drugs\n\n"
        "Filter by `max_phase=4` for approved/marketed drugs. "
        "Use `order_by` to sort results - e.g. by molecular weight."
    ),

    "## Get approved drugs for lung cancer": (
        "## Get Phase 3 clinical candidates for lung cancer, and examine their molecular properties\n\n"
        "- First use `drug_indication` filtering on `efo_term` and `max_phase_for_ind`.\n"
        "- Then feed the resulting molecule IDs into the `molecule` endpoint to get properties.\n"
        "- Unpack nested `molecule_properties` dict columns into individual DataFrame columns."
    ),

    "## Filter drugs by approval year and name": (
        "## Filter drugs by approval year and name using the `drug` endpoint\n\n"
        "The `drug` endpoint is separate from `molecule` and contains additional curated "
        "fields such as `first_approval`, `usan_stem`, and `usan_stem_definition`."
    ),

    "# Activities": (
        "# Activities\n\n"
        "The `activity` API endpoint provides bioactivity data for compounds that have "
        "been measured against target(s) in an assay. Key fields include:\n\n"
        "- `standard_type` - the measurement type (e.g. IC50, Ki, EC50, Kd)\n"
        "- `standard_value` / `standard_units` - the raw measured value and its units\n"
        "- `pchembl_value` - standardised −log₁₀(IC50/Ki/EC50/Kd) value in molar units; "
        "≥5 (~10 µM) is a typical minimum threshold for a hit\n"
        "- `assay_type` - B (binding), F (functional), A (ADMET), T (toxicity), P (physicochemical)\n"
        "- `target_chembl_id` - the ChEMBL target being assayed\n"
        "- `molecule_chembl_id` - the compound being tested"
    ),

    "## Get all IC50 activities related to the hERG target": (
        "## Get all IC50 activities related to the hERG target\n\n"
        "Two-step pattern: first resolve the target by name, then filter activities by its ID."
    ),

    "## Get all activities for a specific target with assay type B (binding):": (
        "## Get all activities for a specific target with assay type B (binding)\n\n"
        "Assay type `B` = binding assay. Other types: `F` (functional), `A` (ADMET)."
    ),

    "## Get all activities with a pChEMBL value for a molecule": (
        "## Get all activities with a pChEMBL value for a molecule\n\n"
        "`pchembl_value__isnull=False` filters to records where pChEMBL is populated - "
        "i.e. activities that have been standardised to a common scale."
    ),

    "## Search for ADMET-related inhibitor assays (type A)": (
        "## Search for ADMET-related inhibitor assays (assay_type A) measured in Rat\n\n"
        "Assay type `A` = ADMET (absorption, distribution, metabolism, excretion, toxicity). "
        "Filter on `assay_organism` to restrict to a specific species."
    ),

    "# Tissues": (
        "# Tissues\n\n"
        "Tissues can be looked up by ontology ID (Uberon, BTO, Caloha, EFO) or by name. "
        "Tissue records link to assays measured in that tissue."
    ),

    "# Cells": (
        "# Cell Lines\n\n"
        "Cell lines can be looked up by Cellosaurus ID or by description. "
        "Cell line records link to assays measured in that cell line."
    ),

    "## Get cell line by cellosaurus id": (
        "## Get cell line by Cellosaurus ID\n\n"
        "Cellosaurus (https://www.cellosaurus.org) is the standard registry for cell lines."
    ),

    "# Targets": (
        "# Targets\n\n"
        "Targets in ChEMBL represent biological entities against which compounds are assayed - "
        "most commonly single proteins, but also protein complexes, cell lines, organisms, and more.\n\n"
        "Key fields: `target_chembl_id`, `pref_name`, `target_type`, `organism`.\n\n"
        "Useful filter patterns:\n"
        "- By gene name: `target.filter(target_synonym__icontains='BRD4')`\n"
        "- By UniProt accession: `target.filter(target_components__accession='P00533')`\n"
        "- By name fragment: `target.filter(pref_name__icontains='kinase')`\n"
        "- By protein family: use `new_client.protein_classification` to get class IDs first"
    ),

    "# References": (
        "# References / Documents\n\n"
        "Document records correspond to scientific papers, patents, and datasets "
        "that contain the bioactivity measurements in ChEMBL. "
        "Filter by `pubmed_id`, `year`, `doc_type`, or `journal`."
    ),
}


def first_line(source) -> str:
    if isinstance(source, list):
        source = "".join(source)
    return source.strip().splitlines()[0].strip()


nb = json.loads(CURRENT.read_text(encoding="utf-8"))
replaced = 0

for cell in nb["cells"]:
    if cell["cell_type"] != "markdown":
        continue
    fl = first_line(cell["source"])
    if fl in REPLACEMENTS:
        cell["source"] = REPLACEMENTS[fl]
        replaced += 1

CURRENT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Replaced {replaced} markdown cells in {CURRENT.name}.")
print(f"Notebook now has {len(nb['cells'])} cells total.")
