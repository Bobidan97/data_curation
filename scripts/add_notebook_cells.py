"""Append verified ChEMBL API examples for uncovered endpoints to chembl_examples.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "documents" / "chembl_examples.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


NEW_CELLS = [

    # ── protein_classification ────────────────────────────────────────────────
    md(
        "## Protein Classification (for family-based target lookups)\n\n"
        "Use `new_client.protein_classification` to find protein class IDs by family name "
        "(e.g. kinase, GPCR), then join to targets.\n"
        "Key fields: `l1`..`l6` (class hierarchy levels), `pref_name`, `protein_class_desc`, `protein_class_id`.\n\n"
        "**Important:** the resource is `protein_classification`, NOT `protein_class`."
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "protein_classification = new_client.protein_classification\n\n"
        "# Find all kinase classes (l2 is the second level of the hierarchy)\n"
        "kinase_classes = protein_classification.filter(l2__icontains='kinase')\n"
        "df = pd.DataFrame(kinase_classes)\n"
        "df"
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "protein_classification = new_client.protein_classification\n"
        "target = new_client.target\n\n"
        "# Step 1: get protein_class_id values for GPCRs\n"
        "gpcr_classes = protein_classification.filter(l1__icontains='gpcr').only(\n"
        "    ['protein_class_id', 'pref_name', 'protein_class_desc'])\n"
        "gpcr_class_ids = [str(x['protein_class_id']) for x in gpcr_classes]\n\n"
        "# Step 2: find human targets belonging to those classes\n"
        "gpcr_targets = target.filter(\n"
        "    target_components__protein_classifications__protein_class_id__in=gpcr_class_ids,\n"
        "    organism='Homo sapiens'\n"
        ").only(['target_chembl_id', 'pref_name', 'target_type', 'organism'])\n"
        "df = pd.DataFrame(gpcr_targets)\n"
        "df"
    ),

    # ── substructure ─────────────────────────────────────────────────────────
    md(
        "## Substructure Search\n\n"
        "Use `new_client.substructure` to find all compounds containing a given SMILES substructure.\n"
        "Pass the substructure SMILES to the `smiles` filter field."
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "substructure = new_client.substructure\n\n"
        "# Find all molecules containing a benzimidazole core\n"
        "res = substructure.filter(smiles='c1ccc2[nH]cnc2c1').only(\n"
        "    ['molecule_chembl_id', 'pref_name'])\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "substructure = new_client.substructure\n"
        "molecule = new_client.molecule\n\n"
        "# Find approved drugs containing a sulfonamide group\n"
        "res = substructure.filter(smiles='NS(=O)(=O)c1ccccc1').only(\n"
        "    ['molecule_chembl_id', 'pref_name', 'max_phase'])\n"
        "df = pd.DataFrame(res)\n"
        "df = df[df['max_phase'] == 4]  # approved only\n"
        "df"
    ),

    # ── atc_class ─────────────────────────────────────────────────────────────
    md(
        "## ATC Classification\n\n"
        "Use `new_client.atc_class` to retrieve WHO ATC classification codes.\n"
        "Key fields: `level1`..`level5`, `level1_description`..`level4_description`, `who_name`.\n\n"
        "ATC codes can also be filtered directly on the molecule endpoint via "
        "`atc_classifications__level1` (etc.)."
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "atc_class = new_client.atc_class\n\n"
        "# Get all ATC class L entries (antineoplastic and immunomodulating agents)\n"
        "res = atc_class.filter(level1='L')\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "# Filter approved drugs directly by ATC level using the molecule endpoint\n"
        "molecule = new_client.molecule\n"
        "res = molecule.filter(\n"
        "    atc_classifications__level1='J',   # antiinfectives for systemic use\n"
        "    max_phase=4                        # approved drugs only\n"
        ").only(['molecule_chembl_id', 'pref_name', 'atc_classifications', 'max_phase'])\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),

    # ── metabolism ───────────────────────────────────────────────────────────
    md(
        "## Metabolism\n\n"
        "Use `new_client.metabolism` to retrieve drug metabolism records.\n"
        "Key fields: `substrate_chembl_id` (drug being metabolised), `metabolite_chembl_id`, "
        "`metabolite_name`, `enzyme_name`, `organism`, `target_chembl_id`.\n\n"
        "Filter by `substrate_chembl_id` to get all metabolites of a given drug."
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "metabolism = new_client.metabolism\n\n"
        "# Get metabolism records for imatinib (CHEMBL941) as the substrate\n"
        "res = metabolism.filter(substrate_chembl_id='CHEMBL941')\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "metabolism = new_client.metabolism\n\n"
        "# Get all human CYP-mediated metabolism records\n"
        "res = metabolism.filter(organism='Homo sapiens')\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),

    # ── target_component ─────────────────────────────────────────────────────
    md(
        "## Target Component\n\n"
        "Use `new_client.target_component` to retrieve protein component data for targets.\n"
        "Key fields: `accession` (UniProt), `component_type`, `description`, `organism`.\n\n"
        "Filter by UniProt accession to find the ChEMBL target associated with a known protein."
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "target_component = new_client.target_component\n\n"
        "# Look up EGFR by UniProt accession (P00533)\n"
        "res = target_component.filter(accession='P00533')\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),

    # ── mechanism ────────────────────────────────────────────────────────────
    md(
        "## Mechanism of Action\n\n"
        "Use `new_client.mechanism` to retrieve curated drug mechanism of action records.\n"
        "Key fields: `action_type`, `mechanism_of_action`, `molecule_chembl_id`, "
        "`target_chembl_id`, `max_phase`.\n\n"
        "Filter by `target_chembl_id` to get all drugs with a known mechanism on a target, "
        "or by `molecule_chembl_id` to get the mechanism(s) for a specific drug."
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "mechanism = new_client.mechanism\n\n"
        "# Get all drugs with a mechanism of action on EGFR (CHEMBL203)\n"
        "res = mechanism.filter(target_chembl_id='CHEMBL203').only(\n"
        "    ['action_type', 'mechanism_of_action', 'molecule_chembl_id', 'max_phase'])\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),
    code(
        "from chembl_webresource_client.new_client import new_client\n"
        "import pandas as pd\n\n"
        "mechanism = new_client.mechanism\n\n"
        "# Get the mechanism of action for imatinib (CHEMBL941)\n"
        "res = mechanism.filter(molecule_chembl_id='CHEMBL941')\n"
        "df = pd.DataFrame(res)\n"
        "df"
    ),
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Guard: don't double-append if script is run twice
existing_text = " ".join(
    "".join(cell.get("source", [])) for cell in nb["cells"]
)
if "protein_classification" in existing_text:
    print("Cells already present — nothing to do.")
else:
    nb["cells"].extend(NEW_CELLS)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Done. Notebook now has {len(nb['cells'])} cells.")
