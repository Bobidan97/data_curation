# 🧪 Small Molecule Data Curation Dashboard

A Streamlit application that translates natural-language queries into ChEMBL bioactivity datasets, applies automated curation, and lets you manually refine the data via an AI chatbot — producing ML-ready CSV files in minutes.

---

## Overview

The dashboard wraps the full data-preparation workflow for computational drug discovery:

1. **Query** — describe what you want in plain English (e.g. `IC50 data for EGFR`).
2. **Automated curation** — filter exact measurements, remove validity issues, compute pIC50, and deduplicate structures with RDKit.
3. **Manual curation** — a sidebar chatbot accepts natural-language instructions (e.g. `keep only rows where pIC50 >= 6`) and applies them as pandas code to your working dataset.
4. **Analyse & export** — explore molecular descriptors, visualise chemical space, and download a clean CSV.

---

## Features

- **Natural-language ChEMBL queries** — RAG over ChEMBL API notebooks + GPT-4o code generation; no API knowledge required
- **Automated data curation** — configurable pipeline: exact-relation filter, standardised-flag filter, data-validity-comment removal, pIC50 computation, RDKit structural deduplication
- **Interactive entity resolution** — when multiple targets or molecules match your query, pick the correct one from a dropdown in the UI
- **AI-powered manual curation chatbot** — GPT-4o generates and executes pandas edits; full edit history shown in the sidebar
- **Structural duplicate analysis** — identify groups of structurally identical molecules, inspect their activity distributions via a per-group box plot, and remove outlier groups through the chatbot
- **Molecular descriptor computation** — append MW, LogP, TPSA, HBA, HBD, HeavyAtomCount, RotatableBonds, RingCount, AromaticRings (RDKit)
- **Chemical space visualisation**
  - PCA biplot on physicochemical descriptors (MW, LogP, TPSA, RotatableBonds, HBA, HBD)
  - t-SNE on MACCS Tanimoto fingerprints + automatic K-means clustering (k chosen by silhouette score)
- **Inline structure rendering** — molecule structures drawn as PNG images directly in the data table
- **CSV download** — export the working dataset at any point in the workflow

---

## How it works

```
User query
    │
    ▼
GPT-4o-mini classifies domain (target / molecule / other)
    │
    ▼
RAG retrieves relevant ChEMBL API examples
(OpenAI text-embedding-3-small + cosine similarity over documents/)
    │
    ▼
GPT-4o generates chembl_webresource_client Python code
    │
    ▼
User confirms resolved entity (target or molecule ChEMBL ID)
    │
    ▼
Code executed → raw DataFrame
    │
    ▼
Automated curation (CurationConfig, configurable in sidebar)
    │
    ▼
Manual curation chatbot → GPT-4o pandas edits → df_working
    │
    ▼
Descriptors / PCA / t-SNE computed on df_working
    │
    ▼
Download CSV
```

---

## Project structure

```
data_curation/
├── frontend/
│   └── app.py                     # Streamlit UI — all 6 phases
├── backend/
│   ├── pipeline.py                # Orchestrates all backend calls
│   ├── llm.py                     # GPT-4o code generation & query classification
│   ├── rag_system.py              # RAG: document loading, chunking, embedding, retrieval
│   ├── curation.py                # Automated curation (CurationConfig, pIC50, dedup)
│   ├── descriptors.py             # RDKit molecular descriptor computation
│   ├── chemical_space.py          # PCA (physicochemical) + t-SNE/K-means (MACCS fps)
│   └── utils/
│       ├── target_resolution.py   # ChEMBL target search & candidate listing
│       └── molecule_resolution.py # ChEMBL molecule search & candidate listing
├── documents/
│   └── chembl_examples.ipynb      # RAG source — ChEMBL API usage examples
├── pyproject.toml
└── README.md
```

---

## Usage walkthrough

### Phase 1 — Query

Type a natural-language description of the data you need and click **Search**.

> Examples:
> - `IC50 data for EGFR`
> - `Ki activities for imatinib`
> - `All approved drugs for lung cancer`

The query is classified by domain (target, molecule, or other), relevant ChEMBL API examples are retrieved via RAG, and GPT-4o generates the data-fetching code.

### Phase 2 — Entity resolution

If your query names a specific target or molecule, the app searches ChEMBL and presents matching candidates in a dropdown. Select the correct entry to confirm which ChEMBL ID to use.

### Phase 3 — Automated curation

A collapsible sidebar panel lets you enable or disable each curation step:

| Option | Default | Effect |
|---|---|---|
| Exact relation only | ✅ | Keep rows where `standard_relation == "="` |
| Standardised only | ✅ | Keep rows where `standard_flag == 1` |
| Remove validity issues | ✅ | Drop rows with a `data_validity_comment` |
| Compute pIC50 | ✅ | Add `pIC50` column from `pchembl_value` or IC50 nM conversion |
| Remove structural duplicates | ✅ | Re-canonicalise SMILES with RDKit, keep first per structure |

Click **Apply curation** to run the selected steps. The result is stored as `df_curated` (immutable reference) and `df_working` (editable copy).

### Phase 4 — Explore & manually curate

The main panel shows:
- **Overview metrics** — raw row count, post-curation count, current working set count
- **Data quality charts** — measurement type distribution, pIC50 distribution, activity value ranges, molecular property histograms, data completeness heatmap
- **Structural duplicate analysis** — if duplicates exist, an expandable section shows a summary table and a box plot of activity measurements per duplicate group (each box labelled with the number of measurements)
- **Working dataset table** — searchable, sortable; includes inline molecule structure images if RDKit is available

**Sidebar chatbot** — type any natural-language instruction to filter or transform `df_working`:

> Examples:
> - `keep only rows where pIC50 >= 6`
> - `remove Group 2 from the duplicate groups`
> - `drop the standard_type column`
> - `keep only IC50 rows`

Each instruction is translated to pandas code by GPT-4o and applied immediately. The edit history is shown below the chat input. Click **Download working dataset (CSV)** at any time to export.

### Phase 5 — Molecular descriptors

Click **Compute Descriptors** to append RDKit descriptors as new columns to `df_working`:

`MW`, `LogP`, `TPSA`, `HBA`, `HBD`, `HeavyAtomCount`, `RotatableBonds`, `RingCount`, `AromaticRings`

A distribution chart and a Lipinski Ro5 violations summary are shown after computation.

### Phase 6 — Chemical space

Two complementary views:

- **PCA** — principal component analysis on the six physicochemical features (MW, LogP, TPSA, RotatableBonds, HBA, HBD). Shows a scree plot (explained variance) and a biplot coloured by pIC50.
- **t-SNE** — dimensionality reduction on 167-bit MACCS fingerprints using Tanimoto (Jaccard) distance. The optimal number of K-means clusters is chosen automatically by silhouette score. Adjust the **perplexity** slider to control local vs global structure emphasis.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `plotly` | Interactive charts |
| `openai` | GPT-4o code generation & `text-embedding-3-small` |
| `langchain` / `langchain-openai` / `langchain-text-splitters` | Document chunking & embedding pipeline |
| `chembl-webresource-client` | ChEMBL REST API client |
| `rdkit` | SMILES canonicalisation, molecular descriptors, MACCS fingerprints, structure images |
| `scikit-learn` | PCA, t-SNE, K-means, cosine similarity |
| `pandas` / `numpy` | Data manipulation |
| `python-dotenv` | `.env` file loading |
| `faiss-cpu` | Vector index (available for future scaling) |
