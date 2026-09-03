# 🧪 Small Molecule Data Curation Dashboard

A Streamlit application that turns a natural-language query (or your own uploaded compound list) into an ML-ready cheminformatics dataset. It covers the full workflow: fetch, curate, explore chemical space, model bioactivity, and prioritise compounds, with an AI assistant for hands-on refinement.

---

## Overview

1. **Get data** - describe what you want in plain English (`IC50 data for EGFR`) or upload a CSV of your own SMILES.
2. **Curate** - automated cleaning (exact measurements, standardisation, salt stripping, deduplication) plus a natural-language assistant for manual edits.
3. **Analyse** - assess dataset quality, explore chemical space and SAR, and diagnose activity/chirality cliffs.
4. **Model** - train and explain QSAR models, then score and rank new compounds.
5. **Export** - download an ML-ready CSV at any stage.

---

## Sections

The app is organised top-to-bottom; each phase feeds the next off a single live *working dataset*.

**Sidebar**
- **Curation Settings** - toggle the automated cleaning rules (exact-only, standardised-only, validity issues, salt stripping, structural-alert flagging, pIC50).
- **SMARTS Highlight** - highlight a substructure across every structure image in the app.
- **Dataset Assistant** - one chat, three modes in plain English: **Edit** (modify the data), **Ask** (answer questions about it), **Plot** (generate a chart).

**1 · Data Source** - Natural-language ChEMBL query (RAG + GPT code generation), or upload your own CSV of compounds by SMILES.

**2 · Entity Selection** - When a query matches several targets/molecules, filter (organism, type, phase) and pick the correct one.

**3 · Bioactivity Overview** - Red/amber/green dataset-quality scorecard, distribution charts, activity cliffs (with shared-scaffold + differing R-group highlighting), chirality cliffs, NLP assay-description vocabulary filtering, and SMARTS substructure filtering.

**4 · Working Dataset** - The live curated table with inline structures, plus a structural validity audit (flags salts, radicals, exotic atoms, isotopes, charges).

**5 · Molecular Descriptors** - Compute standard RDKit descriptors, view their distributions as boxplots, and rank which descriptors correlate with pIC50.

**6 · Chemical Space** - PCA and t-SNE/K-means projections, top Bemis–Murcko scaffolds, R-group decomposition, functional-group impact on activity, and pharmacophore profiling.

**7 · Bioactivity Model** - Train QSAR models (Random Forest, Ridge, Elastic Net, Gradient Boosting) with a choice of fingerprint, train/test split strategy, Optuna tuning, and optional mRMR feature selection. Explain them with SHAP (mapped back to substructures), diagnose error trends, and score new compounds with an applicability-domain check.

**8 · Desirability Ranking** - Combine predicted potency, drug-likeness (QED, Lipinski), and structural liabilities into a single ranked shortlist with per-compound reasons.

---

## Features at a glance

- **Two data sources** - natural-language ChEMBL queries (no API knowledge needed) or direct CSV upload of your own structures
- **Configurable automated curation** - relation/standardisation filters, salt stripping, structural deduplication, PAINS/Brenk/NIH/ZINC alert flagging, pIC50 computation
- **AI Dataset Assistant** - edit, query, or plot the data conversationally; generated code is shown and inspectable
- **Dataset quality assessment** - traffic-light scorecard across sample size, diversity, activity range, class balance, and assay reliability
- **SAR analysis** - activity cliffs (MCS-highlighted), chirality cliffs, R-group decomposition, functional-group impact
- **Chemical space** - PCA, t-SNE + K-means, scaffold galleries, pharmacophore profiling
- **QSAR modelling** - multiple models/fingerprints, scaffold & Butina splits, Optuna tuning, mRMR selection, SHAP explainability, error diagnostics, applicability-domain scoring
- **Compound prioritisation** - multi-criteria desirability ranking
- **Structure rendering everywhere** - inline 2D depictions, SMARTS highlighting, CSV export at every stage

---

## Project structure

```
data_curation/
├── frontend/
│   └── app.py                     # Streamlit UI — all 8 sections + sidebar
├── backend/
│   ├── pipeline.py                # Thin orchestration layer over every backend call
│   ├── llm.py                     # Query classification + code/plot/query generation
│   ├── rag_system.py              # RAG over ChEMBL API examples
│   ├── data_import.py             # Uploaded-CSV validation & reshaping
│   ├── curation.py                # Automated curation, salt stripping, structural alerts
│   ├── structure_audit.py         # Structural validity audit
│   ├── descriptors.py             # RDKit descriptors + descriptor↔pIC50 correlation
│   ├── chemical_space.py          # PCA, t-SNE/K-means, scaffolds, cliffs, pharmacophores
│   ├── functional_groups.py       # Functional-group impact on activity
│   ├── smarts.py                  # SMARTS filtering, R-group decomposition, bit→SMARTS
│   ├── model.py                   # QSAR training (RF / Ridge / Elastic Net / GB, Optuna)
│   ├── explainability.py          # SHAP values + Morgan-bit → substructure mapping
│   ├── error_analysis.py          # Prediction-error trend diagnostics
│   ├── scoring.py                 # Score new compounds + applicability domain
│   ├── desirability.py            # Multi-criteria desirability ranking
│   ├── assay_vocabulary.py        # NLP vocabulary extraction from assay descriptions
│   ├── drawing.py                 # Shared molecule-to-PNG rendering
│   └── utils/
│       ├── fingerprints.py        # Morgan / MACCS / atom-pair fingerprints
│       ├── splits.py              # Random / scaffold / Butina train-test splits
│       ├── feature_selection.py   # mRMR feature selection
│       ├── rdkit_utils.py         # Shared RDKit helpers (scaffolds, alert catalogue)
│       ├── target_resolution.py   # ChEMBL target search & candidate listing
│       └── molecule_resolution.py # ChEMBL molecule search & candidate listing
├── documents/
│   └── chembl_examples.ipynb      # RAG source - ChEMBL API usage examples
├── pyproject.toml
└── README.md
```
