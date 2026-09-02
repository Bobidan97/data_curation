import io
import pickle
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pipeline import (
    run_rag,
    classify_query,
    get_generated_code,
    get_entity_candidates,
    fetch_chembl_data,
    run_descriptor_computation,
    get_descriptor_correlations,
    get_structural_duplicates,
    get_duplicate_activity_rows,
    run_chemical_space_pca,
    run_chemical_space_tsne,
    get_pca_insights,
    get_cluster_insights,
    get_chemical_space_summary,
    get_top_scaffolds,
    get_cluster_chemistry,
    get_pca_outliers,
    get_pca_axis_extremes,
    get_activity_cliff_pairs,
    run_pharmacophore_profiles,
    get_cliff_pair_mcs,
    run_desirability_ranking,
    get_desirability_reasons,
    compute_shap_explanation,
    get_global_top_bits,
    get_top_bits_for_molecule,
    get_atoms_for_bit,
    extract_assay_vocabulary,
    filter_assay_descriptions,
    preview_assay_filter,
    validate_smarts_pattern,
    filter_dataset_by_smarts,
    preview_smarts_match,
    run_rgroup_decomposition,
    get_morgan_bit_smarts,
    score_new_compounds,
    get_training_pic50_distribution,
    analyze_model_errors,
    run_model_training,
    run_model_training_tuned,
    edit_dataframe,
    answer_dataframe_question,
    create_dataframe_plot,
    run_activity_cliffs,
    run_chirality_cliffs,
    detect_upload_columns,
    prepare_uploaded_compounds,
    run_structure_audit,
    get_structure_audit_summary,
    run_functional_group_impact,
    get_example_with_group,
)
from curation import curate_dataframe, CurationConfig, summarise_drops


# Colour palette for the six pharmacophore families (RGB 0-1 floats for RDKit;
# hex versions exposed for legend rendering / Plotly use).
_PHARM_COLORS = {
    "Donor":        (0.22, 0.48, 0.93),  # blue
    "Acceptor":     (0.90, 0.22, 0.22),  # red
    "Hydrophobe":   (0.94, 0.78, 0.18),  # amber
    "Aromatic":     (0.56, 0.22, 0.93),  # violet
    "PosIonizable": (0.10, 0.72, 0.50),  # teal-green
    "NegIonizable": (0.93, 0.52, 0.14),  # orange
}
_PHARM_COLORS_HEX = {
    fam: "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    for fam, (r, g, b) in _PHARM_COLORS.items()
}


@st.cache_data(show_spinner=False)
def _build_pharmacophore_images(smiles_tuple: tuple) -> list:
    """Render each SMILES as a base64 PNG with pharmacophore atoms colour-coded.

    Uses RDKit's BaseFeatures.fdef feature factory to detect Donor / Acceptor /
    Hydrophobe / Aromatic / +Ionisable / −Ionisable atoms, then draws each
    molecule with the matched atoms highlighted in the family colour.

    Falls back to all-None when RDKit is unavailable.
    """
    import os
    try:
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
    except ImportError:
        return [None] * len(smiles_tuple)

    from drawing import mol_to_uri
    from utils.rdkit_utils import mol_from_smiles

    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory   = ChemicalFeatures.BuildFeatureFactory(fdef_path)

    results = []
    for smi in smiles_tuple:
        mol = mol_from_smiles(smi)
        if mol is None:
            results.append(None)
            continue
        try:
            highlight_atoms: list[int] = []
            atom_colors:     dict[int, tuple] = {}
            for feat in factory.GetFeaturesForMol(mol):
                family = feat.GetFamily()
                if family == "LumpedHydrophobe":
                    family = "Hydrophobe"
                color = _PHARM_COLORS.get(family)
                if color is None:
                    continue
                for idx in feat.GetAtomIds():
                    if idx not in atom_colors:
                        highlight_atoms.append(idx)
                        atom_colors[idx] = color
            results.append(mol_to_uri(
                mol,
                highlight_atoms=highlight_atoms,
                atom_colors=atom_colors,
                size=(360, 270),
            ))
        except Exception:
            results.append(None)
    return results


@st.cache_data(show_spinner=False)
def _build_image_column_with_smarts(
    smiles_tuple: tuple,
    smarts: str,
    size: tuple = (400, 300),
) -> list:
    """Render each SMILES with atoms matching ``smarts`` highlighted (amber).

    Thin cached wrapper over ``drawing.render_smiles_smarts``; falls back to a
    plain column when the SMARTS is empty/invalid.
    """
    if not smarts or not smarts.strip():
        return _build_image_column(smiles_tuple, size=size)
    from drawing import render_smiles_smarts
    big = (int(size[0] * 2), int(size[1] * 2))
    return [render_smiles_smarts(s, smarts, size=big) for s in smiles_tuple]


@st.cache_data(show_spinner=False)
def _highlight_atoms_image(smiles: str, atom_indices: list, size: tuple = (240, 180)) -> str | None:
    """Render a molecule with a specific set of atoms highlighted in blue.

    Used by the SHAP explainability tab to show which atoms triggered a given
    Morgan fingerprint bit. Thin wrapper over ``drawing.render_smiles_atoms``.
    """
    from drawing import render_smiles_atoms, BLUE
    return render_smiles_atoms(smiles, atom_indices, BLUE, size=size)


@st.cache_data(show_spinner=False)
def _build_mcs_pair_images(
    smi_1: str,
    smi_2: str,
    atom_compare: str = "elements",
    bond_compare: str = "order_exact",
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
) -> tuple:
    """Render a pair of molecules with their MCS scaffold greyed-out and the
    differing R-groups highlighted in coral.

    Strictness arguments are passed through to ``get_cliff_pair_mcs``; tighter
    settings produce more conservative scaffolds (fewer "shared" atoms, larger
    highlighted R-groups).

    Returns ``(img_1, img_2, mcs_size)``; the images are base64 data URIs (or
    None on failure). ``mcs_size`` is 0 when no MCS was found.
    """
    from drawing import mol_to_uri, CORAL

    mcs = get_cliff_pair_mcs(
        smi_1, smi_2,
        atom_compare=atom_compare,
        bond_compare=bond_compare,
        complete_rings_only=complete_rings_only,
        ring_matches_ring_only=ring_matches_ring_only,
    )
    if mcs is None:
        return (None, None, 0)

    def _render(mol, diff_atoms, diff_bonds):
        return mol_to_uri(
            mol,
            highlight_atoms=diff_atoms,
            atom_colors={a: CORAL for a in diff_atoms},
            highlight_bonds=diff_bonds,
            bond_colors={b: CORAL for b in diff_bonds},
            size=(360, 280),
        )

    img_1 = _render(mcs["mol_1"], mcs["diff_atoms_1"], mcs["diff_bonds_1"])
    img_2 = _render(mcs["mol_2"], mcs["diff_atoms_2"], mcs["diff_bonds_2"])
    return (img_1, img_2, mcs["mcs_size"])


def _raw_df_fingerprint(raw_df: pd.DataFrame) -> tuple:
    """Cheap hash-safe fingerprint for raw ChEMBL DataFrames.

    Streamlit's ``@st.cache_data`` chokes on the dict-valued columns ChEMBL
    sometimes returns (``molecule_properties``, ``activity_properties``).
    We sidestep that by building our own fingerprint and using session_state
    as the cache.
    """
    cols = tuple(raw_df.columns.tolist())
    ids: tuple = ()
    if "molecule_chembl_id" in raw_df.columns and len(raw_df) > 0:
        head = raw_df["molecule_chembl_id"].head(5).astype(str).tolist()
        tail = raw_df["molecule_chembl_id"].tail(5).astype(str).tolist()
        ids = tuple(head + tail)
    return (len(raw_df), cols, ids)


def _cached_get_structural_duplicates(raw_df: pd.DataFrame):
    """Session-state cache for the structural-duplicate scan."""
    fp = _raw_df_fingerprint(raw_df)
    if st.session_state.get("_sd_cache_fp") != fp:
        st.session_state["_sd_cache"]    = get_structural_duplicates(raw_df)
        st.session_state["_sd_cache_fp"] = fp
    return st.session_state["_sd_cache"]


def _cached_get_duplicate_activity_rows(raw_df: pd.DataFrame):
    """Session-state cache for duplicate-row extraction."""
    fp = _raw_df_fingerprint(raw_df)
    if st.session_state.get("_dar_cache_fp") != fp:
        st.session_state["_dar_cache"]    = get_duplicate_activity_rows(raw_df)
        st.session_state["_dar_cache_fp"] = fp
    return st.session_state["_dar_cache"]


@st.cache_data(show_spinner=False)
def _count_unique_scaffolds(smiles_tuple: tuple) -> int | None:
    """Count unique Bemis-Murcko scaffolds across a set of SMILES strings."""
    from utils.rdkit_utils import murcko_scaffold
    return len({s for s in (murcko_scaffold(smi) for smi in smiles_tuple) if s})


@st.cache_data(show_spinner=False)
def _build_image_column(smiles_tuple: tuple, size: tuple = (400, 300)) -> list:
    """Render each SMILES as a high-quality base64 PNG data URI.

    Thin cached wrapper over ``drawing.render_smiles`` (RDKit Cairo backend).
    Images are rendered at 2× the requested size for HiDPI crispness; Streamlit
    downsamples them to the column width. Returns a list of data-URI strings
    (or None for invalid SMILES).
    """
    from drawing import render_smiles
    big = (int(size[0] * 2), int(size[1] * 2))
    return [
        render_smiles(s, size=big, padding=0.08, bond_line_width=2)
        for s in smiles_tuple
    ]


# ── Professional chart palette & layout ───────────────────────────────────────
_PALETTE = [
    "#1B3A6B",  # navy        (primary)
    "#0891B2",  # cyan        (secondary)
    "#059669",  # emerald
    "#D97706",  # amber
    "#7C3AED",  # violet
    "#DC2626",  # red
    "#0D9488",  # teal
    "#B45309",  # brown-amber
]

_PLOT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#FAFAFA",
    font_family="Inter, system-ui, -apple-system, sans-serif",
    font_color="#1E293B",
    title_font_size=14,
    title_font_color="#1B3A6B",
    margin=dict(t=50, b=40, l=60, r=20),
    xaxis=dict(
        gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
        title_font_size=12, tickfont_size=11,
    ),
    yaxis=dict(
        gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
        title_font_size=12, tickfont_size=11,
    ),
)


def _apply_chart_style(fig, height: int | None = None) -> None:
    """Apply the shared professional layout to any Plotly figure."""
    updates = dict(_PLOT_LAYOUT)
    if height:
        updates["height"] = height
    fig.update_layout(**updates)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chemical Data Curation", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Design tokens ────────────────────────────────────────────────────────── */
:root {
    --brand-navy:      #1B3A6B;
    --brand-navy-deep: #0F2444;
    --brand-cyan:      #0891B2;
    --ink:             #1E293B;
    --muted:           #64748B;
    --line:            #E7ECF2;
    --line-strong:     #CBD5E1;
    --surface:         #FFFFFF;
    --bg-soft:         #F6F8FB;
    --radius:          12px;
    --radius-sm:       9px;
    --shadow-sm:       0 1px 2px rgba(15,36,68,0.04), 0 1px 3px rgba(15,36,68,0.05);
    --shadow-md:       0 4px 14px rgba(15,36,68,0.08);
}

/* ── Global font ──────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp, .stMarkdown,
button, input, textarea, select, [data-baseweb] {
    font-family: 'Inter', system-ui, -apple-system, "Segoe UI", sans-serif !important;
}
.stApp { background: var(--bg-soft); }

/* ── Typography ───────────────────────────────────────────────────────────── */
h1 { font-size: 1.55rem !important; font-weight: 800 !important;
     color: var(--brand-navy-deep) !important; letter-spacing: -0.012em; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important;
     color: var(--brand-navy) !important; letter-spacing: 0.005em; }
h3 { font-size: 0.98rem !important; font-weight: 600 !important;
     color: var(--brand-navy) !important; }

/* ── Numbered section header ──────────────────────────────────────────────── */
.sec-head { display:flex; align-items:center; gap:0.7rem; margin:0.1rem 0 1.1rem 0; }
.sec-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:32px; height:32px; border-radius:9px; flex-shrink:0;
    background: linear-gradient(135deg, var(--brand-navy), var(--brand-cyan));
    color:#fff; font-weight:700; font-size:1rem;
    box-shadow: 0 2px 6px rgba(8,145,178,0.28);
}
.sec-title { font-size:1.18rem; font-weight:700; color:var(--brand-navy-deep); }

/* ── Phase containers ─────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: var(--radius) !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1.1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover { box-shadow: var(--shadow-md); }
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700; color: var(--brand-navy-deep) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    border: 1px solid var(--line-strong);
    transition: all 0.15s ease;
}
[data-testid="stButton"] > button:hover:enabled {
    border-color: var(--brand-cyan); color: var(--brand-cyan);
    box-shadow: var(--shadow-sm);
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--brand-navy), var(--brand-cyan));
    border: none; color: #fff;
}
[data-testid="stButton"] > button[kind="primary"]:hover:enabled {
    filter: brightness(1.08); color: #fff; border: none;
}

/* ── Inputs, selects, textareas ───────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="select"] > div,
.stTextArea textarea, [data-baseweb="textarea"] {
    border-radius: var(--radius-sm) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { font-weight: 600; color: var(--muted); }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--brand-navy) !important; }

/* ── Expanders ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm);
    background: var(--surface);
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* ── Plotly chart wrappers ────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] > div {
    border-radius: var(--radius-sm);
    box-shadow: var(--shadow-sm);
    background: var(--surface);
    border: 1px solid var(--line);
}

/* ── Dataframes ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: var(--radius-sm); border: 1px solid var(--line); }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: var(--brand-navy) !important; }

/* ── Misc ─────────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)

# ── Branded header ────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1B3A6B 0%, #0891B2 100%);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.6rem;
    color: white;
    box-shadow: 0 6px 20px rgba(15,36,68,0.16);
">
    <div style="display:flex; align-items:center; gap:0.9rem;">
        <span style="font-size:2.1rem; line-height:1;">🧬</span>
        <div>
            <div style="font-size:1.5rem; font-weight:800; letter-spacing:0.01em;">
                ChEMBL Data Curation
            </div>
            <div style="font-size:0.83rem; opacity:0.82; margin-top:3px;
                        font-weight:500; letter-spacing:0.02em; text-transform:uppercase;">
                Automated pipeline &nbsp;·&nbsp; Structural analysis &nbsp;·&nbsp; Bioactivity modelling
            </div>
        </div>
        <div style="margin-left:auto; font-size:0.72rem; font-weight:600; opacity:0.85;
                    background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.18);
                    border-radius:20px; padding:4px 13px; white-space:nowrap;">
            v1.0
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def _section_header(number, title: str) -> None:
    """Render a professional numbered section header (badge + title)."""
    st.markdown(
        f'<div class="sec-head">'
        f'<span class="sec-badge">{number}</span>'
        f'<span class="sec-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Phase 1: Query ────────────────────────────────────────────────────────────
with st.container(border=True):
    _section_header(1, "Query")
    user_query = st.text_input(
        "Enter your ChEMBL database query",
        placeholder="e.g. IC50 data for EGFR, activities for imatinib, approved drugs for lung cancer",
    )

    _do_search = st.button("🔍 Search", disabled=not user_query,
                           use_container_width=True, type="primary")

    # ── Bring your own compounds (bypasses ChEMBL entity resolution) ───────────
    with st.expander("📁 Or upload your own compounds (CSV)", expanded=False):
        st.caption(
            "A CSV with a SMILES column (and optionally a name/ID column). "
            "Skips ChEMBL entity resolution - everything downstream (curation, "
            "descriptors, chemical space, modelling) works the same as for a "
            "ChEMBL query, minus bioactivity-specific fields like pIC50."
        )
        _uploaded_file = st.file_uploader("CSV file", type=["csv"], key="upload_csv_file")

        if _uploaded_file is not None:
            try:
                _upload_raw = pd.read_csv(_uploaded_file)
            except Exception as _ue:
                st.error(f"Could not read this file as CSV: {_ue}")
                _upload_raw = None

            if _upload_raw is not None and not _upload_raw.empty:
                _det_smiles, _det_name = detect_upload_columns(_upload_raw)
                _up_cols = list(_upload_raw.columns)

                _uc1, _uc2 = st.columns(2)
                with _uc1:
                    _smiles_col = st.selectbox(
                        "SMILES column",
                        options=_up_cols,
                        index=_up_cols.index(_det_smiles) if _det_smiles in _up_cols else 0,
                        key="upload_smiles_col",
                    )
                with _uc2:
                    _name_options = ["(none - auto-number)"] + _up_cols
                    _default_name_idx = (
                        _name_options.index(_det_name) if _det_name in _up_cols else 0
                    )
                    _name_choice = st.selectbox(
                        "Name / ID column (optional)",
                        options=_name_options,
                        index=_default_name_idx,
                        key="upload_name_col",
                    )
                _name_col = None if _name_choice.startswith("(none") else _name_choice

                st.caption(f"{len(_upload_raw):,} rows detected in the file.")
                st.dataframe(_upload_raw.head(5), use_container_width=True, hide_index=True)

                if st.button("✅ Validate & preview", key="btn_upload_validate"):
                    with st.spinner("Validating structures…"):
                        try:
                            st.session_state["_upload_prepared"] = prepare_uploaded_compounds(
                                _upload_raw, _smiles_col, _name_col
                            )
                        except ValueError as _ve:
                            st.error(str(_ve))

                if "_upload_prepared" in st.session_state:
                    _prep = st.session_state["_upload_prepared"]
                    _pu1, _pu2, _pu3 = st.columns(3)
                    _pu1.metric("Rows in file", f"{_prep['n_input']:,}")
                    _pu2.metric("Valid structures", f"{_prep['n_valid']:,}")
                    _pu3.metric("Invalid / dropped", f"{_prep['n_invalid']:,}")

                    if _prep["n_invalid"] > 0:
                        with st.popover(f"👁️ View {min(20, _prep['n_invalid'])} dropped rows"):
                            st.dataframe(
                                _prep["invalid_df"].head(20),
                                use_container_width=True, hide_index=True,
                            )

                    if _prep["n_valid"] > 0:
                        _prev_df = _prep["df"].head(8)
                        _prev_imgs = _build_image_column(
                            tuple(_prev_df["canonical_smiles"]), size=(180, 140)
                        )
                        st.markdown("**Preview**")
                        _up_ncols = min(4, len(_prev_df))
                        _prev_rows = [
                            list(zip(_prev_df["name"], _prev_imgs))[i:i + _up_ncols]
                            for i in range(0, len(_prev_df), _up_ncols)
                        ]
                        for _prow in _prev_rows:
                            _pcols = st.columns(len(_prow))
                            for _pc, (_pname, _pimg) in zip(_pcols, _prow):
                                with _pc:
                                    if _pimg:
                                        st.image(_pimg, use_container_width=True)
                                    st.caption(_pname)

                        if st.button(
                            "📥 Import into dashboard", key="btn_upload_import", type="primary"
                        ):
                            st.session_state["raw_df"] = _prep["df"]
                            st.session_state["domain"] = "molecule"
                            st.session_state["user_query"] = f"Uploaded: {_uploaded_file.name}"
                            st.session_state["entity_name"] = None
                            st.session_state.pop("candidates_df", None)
                            st.session_state.pop("generated_code", None)
                            st.session_state.pop("_upload_prepared", None)
                            st.rerun()
                    else:
                        st.warning(
                            "No valid structures to import - check SMILES "
                            "column selection above."
                        )

    if _do_search:
        with st.spinner("Classifying query and generating code..."):
            domain, entity_name = classify_query(user_query)
            context = run_rag(user_query)
            generated_code = get_generated_code(user_query, context, domain)
            # Only resolve candidates when the domain has a candidate-resolver.
            # The classifier sometimes returns an entity_name (e.g. 'imatinib',
            # 'HeLa') with domain='other'; calling get_entity_candidates in that
            # case raises ValueError.
            candidates_df = (
                get_entity_candidates(domain, entity_name)
                if entity_name and domain in ("target", "molecule")
                else None
            )
            st.session_state["domain"] = domain
            st.session_state["generated_code"] = generated_code
            st.session_state["candidates_df"] = candidates_df
            st.session_state["user_query"] = user_query
            st.session_state["entity_name"] = entity_name
            st.session_state.pop("raw_df", None)

# ── Phase 2: Entity selection (target or molecule) ────────────────────────────
if "candidates_df" in st.session_state:
    with st.container(border=True):
        domain      = st.session_state.get("domain", "other")
        candidates  = st.session_state["candidates_df"]
        entity_name = st.session_state.get("entity_name") or ""

        if candidates is not None and not candidates.empty:
            _n = len(candidates)
            _n_label = f"**{_n} candidate{'s' if _n != 1 else ''}** found" + (
                f" for *{entity_name}*" if entity_name else ""
            )

            # ── TARGET selection ───────────────────────────────────────────
            if domain == "target":
                _section_header(2, "Select Target")
                st.caption(_n_label)

                # Quick filters
                _tf1, _tf2 = st.columns(2)
                _orgs_raw  = candidates["organism"].dropna().unique().tolist()
                _prio_orgs = [o for o in ["Homo sapiens", "Mus musculus", "Rattus norvegicus"]
                              if o in _orgs_raw]
                _rest_orgs = sorted([o for o in _orgs_raw if o not in _prio_orgs])
                _org_opts  = ["All"] + _prio_orgs + _rest_orgs
                _type_opts = ["All"] + sorted(candidates["target_type"].dropna().unique().tolist())

                with _tf1:
                    _org_f  = st.selectbox("Organism",    _org_opts,  key="ph2_org")
                with _tf2:
                    _type_f = st.selectbox("Target type", _type_opts, key="ph2_type")

                # Apply filters
                _filt = candidates.copy()
                if _org_f  != "All":
                    _filt = _filt[_filt["organism"]    == _org_f]
                if _type_f != "All":
                    _filt = _filt[_filt["target_type"] == _type_f]

                if _filt.empty:
                    st.warning("No candidates match these filters - clear one to broaden results.")
                    _filt = candidates.copy()

                # Sort: Homo sapiens SINGLE PROTEINs first
                _filt = _filt.copy()
                _filt["_s"] = (
                    (_filt["organism"]    == "Homo sapiens")  .astype(int) * 10
                    + (_filt["target_type"] == "SINGLE PROTEIN").astype(int) * 5
                )
                _filt = (
                    _filt.sort_values("_s", ascending=False)
                    .drop(columns=["_s"])
                    .reset_index(drop=True)
                )

                # Candidate table
                st.dataframe(
                    _filt[["pref_name", "organism", "target_type", "target_chembl_id"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "pref_name":        st.column_config.TextColumn("Name",       width="large"),
                        "organism":         st.column_config.TextColumn("Organism",   width="medium"),
                        "target_type":      st.column_config.TextColumn("Type",       width="medium"),
                        "target_chembl_id": st.column_config.TextColumn("ChEMBL ID",  width="small"),
                    },
                )

                # Selection dropdown
                _t_opts = {
                    f"{row['pref_name']} ({row['target_chembl_id']})": row["target_chembl_id"]
                    for _, row in _filt.iterrows()
                }
                _t_label       = st.selectbox("Select target", list(_t_opts.keys()), key="ph2_target_sel")
                entity_chembl_id = _t_opts[_t_label]

                # Info strip for selected target
                _sel = _filt[_filt["target_chembl_id"] == entity_chembl_id].iloc[0]
                _ic1, _ic2, _ic3 = st.columns(3)
                _ic1.metric("ChEMBL ID",   _sel["target_chembl_id"])
                _ic2.metric("Organism",    _sel["organism"]    or "—")
                _ic3.metric("Target Type", _sel["target_type"] or "—")

            # ── MOLECULE selection ─────────────────────────────────────────
            else:
                _section_header(2, "Select Molecule")
                st.caption(_n_label)

                # Quick filters
                _mf1, _mf2 = st.columns(2)
                _phase_opts = ["All", "Approved (Phase 4)", "Phase 3+", "Phase 2+"]
                _mtype_opts = ["All"] + sorted(candidates["molecule_type"].dropna().unique().tolist())

                with _mf1:
                    _phase_f = st.selectbox("Clinical phase",  _phase_opts, key="ph2_phase")
                with _mf2:
                    _mtype_f = st.selectbox("Molecule type",   _mtype_opts, key="ph2_mtype")

                # Apply filters
                _filt = candidates.copy()
                _filt["_pn"] = pd.to_numeric(_filt["max_phase"], errors="coerce").fillna(-1)
                if _phase_f == "Approved (Phase 4)":
                    _filt = _filt[_filt["_pn"] == 4]
                elif _phase_f == "Phase 3+":
                    _filt = _filt[_filt["_pn"] >= 3]
                elif _phase_f == "Phase 2+":
                    _filt = _filt[_filt["_pn"] >= 2]
                if _mtype_f != "All":
                    _filt = _filt[_filt["molecule_type"] == _mtype_f]

                if _filt.empty:
                    st.warning("No candidates match these filters - clear one to broaden results.")
                    _filt = candidates.copy()
                    _filt["_pn"] = pd.to_numeric(_filt["max_phase"], errors="coerce").fillna(-1)

                _filt = _filt.sort_values("_pn", ascending=False).reset_index(drop=True)

                # Display table with formatted phase
                _disp = _filt.drop(columns=["_pn"]).copy()
                _disp["max_phase"] = _filt["_pn"].apply(
                    lambda x: str(int(x)) if x >= 0 else "—"
                )
                st.dataframe(
                    _disp[["pref_name", "molecule_type", "max_phase", "molecule_chembl_id"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "pref_name":          st.column_config.TextColumn("Name",      width="large"),
                        "molecule_type":      st.column_config.TextColumn("Type",      width="medium"),
                        "max_phase":          st.column_config.TextColumn("Phase",     width="small"),
                        "molecule_chembl_id": st.column_config.TextColumn("ChEMBL ID", width="small"),
                    },
                )

                # Selection dropdown
                _m_opts = {
                    f"{row['pref_name']} ({row['molecule_chembl_id']})": row["molecule_chembl_id"]
                    for _, row in _filt.iterrows()
                }
                _m_label       = st.selectbox("Select molecule", list(_m_opts.keys()), key="ph2_mol_sel")
                entity_chembl_id = _m_opts[_m_label]

                # Info strip for selected molecule
                _sel    = _filt[_filt["molecule_chembl_id"] == entity_chembl_id].iloc[0]
                _mi1, _mi2, _mi3 = st.columns(3)
                _mi1.metric("ChEMBL ID",    _sel["molecule_chembl_id"])
                _mi2.metric("Molecule Type", _sel["molecule_type"] or "—")
                _mi3.metric("Max Phase",    str(int(_sel["_pn"])) if _sel["_pn"] >= 0 else "—")

            if st.button("📥 Fetch Data"):
                with st.spinner("Fetching data from ChEMBL..."):
                    try:
                        st.session_state["raw_df"] = fetch_chembl_data(
                            st.session_state["generated_code"], domain, entity_chembl_id
                        )
                    except Exception as e:
                        st.error(str(e))

        else:
            # No resolution needed: exact ChEMBL ID / SMILES / 'other' domain
            _section_header(2, "Fetch Data")
            _domain_hints = {
                "molecule": "The query uses an exact identifier (SMILES, InChI key, or ChEMBL ID) - no disambiguation needed.",
                "other":    "This query does not reference a specific entity. The generated code will query ChEMBL directly.",
            }
            st.info(_domain_hints.get(domain, "No entity resolution required. Click below to fetch data."))

            # For target/molecule queries using an exact ChEMBL ID, extract it so the
            # LLM-generated code (which references target_chembl_id / molecule_chembl_id) works.
            _direct_id = None
            if domain in ("target", "molecule"):
                _match = re.search(
                    r"CHEMBL\d+",
                    st.session_state.get("user_query", ""),
                    flags=re.IGNORECASE,
                )
                if _match:
                    _direct_id = _match.group(0).upper()

            if st.button("📥 Fetch Data"):
                with st.spinner("Fetching data from ChEMBL..."):
                    try:
                        st.session_state["raw_df"] = fetch_chembl_data(
                            st.session_state["generated_code"], domain, _direct_id
                        )
                    except Exception as e:
                        st.error(str(e))

# ── Phases 3–7: post-fetch pipeline ───────────────────────────────────────────
if "raw_df" in st.session_state:
    raw_df = st.session_state["raw_df"]

    with st.sidebar.expander("⚙️ Curation Settings", expanded=False):
        st.caption("Automated rules applied before manual review.")
        exact_only = st.checkbox("Exact measurements only  (=)", value=True)
        standardised_only = st.checkbox("Standardised rows only (flag = 1)", value=True)
        remove_validity = st.checkbox("Remove data validity issues", value=True)
        remove_structural_duplicates = st.checkbox(
            "Remove structural duplicates (canonicalise SMILES)", value=True
        )
        strip_salts = st.checkbox(
            "Strip salts / counter-ions  (keep largest fragment)",
            value=True,
            help="Discards counter-ions, solvents and other components, keeping "
                 "the heavy-atom fragment. Applied during SMILES re-canonicalisation."
        )
        flag_structural_alerts_opt = st.checkbox(
            "Flag structural alerts  (PAINS / Brenk / NIH / ZINC)",
            value=False,
            help="Adds a `structural_alerts` column listing any matched alerts "
                 "from the RDKit FilterCatalog. Does NOT remove rows — surfaces "
                 "promiscuous / reactive groups for manual review.",
        )
        compute_pic50 = st.checkbox("Compute pIC50", value=True)

    # ── SMARTS-based structure highlight (cross-cuts every structure view) ────
    with st.sidebar.expander("🔍 SMARTS Highlight", expanded=False):
        st.caption(
            "Enter a SMARTS pattern to highlight matching atoms in every "
            "structure image across the app (working dataset table, scaffolds, "
            "outliers, cliff pairs, etc.). Leave blank to disable."
        )
        _smarts_highlight = st.text_input(
            "SMARTS pattern",
            value="",
            placeholder="e.g. c1ccncc1  (pyridine ring)",
            key="smarts_highlight",
        )
        if _smarts_highlight.strip():
            _ok, _err = validate_smarts_pattern(_smarts_highlight)
            if not _ok:
                st.warning(f"Invalid SMARTS: {_err}")
                _smarts_highlight = ""

    config = CurationConfig(
        exact_relation_only=exact_only,
        standardised_only=standardised_only,
        remove_validity_issues=remove_validity,
        remove_structural_duplicates=remove_structural_duplicates,
        compute_pic50=compute_pic50,
        strip_salts=strip_salts,
        flag_structural_alerts=flag_structural_alerts_opt,
    )

    # ── Session-state cache for curate_dataframe ──────────────────────────────
    # raw_df can contain dict-valued columns from ChEMBL (molecule_properties,
    # activity_properties, …) which Streamlit's @st.cache_data can't hash.
    # Use a cheap fingerprint + session_state instead.
    _config_fp = (
        exact_only, standardised_only, remove_validity, compute_pic50,
        remove_structural_duplicates, strip_salts, flag_structural_alerts_opt,
    )
    _curate_key = (_raw_df_fingerprint(raw_df), _config_fp)
    if st.session_state.get("_curate_cache_key") != _curate_key:
        st.session_state["_curate_cache"] = curate_dataframe(raw_df, config)
        st.session_state["_curate_cache_key"] = _curate_key
    df_curated, df_dropped = st.session_state["_curate_cache"]

    # Invalidate cached results when curated data changes
    _fingerprint = (len(df_curated), tuple(df_curated.columns.tolist()))
    if st.session_state.get("_curated_fingerprint") != _fingerprint:
        st.session_state.pop("df_with_descriptors", None)
        st.session_state.pop("_pca_result", None)
        st.session_state.pop("_tsne_result", None)
        st.session_state.pop("_pharm_result", None)
        st.session_state.pop("_tsne_config", None)
        st.session_state["_curated_fingerprint"] = _fingerprint

    # ── Sidebar: Manual curation chat ─────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.header("💬 Dataset Assistant")
    st.sidebar.caption(
        "**Edit** the data or **Ask** a question about it, in plain English."
    )

    # Initialise or invalidate working copy when df_curated changes
    _wfp = (len(df_curated), tuple(df_curated.columns.tolist()))
    if st.session_state.get("_working_fingerprint") != _wfp:
        st.session_state["df_working"]           = df_curated.copy()
        st.session_state["chat_history"]         = []
        st.session_state["_working_fingerprint"] = _wfp

    df_working = st.session_state["df_working"]

    # Invalidate Phase 5/6/7 caches whenever the working dataset changes shape or columns
    _wdesc_fp = (len(df_working), tuple(df_working.columns.tolist()))
    if st.session_state.get("_working_desc_fp") != _wdesc_fp:
        st.session_state.pop("df_with_descriptors", None)
        st.session_state.pop("_pca_result", None)
        st.session_state.pop("_tsne_result", None)
        st.session_state.pop("_pharm_result", None)
        st.session_state.pop("_model_result", None)
        st.session_state.pop("_model_tuned_result", None)
        st.session_state.pop("_model_history", None)
        st.session_state.pop("_sali_result", None)
        st.session_state.pop("_chir_result", None)
        st.session_state.pop("_audit_df", None)
        st.session_state.pop("_fg_impact", None)
        st.session_state.pop("_desirability_result", None)
        st.session_state.pop("_assay_classified", None)

        st.session_state["_working_desc_fp"] = _wdesc_fp

    removed = len(df_curated) - len(df_working)
    st.sidebar.caption(
        f"Working: **{len(df_working):,}** rows"
        + (f" ({removed:,} removed by manual edits)" if removed else "")
    )

    if st.sidebar.button("↺ Reset to curated data", use_container_width=True):
        st.session_state["df_working"]   = df_curated.copy()
        st.session_state["chat_history"] = []
        st.rerun()

    # ── Chat input form (reliable alternative to st.sidebar.chat_input) ──────
    with st.sidebar.form("manual_curation_form", clear_on_submit=True):
        _chat_mode = st.radio(
            "Mode",
            options=["✏️ Edit", "❓ Ask", "📊 Plot"],
            horizontal=True,
            label_visibility="collapsed",
            help="Edit: modify the working dataset. Ask: get an answer without "
                 "changing the data. Plot: generate a chart from a description.",
        )
        _instruction = st.text_area(
            "Instruction",
            height=80,
            placeholder=(
                'Edit:  "remove rows where pIC50 < 5"\n'
                'Ask:   "what is the median pIC50?"\n'
                'Plot:  "histogram of pIC50 coloured by assay type"'
            ),
            label_visibility="collapsed",
        )
        _submitted = st.form_submit_button("▶ Submit", use_container_width=True)

    _ask_mode  = _chat_mode == "❓ Ask"
    _plot_mode = _chat_mode == "📊 Plot"

    if _submitted and _instruction.strip():
        _instruction = _instruction.strip()
        _mode_prefix = "📊 " if _plot_mode else ("❓ " if _ask_mode else "✏️ ")
        st.session_state["chat_history"].append(
            {"role": "user", "content": _mode_prefix + _instruction, "code": None}
        )

        # Shared read-only reference DataFrames for both Edit and Ask modes.
        # Coerce known numeric columns that ChEMBL returns as object (strings)
        # so LLM-generated aggregations work without extra casting.
        _NUMERIC_COLS = {"standard_value", "pchembl_value", "standard_upper_value"}

        def _coerce_numerics(frame: pd.DataFrame) -> pd.DataFrame:
            frame = frame.copy()
            for _c in _NUMERIC_COLS & set(frame.columns):
                frame[_c] = pd.to_numeric(frame[_c], errors="coerce")
            return frame

        _extra = {"raw_df": _coerce_numerics(raw_df), "df_curated": df_curated}
        _dup = _cached_get_duplicate_activity_rows(raw_df)
        if not _dup.empty:
            _extra["dup_rows"] = _coerce_numerics(_dup)
        if "df_with_descriptors" in st.session_state:
            _extra["df_with_descriptors"] = st.session_state["df_with_descriptors"]

        _reset_kws = {"reset", "restore", "undo all", "revert"}

        if _ask_mode:
            #  Ask: answer a question without modifying the data ──────────────
            with st.spinner("Thinking…"):
                try:
                    _answer, _qcode = answer_dataframe_question(
                        st.session_state["df_working"], _instruction, extra_dfs=_extra
                    )
                    _obj = None
                    if isinstance(_answer, pd.DataFrame):
                        _content = f"Returned a table ({len(_answer):,} rows)."
                        _obj = _answer
                    elif isinstance(_answer, pd.Series):
                        _content = f"Returned a breakdown ({len(_answer):,} rows)."
                        _obj = _answer
                    elif isinstance(_answer, float) or hasattr(_answer, "dtype"):
                        _content = f"**{float(_answer):,.4g}**"
                    else:
                        _content = f"**{_answer}**"
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": _content,
                        "code": _qcode, "answer_obj": _obj,
                    })
                except (ValueError, TypeError) as _exc:
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": f"⚠️ {_exc}", "code": None,
                    })

        elif _plot_mode:
            #  Plot: generate a chart without modifying the data ──────────────
            with st.spinner("Building chart…"):
                try:
                    _fig, _pcode = create_dataframe_plot(
                        st.session_state["df_working"], _instruction, extra_dfs=_extra
                    )
                    _apply_chart_style(_fig, height=360)
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": "Here's your chart:",
                        "code": _pcode, "fig_obj": _fig,
                    })
                except ValueError as _exc:
                    st.session_state["chat_history"].append({
                        "role": "assistant", "content": f"⚠️ {_exc}", "code": None,
                    })

        elif any(kw in _instruction.lower() for kw in _reset_kws):
            st.session_state["df_working"] = df_curated.copy()
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": "Dataset reset to the automated curation result.",
                "code": None,
            })

        else:
            # Edit: modify the working dataset ───────────────────────────────
            with st.spinner("Applying edit…"):
                try:
                    _new_df, _code = edit_dataframe(
                        st.session_state["df_working"], _instruction, extra_dfs=_extra
                    )
                    _rb = len(st.session_state["df_working"])
                    _cb = len(st.session_state["df_working"].columns)
                    st.session_state["df_working"] = _new_df
                    _ra, _ca = len(_new_df), len(_new_df.columns)
                    _parts = []
                    if _rb != _ra:
                        _parts.append(f"Rows: {_rb:,} → {_ra:,} ({_rb - _ra:,} removed).")
                    if _cb != _ca:
                        _parts.append(f"Columns: {_cb} → {_ca} ({_cb - _ca} dropped).")
                    if not _parts:
                        _parts.append(f"Applied - shape unchanged ({_ra:,} × {_ca}).")
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": " ".join(_parts),
                        "code": _code,
                    })
                except ValueError as _exc:
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": f"⚠️ {_exc}",
                        "code": None,
                    })
        st.rerun()

    with st.sidebar.expander("Chat history", expanded=True):
        if not st.session_state["chat_history"]:
            st.caption("No messages yet.")
        for _mi, _msg in enumerate(st.session_state["chat_history"]):
            with st.chat_message(_msg["role"]):
                st.write(_msg["content"])
                _obj = _msg.get("answer_obj")
                if _obj is not None:
                    if isinstance(_obj, pd.Series):
                        st.dataframe(
                            _obj.rename("value").reset_index(),
                            use_container_width=True, hide_index=True,
                        )
                    elif isinstance(_obj, pd.DataFrame):
                        st.dataframe(_obj, use_container_width=True, hide_index=True)
                _fig_obj = _msg.get("fig_obj")
                if _fig_obj is not None:
                    st.plotly_chart(
                        _fig_obj, use_container_width=True,
                        key=f"chat_fig_{_mi}",
                    )
                if _msg.get("code"):
                    with st.expander("code", expanded=False):
                        st.code(_msg["code"], language="python")

    st.sidebar.download_button(
        label="⬇️ Download working dataset",
        data=st.session_state["df_working"].to_csv(index=False),
        file_name="chembl_manual.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Column-presence detection ──────────────────────────────────────────────
    domain = st.session_state.get("domain", "other")
    cols = set(df_working.columns)

    BIOACTIVITY_COLS   = {"pIC50", "assay_type", "assay_chembl_id", "standard_type",
                          "document_year", "document_journal"}
    MOLECULE_PROP_COLS = {"max_phase", "molecule_type", "first_approval"}

    has_bioactivity    = bool(cols & BIOACTIVITY_COLS)
    has_molecule_props = bool(cols & MOLECULE_PROP_COLS)
    has_pic50          = "pIC50" in cols
    has_assay_type     = "assay_type" in cols
    has_assay_id       = "assay_chembl_id" in cols
    has_standard_type  = "standard_type" in cols
    has_bao_label      = "bao_label" in cols
    has_doc_year       = "document_year" in cols
    has_doc_journal    = "document_journal" in cols
    has_smiles         = "canonical_smiles" in cols
    has_max_phase      = "max_phase" in cols
    has_mol_type       = "molecule_type" in cols
    has_first_appr     = "first_approval" in cols

    DOMAIN_LABEL = {"target": "Bioactivity", "molecule": "Molecule", "other": "Data"}
    section_label = DOMAIN_LABEL.get(domain, "Data")

    # ── Structural duplicate analysis ─────────────────────────────────────────
    if has_smiles:
        dup_df = _cached_get_structural_duplicates(raw_df)
        if not dup_df.empty:
            total_groups  = len(dup_df)
            total_removed = int(dup_df["would_remove"].sum())
            with st.expander(
                f"🔍 {total_groups} structural duplicate group(s) found "
                f"- {total_removed} molecule(s) would be removed by deduplication",
                expanded=False,
            ):
                # ── Structure grid (one image per group, capped at 8) ──────────
                group_canons = dup_df["canonical_smiles"].tolist()
                group_labels = [f"Group {i + 1}" for i in range(len(group_canons))]
                display_canons = group_canons[:8]
                display_labels = group_labels[:8]
                if len(group_canons) > 8:
                    st.caption(f"Showing first 8 of {len(group_canons)} groups.")
                grid_cols = st.columns(len(display_canons))
                _grid_imgs = _build_image_column(tuple(display_canons))
                for col, img, label in zip(grid_cols, _grid_imgs, display_labels):
                    if img:
                        col.image(img, caption=label, use_container_width=True)
                    else:
                        col.caption(label)

                # ── Activity strip plot ────────────────────────────────────────
                dup_rows = _cached_get_duplicate_activity_rows(raw_df)
                if (
                    not dup_rows.empty
                    and "standard_value" in dup_rows.columns
                    and dup_rows["standard_value"].notna().any()
                ):
                    plot_df = dup_rows.copy()
                    plot_df["standard_value"] = pd.to_numeric(
                        plot_df["standard_value"], errors="coerce"
                    )
                    plot_df = plot_df.dropna(subset=["standard_value"])

                    if (
                        "_canonical" in plot_df.columns
                        and "canonical_smiles" in df_curated.columns
                        and "canonical_smiles" in df_working.columns
                    ):
                        _manually_removed_smi = (
                            set(df_curated["canonical_smiles"].dropna())
                            - set(df_working["canonical_smiles"].dropna())
                        )
                        if _manually_removed_smi:
                            plot_df = plot_df[
                                ~plot_df["_canonical"].isin(_manually_removed_smi)
                            ]

                    if plot_df.empty:
                        st.info(
                            "All duplicate-group molecules have been removed "
                            "from the working dataset."
                        )
                    else:
                        group_order = sorted(
                            plot_df["_group"].unique(),
                            key=lambda g: int(re.search(r"\d+", g).group()),
                        )

                        total_groups = len(group_order)
                        # Streamlit's slider requires min_value < max_value, so only
                        # render it when there are at least 2 groups to choose between.
                        if total_groups <= 1:
                            n_show = total_groups
                            st.caption(
                                f"{total_groups} duplicate group found."
                            )
                        else:
                            n_show = st.slider(
                                "Groups to display",
                                min_value=1,
                                max_value=total_groups,
                                value=min(20, total_groups),
                                step=1,
                                help=f"{total_groups} duplicate groups found. "
                                     "Showing the first N (Group 1 = most duplicated).",
                            )
                        group_order = group_order[:n_show]
                        if n_show < total_groups:
                            st.caption(
                                f"Showing {n_show} of {total_groups} groups. "
                                "Increase the slider to see more."
                            )

                        sort_key = {g: i for i, g in enumerate(group_order)}
                        plot_df = plot_df.copy()
                        plot_df["_sort"] = plot_df["_group"].map(sort_key)
                        plot_df = (
                            plot_df[plot_df["_group"].isin(group_order)]
                            .sort_values("_sort")
                            .drop(columns=["_sort"])
                        )

                        hover_cols = [
                            c for c in ["molecule_chembl_id", "canonical_smiles",
                                        "standard_type", "standard_units"]
                            if c in dup_rows.columns
                        ]
                        fig_dup = go.Figure()
                        for grp in group_order:
                            grp_rows = plot_df[plot_df["_group"] == grp]
                            n_mols = len(grp_rows)
                            hover_text = grp_rows.apply(
                                lambda r: "<br>".join(
                                    f"{c}: {r[c]}" for c in hover_cols if pd.notna(r.get(c))
                                ),
                                axis=1,
                            ).tolist()
                            fig_dup.add_trace(go.Box(
                                y=grp_rows["standard_value"].tolist(),
                                name=f"{grp} (n={n_mols})",
                                text=hover_text,
                                hovertemplate="%{text}<extra></extra>",
                                boxpoints="all",
                                jitter=0.3,
                                marker=dict(size=7, opacity=0.7, color=_PALETTE[1]),
                                line=dict(color=_PALETTE[0]),
                                fillcolor="rgba(8,145,178,0.15)",
                            ))
                        _apply_chart_style(fig_dup, height=420)
                        fig_dup.update_layout(
                            title="Activity values within structural duplicate groups",
                            xaxis_title="Duplicate group",
                            yaxis_title="Standard value (nM)",
                            yaxis_type="log",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_dup, use_container_width=True)
                else:
                    st.info("No standard_value data available to plot activity.")

                st.caption(
                    "Use **Remove structural duplicates (canonicalise SMILES)** in the "
                    "sidebar to control whether these are removed during curation."
                )
        else:
            st.success("✅ No structural duplicates found in the raw data.")

    # ── Phase 3: Overview ─────────────────────────────────────────────────────
    with st.container(border=True):
        _section_header(3, f"{section_label} Overview")

        # ── Dropped rows reference ─────────────────────────────────────────────
        _n_dropped = len(df_dropped)
        with st.expander(
            f"🗑️ {_n_dropped:,} row{'s' if _n_dropped != 1 else ''} dropped during auto-curation",
            expanded=False,
        ):
            if _n_dropped == 0:
                st.success("No rows were dropped during auto-curation.")
            else:
                _summary = summarise_drops(df_dropped)

                # Headline metrics
                _dm1, _dm2 = st.columns(2)
                _dm1.metric("Total dropped",     f"{_summary['total']:,}")
                _dm2.metric("Multi-reason rows", f"{_summary['multi_reason']:,}")

                if _summary["multi_reason"] > 0:
                    st.caption(
                        f"ℹ️ **{_summary['multi_reason']:,}** row"
                        f"{'s' if _summary['multi_reason'] != 1 else ''} failed more than one check. "
                        "They are counted in **each** applicable reason in the table below."
                    )

                # Per-reason breakdown (overlapping)
                st.markdown("**Drops by reason** _(rows with multiple reasons counted in each)_")
                st.dataframe(
                    _summary["by_reason"],
                    use_container_width=True,
                    hide_index=True,
                )

                # Per-combination breakdown (non-overlapping) — only useful when overlap exists
                if _summary["multi_reason"] > 0:
                    st.markdown("**Drops by reason combination** _(each row counted exactly once)_")
                    st.dataframe(
                        _summary["by_combination"],
                        use_container_width=True,
                        hide_index=True,
                    )

                st.divider()
                st.caption("Full dropped-rows table (first column = reason(s) for removal):")
                st.dataframe(df_dropped, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Raw rows", f"{len(raw_df):,}")
        col2.metric("After auto-curation", f"{len(df_curated):,}")
        _manual_removed = len(df_curated) - len(df_working)
        col3.metric(
            "Working set",
            f"{len(df_working):,}",
            delta=f"−{_manual_removed:,} manual edits" if _manual_removed else None,
            delta_color="off",
        )

        # ── Dataset quality assessment ─────────────────────────────────────────
        st.subheader("📊 Dataset Quality Assessment")

        # Compute the metrics we'll reuse
        _n_rows         = len(df_working)
        _n_uniq_smiles  = df_working["canonical_smiles"].nunique() if has_smiles else None
        _n_uniq_assays  = df_working["assay_chembl_id"].nunique() if has_assay_id else None
        _n_uniq_docs    = (
            df_working["document_chembl_id"].nunique()
            if "document_chembl_id" in cols else None
        )
        _n_uniq_targets = (
            df_working["target_chembl_id"].nunique()
            if "target_chembl_id" in cols else None
        )

        # ── Tier 1: overview cards ─────────────────────────────────────────────
        _oc1, _oc2, _oc3, _oc4 = st.columns(4)
        _oc1.metric("Working rows", f"{_n_rows:,}")
        _oc2.metric(
            "Unique compounds",
            f"{_n_uniq_smiles:,}" if _n_uniq_smiles is not None else "N/A",
            delta=(
                f"{_n_uniq_smiles / _n_rows * 100:.0f}% of rows"
                if _n_uniq_smiles is not None and _n_rows else None
            ),
            delta_color="off",
        )
        _oc3.metric(
            "Unique assays",
            f"{_n_uniq_assays:,}" if _n_uniq_assays is not None else "N/A",
        )
        _oc4.metric(
            "Unique sources",
            f"{_n_uniq_docs:,}" if _n_uniq_docs is not None else "N/A",
        )

        # ── Active-threshold control (only meaningful when pIC50 exists) ───────
        _active_thresh = 5.0
        if has_pic50:
            _ts1, _ts2 = st.columns([3, 2])
            with _ts1:
                _active_thresh = st.slider(
                    "Active threshold (pIC50)",
                    min_value=3.0,
                    max_value=9.0,
                    value=5.0,
                    step=0.5,
                    key="quality_active_thresh",
                )
            _um = 10 ** (6 - _active_thresh)
            if   _um >= 1000: _ic50_str = f"{_um/1000:.3g} mM"
            elif _um >= 1:    _ic50_str = f"{_um:.3g} µM"
            else:             _ic50_str = f"{_um*1000:.3g} nM"
            with _ts2:
                st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical spacer
                st.markdown(
                    f"Equivalent activity: **IC50 ≤ {_ic50_str}**",
                    help="Direct conversion from the chosen pIC50 threshold.",
                )

        # ── Tier 2: traffic-light scorecard ────────────────────────────────────
        _GOOD, _WARN, _POOR = "🟢", "🟡", "🔴"
        _rows = []
        def _add(flag: str, metric: str, value: str, note: str) -> None:
            _rows.append({"": flag, "Metric": metric, "Value": value, "Assessment": note})

        # Sample size
        if _n_rows >= 500:
            _add(_GOOD, "Sample size", f"{_n_rows:,} rows", "Adequate for ML modelling")
        elif _n_rows >= 100:
            _add(_WARN, "Sample size", f"{_n_rows:,} rows", "Borderline - risk of overfit")
        else:
            _add(_POOR, "Sample size", f"{_n_rows:,} rows", "Too small for reliable ML")

        # Compound uniqueness (after auto-curation, ideally close to 100%)
        if _n_uniq_smiles is not None and _n_rows > 0:
            _u = _n_uniq_smiles / _n_rows
            if _u >= 0.95:
                _add(_GOOD, "Compound uniqueness", f"{_u*100:.0f}%", "Clean - minimal residual duplication")
            elif _u >= 0.80:
                _add(_WARN, "Compound uniqueness", f"{_u*100:.0f}%", "Some duplication remains")
            else:
                _add(_POOR, "Compound uniqueness", f"{_u*100:.0f}%", "Heavy duplication - review curation")

        # pIC50-derived metrics
        if has_pic50:
            _pic50 = df_working["pIC50"].dropna()
            _cov   = len(_pic50) / _n_rows * 100 if _n_rows else 0

            if _cov >= 95:
                _add(_GOOD, "pIC50 coverage", f"{_cov:.0f}%", "Near-complete activity values")
            elif _cov >= 70:
                _add(_WARN, "pIC50 coverage", f"{_cov:.0f}%", "Some missing - consider imputation")
            else:
                _add(_POOR, "pIC50 coverage", f"{_cov:.0f}%", "Many missing - limited ML signal")

            if len(_pic50) > 1:
                _rng = float(_pic50.max() - _pic50.min())
                if _rng >= 3:
                    _add(_GOOD, "Activity range",  f"{_rng:.1f} log units", "Wide dynamic range - good for regression")
                elif _rng >= 1.5:
                    _add(_WARN, "Activity range",  f"{_rng:.1f} log units", "Moderate range")
                else:
                    _add(_POOR, "Activity range",  f"{_rng:.1f} log units", "Narrow range - weak ML signal")

                _active = float((_pic50 >= _active_thresh).mean() * 100)
                _af_label = f"Active fraction (pIC50 ≥ {_active_thresh:g})"
                if 20 <= _active <= 80:
                    _add(_GOOD, _af_label, f"{_active:.0f}%", "Balanced active/inactive split")
                elif 5 <= _active <= 95:
                    _add(_WARN, _af_label, f"{_active:.0f}%", "Skewed - consider stratified split")
                else:
                    _add(_POOR, _af_label, f"{_active:.0f}%", "Heavily imbalanced classes")

        # Source diversity
        if _n_uniq_docs is not None:
            if _n_uniq_docs >= 10:
                _add(_GOOD, "Source diversity", f"{_n_uniq_docs:,} documents", "Diverse sources - reduces single-paper bias")
            elif _n_uniq_docs >= 3:
                _add(_WARN, "Source diversity", f"{_n_uniq_docs:,} documents", "Few sources - moderate publication bias")
            else:
                _add(_POOR, "Source diversity", f"{_n_uniq_docs:,} document(s)", "Single/few sources - high bias risk")

        # Assay reliability (binding assays are the gold standard)
        if has_assay_type:
            _pct_b = float((df_working["assay_type"] == "B").mean() * 100)
            if _pct_b >= 70:
                _add(_GOOD, "Binding assays (type B)", f"{_pct_b:.0f}%", "Mostly direct binding - high reliability")
            elif _pct_b >= 30:
                _add(_WARN, "Binding assays (type B)", f"{_pct_b:.0f}%", "Mixed assay types - heterogeneous")
            else:
                _add(_POOR, "Binding assays (type B)", f"{_pct_b:.0f}%", "Mostly functional/ADMET - harder to model")

        # Scaffold diversity (RDKit; skip for very large datasets to keep it snappy)
        if has_smiles and _n_uniq_smiles is not None and 5 <= _n_uniq_smiles <= 5000:
            _uniq_smi_tuple = tuple(df_working["canonical_smiles"].dropna().unique())
            _n_scaf = _count_unique_scaffolds(_uniq_smi_tuple)
            if _n_scaf is not None and _n_uniq_smiles > 0:
                _sd = _n_scaf / _n_uniq_smiles
                if _sd >= 0.30:
                    _add(_GOOD, "Scaffold diversity", f"{_n_scaf:,} scaffolds ({_sd*100:.0f}%)",
                         "Diverse chemistry — broadly applicable model")
                elif _sd >= 0.10:
                    _add(_WARN, "Scaffold diversity", f"{_n_scaf:,} scaffolds ({_sd*100:.0f}%)",
                         "Moderate diversity — some scaffold concentration")
                else:
                    _add(_POOR, "Scaffold diversity", f"{_n_scaf:,} scaffolds ({_sd*100:.0f}%)",
                         "Series-like dataset — model may not generalise")

        # Render scorecard
        _qdf = pd.DataFrame(_rows)
        _good_n = sum(r[""] == _GOOD for r in _rows)
        _warn_n = sum(r[""] == _WARN for r in _rows)
        _poor_n = sum(r[""] == _POOR for r in _rows)
        st.caption(
            f"Quality scorecard: **{_good_n} good**, **{_warn_n} caution**, **{_poor_n} concern**."
        )
        st.dataframe(
            _qdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "":           st.column_config.TextColumn(width="small"),
                "Metric":     st.column_config.TextColumn(width="medium"),
                "Value":      st.column_config.TextColumn(width="medium"),
                "Assessment": st.column_config.TextColumn(width="large"),
            },
        )

        # ── Tier 3: pIC50 statistics (compact reference) ──────────────────────
        if has_pic50:
            _pic50 = df_working["pIC50"].dropna()
            if not _pic50.empty:
                st.markdown("**pIC50 statistics**")
                _p1, _p2, _p3, _p4, _p5 = st.columns(5)
                _p1.metric("Mean",   f"{_pic50.mean():.2f}")
                _p2.metric("Median", f"{_pic50.median():.2f}")
                _p3.metric("Std",    f"{_pic50.std():.2f}")
                _p4.metric("IQR",    f"{_pic50.quantile(0.75) - _pic50.quantile(0.25):.2f}")
                _p5.metric("Range",  f"{_pic50.min():.1f} – {_pic50.max():.1f}")

        # ── Bioactivity charts ─────────────────────────────────────────────────
        if has_bioactivity:
            st.subheader("Bioactivity Analysis")

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if has_pic50:
                    pic50_data = df_working["pIC50"].dropna()
                    if not pic50_data.empty:
                        fig = px.histogram(
                            pic50_data, nbins=40, title="pIC50 Distribution",
                            labels={"value": "pIC50", "count": "Count"},
                            color_discrete_sequence=_PALETTE[:1],
                        )
                        _apply_chart_style(fig)
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            with bcol2:
                if has_assay_type:
                    assay_counts = df_working["assay_type"].value_counts().reset_index()
                    assay_counts.columns = ["assay_type", "count"]
                    fig2 = px.bar(
                        assay_counts, x="assay_type", y="count",
                        title="Assay Type Breakdown",
                        color_discrete_sequence=_PALETTE[:1],
                    )
                    _apply_chart_style(fig2)
                    st.plotly_chart(fig2, use_container_width=True)

            brow2_col1, brow2_col2 = st.columns(2)
            with brow2_col1:
                if has_doc_year:
                    year_counts = df_working["document_year"].value_counts().sort_index().reset_index()
                    year_counts.columns = ["year", "count"]
                    fig3 = px.bar(
                        year_counts, x="year", y="count", title="Publications by Year",
                        color_discrete_sequence=_PALETTE[:1],
                    )
                    _apply_chart_style(fig3)
                    st.plotly_chart(fig3, use_container_width=True)
            with brow2_col2:
                if has_bao_label:
                    bao_counts = (
                        df_working["bao_label"]
                        .dropna()
                        .value_counts()
                        .reset_index()
                    )
                    bao_counts.columns = ["bao_label", "count"]
                    # Shorten long BAO labels for readability
                    bao_counts["bao_label"] = (
                        bao_counts["bao_label"]
                        .str.replace(r"^BAO_\d+\s*", "", regex=True)
                        .str.replace("_", " ")
                        .str.title()
                    )
                    fig_bao = px.bar(
                        bao_counts, x="count", y="bao_label",
                        orientation="h",
                        title="Assay Format (BAO Label)",
                        labels={"bao_label": "", "count": "Count"},
                        color_discrete_sequence=_PALETTE[1:2],
                    )
                    _apply_chart_style(fig_bao)
                    fig_bao.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_bao, use_container_width=True)

            # ── NLP vocabulary explorer for assay descriptions ────────────────
            if "assay_description" in df_working.columns:
                with st.expander(
                    "🔍 Assay description vocabulary",
                    expanded=False,
                ):
                    _n_unique = df_working["assay_description"].fillna("").nunique()
                    st.caption(
                        f"Recurring words and phrases across **{_n_unique:,} unique "
                        f"assay descriptions**. Highlighted categories (Mutation, "
                        f"Wild-type, Cell line, Species, Preparation, Assay format) "
                        f"flag terms a user is likely to want to filter on - e.g. "
                        f"selecting *'mutant'* removes any row whose description "
                        f"mentions a mutant protein, keeping wild-type measurements only."
                    )

                    _vc1, _vc2, _vc3, _vc4 = st.columns([1, 1, 1, 1])
                    with _vc1:
                        _min_rows = st.slider(
                            "Min row count",
                            min_value=1, max_value=50, value=3,
                            key="vocab_min_rows",
                            help="Drop terms that appear in fewer rows than this.",
                        )
                    with _vc2:
                        _inc_uni = st.checkbox(
                            "Unigrams", value=True, key="vocab_uni"
                        )
                    with _vc3:
                        _inc_bi  = st.checkbox(
                            "Bigrams", value=True, key="vocab_bi"
                        )
                    with _vc4:
                        _inc_tri = st.checkbox(
                            "Trigrams", value=False, key="vocab_tri"
                        )

                    _vocab_df = extract_assay_vocabulary(
                        df_working["assay_description"].tolist(),
                        min_row_count=_min_rows,
                        include_unigrams=_inc_uni,
                        include_bigrams=_inc_bi,
                        include_trigrams=_inc_tri,
                    )

                    if _vocab_df.empty:
                        st.info(
                            "No terms meet the current count threshold - lower the "
                            "**Min row count** or enable more n-gram lengths."
                        )
                    else:
                        # Category filter
                        _cats = ["All"] + sorted(_vocab_df["category"].unique().tolist())
                        _cat_sel = st.selectbox(
                            "Show category",
                            options=_cats,
                            key="vocab_cat_filter",
                        )
                        _disp = (
                            _vocab_df
                            if _cat_sel == "All"
                            else _vocab_df[_vocab_df["category"] == _cat_sel]
                        )

                        # Distribution by category
                        _cat_summary = (
                            _vocab_df["category"]
                            .value_counts()
                            .reset_index()
                        )
                        _cat_summary.columns = ["category", "term_count"]

                        _sumcol, _tblcol = st.columns([2, 5])
                        with _sumcol:
                            st.markdown("**Categories**")
                            st.dataframe(
                                _cat_summary,
                                use_container_width=True,
                                hide_index=True,
                            )
                        with _tblcol:
                            st.markdown(
                                f"**Terms** _(showing **{len(_disp):,}** of "
                                f"{len(_vocab_df):,})_"
                            )
                            st.dataframe(
                                _disp,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "n_rows": st.column_config.NumberColumn(
                                        "Rows", format="%d"
                                    ),
                                    "n_descriptions": st.column_config.NumberColumn(
                                        "Descs", format="%d"
                                    ),
                                    "example_description": st.column_config.TextColumn(
                                        "Example description", width="large"
                                    ),
                                },
                            )

                        # Term-picker for filtering
                        st.markdown("---")
                        st.markdown("**Filter the working dataset by removing rows mentioning…**")

                        _picked_terms = st.multiselect(
                            "Pick terms to exclude",
                            options=_vocab_df["term"].tolist(),
                            default=[],
                            key="vocab_pick_terms",
                            help="Case-insensitive substring match: 'mutant' also "
                                 "catches 'mutants' and 'G12D mutant'.",
                        )

                        if _picked_terms:
                            _impact = preview_assay_filter(df_working, _picked_terms)
                            st.caption(
                                f"This would remove **{_impact['n_remove']:,} rows** "
                                f"({_impact['pct_remove']:.1f}% of working set), "
                                f"keeping **{_impact['n_keep']:,}**."
                            )

                            _pcol1, _pcol2 = st.columns([1, 1])
                            with _pcol1:
                                if st.button(
                                    "🚫 Remove rows mentioning selected terms",
                                    key="btn_apply_vocab_filter",
                                    disabled=_impact["n_remove"] == 0,
                                ):
                                    _kept, _removed = filter_assay_descriptions(
                                        df_working, _picked_terms
                                    )
                                    st.session_state["df_working"] = _kept
                                    st.success(
                                        f"Removed {len(_removed):,} rows. "
                                        f"Working set now has {len(_kept):,} rows."
                                    )
                                    st.rerun()
                            with _pcol2:
                                # Show a preview of removal candidates
                                _, _to_remove = filter_assay_descriptions(
                                    df_working, _picked_terms
                                )
                                if not _to_remove.empty and "assay_description" in _to_remove.columns:
                                    with st.popover(
                                        f"👁️ Preview {min(20, len(_to_remove))} rows that would be removed"
                                    ):
                                        _preview_cols = [
                                            c for c in (
                                                "molecule_chembl_id",
                                                "pIC50",
                                                "standard_type",
                                                "assay_description",
                                            ) if c in _to_remove.columns
                                        ]
                                        st.dataframe(
                                            _to_remove[_preview_cols].head(20),
                                            use_container_width=True,
                                            hide_index=True,
                                        )

            # ── SMARTS substructure filter ────────────────────────────────────
            if "canonical_smiles" in df_working.columns:
                @st.fragment
                def _smarts_filter_panel():
                    """Fragment: SMARTS controls + preview re-run only on internal
                    widget changes (not full-app reruns). Apply forces a full
                    rerun once the working dataset has actually mutated."""
                    df_w = st.session_state["df_working"]
                    with st.expander("🧬 SMARTS substructure filter", expanded=False):
                        st.caption(
                            "Enter a SMARTS pattern to keep or remove rows whose "
                            "structure contains a specific substructure. "
                            "Examples: `c1ccncc1` (pyridine), `C=CC(=O)` (Michael "
                            "acceptor), `[F,Cl,Br,I]` (any halogen)."
                        )
                        _sc1, _sc2 = st.columns([3, 1])
                        with _sc1:
                            _smarts_input = st.text_input(
                                "SMARTS pattern",
                                value="",
                                placeholder="e.g. c1ccncc1",
                                key="smarts_filter_input",
                            )
                        with _sc2:
                            _mode = st.radio(
                                "Action",
                                options=["Keep matches", "Remove matches"],
                                key="smarts_filter_mode",
                                horizontal=False,
                            )

                        if _smarts_input.strip():
                            _ok, _err = validate_smarts_pattern(_smarts_input)
                            if not _ok:
                                st.error(f"Invalid SMARTS: {_err}")
                                return
                            _stats = preview_smarts_match(df_w, _smarts_input)
                            _keep = _mode == "Keep matches"
                            _n_kept    = _stats["n_match"] if _keep else _stats["n_total"] - _stats["n_match"]
                            _n_removed = _stats["n_total"] - _n_kept
                            st.caption(
                                f"{_stats['n_match']:,} of {_stats['n_total']:,} rows "
                                f"match this SMARTS ({_stats['pct_match']:.1f}%). "
                                f"Applying would keep **{_n_kept:,}** and remove **{_n_removed:,}**."
                            )

                            _, _hits = filter_dataset_by_smarts(
                                df_w, _smarts_input, keep_matches=True
                            )
                            if not _hits.empty:
                                _preview_n = min(8, len(_hits))
                                _preview_imgs = _build_image_column_with_smarts(
                                    tuple(_hits["canonical_smiles"].head(_preview_n)),
                                    _smarts_input,
                                    size=(200, 160),
                                )
                                st.markdown(f"**First {_preview_n} matches:**")
                                _ncols = min(4, _preview_n)
                                _rows = [
                                    list(zip(
                                        _hits.head(_preview_n).iterrows(),
                                        _preview_imgs,
                                    ))[i:i + _ncols]
                                    for i in range(0, _preview_n, _ncols)
                                ]
                                for _row in _rows:
                                    _cols = st.columns(len(_row))
                                    for _col, ((_, _r), _img) in zip(_cols, _row):
                                        with _col:
                                            if _img:
                                                st.image(_img, use_container_width=True)
                                            _id = _r.get("molecule_chembl_id", "—")
                                            st.caption(f"**{_id}**")

                            if st.button(
                                "🚀 Apply SMARTS filter",
                                key="btn_apply_smarts_filter",
                                disabled=_n_kept == 0 or _n_removed == 0,
                            ):
                                _kept_df, _ = filter_dataset_by_smarts(
                                    df_w, _smarts_input, keep_matches=_keep,
                                )
                                st.session_state["df_working"] = _kept_df
                                # Full app rerun so Phase 4-8 see the new df_working
                                st.rerun()

                _smarts_filter_panel()

            if has_assay_id and has_pic50:
                cv_df = (
                    df_working.groupby("assay_chembl_id")["pIC50"]
                    .agg(["mean", "std", "count"])
                    .query("count >= 2")
                    .dropna()
                    .assign(cv=lambda x: x["std"] / x["mean"].abs() * 100)
                    .reset_index()
                )
                if not cv_df.empty:
                    cv_col1, cv_col2 = st.columns(2)
                    with cv_col1:
                        fig_cv1 = px.histogram(
                            cv_df, x="cv", nbins=30,
                            title="CV Distribution Across Assays",
                            labels={"cv": "CV (%)", "count": "# Assays"},
                            color_discrete_sequence=_PALETTE[:1],
                        )
                        _apply_chart_style(fig_cv1)
                        fig_cv1.update_layout(showlegend=False)
                        st.plotly_chart(fig_cv1, use_container_width=True)
                    with cv_col2:
                        top10_cv = cv_df.nlargest(10, "cv")
                        fig_cv2 = px.bar(
                            top10_cv, x="cv", y="assay_chembl_id", orientation="h",
                            title="Top 10 Most Variable Assays (CV %)",
                            labels={"cv": "CV (%)", "assay_chembl_id": "Assay"},
                            color_discrete_sequence=[_PALETTE[5]],
                        )
                        _apply_chart_style(fig_cv2)
                        fig_cv2.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_cv2, use_container_width=True)

            if has_doc_journal:
                journal_counts = (
                    df_working["document_journal"].value_counts().head(10).reset_index()
                )
                journal_counts.columns = ["journal", "count"]
                fig5 = px.bar(
                    journal_counts, x="count", y="journal", orientation="h",
                    title="Top 10 Journals",
                    labels={"count": "Count", "journal": "Journal"},
                    color_discrete_sequence=_PALETTE[:1],
                )
                _apply_chart_style(fig5)
                fig5.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig5, use_container_width=True)

        # ── Molecule quality charts ────────────────────────────────────────────
        if has_molecule_props:
            st.subheader("Molecule Overview")

            mcol1, mcol2 = st.columns(2)
            with mcol1:
                if has_max_phase:
                    phase_counts = (
                        pd.to_numeric(df_working["max_phase"], errors="coerce")
                        .fillna(-1).astype(float).astype(int)
                        .value_counts().sort_index().reset_index()
                    )
                    phase_counts.columns = ["max_phase", "count"]
                    phase_counts["max_phase"] = (
                        phase_counts["max_phase"].replace(-1, "N/A").astype(str)
                    )
                    fig_phase = px.bar(
                        phase_counts, x="max_phase", y="count",
                        title="Development Phase Distribution",
                        labels={"max_phase": "Max Phase", "count": "# Molecules"},
                        color_discrete_sequence=_PALETTE[:1],
                        category_orders={"max_phase": ["N/A", "0", "1", "2", "3", "4"]},
                    )
                    _apply_chart_style(fig_phase)
                    st.plotly_chart(fig_phase, use_container_width=True)
            with mcol2:
                if has_mol_type:
                    mol_type_counts = df_working["molecule_type"].value_counts().reset_index()
                    mol_type_counts.columns = ["molecule_type", "count"]
                    fig_mol_type = px.bar(
                        mol_type_counts, x="molecule_type", y="count",
                        title="Molecule Type Breakdown",
                        labels={"molecule_type": "Type", "count": "# Molecules"},
                        color_discrete_sequence=_PALETTE[:1],
                    )
                    _apply_chart_style(fig_mol_type)
                    st.plotly_chart(fig_mol_type, use_container_width=True)

            if has_first_appr:
                appr_counts = (
                    df_working["first_approval"].dropna()
                    .astype(int).value_counts().sort_index().reset_index()
                )
                appr_counts.columns = ["year", "count"]
                fig_appr = px.bar(
                    appr_counts, x="year", y="count",
                    title="First Approval Year",
                    labels={"year": "Year", "count": "# Molecules"},
                    color_discrete_sequence=_PALETTE[:1],
                )
                _apply_chart_style(fig_appr)
                st.plotly_chart(fig_appr, use_container_width=True)

        # ── Data completeness ──────────────────────────────────────────────────
        st.subheader("Data Completeness")
        completeness = (df_working.notna().mean() * 100).sort_values()
        completeness_df = completeness.reset_index()
        completeness_df.columns = ["column", "pct_complete"]
        fig6 = px.bar(
            completeness_df, x="pct_complete", y="column", orientation="h",
            title="Data Completeness (% non-null per column)",
            color="pct_complete",
            color_continuous_scale=["#DC2626", "#D97706", "#059669"],
            range_color=[0, 100],
            labels={"pct_complete": "% Complete", "column": "Column"},
        )
        _apply_chart_style(fig6)
        fig6.update_layout(coloraxis_showscale=False, xaxis_range=[0, 100])
        st.plotly_chart(fig6, use_container_width=True)

        # ── Activity Cliffs (SALI) ─────────────────────────────────────────────
        if has_smiles and has_pic50:
            st.subheader("Activity Cliffs (Structural Similarity vs. Potency Difference)")
            st.markdown(
                "Computes pairwise Tanimoto similarity (ECFP4) and |ΔpIC50|. Points "
                "in the **top-right** (Tanimoto > 0.4 and |ΔpIC50| > 2.0) are activity "
                "cliffs - structurally similar molecules with large potency "
                "differences. Every molecule pair is scanned, but only pairs above "
                "the similarity floor below are retained (that's where cliffs live) - "
                "so the whole dataset is analysed without missing cliffs to sampling."
            )
            _sali_c1, _sali_c2 = st.columns([2, 3])
            with _sali_c1:
                _sali_floor = st.slider(
                    "Similarity floor (min Tanimoto to retain)",
                    min_value=0.2, max_value=0.7, value=0.3, step=0.05,
                    key="sali_floor",
                    help="Only pairs at least this similar are kept and plotted. "
                         "Lower = more pairs (heavier); cliffs sit at ≥ 0.4.",
                )
            if st.button("🔭 Compute Activity Cliffs", key="btn_sali"):
                with st.spinner("Scanning all molecule pairs…"):
                    try:
                        st.session_state["_sali_result"] = run_activity_cliffs(
                            df_working, min_similarity=_sali_floor
                        )
                    except ValueError as _se:
                        st.error(f"Activity cliff computation failed: {_se}")

            if "_sali_result" in st.session_state:
                _sali = st.session_state["_sali_result"]
                _pairs = _sali["pairs_df"]
                _scan_note = (
                    f"{_sali['n_molecules']:,} molecules scanned · "
                    f"{_sali.get('n_total_pairs', 0):,} pairs compared · "
                    f"**{_sali['n_pairs']:,}** retained at Tanimoto ≥ "
                    f"{_sali.get('min_similarity', 0.3):g}."
                )
                if _sali["truncated"]:
                    st.info(
                        f"Dataset has {_sali['n_original']:,} molecules - sampled "
                        f"{_sali['n_molecules']:,} for the pairwise scan (runtime cap). "
                        + _scan_note
                    )
                else:
                    st.caption(_scan_note)
                if not _pairs.empty:
                    fig_sali = px.scatter(
                        _pairs,
                        x="tanimoto",
                        y="delta_pic50",
                        color="delta_pic50",
                        color_continuous_scale="plasma",
                        opacity=0.4,
                        title="Activity Cliffs: Structural Similarity vs Potency Difference",
                        labels={
                            "tanimoto":    "Tanimoto Similarity (ECFP4)",
                            "delta_pic50": "|ΔpIC50|",
                        },
                    )
                    fig_sali.add_vline(
                        x=0.4, line_dash="dash", line_color="#64748B",
                        annotation_text="Tanimoto = 0.4",
                        annotation_position="top right",
                    )
                    fig_sali.add_hline(
                        y=2.0, line_dash="dash", line_color="#64748B",
                        annotation_text="|ΔpIC50| = 2.0",
                        annotation_position="top left",
                    )
                    _apply_chart_style(fig_sali, height=500)
                    fig_sali.update_layout(
                        coloraxis_colorbar=dict(title="|ΔpIC50|"),
                    )
                    st.plotly_chart(fig_sali, use_container_width=True)

                    # ── (D) Top cliff pairs with structures ────────────────────
                    try:
                        _cliff_pairs_df = get_activity_cliff_pairs(
                            df_working,
                            sim_threshold=0.6,
                            delta_threshold=1.0,
                            top_n=6,
                        )
                    except Exception as _cpe:
                        st.warning(f"Could not extract top cliff pairs: {_cpe}")
                        _cliff_pairs_df = pd.DataFrame()

                    if not _cliff_pairs_df.empty:
                        @st.fragment
                        def _cliff_pairs_block(cliff_pairs_df):
                            """Fragment: MCS strictness controls + cliff-pair gallery
                            re-render together, isolated from the full app rerun."""
                            st.subheader("Top Activity Cliff Pairs")
                            st.caption(
                                "Pairs ranked by **Tanimoto × |ΔpIC50|** (stronger cliffs "
                                "first). The shared scaffold (Maximum Common Substructure) "
                                "is drawn in default colours; the differing **R-groups** are "
                                "highlighted in coral, so the structural change driving the "
                                "activity cliff jumps out."
                            )

                            with st.expander("⚙️ MCS strictness", expanded=False):
                                _mcs_c1, _mcs_c2 = st.columns(2)
                                with _mcs_c1:
                                    atom_cmp = st.selectbox(
                                        "Atom match",
                                        options=["elements", "any", "isotopes"],
                                        index=0,
                                        key="mcs_atom_cmp",
                                        help=(
                                            "'elements' = require same atom type. "
                                            "'any' = match any atom (broader scaffolds). "
                                            "'isotopes' = also require isotope match."
                                        ),
                                    )
                                with _mcs_c2:
                                    bond_cmp = st.selectbox(
                                        "Bond match",
                                        options=["order_exact", "order", "any"],
                                        index=0,
                                        key="mcs_bond_cmp",
                                        help=(
                                            "'order_exact' = require identical bond orders "
                                            "(strictest). 'order' = same nominal order. "
                                            "'any' = any bond matches (broadest)."
                                        ),
                                    )
                                ring_only = st.checkbox(
                                    "Require complete rings (recommended)",
                                    value=True,
                                    key="mcs_ring_only",
                                )

                            has_pair_ids = "id_1" in cliff_pairs_df.columns
                            for _i, _p in cliff_pairs_df.iterrows():
                                _img_1, _img_2, _mcs_size = _build_mcs_pair_images(
                                    _p["smiles_1"], _p["smiles_2"],
                                    atom_compare=atom_cmp,
                                    bond_compare=bond_cmp,
                                    complete_rings_only=ring_only,
                                    ring_matches_ring_only=ring_only,
                                )
                                _mcs_note = (
                                    f"shared scaffold = {_mcs_size} atoms"
                                    if _mcs_size else "no MCS found"
                                )
                                st.markdown(
                                    f"**Pair {_i + 1}** · Tanimoto = **{_p['tanimoto']:.2f}** "
                                    f"· |ΔpIC50| = **{_p['delta_pic50']:.2f}** "
                                    f"· {_mcs_note}"
                                )
                                if _img_1 is None or _img_2 is None:
                                    _plain = _build_image_column(
                                        (_p["smiles_1"], _p["smiles_2"]),
                                        size=(260, 200),
                                    )
                                    _img_1 = _img_1 or _plain[0]
                                    _img_2 = _img_2 or _plain[1]
                                _pc1, _pc2 = st.columns(2)
                                for _col_, _img, _pic, _which in (
                                    (_pc1, _img_1, _p["pIC50_1"], "1"),
                                    (_pc2, _img_2, _p["pIC50_2"], "2"),
                                ):
                                    with _col_:
                                        if _img:
                                            st.image(_img, use_container_width=True)
                                        _id = _p[f"id_{_which}"] if has_pair_ids else "—"
                                        st.caption(f"**{_id}**  · pIC50 {_pic:.2f}")
                                st.divider()

                        _cliff_pairs_block(_cliff_pairs_df)

        # ── Chirality Cliffs (stereochemistry vs potency) ──────────────────────
        if has_smiles and has_pic50:
            st.subheader("Chirality Cliffs (Stereochemistry vs. Potency)")
            st.markdown(
                "Finds pairs that are **identical except for stereochemistry** "
                "(enantiomers, diastereomers, or E/Z isomers) yet differ in "
                "potency. Because the *only* difference is 3D configuration, these "
                "are the cleanest possible evidence that stereochemistry drives "
                "binding - and can flag data where a racemate and a pure enantiomer "
                "were conflated."
            )
            _cc_thresh = st.slider(
                "Cliff threshold  |ΔpIC50| ≥",
                min_value=0.5, max_value=3.0, value=1.0, step=0.5,
                key="chir_thresh",
                help="A stereoisomer pair is flagged a cliff when their pIC50 "
                     "differ by at least this much.",
            )
            if st.button("🔬 Find Chirality Cliffs", key="btn_chir"):
                with st.spinner("Scanning for stereoisomer pairs…"):
                    try:
                        st.session_state["_chir_result"] = run_chirality_cliffs(
                            df_working, delta_threshold=_cc_thresh
                        )
                    except ValueError as _ce:
                        st.error(f"Chirality cliff computation failed: {_ce}")

            if "_chir_result" in st.session_state:
                _chir = st.session_state["_chir_result"]
                _cp = _chir["pairs_df"]

                if _chir["truncated"]:
                    st.info(
                        f"Dataset has {_chir['n_original']:,} molecules - "
                        f"randomly sampled 5,000 for the stereo scan."
                    )

                # Cliffs recomputed live against the current slider threshold
                _cliffs = _cp[_cp["delta_pic50"] >= _cc_thresh] if not _cp.empty else _cp

                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("Stereo-bearing molecules", f"{_chir['n_stereo_molecules']:,}")
                _mc2.metric("Stereoisomer pairs", f"{_chir['n_stereo_pairs']:,}")
                _mc3.metric(f"Cliffs (≥ {_cc_thresh:g})", f"{len(_cliffs):,}")

                if _chir["n_stereo_pairs"] == 0:
                    if _chir["n_stereo_molecules"] == 0:
                        st.success(
                            "No stereochemistry present - every molecule is achiral "
                            "or has undefined stereo, so chirality cliffs don't apply."
                        )
                    else:
                        st.info(
                            "Stereo-bearing molecules exist, but none share a 2D "
                            "structure with a *different* stereo variant - so there "
                            "are no stereoisomer pairs to compare."
                        )
                elif _cliffs.empty:
                    st.caption(
                        f"Found {_chir['n_stereo_pairs']:,} stereoisomer pair(s), but "
                        f"none differ by ≥ {_cc_thresh:g} pIC50. Lower the threshold "
                        "to see smaller stereo effects."
                    )
                else:
                    _has_ids = "id_1" in _cliffs.columns
                    st.caption(
                        "Each pair below is the same 2D structure drawn with its two "
                        "stereo configurations - the wedge/geometry difference is the "
                        "only change driving the potency gap."
                    )
                    for _i, _cr in _cliffs.head(8).reset_index(drop=True).iterrows():
                        st.markdown(
                            f"**Pair {_i + 1}** · |ΔpIC50| = **{_cr['delta_pic50']:.2f}**"
                        )
                        _cimgs = _build_image_column(
                            (_cr["smiles_1"], _cr["smiles_2"]), size=(260, 200)
                        )
                        _ccol1, _ccol2 = st.columns(2)
                        for _col_, _img, _pic, _w in (
                            (_ccol1, _cimgs[0], _cr["pIC50_1"], "1"),
                            (_ccol2, _cimgs[1], _cr["pIC50_2"], "2"),
                        ):
                            with _col_:
                                if _img:
                                    st.image(_img, use_container_width=True)
                                _lbl = _cr[f"id_{_w}"] if _has_ids else "—"
                                st.caption(f"**{_lbl}** · pIC50 {_pic:.2f}")
                        st.divider()

                    st.download_button(
                        label="⬇️ Download stereoisomer pairs (CSV)",
                        data=_cp.to_csv(index=False),
                        file_name="chirality_cliffs.csv",
                        mime="text/csv",
                        key="dl_chir",
                    )

    # ── Phase 4: Working Dataset ───────────────────────────────────────────────
    with st.container(border=True):
        _section_header(4, "Working Dataset")
        if _manual_removed:
            st.caption(
                f"Showing the working set ({len(df_working):,} rows) - "
                f"{_manual_removed:,} rows removed by manual edits. "
                "Use **↺ Reset** in the sidebar to restore the auto-curated dataset."
            )

        @st.fragment
        def _working_dataset_table():
            """Fragment: toggle + row-height slider + table all re-run together
            on their own widget changes, without triggering the rest of Phase 4-8."""
            df_w     = st.session_state["df_working"]
            smarts_h = st.session_state.get("smarts_highlight", "")
            show = st.toggle(
                "Show structures inline",
                value=True,
                key="wd_show_structures",
                help="Renders each molecule as an inline 2D structure image. "
                     "Disable for faster rendering on very large datasets.",
            )
            if not (show and "canonical_smiles" in df_w.columns):
                st.dataframe(df_w, use_container_width=True)
                return

            row_h = st.slider(
                "Structure row height (px)",
                min_value=80, max_value=240, value=160, step=20,
                key="wd_row_height",
                help="Larger rows make structures easier to read but mean fewer "
                     "rows fit on screen at once.",
            )
            disp = df_w.copy()
            size = (int(row_h * 1.6), row_h)
            imgs = (
                _build_image_column_with_smarts(
                    tuple(disp["canonical_smiles"]), smarts_h, size=size,
                )
                if smarts_h
                else _build_image_column(tuple(disp["canonical_smiles"]), size=size)
            )
            disp.insert(0, "structure", imgs)
            if smarts_h:
                st.caption(f"Highlighting atoms matching SMARTS `{smarts_h}` in amber.")
            st.dataframe(
                disp, use_container_width=True, row_height=row_h,
                column_config={
                    "structure": st.column_config.ImageColumn(
                        "Structure", width="large",
                    ),
                },
            )

        _working_dataset_table()

        st.download_button(
            label="⬇️ Download working dataset (CSV)",
            data=df_working.to_csv(index=False),
            file_name="chembl_working.csv",
            mime="text/csv",
        )

        #Structural validity audit ──────────────────────────────────────────
        if "canonical_smiles" in df_working.columns:
            with st.expander("🔬 Structural validity audit", expanded=False):
                st.caption(
                    "Scans every structure for likely problems: unparseable SMILES, "
                    "disconnected fragments (unstripped salts/mixtures), inorganic or "
                    "exotic-element species, radicals, isotopes, and net-charged "
                    "molecules. Flags are advisory - review before deciding to remove."
                )
                if st.button("Run structural audit", key="btn_audit"):
                    with st.spinner("Auditing structures…"):
                        st.session_state["_audit_df"] = run_structure_audit(df_working)

                if "_audit_df" in st.session_state:
                    _aud = st.session_state["_audit_df"]
                    _asum = get_structure_audit_summary(_aud)

                    _au1, _au2, _au3 = st.columns(3)
                    _au1.metric("Audited", f"{_asum['n_total']:,}")
                    _au2.metric("Flagged", f"{_asum['n_flagged']:,}")
                    _au3.metric("Clean", f"{_asum['n_clean']:,}")

                    if _asum["n_flagged"] == 0:
                        st.success("No structural issues detected - every molecule looks clean.")
                    else:
                        _ac1, _ac2 = st.columns([2, 3])
                        with _ac1:
                            st.markdown("**Issues by type**")
                            st.dataframe(
                                _asum["by_issue"], use_container_width=True, hide_index=True,
                            )
                        with _ac2:
                            st.markdown("**Flagged rows**")
                            _flagged = _aud[_aud["validity_issues"].str.len() > 0]
                            _show_cols = [
                                c for c in ("molecule_chembl_id", "canonical_smiles",
                                            "validity_issues")
                                if c in _flagged.columns
                            ]
                            st.dataframe(
                                _flagged[_show_cols].head(50),
                                use_container_width=True, hide_index=True,
                            )

                        if st.button(
                            "🚫 Remove all flagged rows from working set",
                            key="btn_remove_flagged",
                        ):
                            _clean = (
                                _aud[_aud["validity_issues"].str.len() == 0]
                                .drop(columns=["validity_issues"])
                                .reset_index(drop=True)
                            )
                            st.session_state["df_working"] = _clean
                            st.session_state.pop("_audit_df", None)
                            st.success(
                                f"Removed {_asum['n_flagged']:,} flagged rows. "
                                f"Working set now has {len(_clean):,} rows."
                            )
                            st.rerun()

    # ── Phase 5: Molecular Descriptors ────────────────────────────────────────
    if has_smiles:
        with st.container(border=True):
            _section_header(5, "Molecular Descriptors")
            st.markdown(
                "Computes **9 standard RDKit descriptors**: MW, LogP, TPSA, HBA, HBD, "
                "HeavyAtomCount, RotatableBonds, RingCount, AromaticRings."
            )
            st.caption(
                "💡 Need more? Ask the **sidebar chat** - e.g. *“add QED and "
                "FractionCSP3 columns”* or *“add the MaxPartialCharge descriptor”*. "
                "RDKit is available inside the chat's exec environment, so any of "
                "the 200+ `Descriptors.descList` entries can be appended on demand."
            )

            if st.button("⚗️ Compute Descriptors"):
                with st.spinner("Computing RDKit descriptors..."):
                    try:
                        st.session_state["df_with_descriptors"] = run_descriptor_computation(
                            df_working
                        )
                    except ImportError as e:
                        st.error(str(e))

            if "df_with_descriptors" in st.session_state:
                df_desc = st.session_state["df_with_descriptors"]
                all_desc_cols = [c for c in df_desc.columns if c not in df_working.columns]

                kept_cols = st.multiselect(
                    "Remove any descriptor columns you don't need:",
                    options=all_desc_cols,
                    default=all_desc_cols,
                )

                df_final = df_desc[list(df_working.columns) + kept_cols]

                # Render molecule images for the working dataset
                _desc_imgs = _build_image_column(tuple(df_final["canonical_smiles"]))
                df_final_display = df_final.copy()
                df_final_display.insert(0, "structure", _desc_imgs)
                st.dataframe(
                    df_final_display,
                    column_config={"structure": st.column_config.ImageColumn("Structure", width="large")},
                    use_container_width=True,
                )
                st.download_button(
                    label="⬇️ Download dataset with descriptors (CSV)",
                    data=df_final.to_csv(index=False),
                    file_name="chembl_curated_descriptors.csv",
                    mime="text/csv",
                )

                # ── Descriptor distribution grid ───────────────────────────────
                _desc_only_cols = [
                    c for c in df_desc.columns
                    if c not in df_working.columns
                    and pd.api.types.is_numeric_dtype(df_desc[c])
                ]
                if _desc_only_cols:
                    st.subheader("Descriptor Distributions")
                    st.caption(
                        "Box = interquartile range (25th–75th percentile); the line "
                        "inside is the median and the dashed line the mean. Whiskers "
                        "span 1.5×IQR; points beyond are outliers."
                    )
                    _grid_rows = [
                        _desc_only_cols[i:i + 3]
                        for i in range(0, len(_desc_only_cols), 3)
                    ]
                    for _row_cols in _grid_rows:
                        _gcols = st.columns(3)
                        for _gc, _dc in zip(_gcols, _row_cols):
                            with _gc:
                                _col_data = df_desc[_dc].dropna()
                                if not _col_data.empty:
                                    _fb = px.box(
                                        y=_col_data, points="outliers",
                                        title=_dc,
                                        color_discrete_sequence=_PALETTE[:1],
                                    )
                                    _fb.update_traces(boxmean=True)
                                    _apply_chart_style(_fb, height=280)
                                    _fb.update_layout(
                                        showlegend=False,
                                        margin=dict(t=40, b=20, l=20, r=20),
                                        yaxis_title=_dc, xaxis_title="",
                                    )
                                    st.plotly_chart(_fb, use_container_width=True)

                # ── Descriptor ↔ pIC50 correlation ─────────────────────────────
                if "pIC50" in df_desc.columns and _desc_only_cols:
                    corr_df = get_descriptor_correlations(df_desc, _desc_only_cols)
                    if not corr_df.empty:
                        st.subheader("Descriptor ↔ pIC50 Correlation")
                        st.caption(
                            "Pearson correlation of each descriptor with pIC50, ranked "
                            "by strength. Bars to the right mean the descriptor rises "
                            "with potency; to the left, it falls. Spearman ρ (table) "
                            "captures monotonic trends robust to outliers."
                        )

                        _c1, _c2 = st.columns([3, 2])
                        with _c1:
                            _plot_df = corr_df.sort_values("pearson_r")
                            _fig = px.bar(
                                _plot_df, x="pearson_r", y="descriptor",
                                orientation="h", color="pearson_r",
                                color_continuous_scale="RdBu", range_color=[-1, 1],
                                title="Pearson r vs pIC50",
                            )
                            _apply_chart_style(_fig, height=max(300, 42 * len(_plot_df)))
                            _fig.update_layout(
                                coloraxis_showscale=False,
                                xaxis_title="Pearson r", yaxis_title="",
                                xaxis_range=[-1, 1],
                            )
                            _fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
                            st.plotly_chart(_fig, use_container_width=True)
                        with _c2:
                            _show = corr_df.copy()
                            _show["pearson_p"] = _show["pearson_p"].apply(lambda v: f"{v:.2g}")
                            st.dataframe(
                                _show, use_container_width=True, hide_index=True,
                                column_config={
                                    "descriptor":   st.column_config.TextColumn("Descriptor"),
                                    "pearson_r":    st.column_config.NumberColumn("Pearson r", format="%.2f"),
                                    "pearson_p":    st.column_config.TextColumn("p-value"),
                                    "spearman_rho": st.column_config.NumberColumn("Spearman ρ", format="%.2f"),
                                    "n":            st.column_config.NumberColumn("n", format="%d"),
                                },
                            )
                            _top = corr_df.iloc[0]
                            _dir = "positively" if _top["pearson_r"] > 0 else "negatively"
                            _strength = (
                                "strong" if abs(_top["pearson_r"]) >= 0.5
                                else "moderate" if abs(_top["pearson_r"]) >= 0.3
                                else "weak"
                            )
                            st.caption(
                                f"Strongest: **{_top['descriptor']}** — {_strength}, "
                                f"{_dir} correlated (r = {_top['pearson_r']:+.2f})."
                            )

    # ── Phase 6: Chemical Space ────────────────────────────────────────────────
    if has_smiles:
        with st.container(border=True):
            _section_header(6, "Chemical Space")

            # ── (C) Combined report card — synthesises PCA + cluster metrics ───
            _pca_insights_cached = None
            _cluster_insights_cached = None
            if "_pca_result" in st.session_state:
                try:
                    _pca_insights_cached = get_pca_insights(
                        st.session_state["_pca_result"]
                    )
                except Exception:
                    _pca_insights_cached = None
            if "_tsne_result" in st.session_state:
                try:
                    _cluster_insights_cached = get_cluster_insights(
                        st.session_state["_tsne_result"]
                    )
                except Exception:
                    _cluster_insights_cached = None

            _have_any = (_pca_insights_cached is not None) or (_cluster_insights_cached is not None)
            with st.expander("📋 Chemical-space report card", expanded=_have_any):
                _summary = get_chemical_space_summary(
                    _pca_insights_cached, _cluster_insights_cached
                )
                st.markdown(_summary)
                if not _have_any:
                    st.caption(
                        "Run PCA and/or t-SNE in the tabs below - the report card "
                        "updates automatically as results become available."
                    )

            # ── (C) Dataset-wide scaffold gallery ──────────────────────────────
            with st.expander("🧪 Top scaffolds in the dataset", expanded=False):
                st.caption(
                    "Most common Bemis-Murcko scaffolds across the working set, "
                    "ranked by frequency. Click to inspect."
                )
                try:
                    _scaf_df = get_top_scaffolds(df_working, top_n=10)
                except Exception as _sse:
                    st.warning(f"Could not compute scaffolds: {_sse}")
                    _scaf_df = pd.DataFrame()

                if _scaf_df.empty:
                    st.info("No parseable scaffolds found in this dataset.")
                else:
                    _scaf_imgs = _build_image_column(
                        tuple(_scaf_df["scaffold_smiles"].tolist()),
                        size=(240, 180),
                    )
                    _ncols = 5
                    _rows = [_scaf_df.iloc[i:i + _ncols] for i in range(0, len(_scaf_df), _ncols)]
                    _img_idx = 0
                    for _row in _rows:
                        _cols = st.columns(len(_row))
                        for _col, (_, _r) in zip(_cols, _row.iterrows()):
                            with _col:
                                if _scaf_imgs[_img_idx]:
                                    st.image(_scaf_imgs[_img_idx], use_container_width=True)
                                _meta = f"**{int(_r['count'])} mols** ({_r['pct']:.0f}%)"
                                if "mean_pIC50" in _row.columns and pd.notna(_r.get("mean_pIC50")):
                                    _meta += f" · pIC50 {_r['mean_pIC50']:.1f}"
                                if "pct_active" in _row.columns and pd.notna(_r.get("pct_active")):
                                    _meta += f" · {_r['pct_active']:.0f}% active"
                                st.caption(_meta)
                            _img_idx += 1

            # ── R-group decomposition (SMARTS-driven SAR table) ────────────────
            with st.expander("🧩 R-group decomposition", expanded=False):
                st.caption(
                    "Define a **core SMARTS** to decompose every matching "
                    "molecule into core + R-groups. Use this to build a SAR "
                    "table — e.g. core = a pyrimidine scaffold, then look at "
                    "what R-groups correlate with potency. Examples: "
                    "`c1ccncn1` (pyrimidine), `c1ccc2[nH]cnc2c1` (benzimidazole)."
                )

                _rg_input = st.text_input(
                    "Core SMARTS",
                    value="",
                    placeholder="e.g. c1ccncn1",
                    key="rgroup_core_input",
                )

                if _rg_input.strip():
                    _ok, _err = validate_smarts_pattern(_rg_input)
                    if not _ok:
                        st.error(f"Invalid SMARTS: {_err}")
                    else:
                        if st.button("▶ Decompose", key="btn_rgroup_decompose"):
                            with st.spinner("Decomposing R-groups…"):
                                try:
                                    st.session_state["_rgroup_result"] = (
                                        run_rgroup_decomposition(df_working, _rg_input)
                                    )
                                except Exception as _re:
                                    st.error(f"Decomposition failed: {_re}")

                if "_rgroup_result" in st.session_state:
                    _rg = st.session_state["_rgroup_result"]
                    _rg_df = _rg["rgroups_df"]

                    _rm1, _rm2, _rm3 = st.columns(3)
                    _rm1.metric("Decomposed", f"{_rg['n_decomposed']:,}")
                    _rm2.metric("Unmatched",  f"{_rg['n_unmatched']:,}")
                    _rm3.metric("R-groups",   f"{len(_rg['r_columns'])}")

                    if _rg_df.empty:
                        st.info(
                            "No molecules matched the core. Try a less restrictive "
                            "SMARTS pattern (e.g. drop ring constraints, allow "
                            "any-atom positions with `*`)."
                        )
                    else:
                        # Render structures for each row
                        st.markdown("**Decomposition table**")
                        _rg_disp = _rg_df.copy()

                        # Compound structure
                        if "canonical_smiles" in _rg_disp.columns:
                            _rg_disp.insert(
                                1, "structure",
                                _build_image_column(
                                    tuple(_rg_disp["canonical_smiles"]),
                                    size=(180, 140),
                                ),
                            )

                        # R-group structures
                        for _rc in _rg["r_columns"]:
                            if _rc in _rg_disp.columns:
                                _rg_disp[f"{_rc}_img"] = _build_image_column(
                                    tuple(_rg_disp[_rc].fillna("")),
                                    size=(140, 110),
                                )

                        _column_config = {}
                        if "structure" in _rg_disp.columns:
                            _column_config["structure"] = st.column_config.ImageColumn(
                                "Compound", width="medium",
                            )
                        for _rc in _rg["r_columns"]:
                            if f"{_rc}_img" in _rg_disp.columns:
                                _column_config[f"{_rc}_img"] = st.column_config.ImageColumn(
                                    _rc, width="small",
                                )
                        _column_config["canonical_smiles"] = st.column_config.TextColumn(
                            "SMILES", width="medium",
                        )

                        # Hide the raw R-group SMARTS columns from the main table
                        # they're useful for export but visually noisy
                        _hidden = _rg["r_columns"] + ["Core"]
                        _visible = [c for c in _rg_disp.columns if c not in _hidden]

                        st.dataframe(
                            _rg_disp[_visible],
                            use_container_width=True,
                            hide_index=True,
                            row_height=150,
                            column_config=_column_config,
                        )

                        # If pIC50 exists, show per-R-group activity summary
                        if "pIC50" in _rg_df.columns:
                            st.markdown("**Per-R-group activity summary**")
                            _sar_tabs = st.tabs([f"📊 {rc}" for rc in _rg["r_columns"]])
                            for _i, _rc in enumerate(_rg["r_columns"]):
                                with _sar_tabs[_i]:
                                    _sar = (
                                        _rg_df
                                        .dropna(subset=[_rc, "pIC50"])
                                        .groupby(_rc)["pIC50"]
                                        .agg(["count", "mean", "median", "std"])
                                        .reset_index()
                                        .rename(columns={
                                            _rc:      "R-group SMILES",
                                            "count":  "n",
                                            "mean":   "mean_pIC50",
                                            "median": "median_pIC50",
                                            "std":    "std_pIC50",
                                        })
                                        .sort_values("mean_pIC50", ascending=False)
                                    )
                                    if not _sar.empty:
                                        _sar["mean_pIC50"]   = _sar["mean_pIC50"].round(2)
                                        _sar["median_pIC50"] = _sar["median_pIC50"].round(2)
                                        _sar["std_pIC50"]    = _sar["std_pIC50"].round(2)
                                        _sar.insert(
                                            0, "R-group",
                                            _build_image_column(
                                                tuple(_sar["R-group SMILES"]),
                                                size=(140, 110),
                                            ),
                                        )
                                        st.dataframe(
                                            _sar,
                                            use_container_width=True,
                                            hide_index=True,
                                            row_height=120,
                                            column_config={
                                                "R-group": st.column_config.ImageColumn(
                                                    "R-group", width="small",
                                                ),
                                            },
                                        )

                        st.download_button(
                            label="⬇️ Download R-group decomposition (CSV)",
                            data=_rg_df.to_csv(index=False),
                            file_name="rgroup_decomposition.csv",
                            mime="text/csv",
                            key="dl_rgroup",
                        )

            # ── Functional-group impact (model-free SAR) ───────────────────────
            if "pIC50" in df_working.columns:
                with st.expander("🧪 Functional-group impact on activity", expanded=False):
                    st.caption(
                        "Model-free SAR: for each common functional group, molecules "
                        "are split by whether they contain it, and the **mean pIC50 "
                        "difference** (present − absent) is measured with a Mann–Whitney "
                        "significance test. A positive impact means the group is "
                        "associated with *higher* potency in this dataset."
                    )
                    if st.button("▶ Analyse functional groups", key="btn_fg"):
                        with st.spinner("Screening functional groups…"):
                            st.session_state["_fg_impact"] = run_functional_group_impact(df_working)

                    if "_fg_impact" in st.session_state:
                        _fg = st.session_state["_fg_impact"]
                        if _fg.empty:
                            st.info(
                                "No functional group had enough molecules on both "
                                "sides of the split to compare - try a larger dataset."
                            )
                        else:
                            _fg1, _fg2 = st.columns([3, 2])
                            with _fg1:
                                _fg_plot = _fg.sort_values("impact")
                                _fig_fg = px.bar(
                                    _fg_plot, x="impact", y="group",
                                    orientation="h", color="impact",
                                    color_continuous_scale="RdBu", range_color=[
                                        -_fg["impact"].abs().max(), _fg["impact"].abs().max()
                                    ],
                                    title="Impact on pIC50  (present − absent)",
                                )
                                _apply_chart_style(_fig_fg, height=max(320, 26 * len(_fg_plot)))
                                _fig_fg.update_layout(
                                    coloraxis_showscale=False,
                                    xaxis_title="Δ mean pIC50", yaxis_title="",
                                )
                                _fig_fg.add_vline(x=0, line_color="#94A3B8", line_width=1)
                                st.plotly_chart(_fig_fg, use_container_width=True)
                            with _fg2:
                                _fg_show = _fg[[
                                    "group", "n_present", "impact", "p_value",
                                ]].copy()
                                _fg_show["p_value"] = _fg_show["p_value"].apply(
                                    lambda v: f"{v:.2g}" if pd.notna(v) else "—"
                                )
                                st.dataframe(
                                    _fg_show, use_container_width=True, hide_index=True,
                                    column_config={
                                        "group":     st.column_config.TextColumn("Group"),
                                        "n_present": st.column_config.NumberColumn("n", format="%d"),
                                        "impact":    st.column_config.NumberColumn("Δ pIC50", format="%.2f"),
                                        "p_value":   st.column_config.TextColumn("p"),
                                    },
                                )

                            # Highlighted example for the top impactful group
                            _sig = _fg[_fg["p_value"] < 0.05] if _fg["p_value"].notna().any() else _fg
                            _top_fg = (_sig if not _sig.empty else _fg).iloc[0]
                            _ex = get_example_with_group(df_working, _top_fg["smarts"])
                            if _ex:
                                _direction = "raises" if _top_fg["impact"] > 0 else "lowers"
                                st.markdown(
                                    f"**{_top_fg['group']}** most strongly {_direction} "
                                    f"potency here (Δ = {_top_fg['impact']:+.2f}). "
                                    f"Example molecule with the group highlighted:"
                                )
                                _ex_img = _build_image_column_with_smarts(
                                    (_ex,), _top_fg["smarts"], size=(300, 220)
                                )[0]
                                if _ex_img:
                                    st.image(_ex_img, width=320)

                            st.download_button(
                                label="⬇️ Download functional-group impact (CSV)",
                                data=_fg.to_csv(index=False),
                                file_name="functional_group_impact.csv",
                                mime="text/csv",
                                key="dl_fg",
                            )

            pca_tab, tsne_tab, pharm_tab = st.tabs(
                ["🔵 PCA", "🟠 t-SNE + K-means", "🔴 Pharmacophore"]
            )

            # ── PCA tab ───────────────────────────────────────────────────────
            with pca_tab:
                st.markdown(
                    "PCA of physicochemical descriptors (MW, LogP, TPSA, "
                    "RotatableBonds, HBA, HBD). Descriptors are computed automatically "
                    "if not already present."
                )
                if st.button("▶ Run PCA", key="btn_pca"):
                    with st.spinner("Running PCA…"):
                        try:
                            st.session_state["_pca_result"] = run_chemical_space_pca(df_working)
                        except Exception as e:
                            st.error(f"PCA failed: {e}")

                if "_pca_result" in st.session_state:
                    pca_res = st.session_state["_pca_result"]
                    pc_df = pca_res["pc_df"]
                    loadings = pca_res["loadings"]
                    explained_var = pca_res["explained_var"]
                    cum_var = pca_res["cum_var"]
                    n_pcs = pca_res["n_components"]

                    st.caption(
                        f"{pca_res['n_samples']} molecules · "
                        f"{n_pcs} components · "
                        f"PC1+PC2 explain "
                        f"{(explained_var[0] + explained_var[1]) * 100:.1f}% of variance"
                    )

                    col_scree, col_biplot = st.columns(2)

                    # Scree plot
                    with col_scree:
                        pc_labels = [f"PC{i + 1}" for i in range(n_pcs)]
                        fig_scree = go.Figure()
                        fig_scree.add_bar(
                            x=pc_labels,
                            y=(explained_var * 100).tolist(),
                            name="Explained variance",
                            marker_color=_PALETTE[0],
                        )
                        fig_scree.add_scatter(
                            x=pc_labels,
                            y=(cum_var * 100).tolist(),
                            name="Cumulative",
                            mode="lines+markers",
                            line=dict(color=_PALETTE[3], width=2),
                        )
                        _apply_chart_style(fig_scree, height=400)
                        fig_scree.update_layout(
                            title="Scree Plot",
                            xaxis_title="Principal Component",
                            yaxis_title="% Variance Explained",
                            legend=dict(orientation="h", y=-0.25),
                        )
                        st.plotly_chart(fig_scree, use_container_width=True)

                    # Biplot
                    with col_biplot:
                        hover_cols = [
                            c for c in ["molecule_chembl_id", "pIC50", "canonical_smiles"]
                            if c in pc_df.columns
                        ]
                        color_col = "pIC50" if "pIC50" in pc_df.columns else None
                        pc1_label = f"PC1 ({explained_var[0] * 100:.1f}%)"
                        pc2_label = f"PC2 ({explained_var[1] * 100:.1f}%)"

                        fig_biplot = go.Figure()

                        # Scatter of normalised scores
                        scatter_kw = dict(
                            x=pc_df["PC1_norm"],
                            y=pc_df["PC2_norm"],
                            mode="markers",
                            marker=dict(size=7, opacity=0.7),
                            customdata=pc_df[hover_cols].values if hover_cols else None,
                            hovertemplate=(
                                "<br>".join(f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols))
                                + "<extra></extra>"
                            ) if hover_cols else None,
                            name="Molecules",
                        )
                        if color_col:
                            scatter_kw["marker"]["color"] = pc_df[color_col]
                            scatter_kw["marker"]["colorscale"] = "Viridis"
                            scatter_kw["marker"]["showscale"] = True
                            scatter_kw["marker"]["colorbar"] = dict(title="pIC50", thickness=12)
                        fig_biplot.add_scatter(**scatter_kw)

                        # Loading vectors + labels
                        for feat in pca_res["feature_names"]:
                            lx, ly = loadings.loc[feat, "PC1"], loadings.loc[feat, "PC2"]
                            fig_biplot.add_annotation(
                                ax=0, ay=0,
                                x=lx, y=ly,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True,
                                arrowhead=3, arrowsize=1.2,
                                arrowwidth=1.5, arrowcolor="#1E293B",
                            )
                            fig_biplot.add_annotation(
                                x=lx * 1.2, y=ly * 1.2,
                                text=feat,
                                showarrow=False,
                                font=dict(size=12, color="#1E293B"),
                            )

                        # Unit circle
                        theta = np.linspace(0, 2 * np.pi, 200)
                        fig_biplot.add_scatter(
                            x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
                            mode="lines",
                            line=dict(color="#CBD5E1", dash="dash", width=1),
                            showlegend=False,
                            hoverinfo="skip",
                        )

                        _apply_chart_style(fig_biplot, height=500)
                        fig_biplot.update_layout(
                            title="PCA Biplot",
                            xaxis=dict(
                                title=pc1_label, zeroline=True,
                                zerolinecolor="#CBD5E1", range=[-1.35, 1.35],
                            ),
                            yaxis=dict(
                                title=pc2_label, zeroline=True,
                                zerolinecolor="#CBD5E1", scaleanchor="x",
                                scaleratio=1, range=[-1.35, 1.35],
                            ),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_biplot, use_container_width=True)

                    # ── (A) PCA insight cards ─────────────────────────────────
                    try:
                        _pi = get_pca_insights(pca_res)
                    except Exception as _pie:
                        st.warning(f"Could not compute PCA insights: {_pie}")
                        _pi = None

                    if _pi:
                        st.subheader("Insights")
                        _ic1, _ic2, _ic3, _ic4 = st.columns(4)
                        _ic1.metric(
                            "Effective dimensions",
                            f"{_pi['effective_dim']}",
                            help="Number of PCs required for cumulative variance ≥ 90 %.",
                        )
                        _pc1_dom = _pi["dominant_features"].get("PC1") or []
                        _pc1_label = (
                            f"{_pc1_dom[0][0]} ({_pc1_dom[0][1]:+.2f})"
                            if _pc1_dom else "—"
                        )
                        _ic2.metric("PC1 driver", _pc1_label,
                                    help="Feature with the largest |loading| on PC1.")
                        # Strongest pIC50 correlation
                        if _pi["pic50_correlations"]:
                            _best_corr = max(_pi["pic50_correlations"], key=lambda c: abs(c["r"]))
                            _sig = " *" if _best_corr["p"] < 0.05 else ""
                            _ic3.metric(
                                f"pIC50 ↔ {_best_corr['PC']}",
                                f"r = {_best_corr['r']:+.2f}{_sig}",
                                help=f"p = {_best_corr['p']:.3g}. Asterisk = p < 0.05.",
                            )
                        else:
                            _ic3.metric("pIC50 ↔ PC", "-",
                                        help="pIC50 column missing or insufficient values.")
                        _ic4.metric(
                            "Outliers (> 3σ)",
                            f"{_pi['n_outliers']}",
                            help="Compounds beyond 3σ of the PC1-PC2 centroid - review candidates.",
                        )

                        # Full pIC50-correlation table + outlier list + quadrant breakdown
                        _details_cols = st.columns(3) if _pi["quadrant_enrichment"] is not None else st.columns(2)

                        with _details_cols[0]:
                            if _pi["pic50_correlations"]:
                                st.markdown("**pIC50 ↔ PC correlations**")
                                _corr_df = pd.DataFrame(_pi["pic50_correlations"])
                                _corr_df["r"] = _corr_df["r"].round(3)
                                _corr_df["p"] = _corr_df["p"].apply(lambda x: f"{x:.2g}")
                                st.dataframe(_corr_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("**pIC50 ↔ PC correlations**")
                                st.caption("Not available (no pIC50 column).")

                        with _details_cols[1]:
                            st.markdown("**Outlier compounds**")
                            if _pi["outlier_ids"]:
                                _olist = ", ".join(_pi["outlier_ids"])
                                if _pi["n_outliers"] > len(_pi["outlier_ids"]):
                                    _olist += f" *(showing first {len(_pi['outlier_ids'])} of {_pi['n_outliers']})*"
                                st.caption(_olist)
                            else:
                                st.caption("No compounds beyond 3σ - distribution is well-concentrated.")

                        if _pi["quadrant_enrichment"] is not None:
                            with _details_cols[2]:
                                st.markdown(f"**Active enrichment by quadrant**")
                                _qdf = _pi["quadrant_enrichment"].copy()
                                st.dataframe(_qdf, use_container_width=True, hide_index=True)
                                if _pi["best_quadrant"]:
                                    st.caption(
                                        f"Highest active fraction: **{_pi['best_quadrant']}**"
                                    )

                    # ── (B) PCA outlier structures ────────────────────────────
                    try:
                        _outliers = get_pca_outliers(pca_res, max_outliers=6)
                    except Exception:
                        _outliers = []
                    if _outliers and "canonical_smiles" in pc_df.columns:
                        st.subheader("Outlier Structures")
                        st.caption(
                            "Compounds beyond 3σ in PC1-PC2 - review to decide whether they "
                            "represent genuine chemical diversity or annotation errors."
                        )
                        _out_imgs = _build_image_column(
                            tuple(o["canonical_smiles"] for o in _outliers),
                            size=(220, 170),
                        )
                        _ncols = min(6, len(_outliers))
                        _cols  = st.columns(_ncols)
                        for _ci_, _o, _img in zip(_cols, _outliers, _out_imgs):
                            with _ci_:
                                if _img:
                                    st.image(_img, use_container_width=True)
                                _label = _o.get("molecule_chembl_id", "-")
                                _meta = f"**{_label}**  · z={_o['radial_z']:.2f}"
                                if _o.get("pIC50") is not None:
                                    _meta += f"  · pIC50 {_o['pIC50']:.2f}"
                                st.caption(_meta)

                    # ── (E) PC axis extremes ──────────────────────────────────
                    try:
                        _ext = get_pca_axis_extremes(pca_res)
                    except Exception:
                        _ext = {}
                    if _ext:
                        st.subheader("PC Axis Extremes")
                        st.caption(
                            "The compounds at each end of PC1 / PC2 - concrete "
                            "examples of what each axis represents."
                        )
                        for _pc in ("PC1", "PC2"):
                            _max_key, _min_key = f"{_pc}_max", f"{_pc}_min"
                            if _max_key not in _ext or _min_key not in _ext:
                                continue
                            _drv = _pi["dominant_features"].get(_pc, []) if _pi else []
                            _drv_lbl = (
                                f" — top driver: **{_drv[0][0]}** ({_drv[0][1]:+.2f})"
                                if _drv else ""
                            )
                            st.markdown(f"**{_pc} axis**{_drv_lbl}")
                            _ec1, _ec2 = st.columns(2)
                            for _col_, _key, _arrow in (
                                (_ec1, _max_key, "↑ HIGH"),
                                (_ec2, _min_key, "↓ LOW"),
                            ):
                                _e = _ext[_key]
                                _img_list = _build_image_column(
                                    (_e["canonical_smiles"],), size=(260, 200)
                                )
                                with _col_:
                                    st.markdown(f"*{_arrow} {_pc} = {_e['pc_value']:+.2f}*")
                                    if _img_list[0]:
                                        st.image(_img_list[0])
                                    _label = _e.get("molecule_chembl_id", "—")
                                    _meta  = f"**{_label}**"
                                    if _e.get("pIC50") is not None:
                                        _meta += f"  · pIC50 {_e['pIC50']:.2f}"
                                    st.caption(_meta)

            # ── t-SNE tab ─────────────────────────────────────────────────────
            with tsne_tab:
                st.markdown(
                    "t-SNE embedding based on MACCS-key Tanimoto distances. "
                    "Molecules are compared by structural features, not physicochemical properties. "
                    "K-means clustering (k chosen by silhouette score) is overlaid."
                )

                _col_perp, _col_dims = st.columns([3, 1])
                with _col_perp:
                    perplexity = st.slider(
                        "Perplexity", min_value=5, max_value=100, value=30, step=5,
                        help="Controls the balance between local and global structure. "
                             "Typical values: 5–50. Will be clamped if the dataset is small.",
                    )
                with _col_dims:
                    tsne_dims = st.radio("Dimensions", ["2D", "3D"], horizontal=True)
                n_components = 3 if tsne_dims == "3D" else 2

                # Invalidate cached t-SNE if perplexity or dimensions changed
                _tsne_config = (perplexity, n_components)
                if st.session_state.get("_tsne_config") != _tsne_config:
                    st.session_state.pop("_tsne_result", None)
                    st.session_state["_tsne_config"] = _tsne_config

                if st.button("▶ Run t-SNE + K-means", key="btn_tsne"):
                    with st.spinner("Running t-SNE and K-means… (this may take a moment)"):
                        try:
                            st.session_state["_tsne_result"] = run_chemical_space_tsne(
                                df_working, perplexity=perplexity, n_components=n_components
                            )
                        except Exception as e:
                            st.error(f"t-SNE failed: {e}")

                if "_tsne_result" in st.session_state:
                    tsne_res = st.session_state["_tsne_result"]
                    tsne_df = tsne_res["tsne_df"]
                    sil_scores = tsne_res["silhouette_scores"]
                    best_k = tsne_res["best_k"]
                    eff_perp = tsne_res["effective_perp"]

                    if eff_perp != perplexity:
                        st.info(
                            f"Perplexity was clamped from {perplexity} → {eff_perp} "
                            f"(dataset has {tsne_res['n_samples']} molecules)."
                        )

                    st.caption(
                        f"{tsne_res['n_samples']} molecules · "
                        f"perplexity = {eff_perp} · "
                        f"best k = {best_k} clusters "
                        f"(silhouette = {sil_scores.get(best_k, 0):.3f})"
                    )

                    col_sil, col_tsne = st.columns(2)

                    # Silhouette score bar chart
                    with col_sil:
                        if sil_scores:
                            ks = list(sil_scores.keys())
                            scores = list(sil_scores.values())
                            colours = [_PALETTE[0] if k == best_k else "#CBD5E1" for k in ks]
                            fig_sil = go.Figure(go.Bar(
                                x=[str(k) for k in ks],
                                y=scores,
                                marker_color=colours,
                                hovertemplate="k=%{x}<br>silhouette=%{y:.3f}<extra></extra>",
                            ))
                            _apply_chart_style(fig_sil, height=400)
                            fig_sil.update_layout(
                                title=f"Silhouette Scores (best k={best_k})",
                                xaxis_title="Number of clusters (k)",
                                yaxis_title="Silhouette score",
                            )
                            st.plotly_chart(fig_sil, use_container_width=True)

                    # t-SNE scatter (2D or 3D)
                    with col_tsne:
                        hover_cols_tsne = [
                            c for c in ["molecule_chembl_id", "pIC50", "canonical_smiles"]
                            if c in tsne_df.columns
                        ]
                        _n_comp = tsne_res.get("n_components", 2)
                        if _n_comp == 3:
                            fig_tsne = px.scatter_3d(
                                tsne_df,
                                x="TC1", y="TC2", z="TC3",
                                color="Cluster",
                                hover_data={c: True for c in hover_cols_tsne},
                                title=f"t-SNE (3D) coloured by K-means cluster (k={best_k})",
                                labels={"TC1": "tSNE 1", "TC2": "tSNE 2", "TC3": "tSNE 3"},
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                height=600,
                            )
                            fig_tsne.update_traces(marker=dict(size=4, opacity=0.8))
                        else:
                            fig_tsne = px.scatter(
                                tsne_df,
                                x="TC1", y="TC2",
                                color="Cluster",
                                hover_data={c: True for c in hover_cols_tsne},
                                title=f"t-SNE coloured by K-means cluster (k={best_k})",
                                labels={"TC1": "tSNE 1", "TC2": "tSNE 2"},
                                color_discrete_sequence=px.colors.qualitative.Set2,
                                height=500,
                            )
                            fig_tsne.update_traces(marker=dict(size=7, opacity=0.8))
                        _apply_chart_style(fig_tsne)
                        fig_tsne.update_layout(legend_title_text="Cluster")
                        st.plotly_chart(fig_tsne, use_container_width=True)

                    # ── (B) Per-cluster insights ───────────────────────────────
                    try:
                        _ci = get_cluster_insights(tsne_res)
                    except Exception as _cie:
                        st.warning(f"Could not compute cluster insights: {_cie}")
                        _ci = None

                    if _ci and not _ci["per_cluster_df"].empty:
                        st.subheader("Cluster Insights")

                        # Winner cards
                        _wc1, _wc2, _wc3, _wc4 = st.columns(4)
                        _wc1.metric(
                            "Most potent",
                            f"Cluster {_ci['most_potent_cluster']}" if _ci["most_potent_cluster"] is not None else "-",
                            help="Cluster with the highest mean pIC50.",
                        )
                        _wc2.metric(
                            "Hit-enriched",
                            f"Cluster {_ci['hit_enriched_cluster']}" if _ci["hit_enriched_cluster"] is not None else "-",
                            help="Cluster with the highest % of compounds at pIC50 ≥ 5.",
                        )
                        _wc3.metric(
                            "Most diverse",
                            f"Cluster {_ci['most_diverse_cluster']}" if _ci["most_diverse_cluster"] is not None else "-",
                            help="Cluster with the widest pIC50 range.",
                        )
                        _wc4.metric(
                            "Singleton clusters",
                            f"{len(_ci['singleton_clusters'])}",
                            help="Clusters of size 1 - likely outliers worth manual review.",
                        )

                        if _ci["singleton_clusters"]:
                            st.caption(
                                f"⚠️ Singleton cluster ID(s): {', '.join(_ci['singleton_clusters'])}"
                            )

                        # Per-cluster table
                        st.markdown("**Per-cluster breakdown**")
                        st.dataframe(
                            _ci["per_cluster_df"],
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ── (A) Per-cluster structure gallery ──────────────────
                        try:
                            _chem = get_cluster_chemistry(tsne_res, max_samples=3)
                        except Exception as _cce:
                            st.warning(f"Could not build cluster chemistry: {_cce}")
                            _chem = {}

                        if _chem:
                            st.markdown("**Per-cluster structures**")
                            st.caption(
                                "For each cluster: the dominant Bemis-Murcko scaffold "
                                "(left) and representative compounds that share it."
                            )
                            _per_cluster = _ci["per_cluster_df"].set_index("Cluster")
                            for _cid, _data in _chem.items():
                                if not _data.get("dominant_scaffold"):
                                    continue
                                _stats = _per_cluster.loc[_cid] if _cid in _per_cluster.index else None

                                # Cluster header
                                _header_bits = [f"**Cluster {_cid}**"]
                                if _stats is not None:
                                    _header_bits.append(f"n = {int(_stats['Size'])}")
                                    if pd.notna(_stats.get("Mean pIC50")):
                                        _header_bits.append(f"mean pIC50 {_stats['Mean pIC50']:.2f}")
                                    if pd.notna(_stats.get("% active")):
                                        _header_bits.append(f"{_stats['% active']:.0f}% active")
                                _header_bits.append(
                                    f"dominant scaffold shared by **{_data['scaffold_pct']:.0f}%** "
                                    f"of cluster ({_data['n_unique']} unique scaffolds)"
                                )
                                st.markdown(" · ".join(_header_bits))

                                _scaf_img = _build_image_column(
                                    (_data["dominant_scaffold"],), size=(220, 170)
                                )[0]
                                _sample_smis = tuple(s["smiles"] for s in _data["samples"])
                                _sample_imgs = _build_image_column(_sample_smis, size=(220, 170))

                                _cluster_cols = st.columns(1 + len(_data["samples"]))
                                with _cluster_cols[0]:
                                    if _scaf_img:
                                        st.image(_scaf_img, use_container_width=True)
                                    st.caption("**Dominant scaffold**")
                                for _scol, _samp, _simg in zip(
                                    _cluster_cols[1:], _data["samples"], _sample_imgs
                                ):
                                    with _scol:
                                        if _simg:
                                            st.image(_simg, use_container_width=True)
                                        _id    = _samp.get("molecule_chembl_id") or "-"
                                        _pic   = _samp.get("pIC50")
                                        _meta  = f"**{_id}**"
                                        if _pic is not None:
                                            _meta += f"  · pIC50 {_pic:.2f}"
                                        st.caption(_meta)
                                st.divider()

                    # ── Cluster activity profiles ──────────────────────────────
                    if (
                        "pIC50" in tsne_df.columns
                        and tsne_df["pIC50"].notna().any()
                    ):
                        st.subheader("pIC50 Distribution by K-means Cluster")
                        fig_clust = px.box(
                            tsne_df.dropna(subset=["pIC50"]),
                            x="Cluster",
                            y="pIC50",
                            color="Cluster",
                            points="outliers",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            title="pIC50 Distribution by K-means Cluster",
                            labels={"Cluster": "Cluster", "pIC50": "pIC50"},
                        )
                        _apply_chart_style(fig_clust, height=420)
                        fig_clust.update_layout(showlegend=False)
                        st.plotly_chart(fig_clust, use_container_width=True)

            # ── Pharmacophore tab ─────────────────────────────────────────────
            with pharm_tab:
                st.markdown(
                    "Detects pharmacophore features per molecule using RDKit's "
                    "BaseFeatures definition file: H-bond **Donors** (blue), "
                    "**Acceptors** (red), **Hydrophobic** centres (amber), "
                    "**Aromatic** rings (violet), **+Ionisable** (teal) and "
                    "**−Ionisable** (orange) groups."
                )

                if st.button("▶ Compute Pharmacophore Profiles", key="btn_pharm"):
                    with st.spinner("Detecting pharmacophore features…"):
                        try:
                            st.session_state["_pharm_result"] = run_pharmacophore_profiles(
                                df_working
                            )
                        except Exception as _pe:
                            st.error(f"Pharmacophore computation failed: {_pe}")

                if "_pharm_result" in st.session_state:
                    _pharm   = st.session_state["_pharm_result"]
                    _prof_df = _pharm["profiles_df"]
                    _means   = _pharm["family_means"]

                    if _pharm["truncated"]:
                        st.info(
                            f"Dataset has {_pharm['n_original']:,} molecules - "
                            f"randomly sampled 500 for profile computation."
                        )
                    else:
                        st.caption(f"{_pharm['n_molecules']:,} molecules profiled.")

                    # ── 1. Dataset-level feature summary bar chart ────────────
                    st.subheader("Dataset Pharmacophore Profile")
                    _summary_df = pd.DataFrame({
                        "Feature":    list(_means.keys()),
                        "Mean count": list(_means.values()),
                    })
                    _summary_fig = px.bar(
                        _summary_df,
                        x="Feature", y="Mean count",
                        title="Mean pharmacophore feature count per molecule",
                        color="Feature",
                        color_discrete_sequence=list(_PHARM_COLORS_HEX.values()),
                    )
                    _apply_chart_style(_summary_fig, height=340)
                    _summary_fig.update_layout(showlegend=False)
                    st.plotly_chart(_summary_fig, use_container_width=True)

                    # ── 2. PCA scatter coloured by selected feature ───────────
                    if "_pca_result" in st.session_state:
                        @st.fragment
                        def _pharm_scatter_fragment(prof_df, pc_df_orig, ev):
                            """Fragment: feature selector + scatter re-render together,
                            isolated from the rest of the chemical-space tabs."""
                            st.subheader("Chemical Space Coloured by Pharmacophore Feature")
                            families_present = [
                                f for f in _PHARM_COLORS_HEX.keys() if f in prof_df.columns
                            ]
                            if len(prof_df) != len(pc_df_orig):
                                st.caption(
                                    "Could not align pharmacophore profiles with the PCA result - "
                                    "re-run PCA after computing pharmacophore profiles."
                                )
                                return
                            pc_df_pharm = pc_df_orig.copy()
                            for _fam in families_present:
                                pc_df_pharm[_fam] = prof_df[_fam].values

                            color_by = st.selectbox(
                                "Colour scatter by",
                                options=families_present,
                                key="pharm_color_select",
                            )
                            hover_cols = [
                                c for c in ("molecule_chembl_id", "canonical_smiles", "pIC50")
                                if c in pc_df_pharm.columns
                            ]
                            pharm_scatter = px.scatter(
                                pc_df_pharm,
                                x="PC1_norm", y="PC2_norm",
                                color=color_by,
                                color_continuous_scale="Blues",
                                hover_data={c: True for c in hover_cols},
                                title=f"PCA — coloured by {color_by} count",
                                labels={
                                    "PC1_norm": f"PC1 ({ev[0] * 100:.1f}%)",
                                    "PC2_norm": f"PC2 ({ev[1] * 100:.1f}%)",
                                    color_by:   color_by,
                                },
                                opacity=0.78,
                            )
                            pharm_scatter.update_traces(marker=dict(size=8))
                            _apply_chart_style(pharm_scatter, height=460)
                            pharm_scatter.update_layout(
                                coloraxis_colorbar=dict(title=color_by),
                            )
                            st.plotly_chart(pharm_scatter, use_container_width=True)

                        _pharm_scatter_fragment(
                            _prof_df,
                            st.session_state["_pca_result"]["pc_df"],
                            st.session_state["_pca_result"]["explained_var"],
                        )
                    else:
                        st.info(
                            "Run **PCA** first to see the pharmacophore-coloured "
                            "chemical-space scatter."
                        )

                    # ── 3. Structure gallery with highlighted features ────────
                    st.subheader("Highlighted Structures")
                    _legend_cols = st.columns(len(_PHARM_COLORS_HEX))
                    for _lc, (_fam, _hex) in zip(_legend_cols, _PHARM_COLORS_HEX.items()):
                        _lc.markdown(
                            f'<span style="display:inline-block;width:12px;height:12px;'
                            f'background:{_hex};border-radius:3px;margin-right:4px;'
                            f'vertical-align:middle;"></span>'
                            f'**{_fam}**',
                            unsafe_allow_html=True,
                        )

                    if "canonical_smiles" in _prof_df.columns:
                        _gallery_smis = (
                            _prof_df["canonical_smiles"]
                            .dropna()
                            .sample(min(12, len(_prof_df)), random_state=7)
                            .tolist()
                        )
                        if _gallery_smis:
                            _gallery_imgs = _build_pharmacophore_images(tuple(_gallery_smis))
                            _img_idx = 0
                            _grid_rows = [
                                _gallery_smis[i:i + 4]
                                for i in range(0, len(_gallery_smis), 4)
                            ]
                            for _row_smis in _grid_rows:
                                _gcols = st.columns(len(_row_smis))
                                for _gc, _smi in zip(_gcols, _row_smis):
                                    _img = _gallery_imgs[_img_idx]
                                    if _img:
                                        _gc.image(_img, use_container_width=True)
                                    else:
                                        _gc.caption("(image unavailable)")
                                    _img_idx += 1

    # ── Phase 7: Bioactivity Model ─────────────────────────────────────────────
    def _render_model_results(res: dict, key_prefix: str) -> None:
        """Render Performance / Feature Importance / Predictions / History tabs
        for a completed model run. key_prefix avoids duplicate widget keys."""

        _tab_perf, _tab_feat, _tab_shap, _tab_pred, _tab_err, _tab_score, _tab_hist = st.tabs([
            "📊 Performance",
            "🔬 Feature Importance",
            "🧠 Explainability",
            "📋 Predictions",
            "🔎 Error Analysis",
            "🧪 Score New",
            "📈 History",
        ])

        # ── Performance ────────────────────────────────────────────────────────
        with _tab_perf:
            mode_label = "Tuned" if res.get("mode") == "tuned" else "Quick"
            st.caption(
                f"**{res['model_type']}** ({mode_label})  ·  "
                f"Trained: {res['timestamp']}  ·  "
                f"Train n = {res['n_train']}, Test n = {res['n_test']}"
            )

            # Best params banner (tuned only)
            if res.get("best_params"):
                _bp_lines = "  |  ".join(
                    f"**{k}** = {v}" for k, v in res["best_params"].items()
                )
                st.info(f"Best hyperparameters found:  {_bp_lines}")

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric("Test R²",  f"{res['test_r2']:.4f}")
            _mc2.metric("Test RMSE", f"{res['test_rmse']:.4f}")
            _mc3.metric("Test MAE",  f"{res['test_mae']:.4f}")
            _mc4.metric(
                f"CV R² ({res['cv_folds']}-fold)",
                f"{res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}",
            )

            _preds = res["predictions_df"]
            _lo = float(min(_preds["actual_pIC50"].min(), _preds["predicted_pIC50"].min()))
            _hi = float(max(_preds["actual_pIC50"].max(), _preds["predicted_pIC50"].max()))
            _avp_fig = go.Figure()
            _avp_fig.add_scatter(
                x=_preds["actual_pIC50"].tolist(),
                y=_preds["predicted_pIC50"].tolist(),
                mode="markers",
                marker=dict(size=7, opacity=0.7, color=_PALETTE[1]),
                name="Test molecules",
                hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
            )
            _avp_fig.add_scatter(
                x=[_lo, _hi], y=[_lo, _hi],
                mode="lines",
                line=dict(color=_PALETTE[5], dash="dash", width=1.5),
                name="Perfect fit", hoverinfo="skip",
            )
            _apply_chart_style(_avp_fig, height=480)
            _avp_fig.update_layout(
                title="Actual vs Predicted pIC50 (test set)",
                xaxis_title="Actual pIC50", yaxis_title="Predicted pIC50",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(_avp_fig, use_container_width=True)

            _residuals = _preds["predicted_pIC50"] - _preds["actual_pIC50"]
            _resid_fig = go.Figure()
            _resid_fig.add_scatter(
                x=_preds["predicted_pIC50"].tolist(),
                y=_residuals.tolist(),
                mode="markers",
                marker=dict(size=7, opacity=0.7, color=_PALETTE[1]),
                hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
            )
            _resid_fig.add_hline(y=0, line_dash="dash",
                                  line_color=_PALETTE[5], line_width=1.5)
            _apply_chart_style(_resid_fig, height=420)
            _resid_fig.update_layout(
                title="Residuals (Predicted − Actual)",
                xaxis_title="Predicted pIC50",
                yaxis_title="Residual",
                showlegend=False,
            )
            st.plotly_chart(_resid_fig, use_container_width=True)
            st.caption(
                "Residuals should scatter randomly around zero. "
                "A funnel shape → heteroscedasticity; a curve → missing non-linear structure."
            )

        # ── Feature importance ─────────────────────────────────────────────────
        with _tab_feat:
            if res["feature_importances"] is not None:
                _fi = res["feature_importances"]
                _top_idx = np.argsort(_fi)[::-1][:20]
                _fi_fig = go.Figure(go.Bar(
                    x=[f"Bit {b}" for b in _top_idx],
                    y=_fi[_top_idx].tolist(),
                    marker_color=_PALETTE[2],
                    hovertemplate="ECFP4 bit %{x}<br>Importance: %{y:.6f}<extra></extra>",
                ))
                _apply_chart_style(_fi_fig, height=450)
                _fi_fig.update_layout(
                    title=f"Top 20 ECFP4 Bit Importances - {res['model_type']}",
                    xaxis_title="Fingerprint Bit",
                    yaxis_title="Importance (MDI)",
                    xaxis=dict(tickangle=-45),
                )
                st.plotly_chart(_fi_fig, use_container_width=True)
                st.caption(
                    "Mean decrease in impurity (MDI). Higher = more predictive. "
                    "Bits correspond to Morgan (ECFP4) circular substructure features."
                )
            else:
                st.info(
                    "Linear models (Ridge / Elastic Net) don't expose tree-style "
                    "feature importances. Use **Random Forest** or **Gradient "
                    "Boosting** for bit importances, or the **Explainability** tab "
                    "(SHAP) which works for every model."
                )

        # ── SHAP Explainability ────────────────────────────────────────────────
        with _tab_shap:
            st.markdown(
                "**SHAP attribution** identifies which Morgan fingerprint bits "
                "drove each prediction. Each bit corresponds to an atom "
                "neighbourhood (substructure) - so high-impact bits become "
                "visible chemistry."
            )

            _shap_key = f"_shap_{key_prefix}"
            _can_explain = (
                res.get("X_train") is not None and
                res.get("X_test")  is not None and
                res.get("model")   is not None
            )
            if not _can_explain:
                st.info("Re-train the model to enable SHAP - the cached result was produced before SHAP was wired in.")
            else:
                if st.button("▶ Compute SHAP values", key=f"btn_shap_{key_prefix}"):
                    with st.spinner("Computing SHAP attributions…"):
                        try:
                            st.session_state[_shap_key] = compute_shap_explanation(res)
                        except Exception as _se:
                            st.error(f"SHAP computation failed: {_se}")

                if _shap_key in st.session_state:
                    _shap_res = st.session_state[_shap_key]
                    _is_morgan = res.get("fp_is_morgan", True)
                    _fp_radius = res.get("fp_radius", 2)
                    # When mRMR reduced the features, SHAP bit ids index the
                    # reduced space — map them back to original Morgan bits for
                    # the substructure lookup.
                    _sel_feats = res.get("selected_features")

                    def _orig_bit(reduced_bit):
                        return int(_sel_feats[reduced_bit]) if _sel_feats is not None else int(reduced_bit)

                    if not _is_morgan:
                        st.info(
                            f"This model uses a **{res.get('fingerprint', 'non-Morgan')}** "
                            "fingerprint, whose bits don't map to a single atom "
                            "neighbourhood - so the per-bit **structure** images below "
                            "are omitted. The SHAP values themselves are still valid. "
                            "Switch to an ECFP (Morgan) fingerprint to see the "
                            "substructure behind each bit."
                        )

                    # ── Global view: top contributing bits across the test set ─
                    st.subheader("Global - most important bits")
                    st.caption(
                        f"Ranked by mean |SHAP| across {len(_shap_res.shap_values):,} "
                        f"test molecules. Explainer: **{_shap_res.explainer_type}**."
                    )

                    _gtop = get_global_top_bits(_shap_res, top_n=12)
                    _test_smiles = res.get("test_smiles", [])

                    # For each top bit, render the substructure on a sample molecule
                    # that contains it
                    _X_test = res["X_test"]
                    _bit_images: list = []
                    for _bid in _gtop["bit_id"].tolist():
                        # Find the first test molecule with this bit ON
                        _sample_smi = None
                        for _mi, _smi in enumerate(_test_smiles):
                            if _mi >= _X_test.shape[0]:
                                break
                            if _X_test[_mi, _bid]:
                                _sample_smi = _smi
                                break
                        if _sample_smi and _is_morgan:
                            _atoms = get_atoms_for_bit(_sample_smi, _orig_bit(_bid), radius=_fp_radius)
                            _img = _highlight_atoms_image(_sample_smi, _atoms)
                            _bit_images.append((_bid, _sample_smi, _atoms, _img))
                        else:
                            _bit_images.append((_bid, None, [], None))

                    # Two rows of six
                    _ncols = 6
                    for _row_start in range(0, len(_bit_images), _ncols):
                        _row = _bit_images[_row_start:_row_start + _ncols]
                        _cols = st.columns(len(_row))
                        for _col_, (_bid, _smi, _atoms, _img) in zip(_cols, _row):
                            _bit_row = _gtop[_gtop["bit_id"] == _bid].iloc[0]
                            _signed   = _bit_row["mean_signed_shap"]
                            _arrow    = "↑" if _signed > 0 else "↓"
                            with _col_:
                                if _img:
                                    st.image(_img, use_container_width=True)
                                else:
                                    st.caption("(no exemplar)")
                                st.caption(
                                    f"**Bit {_bid}**  ·  {_arrow} {_signed:+.3f}  "
                                    f"(|μ|={_bit_row['mean_abs_shap']:.3f})"
                                )
                                # SMARTS pattern for this bit
                                _bit_smarts = (
                                    get_morgan_bit_smarts(_smi, _orig_bit(_bid), radius=_fp_radius)
                                    if (_smi and _is_morgan) else None
                                )
                                if _bit_smarts:
                                    st.code(_bit_smarts, language="text")

                    st.divider()

                    # ── Local view: per-molecule explanation ─────────────────
                    st.subheader("Local - per-molecule attribution")
                    if not _test_smiles:
                        st.info("Test SMILES are unavailable for this cached result.")
                    else:
                        # Build a list of options (index → label)
                        _preds_df = res["predictions_df"]
                        _has_ids  = "molecule_chembl_id" in _preds_df.columns

                        # The predictions_df is reordered by |error| desc; we need
                        # SHAP indices which are aligned to X_test (original test order).
                        # Map predictions back via SMILES match.
                        _smi_to_test_idx = {s: i for i, s in enumerate(_test_smiles)}

                        _options = []
                        for _r_idx, _p in _preds_df.iterrows():
                            _label_parts = []
                            if _has_ids:
                                _label_parts.append(str(_p["molecule_chembl_id"]))
                            _label_parts.append(
                                f"pred {_p['predicted_pIC50']:.2f}  ·  err {_p['error']:+.2f}"
                            )
                            _options.append(" — ".join(_label_parts))

                        _sel = st.selectbox(
                            "Pick a test molecule",
                            options=range(len(_options)),
                            format_func=lambda i: _options[i],
                            key=f"shap_select_{key_prefix}",
                        )

                        # Look up which test row this corresponds to
                        _picked = _preds_df.iloc[_sel]
                        _pick_smi = _picked.get("canonical_smiles")
                        _shap_row_idx = _smi_to_test_idx.get(_pick_smi)

                        if _shap_row_idx is None or _pick_smi is None:
                            st.warning("Could not align this row to a SHAP value.")
                        else:
                            _shap_row = _shap_res.shap_values[_shap_row_idx]
                            _fp_row   = _X_test[_shap_row_idx]
                            _top_bits = get_top_bits_for_molecule(
                                _shap_res, _fp_row, _shap_row_idx, top_n=6
                            )

                            # Header strip
                            _id_str = str(_picked["molecule_chembl_id"]) if _has_ids else "—"
                            _hc1, _hc2, _hc3, _hc4 = st.columns(4)
                            _hc1.metric("Molecule", _id_str)
                            _hc2.metric("Actual",   f"{_picked['actual_pIC50']:.3f}")
                            _hc3.metric("Predicted",f"{_picked['predicted_pIC50']:.3f}")
                            _hc4.metric("Error",    f"{_picked['error']:+.3f}")

                            # Render top bits as a row of highlighted structures
                            if not _top_bits.empty:
                                _bit_galls: list = []
                                for _, _br in _top_bits.iterrows():
                                    _b      = int(_br["bit_id"])
                                    _atoms  = (
                                        get_atoms_for_bit(_pick_smi, _orig_bit(_b), radius=_fp_radius)
                                        if (_br["present"] and _is_morgan) else []
                                    )
                                    _img    = (
                                        _highlight_atoms_image(_pick_smi, _atoms)
                                        if _is_morgan else None
                                    )
                                    _bit_galls.append((_b, _br, _atoms, _img))

                                _ngc = min(6, len(_bit_galls))
                                _cols = st.columns(_ngc)
                                for _col_, (_b, _br, _atoms, _img) in zip(_cols, _bit_galls):
                                    with _col_:
                                        if _img:
                                            st.image(_img, use_container_width=True)
                                        _sv = _br["shap_value"]
                                        _arrow = "↑" if _sv > 0 else "↓"
                                        _present = "on" if _br["present"] else "off"
                                        st.caption(
                                            f"**Bit {_b}**  ·  {_arrow} {_sv:+.3f}  ·  {_present}"
                                        )
                                        if _br["present"] and _is_morgan:
                                            _bit_smarts = get_morgan_bit_smarts(
                                                _pick_smi, _orig_bit(_b), radius=_fp_radius
                                            )
                                            if _bit_smarts:
                                                st.code(_bit_smarts, language="text")
                                st.caption(
                                    "Atoms in blue triggered the bit on this molecule. "
                                    "Up-arrows ↑ raised the predicted pIC50, down-arrows ↓ lowered it."
                                )

        # ── Predictions ────────────────────────────────────────────────────────
        with _tab_pred:
            st.caption(
                f"Test-set predictions ({res['n_test']} molecules), "
                "sorted by |error| descending — worst predictions first."
            )
            _disp = res["predictions_df"].copy()
            for _c in ("actual_pIC50", "predicted_pIC50", "error"):
                _disp[_c] = _disp[_c].round(4)
            st.dataframe(_disp, use_container_width=True)

        # ── Error Analysis ─────────────────────────────────────────────────────
        with _tab_err:
            st.markdown(
                "Diagnose **what the model gets wrong**. Are the worst-predicted "
                "compounds structurally unusual (outside the applicability domain), "
                "do they share physicochemical properties, or does the model just "
                "struggle at the activity extremes?"
            )
            _err_key = f"_erra_{key_prefix}"
            if st.button("▶ Run error analysis", key=f"btn_erra_{key_prefix}"):
                with st.spinner("Analysing prediction errors…"):
                    try:
                        st.session_state[_err_key] = analyze_model_errors(res)
                    except Exception as _ee:
                        st.error(f"Error analysis failed: {_ee}")

            if _err_key in st.session_state and st.session_state[_err_key]:
                _ea = st.session_state[_err_key]
                _edf = _ea["error_df"]

                # Headline read
                st.info(_ea["summary"])

                _em1, _em2, _em3 = st.columns(3)
                _em1.metric("High-error compounds",
                            f"{_ea['n_high']} / {len(_edf)}")
                _em2.metric("|error| threshold", f"{_ea['threshold']:.2f}")
                if _ea["ad_corr"]:
                    _sig = " *" if _ea["ad_corr"]["p"] < 0.05 else ""
                    _em3.metric("Similarity ↔ |error|",
                                f"r = {_ea['ad_corr']['r']:+.2f}{_sig}")
                else:
                    _em3.metric("Similarity ↔ |error|", "N/A")

                # ── 1. Applicability domain: similarity vs |error| ─────────────
                if _ea["has_ad"] and "nn_tanimoto" in _edf.columns:
                    st.subheader("Prediction Error vs Training Similarity")
                    st.caption(
                        "Each point is a test compound. A downward trend means the "
                        "model errs more on compounds dissimilar from its training "
                        "set - the classic applicability-domain effect."
                    )
                    _ad_plot = _edf.dropna(subset=["nn_tanimoto", "abs_error"])
                    _fig_ad = px.scatter(
                        _ad_plot, x="nn_tanimoto", y="abs_error",
                        color="error_group",
                        color_discrete_map={"High": _PALETTE[5], "Low": _PALETTE[1]},
                        hover_data=[c for c in ("molecule_chembl_id",)
                                    if c in _ad_plot.columns],
                        title="|error| vs nearest-neighbour Tanimoto",
                        labels={"nn_tanimoto": "Nearest-neighbour Tanimoto (to training set)",
                                "abs_error": "|error| (pIC50)"},
                        opacity=0.7,
                    )
                    _xv = _ad_plot["nn_tanimoto"].to_numpy(dtype=float)
                    _yv = _ad_plot["abs_error"].to_numpy(dtype=float)
                    if len(_ad_plot) >= 2 and np.std(_xv) > 0:
                        _s, _i = np.polyfit(_xv, _yv, 1)
                        _xln = np.array([_xv.min(), _xv.max()])
                        _fig_ad.add_scatter(
                            x=_xln, y=_s * _xln + _i, mode="lines",
                            line=dict(color="#64748B", dash="dash", width=2),
                            name="Trend",
                        )
                    _apply_chart_style(_fig_ad, height=420)
                    st.plotly_chart(_fig_ad, use_container_width=True)

                # ── 2. Descriptor ↔ |error| correlation ────────────────────────
                _dc = _ea["descriptor_corr"]
                if _dc is not None and not _dc.empty:
                    st.subheader("What Properties Track Prediction Error?")
                    _cA, _cB = st.columns([3, 2])
                    with _cA:
                        _dc_plot = _dc.sort_values("pearson_r")
                        _fig_dc = px.bar(
                            _dc_plot, x="pearson_r", y="descriptor",
                            orientation="h", color="pearson_r",
                            color_continuous_scale="RdBu", range_color=[-1, 1],
                            title="Pearson r  ( descriptor vs |error| )",
                        )
                        _apply_chart_style(_fig_dc, height=max(280, 42 * len(_dc_plot)))
                        _fig_dc.update_layout(coloraxis_showscale=False,
                                              xaxis_range=[-1, 1],
                                              xaxis_title="Pearson r", yaxis_title="")
                        _fig_dc.add_vline(x=0, line_color="#94A3B8", line_width=1)
                        st.plotly_chart(_fig_dc, use_container_width=True)
                    with _cB:
                        _gc = _ea["group_comparison"]
                        if _gc is not None and not _gc.empty:
                            st.markdown("**High vs low-error group means**")
                            st.dataframe(
                                _gc, use_container_width=True, hide_index=True,
                                column_config={
                                    "descriptor":      st.column_config.TextColumn("Descriptor"),
                                    "high_error_mean": st.column_config.NumberColumn("High-err mean", format="%.2f"),
                                    "low_error_mean":  st.column_config.NumberColumn("Low-err mean", format="%.2f"),
                                    "delta":           st.column_config.NumberColumn("Δ", format="%.2f"),
                                },
                            )

                # ── 3. Worst-predicted structures ──────────────────────────────
                _worst = _ea["worst_df"]
                if "canonical_smiles" in _worst.columns and not _worst.empty:
                    st.subheader("Worst-Predicted Compounds")
                    _wimgs = _build_image_column(
                        tuple(_worst["canonical_smiles"]), size=(200, 160)
                    )
                    _ncol = min(6, len(_worst))
                    _wcols = st.columns(_ncol)
                    for _wc, (_, _wr), _wi in zip(_wcols, _worst.iterrows(), _wimgs):
                        with _wc:
                            if _wi:
                                st.image(_wi, use_container_width=True)
                            _wid = _wr.get("molecule_chembl_id", "—")
                            _meta = (f"**{_wid}**  \n"
                                     f"act {_wr['actual_pIC50']:.2f} · "
                                     f"pred {_wr['predicted_pIC50']:.2f}  \n"
                                     f"err {_wr['error']:+.2f}")
                            if "nn_tanimoto" in _wr.index and pd.notna(_wr.get("nn_tanimoto")):
                                _meta += f" · NN {_wr['nn_tanimoto']:.2f}"
                            st.caption(_meta)

                    _hs = _ea["high_error_scaffolds"]
                    if _hs is not None and not _hs.empty and len(_hs) > 0:
                        st.caption(
                            f"Dominant scaffold among high-error compounds appears "
                            f"**{int(_hs.iloc[0]['count'])}×**."
                        )

        # ── Score New Compounds ────────────────────────────────────────────────
        with _tab_score:
            st.markdown(
                "Score your own compounds against this trained model. Paste one "
                "SMILES per line, then compare the predicted pIC50 against the "
                "distribution the model was trained on."
            )
            _can_score = (
                res.get("model") is not None and res.get("X_train") is not None
            )
            if not _can_score:
                st.info(
                    "This cached result predates the scoring feature. Re-train the "
                    "model (Quick or Detailed) to enable compound scoring."
                )
            else:
                _smiles_text = st.text_area(
                    "SMILES (one per line)",
                    height=140,
                    key=f"score_input_{key_prefix}",
                    placeholder="CC(=O)Oc1ccccc1C(=O)O\nCC(C)Cc1ccc(C(C)C(=O)O)cc1",
                )
                _ad_thresh = st.slider(
                    "Applicability-domain threshold (max Tanimoto to training set)",
                    min_value=0.10, max_value=0.60, value=0.30, step=0.05,
                    key=f"score_ad_{key_prefix}",
                    help="A compound whose nearest training-set neighbour is below "
                         "this similarity is flagged as outside the model's domain - "
                         "its prediction is an extrapolation and less reliable.",
                )

                if st.button("▶ Score compounds", key=f"btn_score_{key_prefix}"):
                    _lines = [
                        ln.strip() for ln in _smiles_text.splitlines() if ln.strip()
                    ]
                    if not _lines:
                        st.warning("Enter at least one SMILES string.")
                    else:
                        with st.spinner(f"Scoring {len(_lines)} compound(s)…"):
                            try:
                                _scored = score_new_compounds(
                                    _lines, res, ad_threshold=_ad_thresh
                                )
                                st.session_state[f"_score_result_{key_prefix}"] = _scored
                                _n_invalid = len(_lines) - len(_scored)
                                if _n_invalid > 0:
                                    st.warning(
                                        f"{_n_invalid} SMILES could not be parsed "
                                        "and were skipped."
                                    )
                            except ValueError as _se:
                                st.error(f"Scoring failed: {_se}")

                _score_key = f"_score_result_{key_prefix}"
                if _score_key in st.session_state:
                    _scored = st.session_state[_score_key]
                    if _scored.empty:
                        st.info("No valid compounds to display.")
                    else:
                        # Summary metrics
                        _n_in  = int(_scored["in_domain"].sum())
                        _n_out = len(_scored) - _n_in
                        _sm1, _sm2, _sm3 = st.columns(3)
                        _sm1.metric("Scored", f"{len(_scored)}")
                        _sm2.metric("In domain", f"{_n_in}")
                        _sm3.metric("Outside domain", f"{_n_out}")

                        # Results table with structures
                        _scored_disp = _scored.copy()
                        _scored_disp.insert(
                            0, "structure",
                            _build_image_column(
                                tuple(_scored_disp["input_smiles"]), size=(180, 140)
                            ),
                        )

                        # Nearest-neighbour structure column (when available)
                        _has_nn = "nn_smiles" in _scored_disp.columns
                        if _has_nn:
                            _scored_disp["nn_structure"] = _build_image_column(
                                tuple(_scored_disp["nn_smiles"]), size=(180, 140)
                            )
                        else:
                            st.info(
                                "ℹ️ Nearest-neighbour structures are unavailable because this "
                                "model was trained before the feature was added. **Re-train the "
                                "model** (after fully restarting Streamlit) to enable them."
                            )

                        _scored_disp["in_domain"] = _scored_disp["in_domain"].map(
                            {True: "✅ in", False: "⚠️ outside"}
                        )

                        # Order columns: input structure first, NN structure right
                        # after the Tanimoto so the comparison reads left-to-right
                        _col_order = [
                            "structure", "input_smiles", "predicted_pIC50",
                            "nn_tanimoto",
                        ]
                        if _has_nn:
                            _col_order += ["nn_structure"]
                            if "nn_id" in _scored_disp.columns:
                                _col_order += ["nn_id"]
                            if "nn_pIC50" in _scored_disp.columns:
                                _col_order += ["nn_pIC50"]
                        _col_order += ["in_domain"]
                        _col_order = [c for c in _col_order if c in _scored_disp.columns]

                        _col_config = {
                            "structure": st.column_config.ImageColumn(
                                "Your compound", width="medium",
                            ),
                            "input_smiles":    st.column_config.TextColumn("SMILES", width="medium"),
                            "predicted_pIC50": st.column_config.NumberColumn("Predicted pIC50", format="%.2f"),
                            "nn_tanimoto":     st.column_config.NumberColumn("NN Tanimoto", format="%.2f"),
                            "in_domain":       st.column_config.TextColumn("Domain"),
                        }
                        if _has_nn:
                            _col_config["nn_structure"] = st.column_config.ImageColumn(
                                "Nearest training neighbour", width="medium",
                            )
                            if "nn_id" in _scored_disp.columns:
                                _col_config["nn_id"] = st.column_config.TextColumn(
                                    "NN ChEMBL ID", width="small",
                                )
                            if "nn_pIC50" in _scored_disp.columns:
                                _col_config["nn_pIC50"] = st.column_config.NumberColumn(
                                    "NN measured pIC50", format="%.2f",
                                )

                        st.dataframe(
                            _scored_disp[_col_order],
                            use_container_width=True,
                            hide_index=True,
                            row_height=150,
                            column_config=_col_config,
                        )
                        if _has_nn:
                            st.caption(
                                "The **nearest training neighbour** is the most similar "
                                "compound the model was trained on. Compare its measured "
                                "pIC50 against the prediction for a sanity check - and the "
                                "lower the Tanimoto, the more the model is extrapolating."
                            )

                        # Position on the test-set scatter / distribution
                        _dist = get_training_pic50_distribution(res)
                        if _dist:
                            st.markdown("**Where your compounds fall vs the training distribution**")
                            _vfig = go.Figure()
                            # Training pIC50 as a violin/box backdrop
                            _vfig.add_box(
                                y=_dist["values"],
                                name="Training set",
                                marker_color=_PALETTE[1],
                                boxpoints="outliers",
                                line=dict(width=1.5),
                            )
                            # Scored compounds as overlaid points
                            _vfig.add_scatter(
                                x=["Training set"] * len(_scored),
                                y=_scored["predicted_pIC50"].tolist(),
                                mode="markers",
                                marker=dict(
                                    size=11,
                                    color=[
                                        _PALETTE[2] if d else _PALETTE[5]
                                        for d in _scored["in_domain"]
                                    ],
                                    line=dict(width=1.5, color="white"),
                                    symbol="diamond",
                                ),
                                name="Your compounds",
                                text=_scored["input_smiles"].tolist(),
                                hovertemplate=(
                                    "Predicted pIC50: %{y:.2f}<br>%{text}<extra></extra>"
                                ),
                            )
                            _apply_chart_style(_vfig, height=420)
                            _vfig.update_layout(
                                title="Predicted pIC50 vs training distribution",
                                yaxis_title="pIC50",
                                showlegend=True,
                                legend=dict(orientation="h", y=-0.15),
                            )
                            st.plotly_chart(_vfig, use_container_width=True)
                            st.caption(
                                "Green diamonds are inside the applicability domain; "
                                "red diamonds are extrapolations. Compounds far above the "
                                "training box are predicted more potent than anything the "
                                "model has seen - treat with caution."
                            )

                        st.download_button(
                            label="⬇️ Download scores (CSV)",
                            data=_scored.to_csv(index=False),
                            file_name="compound_scores.csv",
                            mime="text/csv",
                            key=f"dl_scores_{key_prefix}",
                        )

        # ── History ────────────────────────────────────────────────────────────
        with _tab_hist:
            if st.session_state.get("_model_history"):
                _hist_df = pd.DataFrame(st.session_state["_model_history"])
                st.dataframe(_hist_df, use_container_width=True)
                st.caption(
                    f"{len(_hist_df)} run(s) this session. "
                    "History resets when the working dataset changes."
                )
            else:
                st.info("No model runs recorded yet this session.")

        # ── Download ───────────────────────────────────────────────────────────
        _model_buf = io.BytesIO()
        pickle.dump(res["model"], _model_buf)
        _safe_name = res["model_type"].lower().replace(" ", "_")
        st.download_button(
            label="⬇️ Download trained model (.pkl)",
            data=_model_buf.getvalue(),
            file_name=f"bioactivity_{_safe_name}.pkl",
            mime="application/octet-stream",
            key=f"{key_prefix}_download_btn",
        )

    if has_smiles and "pIC50" in df_working.columns:
        with st.container(border=True):
            _section_header(7, "Bioactivity Model")
            st.markdown(
                "Predict **pIC50** from Morgan (ECFP4) fingerprints (2048-bit, radius 2). "
                "**Quick** trains immediately with sensible defaults. "
                "**Detailed** exposes full configuration and runs automated "
                "hyperparameter tuning to find the best settings for your dataset."
            )

            # ── Split method (shared between Quick + Detailed tabs) ────────────
            _SPLIT_LABELS = {
                "Random":                  "random",
                "Scaffold (Bemis-Murcko)": "scaffold",
                "Butina clustering":       "butina",
            }
            _sm_col, _sc_col = st.columns([2, 3])
            with _sm_col:
                _split_label = st.selectbox(
                    "Train/test split method",
                    options=list(_SPLIT_LABELS.keys()),
                    key="split_method",
                )
            _split_method = _SPLIT_LABELS[_split_label]

            # Butina cutoff only relevant when Butina is selected
            _cluster_cutoff = 0.35
            if _split_method == "butina":
                with _sc_col:
                    _cluster_cutoff = st.slider(
                        "Butina similarity cutoff (Tanimoto)",
                        min_value=0.40, max_value=0.90, value=0.65, step=0.05,
                        key="butina_sim_cutoff",
                    )
                    # Display as Tanimoto similarity for intuition, convert to
                    # distance for the backend.
                    _cluster_cutoff = 1.0 - _cluster_cutoff
            else:
                with _sc_col:
                    _split_descriptions = {
                        "random":   "Uniform random - most optimistic, easiest baseline.",
                        "scaffold": "Bemis–Murcko scaffolds split as whole groups - test gets rarer scaffolds.",
                    }
                    st.caption(_split_descriptions.get(_split_method, ""))

            # ── Fingerprint method (shared between Quick + Detailed) ───────────
            _fp_col, _fpd_col = st.columns([2, 3])
            with _fp_col:
                _fp_method = st.selectbox(
                    "Molecular fingerprint",
                    options=[
                        "ECFP4 (Morgan r2)", "ECFP6 (Morgan r3)",
                        "MACCS keys", "Atom Pair",
                    ],
                    key="fp_method",
                    help="How each molecule is encoded for the model. ECFP4 is the "
                         "standard default; try others to see if they improve accuracy.",
                )
            with _fpd_col:
                _fp_descriptions = {
                    "ECFP4 (Morgan r2)": "Circular substructures, radius 2 - the QSAR standard. Full SHAP bit→structure mapping.",
                    "ECFP6 (Morgan r3)": "Larger circular substructures, radius 3 - captures bigger motifs. SHAP mapping supported.",
                    "MACCS keys":        "166 predefined structural keys - compact, interpretable, but no per-bit structure mapping in SHAP.",
                    "Atom Pair":         "Encodes atom-pair distances - good for shape/topology. No per-bit structure mapping in SHAP.",
                }
                st.caption(_fp_descriptions.get(_fp_method, ""))

            # ── Feature selection (shared between Quick + Detailed) ────────────
            _fs_col, _fsn_col = st.columns([2, 3])
            with _fs_col:
                _fs_choice = st.selectbox(
                    "Feature selection",
                    options=["None", "mRMR (top-K bits)"],
                    key="feat_sel",
                    help="mRMR keeps the most informative, least-redundant "
                         "fingerprint bits - can sharpen linear models and speed "
                         "up training. Fit on the training fold only.",
                )
            _feature_selection = "mrmr" if _fs_choice.startswith("mRMR") else "none"
            _n_features = 100
            if _feature_selection == "mrmr":
                with _fsn_col:
                    _n_features = st.slider(
                        "Bits to keep (K)",
                        min_value=20, max_value=500, value=100, step=20,
                        key="feat_sel_k",
                    )
            else:
                with _fsn_col:
                    st.caption("Using the full fingerprint (no reduction).")

            _quick_tab, _detail_tab, _viz_tab = st.tabs(
                ["⚡ Quick", "🔬 Detailed", "📊 Optuna Visualisation"]
            )

            # ── Quick tab ──────────────────────────────────────────────────────
            with _quick_tab:
                st.markdown(
                    "Pick a model and train in one click. "
                    "Defaults: 20 % test split · 5-fold CV · 100 estimators."
                )
                _q_model = st.selectbox(
                    "Model",
                    options=["Random Forest", "Ridge Regression", "Elastic Net", "Gradient Boosting"],
                    key="quick_model_type",
                    help=(
                        "Random Forest: robust ensemble, feature importances. "
                        "Ridge: fast linear baseline. "
                        "Gradient Boosting: often highest accuracy but slower."
                    ),
                )
                if st.button("▶ Train", key="btn_quick_train"):
                    with st.spinner(f"Training {_q_model}…"):
                        try:
                            _qres = run_model_training(
                                df_working,
                                model_type=_q_model,
                                test_size=0.20,
                                n_estimators=100,
                                cv_folds=5,
                                split_method=_split_method,
                                cluster_cutoff=_cluster_cutoff,
                                fingerprint=_fp_method,
                                feature_selection=_feature_selection,
                                n_features=_n_features,
                            )
                            st.session_state["_model_result"] = _qres
                            _history_row = {
                                "Mode":       "Quick",
                                "Timestamp":  _qres["timestamp"],
                                "Model":      _qres["model_type"],
                                "Fingerprint": _fp_method,
                                "Split":      _split_label,
                                "Test %":     int(_qres["test_size"] * 100),
                                "CV folds":   _qres["cv_folds"],
                                "N train":    _qres["n_train"],
                                "N test":     _qres["n_test"],
                                "Test R²":    round(_qres["test_r2"],    4),
                                "RMSE":       round(_qres["test_rmse"],  4),
                                "MAE":        round(_qres["test_mae"],   4),
                                "CV R² mean": round(_qres["cv_r2_mean"], 4),
                                "CV R² std":  round(_qres["cv_r2_std"],  4),
                            }
                            if "_model_history" not in st.session_state:
                                st.session_state["_model_history"] = []
                            st.session_state["_model_history"].append(_history_row)
                            if _qres["n_invalid_smiles"] > 0:
                                st.warning(
                                    f"{_qres['n_invalid_smiles']} molecule(s) with "
                                    "unparseable SMILES were excluded."
                                )
                        except ValueError as _ve:
                            st.error(f"Training failed: {_ve}")

                if "_model_result" in st.session_state:
                    _render_model_results(
                        st.session_state["_model_result"], key_prefix="quick"
                    )

            # ── Detailed tab ───────────────────────────────────────────────────
            with _detail_tab:
                st.markdown(
                    "Configure the train/test split and cross-validation folds, "
                    "then let **Optuna** (TPE sampler) optimise the hyperparameters. "
                    "TPE learns from earlier trials to focus on promising regions of the search space - "
                    "more efficient than random search at the same trial budget."
                )

                _d_col1, _d_col2, _d_col3 = st.columns(3)
                with _d_col1:
                    _d_model = st.selectbox(
                        "Model",
                        options=["Random Forest", "Ridge Regression", "Elastic Net", "Gradient Boosting"],
                        key="detail_model_type",
                    )
                with _d_col2:
                    _d_test_pct = st.slider(
                        "Test split (%)",
                        min_value=10, max_value=40, value=20, step=5,
                        key="detail_test_split",
                        help="Percentage of molecules held out for evaluation.",
                    )
                with _d_col3:
                    _d_cv = st.slider(
                        "CV folds",
                        min_value=2, max_value=10, value=5,
                        key="detail_cv_folds",
                        help="Inner cross-validation folds used during the search.",
                    )

                if _d_model != "Ridge Regression":
                    _d_n_iter = st.slider(
                        "Optuna trials",
                        min_value=5, max_value=100, value=20, step=5,
                        key="detail_n_iter",
                        help=(
                            "Number of hyperparameter configurations Optuna evaluates. "
                            "TPE uses earlier trials to guide later ones, so more trials → "
                            "better optimisation. Each trial runs full CV, so wall time scales linearly."
                        ),
                    )
                    _param_preview = {
                        "Random Forest":     "n_estimators · max_depth · min_samples_split · max_features",
                        "Gradient Boosting": "n_estimators · learning_rate (log) · max_depth · subsample",
                        "Elastic Net":       "α (log) · l1_ratio (L1/L2 mix)",
                    }
                    st.caption(
                        f"Parameters optimised: {_param_preview.get(_d_model, '')}"
                    )
                else:
                    _d_n_iter = 15  # capped internally for Ridge — one param to tune
                    st.caption(
                        "Parameters optimised: **α** (regularisation strength) - "
                        "Optuna samples on a log scale from 10⁻³ to 10³."
                    )

                if st.button("▶ Run Optuna Tuning", key="btn_detail_train"):
                    with st.spinner(
                        f"Optuna optimising {_d_model} hyperparameters… this may take a moment."
                    ):
                        try:
                            _dres = run_model_training_tuned(
                                df_working,
                                model_type=_d_model,
                                test_size=_d_test_pct / 100.0,
                                cv_folds=_d_cv,
                                n_iter=_d_n_iter,
                                split_method=_split_method,
                                cluster_cutoff=_cluster_cutoff,
                                fingerprint=_fp_method,
                                feature_selection=_feature_selection,
                                n_features=_n_features,
                            )
                            st.session_state["_model_tuned_result"] = _dres
                            _history_row = {
                                "Mode":       "Tuned",
                                "Timestamp":  _dres["timestamp"],
                                "Model":      _dres["model_type"],
                                "Fingerprint": _fp_method,
                                "Split":      _split_label,
                                "Test %":     int(_dres["test_size"] * 100),
                                "CV folds":   _dres["cv_folds"],
                                "N train":    _dres["n_train"],
                                "N test":     _dres["n_test"],
                                "Test R²":    round(_dres["test_r2"],    4),
                                "RMSE":       round(_dres["test_rmse"],  4),
                                "MAE":        round(_dres["test_mae"],   4),
                                "CV R² mean": round(_dres["cv_r2_mean"], 4),
                                "CV R² std":  round(_dres["cv_r2_std"],  4),
                            }
                            if "_model_history" not in st.session_state:
                                st.session_state["_model_history"] = []
                            st.session_state["_model_history"].append(_history_row)
                            if _dres["n_invalid_smiles"] > 0:
                                st.warning(
                                    f"{_dres['n_invalid_smiles']} molecule(s) with "
                                    "unparseable SMILES were excluded."
                                )
                        except ValueError as _ve:
                            st.error(f"Tuning failed: {_ve}")

                if "_model_tuned_result" in st.session_state:
                    _dres = st.session_state["_model_tuned_result"]

                    # Top-10 candidate table
                    if _dres.get("tuning_cv_results") is not None:
                        with st.expander("🔎 Top hyperparameter candidates", expanded=False):
                            st.dataframe(
                                _dres["tuning_cv_results"], use_container_width=True
                            )

                    _render_model_results(_dres, key_prefix="tuned")

            # ── Optuna Visualisation tab ───────────────────────────────────────
            with _viz_tab:
                if "_model_tuned_result" not in st.session_state:
                    st.info(
                        "Run **Optuna tuning** in the 🔬 Detailed tab first - "
                        "visualisations populate from the most recent study."
                    )
                else:
                    _vres = st.session_state["_model_tuned_result"]
                    _study = _vres.get("study")

                    if _study is None:
                        st.warning(
                            "The most recent tuning result has no `study` object. "
                            "Likely cause: Streamlit cached the old backend module before "
                            "Optuna integration. **Fully restart Streamlit** (Ctrl+C → "
                            "`streamlit run app.py`), then click **▶ Run Optuna Tuning** again."
                        )
                        with st.expander("🔧 Debug - keys present in the cached result"):
                            st.write({
                                "keys": list(_vres.keys()),
                                "tuner": _vres.get("tuner", "(missing)"),
                                "mode":  _vres.get("mode",  "(missing)"),
                            })
                    else:
                        try:
                            import optuna.visualization as _ovis
                            _viz_ok = True
                        except ImportError as _vie:
                            st.error(f"Optuna visualisation unavailable: {_vie}")
                            _viz_ok = False

                        if _viz_ok:
                            _completed = [
                                t for t in _study.trials if t.state.name == "COMPLETE"
                            ]
                            _n_done = len(_completed)
                            _n_params = len(_study.best_params)
                            _model_label = _vres["model_type"]

                            # Header summary
                            _vh1, _vh2, _vh3 = st.columns(3)
                            _vh1.metric("Model", _model_label)
                            _vh2.metric("Completed trials", f"{_n_done}")
                            _vh3.metric("Best CV R²", f"{_vres['best_cv_r2']:.4f}")

                            st.caption(
                                "All plots below are interactive - hover for trial details, "
                                "click legend entries to toggle traces."
                            )

                            # ── 1. Optimisation history (always shown) ─────────
                            st.subheader("Optimisation History")
                            st.markdown(
                                "Best CV R² achieved as trials progress. The flat regions are "
                                "where TPE was unable to improve on the current best."
                            )
                            try:
                                _fig_hist = _ovis.plot_optimization_history(_study)
                                _apply_chart_style(_fig_hist, height=380)
                                st.plotly_chart(_fig_hist, use_container_width=True)
                            except Exception as _e:
                                st.warning(f"Could not render history plot: {_e}")

                            if _n_params >= 2:
                                # ── 2. Parameter importances + slice (side by side) ─
                                _v1, _v2 = st.columns(2)
                                with _v1:
                                    st.subheader("Parameter Importances")
                                    st.markdown(
                                        "Relative impact of each hyperparameter on CV R² "
                                        "(fANOVA-based)."
                                    )
                                    try:
                                        _fig_imp = _ovis.plot_param_importances(_study)
                                        _apply_chart_style(_fig_imp, height=400)
                                        st.plotly_chart(_fig_imp, use_container_width=True)
                                    except Exception as _e:
                                        st.info(
                                            f"Importance plot needs more diverse trials "
                                            f"({_e})."
                                        )

                                with _v2:
                                    st.subheader("Slice View")
                                    st.markdown(
                                        "CV R² vs each hyperparameter - one panel per param. "
                                        "Steep slopes indicate sensitivity."
                                    )
                                    try:
                                        _fig_slice = _ovis.plot_slice(_study)
                                        _apply_chart_style(_fig_slice, height=400)
                                        st.plotly_chart(_fig_slice, use_container_width=True)
                                    except Exception as _e:
                                        st.warning(f"Slice plot unavailable: {_e}")

                                # ── 3. Parallel coordinates (full width) ───────
                                st.subheader("Parallel Coordinates")
                                st.markdown(
                                    "Each line is one trial; colour shows CV R². Look for "
                                    "narrow bands of dark lines — those reveal the "
                                    "high-performing parameter combinations."
                                )
                                try:
                                    _fig_par = _ovis.plot_parallel_coordinate(_study)
                                    _apply_chart_style(_fig_par, height=460)
                                    st.plotly_chart(_fig_par, use_container_width=True)
                                except Exception as _e:
                                    st.warning(f"Parallel coordinates unavailable: {_e}")
                            else:
                                # Ridge — only one hyperparameter
                                st.subheader("Slice View")
                                st.markdown(
                                    "CV R² vs the searched hyperparameter "
                                    "(α, log scale)."
                                )
                                try:
                                    _fig_slice = _ovis.plot_slice(_study)
                                    _apply_chart_style(_fig_slice, height=400)
                                    st.plotly_chart(_fig_slice, use_container_width=True)
                                except Exception as _e:
                                    st.warning(f"Slice plot unavailable: {_e}")

                                st.caption(
                                    "📌 Importance / parallel-coordinate plots are skipped "
                                    "for Ridge - it has only one hyperparameter."
                                )

    # ── Phase 8: Desirability Ranking ─────────────────────────────────────────
    if has_smiles:
        with st.container(border=True):
            _section_header(8, "Desirability Ranking")
            st.markdown(
                "Combines several drug-discovery criteria into a single 0-100 "
                "**desirability score** per compound. Higher = more attractive "
                "for follow-up. When a trained model is present, its predicted "
                "pIC50 dominates the score; otherwise the other components are "
                "reweighted automatically."
            )

            _has_model = "_model_result" in st.session_state or "_model_tuned_result" in st.session_state

            # Component-weight controls
            with st.expander("⚖️ Component weights", expanded=False):
                _ws_cols = st.columns(5)
                _w_pic50 = _ws_cols[0].slider(
                    "Predicted pIC50",
                    min_value=0.0, max_value=1.0,
                    value=0.40 if _has_model else 0.0,
                    step=0.05, key="dwt_pic50",
                    disabled=not _has_model,
                )
                _w_qed   = _ws_cols[1].slider(
                    "QED (drug-likeness)",
                    min_value=0.0, max_value=1.0,
                    value=0.30, step=0.05, key="dwt_qed",
                )
                _w_lip   = _ws_cols[2].slider(
                    "Lipinski",
                    min_value=0.0, max_value=1.0,
                    value=0.15, step=0.05, key="dwt_lip",
                )
                _w_alert = _ws_cols[3].slider(
                    "Alerts (PAINS/Brenk)",
                    min_value=0.0, max_value=1.0,
                    value=0.10, step=0.05, key="dwt_alerts",
                )
                _w_size  = _ws_cols[4].slider(
                    "Size",
                    min_value=0.0, max_value=1.0,
                    value=0.05, step=0.05, key="dwt_size",
                )
                if not _has_model:
                    st.caption(
                        "ℹ️ Train a model in **Section 7** to enable the *Predicted pIC50* "
                        "component - the most reliable signal when available."
                    )

            _custom_weights = {
                "predicted_pIC50":   _w_pic50,
                "QED":               _w_qed,
                "lipinski":          _w_lip,
                "structural_alerts": _w_alert,
                "size":              _w_size,
            }

            if st.button("🏆 Compute Desirability Ranking", key="btn_desirability"):
                with st.spinner("Scoring molecules…"):
                    try:
                        # Try to pull predicted pIC50 from a trained model
                        _preds = None
                        if _has_model:
                            _src = st.session_state.get("_model_tuned_result") \
                                   or st.session_state.get("_model_result")
                            _model = _src.get("model") if _src else None
                            if _model is not None:
                                from utils.fingerprints import compute_fp_array
                                _fp, _vm = compute_fp_array(
                                    df_working["canonical_smiles"], fp_type="morgan"
                                )
                                _pred_vals = np.full(len(df_working), np.nan)
                                _pred_vals[_vm] = _model.predict(_fp)
                                _preds = pd.Series(_pred_vals)
                        _ranked = run_desirability_ranking(
                            df_working,
                            predicted_pic50=_preds,
                            weights=_custom_weights,
                        )
                        st.session_state["_desirability_result"] = _ranked
                    except Exception as _de:
                        st.error(f"Desirability scoring failed: {_de}")

            if "_desirability_result" in st.session_state:
                _ranked = st.session_state["_desirability_result"]

                # Top-line counts
                _dm1, _dm2, _dm3 = st.columns(3)
                _dm1.metric("Scored", f"{len(_ranked):,}")
                _dm2.metric(
                    "Mean desirability",
                    f"{_ranked['desirability_pct'].mean():.1f}",
                )
                _high = (_ranked["desirability_pct"] >= 70).sum()
                _dm3.metric("Score ≥ 70", f"{_high:,}")

                # Top-N gallery
                _top_n = st.slider(
                    "Top N to display",
                    min_value=5, max_value=20, value=10, step=5,
                    key="desirability_top_n",
                )
                _top = _ranked.head(_top_n).reset_index(drop=True)

                st.subheader("Top candidates")
                _top_imgs = _build_image_column(
                    tuple(_top["canonical_smiles"].tolist()),
                    size=(220, 170),
                )
                _ncols = 5
                _rows  = [_top.iloc[i:i + _ncols] for i in range(0, len(_top), _ncols)]
                _idx = 0
                for _row in _rows:
                    _cols = st.columns(len(_row))
                    for _col_, (_, _r) in zip(_cols, _row.iterrows()):
                        with _col_:
                            if _top_imgs[_idx]:
                                st.image(_top_imgs[_idx], use_container_width=True)
                            _id_label = _r.get("molecule_chembl_id", "—")
                            st.markdown(
                                f"**{_id_label}**  · score **{_r['desirability_pct']:.0f}**"
                            )
                            _reasons = get_desirability_reasons(_r, top_k=3)
                            if _reasons:
                                st.caption(" · ".join(_reasons))
                        _idx += 1

                # Full ranked table
                st.subheader("Full ranking")
                _show_cols = [
                    c for c in (
                        "molecule_chembl_id", "canonical_smiles",
                        "desirability_pct", "predicted_pIC50", "pIC50",
                        "QED", "lipinski_violations",
                        "n_structural_alerts", "heavy_atoms",
                    ) if c in _ranked.columns
                ]
                st.dataframe(
                    _ranked[_show_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "desirability_pct": st.column_config.ProgressColumn(
                            "Desirability", min_value=0, max_value=100,
                            format="%.0f",
                        ),
                    },
                )

                st.download_button(
                    label="⬇️ Download desirability ranking (CSV)",
                    data=_ranked.to_csv(index=False),
                    file_name="desirability_ranking.csv",
                    mime="text/csv",
                    key="dl_desirability",
                )
