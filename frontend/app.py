import io
import pickle
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
    get_structural_duplicates,
    get_duplicate_activity_rows,
    run_chemical_space_pca,
    run_chemical_space_tsne,
    run_model_training,
    edit_dataframe,
    run_activity_cliffs,
)
from curation import curate_dataframe, CurationConfig


@st.cache_data(show_spinner=False)
def _build_image_column(smiles_tuple: tuple) -> list:
    """Render each SMILES as a base64 PNG data URI for st.column_config.ImageColumn.

    Results are cached by Streamlit so images are only generated once per unique
    set of SMILES strings. Returns a list of data-URI strings (or None for
    invalid / unparseable SMILES). Silently returns all-None if RDKit is absent.
    """
    import base64
    import io

    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import MolToImage
    except ImportError:
        return [None] * len(smiles_tuple)

    images = []
    for smi in smiles_tuple:
        try:
            mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
            if mol is None:
                images.append(None)
                continue
            img = MolToImage(mol, size=(400, 300))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            images.append(f"data:image/png;base64,{b64}")
        except Exception:
            images.append(None)
    return images


st.set_page_config(page_title="ChEMBL Data Curation", layout="wide")
st.title("🧪 ChEMBL Data Curation Dashboard")

# ── Phase 1: Query ────────────────────────────────────────────────────────────
st.header("1. Query")
user_query = st.text_input(
    "Enter your ChEMBL query",
    placeholder="e.g. IC50 data for EGFR, activities for imatinib, approved drugs for lung cancer",
)

if st.button("🔍 Search", disabled=not user_query):
    with st.spinner("Classifying query and generating code..."):
        domain, entity_name = classify_query(user_query)
        context = run_rag(user_query)
        generated_code = get_generated_code(user_query, context, domain)
        candidates_df = get_entity_candidates(domain, entity_name) if entity_name else None
        st.session_state["domain"] = domain
        st.session_state["generated_code"] = generated_code
        st.session_state["candidates_df"] = candidates_df
        st.session_state.pop("raw_df", None)

# ── Phase 2: Entity selection (target or molecule) ────────────────────────────
if "candidates_df" in st.session_state:
    domain = st.session_state.get("domain", "other")
    candidates = st.session_state["candidates_df"]

    if candidates is not None:
        if domain == "target":
            st.header("2. Select Target")
            options = {
                f"{row['pref_name']} | {row['organism']} | {row['target_type']} ({row['target_chembl_id']})": row["target_chembl_id"]
                for _, row in candidates.iterrows()
            }
            selected_label = st.selectbox("Target candidates", list(options.keys()))
            entity_chembl_id = options[selected_label]
        else:
            st.header("2. Select Molecule")
            options = {
                f"{row['pref_name']} | {row['molecule_type']} | Phase {row.get('max_phase', 'N/A')} ({row['molecule_chembl_id']})": row["molecule_chembl_id"]
                for _, row in candidates.iterrows()
            }
            selected_label = st.selectbox("Molecule candidates", list(options.keys()))
            entity_chembl_id = options[selected_label]

        if st.button("📥 Fetch Data"):
            with st.spinner("Fetching data from ChEMBL..."):
                try:
                    st.session_state["raw_df"] = fetch_chembl_data(
                        st.session_state["generated_code"], domain, entity_chembl_id
                    )
                except ValueError as e:
                    st.error(str(e))
    else:
        st.header("2. Fetch Data")
        _domain_hints = {
            "molecule": "The query uses an exact identifier (e.g. SMILES, InChI key, or ChEMBL ID) — no disambiguation needed. The generated code will use it directly.",
            "other":    "This query does not reference a specific entity (target or molecule). The generated code will query ChEMBL directly.",
        }
        st.info(_domain_hints.get(domain, "No entity resolution required. Click below to fetch data."))
        if st.button("📥 Fetch Data"):
            with st.spinner("Fetching data from ChEMBL..."):
                try:
                    st.session_state["raw_df"] = fetch_chembl_data(
                        st.session_state["generated_code"], domain
                    )
                except ValueError as e:
                    st.error(str(e))

# ── Phase 3: Curation + Visualisation ────────────────────────────────────────
if "raw_df" in st.session_state:
    raw_df = st.session_state["raw_df"]

    st.sidebar.header("⚙️ Curation Settings")
    exact_only = st.sidebar.checkbox("Exact measurements only  (=)", value=True)
    standardised_only = st.sidebar.checkbox("Standardised rows only (flag = 1)", value=True)
    remove_validity = st.sidebar.checkbox("Remove data validity issues", value=True)
    remove_structural_duplicates = st.sidebar.checkbox(
        "Remove structural duplicates (canonicalise SMILES)", value=True
    )
    compute_pic50 = st.sidebar.checkbox("Compute pIC50", value=True)

    config = CurationConfig(
        exact_relation_only=exact_only,
        standardised_only=standardised_only,
        remove_validity_issues=remove_validity,
        remove_structural_duplicates=remove_structural_duplicates,
        compute_pic50=compute_pic50,
    )

    df_curated, df_dropped = curate_dataframe(raw_df, config)

    # Invalidate cached results when curated data changes
    _fingerprint = (len(df_curated), tuple(df_curated.columns.tolist()))
    if st.session_state.get("_curated_fingerprint") != _fingerprint:
        st.session_state.pop("df_with_descriptors", None)
        st.session_state.pop("_pca_result", None)
        st.session_state.pop("_tsne_result", None)
        st.session_state.pop("_tsne_config", None)
        st.session_state["_curated_fingerprint"] = _fingerprint

    # ── Sidebar: Manual curation chat ─────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.header("💬 Manual Curation")

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
        st.session_state.pop("_model_result", None)
        st.session_state.pop("_model_history", None)
        st.session_state.pop("_sali_result", None)
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
        _instruction = st.text_area(
            "Instruction",
            height=80,
            placeholder=(
                'e.g. "remove rows where pIC50 < 5"\n'
                '"drop column assay_type"\n'
                '"keep only IC50 measurements"'
            ),
            label_visibility="collapsed",
        )
        _submitted = st.form_submit_button("▶ Apply", use_container_width=True)

    if _submitted and _instruction.strip():
        _instruction = _instruction.strip()
        st.session_state["chat_history"].append(
            {"role": "user", "content": _instruction, "code": None}
        )
        _reset_kws = {"reset", "restore", "undo all", "revert"}
        if any(kw in _instruction.lower() for kw in _reset_kws):
            st.session_state["df_working"]   = df_curated.copy()
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": "Dataset reset to the automated curation result.",
                "code": None,
            })
        else:
            with st.spinner("Applying edit…"):
                try:
                    # Collect all DataFrames currently available in the session.
                    # Coerce known numeric columns that ChEMBL returns as object
                    # (strings) so that LLM-generated aggregations (median, mean,
                    # comparisons) work without extra casting in the generated code.
                    _NUMERIC_COLS = {"standard_value", "pchembl_value", "standard_upper_value"}

                    def _coerce_numerics(frame: pd.DataFrame) -> pd.DataFrame:
                        frame = frame.copy()
                        for _c in _NUMERIC_COLS & set(frame.columns):
                            frame[_c] = pd.to_numeric(frame[_c], errors="coerce")
                        return frame

                    _extra = {
                        "raw_df": _coerce_numerics(raw_df),
                        "df_curated": df_curated,
                    }
                    _dup = get_duplicate_activity_rows(raw_df)
                    if not _dup.empty:
                        _extra["dup_rows"] = _coerce_numerics(_dup)
                    if "df_with_descriptors" in st.session_state:
                        _extra["df_with_descriptors"] = st.session_state["df_with_descriptors"]

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
                        _parts.append(f"Applied — shape unchanged ({_ra:,} × {_ca}).")
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
            st.caption("No edits yet.")
        for _msg in st.session_state["chat_history"]:
            with st.chat_message(_msg["role"]):
                st.write(_msg["content"])
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

    # ── Column-presence detection ──────────────────────────────────────────────
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
        dup_df = get_structural_duplicates(raw_df)
        if not dup_df.empty:
            total_groups  = len(dup_df)
            total_removed = int(dup_df["would_remove"].sum())
            with st.expander(
                f"🔍 {total_groups} structural duplicate group(s) found "
                f"— {total_removed} molecule(s) would be removed by deduplication",
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
                dup_rows = get_duplicate_activity_rows(raw_df)
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

                    # Hide all raw measurements for any canonical SMILES that was
                    # manually removed from df_working via the chatbot.
                    # Filter by _canonical in plot_df (= RDKit-canonical SMILES, same
                    # canonicalisation as canonical_smiles in df_curated/df_working) so
                    # that ALL measurements for a removed structure are hidden —
                    # including rows from auto-deduped molecule_chembl_ids that were
                    # never present in df_curated and therefore invisible to a
                    # molecule_chembl_id set-difference filter.
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
                        # Order groups by group number (Group 1, Group 2, …)
                        import re
                        group_order = sorted(
                            plot_df["_group"].unique(),
                            key=lambda g: int(re.search(r"\d+", g).group()),
                        )

                        total_groups = len(group_order)
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
                            n_mols = len(grp_rows)  # number of activity measurements in group
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
                                marker=dict(size=7, opacity=0.7, color="#1f77b4"),
                                line=dict(color="#1f77b4"),
                                fillcolor="rgba(31,119,180,0.25)",
                            ))
                        fig_dup.update_layout(
                            title="Activity values within structural duplicate groups",
                            xaxis_title="Duplicate group",
                            yaxis_title="Standard value (nM)",
                            yaxis_type="log",
                            showlegend=False,
                            height=420,
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

    # ── Tier 1: pipeline counts ────────────────────────────────────────────────
    st.header(f"3. {section_label} Overview")

    # ── Dropped rows reference ─────────────────────────────────────────────────
    _n_dropped = len(df_dropped)
    with st.expander(
        f"🗑️ {_n_dropped:,} row{'s' if _n_dropped != 1 else ''} dropped during auto-curation",
        expanded=False,
    ):
        if _n_dropped == 0:
            st.success("No rows were dropped during auto-curation.")
        else:
            _reason_counts = df_dropped["reason"].value_counts().reset_index()
            _reason_counts.columns = ["Reason", "Count"]
            st.dataframe(_reason_counts, use_container_width=True, hide_index=True)
            st.divider()
            st.caption("Full dropped-rows table (first column = reason for removal):")
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

    # ── Tier 2: quality metrics ────────────────────────────────────────────────
    st.subheader("Quality Metrics")
    q1, q2, q3, q4 = st.columns(4)

    unique_smiles = df_working["canonical_smiles"].nunique() if has_smiles else None
    dup_rate = (
        (1 - unique_smiles / len(df_working)) * 100
        if unique_smiles is not None and len(df_working) > 0 else None
    )
    pct_valid_pic50 = df_working["pIC50"].notna().mean() * 100 if has_pic50 else None
    unique_assays   = df_working["assay_chembl_id"].nunique() if has_assay_id else None

    q1.metric("Unique compounds", f"{unique_smiles:,}" if unique_smiles is not None else "N/A")
    q2.metric("Duplication rate", f"{dup_rate:.1f}%" if dup_rate is not None else "N/A")
    q3.metric("Valid pIC50", f"{pct_valid_pic50:.1f}%" if pct_valid_pic50 is not None else "N/A")
    q4.metric("Unique assays", f"{unique_assays:,}" if unique_assays is not None else "N/A")

    if has_pic50:
        pic50 = df_working["pIC50"].dropna()
        if not pic50.empty:
            st.subheader("pIC50 Statistics")
            p1, p2, p3, p4 = st.columns(4)
            iqr = pic50.quantile(0.75) - pic50.quantile(0.25)
            p1.metric("Mean", f"{pic50.mean():.2f}")
            p2.metric("Std", f"{pic50.std():.2f}")
            p3.metric("Median", f"{pic50.median():.2f}")
            p4.metric("IQR", f"{iqr:.2f}")

    # ── Tier 3A: bioactivity charts ────────────────────────────────────────────
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
                        color_discrete_sequence=["#1f77b4"],
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        with bcol2:
            if has_assay_type:
                assay_counts = df_working["assay_type"].value_counts().reset_index()
                assay_counts.columns = ["assay_type", "count"]
                fig2 = px.bar(
                    assay_counts, x="assay_type", y="count",
                    title="Assay Type Breakdown",
                    color_discrete_sequence=["#2ca02c"],
                )
                st.plotly_chart(fig2, use_container_width=True)

        brow2_col1, brow2_col2 = st.columns(2)
        with brow2_col1:
            if has_doc_year:
                year_counts = df_working["document_year"].value_counts().sort_index().reset_index()
                year_counts.columns = ["year", "count"]
                fig3 = px.bar(
                    year_counts, x="year", y="count", title="Publications by Year",
                    color_discrete_sequence=["#ff7f0e"],
                )
                st.plotly_chart(fig3, use_container_width=True)
        with brow2_col2:
            if has_standard_type:
                type_counts = df_working["standard_type"].value_counts().reset_index()
                type_counts.columns = ["standard_type", "count"]
                fig4 = px.bar(
                    type_counts, x="standard_type", y="count",
                    title="Measurement Type Breakdown",
                    labels={"standard_type": "Type", "count": "Count"},
                    color_discrete_sequence=["#9467bd"],
                )
                st.plotly_chart(fig4, use_container_width=True)

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
                        color_discrete_sequence=["#e377c2"],
                    )
                    fig_cv1.update_layout(showlegend=False)
                    st.plotly_chart(fig_cv1, use_container_width=True)
                with cv_col2:
                    top10_cv = cv_df.nlargest(10, "cv")
                    fig_cv2 = px.bar(
                        top10_cv, x="cv", y="assay_chembl_id", orientation="h",
                        title="Top 10 Most Variable Assays (CV %)",
                        labels={"cv": "CV (%)", "assay_chembl_id": "Assay"},
                        color_discrete_sequence=["#d62728"],
                    )
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
                color_discrete_sequence=["#8c564b"],
            )
            fig5.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig5, use_container_width=True)

    # ── Tier 3B: molecule quality charts ──────────────────────────────────────
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
                    color_discrete_sequence=["#1f77b4"],
                    category_orders={"max_phase": ["N/A", "0", "1", "2", "3", "4"]},
                )
                st.plotly_chart(fig_phase, use_container_width=True)
        with mcol2:
            if has_mol_type:
                mol_type_counts = df_working["molecule_type"].value_counts().reset_index()
                mol_type_counts.columns = ["molecule_type", "count"]
                fig_mol_type = px.bar(
                    mol_type_counts, x="molecule_type", y="count",
                    title="Molecule Type Breakdown",
                    labels={"molecule_type": "Type", "count": "# Molecules"},
                    color_discrete_sequence=["#2ca02c"],
                )
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
                color_discrete_sequence=["#ff7f0e"],
            )
            st.plotly_chart(fig_appr, use_container_width=True)

    # ── Tier 4: data completeness (always shown) ───────────────────────────────
    st.subheader("Data Completeness")
    completeness = (df_working.notna().mean() * 100).sort_values()
    completeness_df = completeness.reset_index()
    completeness_df.columns = ["column", "pct_complete"]
    fig6 = px.bar(
        completeness_df, x="pct_complete", y="column", orientation="h",
        title="Data Completeness (% non-null per column)",
        color="pct_complete",
        color_continuous_scale=["#d62728", "#ff7f0e", "#2ca02c"],
        range_color=[0, 100],
        labels={"pct_complete": "% Complete", "column": "Column"},
    )
    fig6.update_layout(coloraxis_showscale=False, xaxis_range=[0, 100])
    st.plotly_chart(fig6, use_container_width=True)

    # ── Activity Cliffs (SALI) ─────────────────────────────────────────────────
    if has_smiles and has_pic50:
        st.subheader("Activity Cliffs (Structural Similarity vs. Potency Difference)")
        st.markdown(
            "Computes pairwise Tanimoto similarity (ECFP4) and |ΔpIC50| for all "
            "molecule pairs. Points in the **top-right quadrant** "
            "(Tanimoto > 0.4 and |ΔpIC50| > 2.0) are activity cliffs — "
            "structurally similar molecules with large potency differences."
        )
        if st.button("🔭 Compute Activity Cliffs", key="btn_sali"):
            with st.spinner("Computing pairwise similarities… (may take a moment for large datasets)"):
                try:
                    st.session_state["_sali_result"] = run_activity_cliffs(df_working)
                except ValueError as _se:
                    st.error(f"Activity cliff computation failed: {_se}")

        if "_sali_result" in st.session_state:
            _sali = st.session_state["_sali_result"]
            _pairs = _sali["pairs_df"]
            if _sali["truncated"]:
                st.info(
                    f"Dataset has {_sali['n_original']:,} molecules — "
                    f"randomly sampled 500 for cliff analysis "
                    f"({_sali['n_pairs']:,} pairs computed)."
                )
            else:
                st.caption(
                    f"{_sali['n_molecules']:,} molecules · "
                    f"{_sali['n_pairs']:,} pairs computed."
                )
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
                    x=0.4, line_dash="dash", line_color="grey",
                    annotation_text="Tanimoto = 0.4",
                    annotation_position="top right",
                )
                fig_sali.add_hline(
                    y=2.0, line_dash="dash", line_color="grey",
                    annotation_text="|ΔpIC50| = 2.0",
                    annotation_position="top left",
                )
                fig_sali.update_layout(
                    height=500,
                    coloraxis_colorbar=dict(title="|ΔpIC50|"),
                )
                st.plotly_chart(fig_sali, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    st.header("4. Working Dataset")
    if _manual_removed:
        st.caption(
            f"Showing the working set ({len(df_working):,} rows) — "
            f"{_manual_removed:,} rows removed by manual edits. "
            "Use **↺ Reset** in the sidebar to restore the auto-curated dataset."
        )
    st.dataframe(df_working, use_container_width=True)

    # ── Download working dataset CSV ──────────────────────────────────────────
    st.download_button(
        label="⬇️ Download working dataset (CSV)",
        data=df_working.to_csv(index=False),
        file_name="chembl_working.csv",
        mime="text/csv",
    )

    # ── Phase 5: Molecular Descriptors ────────────────────────────────────────
    if has_smiles:
        st.header("5. Molecular Descriptors")

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

            # ── Descriptor distribution grid ───────────────────────────────────
            _desc_only_cols = [
                c for c in df_desc.columns
                if c not in df_working.columns
                and pd.api.types.is_numeric_dtype(df_desc[c])
            ]
            if _desc_only_cols:
                st.subheader("Descriptor Distributions")
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
                                _fh = px.histogram(
                                    _col_data, nbins=30, title=_dc,
                                    color_discrete_sequence=["#1f77b4"],
                                )
                                _fh.update_layout(
                                    showlegend=False, height=280,
                                    margin=dict(t=40, b=20, l=20, r=20),
                                    xaxis_title=_dc, yaxis_title="Count",
                                )
                                st.plotly_chart(_fh, use_container_width=True)

    # ── Phase 6: Chemical Space ────────────────────────────────────────────────
    if has_smiles:
        st.header("6. Chemical Space")
        pca_tab, tsne_tab = st.tabs(["🔵 PCA", "🟠 t-SNE + K-means"])

        # ── PCA tab ───────────────────────────────────────────────────────────
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
                        marker_color="#1f77b4",
                    )
                    fig_scree.add_scatter(
                        x=pc_labels,
                        y=(cum_var * 100).tolist(),
                        name="Cumulative",
                        mode="lines+markers",
                        line=dict(color="#d62728", width=2),
                    )
                    fig_scree.update_layout(
                        title="Scree Plot",
                        xaxis_title="Principal Component",
                        yaxis_title="% Variance Explained",
                        legend=dict(orientation="h", y=-0.25),
                        height=400,
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
                            arrowwidth=1.5, arrowcolor="black",
                        )
                        fig_biplot.add_annotation(
                            x=lx * 1.2, y=ly * 1.2,
                            text=feat,
                            showarrow=False,
                            font=dict(size=12, color="black"),
                        )

                    # Unit circle
                    theta = np.linspace(0, 2 * np.pi, 200)
                    fig_biplot.add_scatter(
                        x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
                        mode="lines",
                        line=dict(color="grey", dash="dash", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )

                    fig_biplot.update_layout(
                        title="PCA Biplot",
                        xaxis=dict(title=pc1_label, zeroline=True, zerolinecolor="lightgrey"),
                        yaxis=dict(title=pc2_label, zeroline=True, zerolinecolor="lightgrey",
                                   scaleanchor="x", scaleratio=1),
                        height=500,
                        showlegend=False,
                    )
                    fig_biplot.update_xaxes(range=[-1.35, 1.35])
                    fig_biplot.update_yaxes(range=[-1.35, 1.35])
                    st.plotly_chart(fig_biplot, use_container_width=True)

        # ── t-SNE tab ─────────────────────────────────────────────────────────
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
                        colours = ["#d62728" if k == best_k else "#aec7e8" for k in ks]
                        fig_sil = go.Figure(go.Bar(
                            x=[str(k) for k in ks],
                            y=scores,
                            marker_color=colours,
                            hovertemplate="k=%{x}<br>silhouette=%{y:.3f}<extra></extra>",
                        ))
                        fig_sil.update_layout(
                            title=f"Silhouette Scores (best k={best_k})",
                            xaxis_title="Number of clusters (k)",
                            yaxis_title="Silhouette score",
                            height=400,
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
                    fig_tsne.update_layout(legend_title_text="Cluster")
                    st.plotly_chart(fig_tsne, use_container_width=True)

                # ── Cluster activity profiles ──────────────────────────────────
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
                    fig_clust.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_clust, use_container_width=True)

    # ── Phase 7: Bioactivity Model ─────────────────────────────────────────────
    if has_smiles and "pIC50" in df_working.columns:
        st.header("7. Bioactivity Model")
        st.markdown(
            "Train a regression model to predict **pIC50** from "
            "Morgan (ECFP4) fingerprints (2048-bit, radius 2). "
            "Choose a model type, set the train/test split and cross-validation "
            "folds, then click **▶ Train Model**. Results from each run are "
            "accumulated in the History tab for comparison."
        )

        # ── Hyperparameter controls ────────────────────────────────────────────
        _m_col1, _m_col2, _m_col3 = st.columns(3)
        with _m_col1:
            _model_type = st.selectbox(
                "Model",
                options=["Random Forest", "Ridge Regression", "Gradient Boosting"],
                key="model_type_select",
                help=(
                    "Random Forest: robust ensemble, provides feature importances. "
                    "Ridge: fast linear baseline. "
                    "Gradient Boosting: often highest accuracy but slower."
                ),
            )
        with _m_col2:
            _test_split_pct = st.slider(
                "Test split (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                key="model_test_split",
                help="Percentage of molecules held out for evaluation.",
            )
        with _m_col3:
            _cv_folds = st.slider(
                "CV folds",
                min_value=2,
                max_value=10,
                value=5,
                key="model_cv_folds",
                help="K-fold cross-validation on the training set.",
            )

        # n_estimators shown only for tree-based models
        if _model_type in {"Random Forest", "Gradient Boosting"}:
            _n_estimators = st.slider(
                "Number of estimators (trees)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                key="model_n_estimators",
            )
        else:
            _n_estimators = 100  # Ridge: not used, kept as harmless default

        # ── Run button ─────────────────────────────────────────────────────────
        if st.button("▶ Train Model", key="btn_train_model"):
            with st.spinner(f"Training {_model_type}…"):
                try:
                    _ml_result = run_model_training(
                        df_working,
                        model_type=_model_type,
                        test_size=_test_split_pct / 100.0,
                        n_estimators=_n_estimators,
                        cv_folds=_cv_folds,
                    )
                    st.session_state["_model_result"] = _ml_result

                    # Append a compact summary to the run history
                    _history_row = {
                        "Timestamp":  _ml_result["timestamp"],
                        "Model":      _ml_result["model_type"],
                        "Estimators": _ml_result["n_estimators"] if _ml_result["n_estimators"] else "N/A",
                        "Test %":     int(_ml_result["test_size"] * 100),
                        "CV folds":   _ml_result["cv_folds"],
                        "N train":    _ml_result["n_train"],
                        "N test":     _ml_result["n_test"],
                        "Test R²":    round(_ml_result["test_r2"],    4),
                        "RMSE":       round(_ml_result["test_rmse"],  4),
                        "MAE":        round(_ml_result["test_mae"],   4),
                        "CV R² mean": round(_ml_result["cv_r2_mean"], 4),
                        "CV R² std":  round(_ml_result["cv_r2_std"],  4),
                    }
                    if "_model_history" not in st.session_state:
                        st.session_state["_model_history"] = []
                    st.session_state["_model_history"].append(_history_row)

                    if _ml_result["n_invalid_smiles"] > 0:
                        st.warning(
                            f"{_ml_result['n_invalid_smiles']} molecule(s) had "
                            "unparseable SMILES and were excluded from training."
                        )
                except ValueError as _ve:
                    st.error(f"Model training failed: {_ve}")

        # ── Results ────────────────────────────────────────────────────────────
        if "_model_result" in st.session_state:
            _res = st.session_state["_model_result"]

            _tab_perf, _tab_feat, _tab_pred, _tab_hist = st.tabs([
                "📊 Performance",
                "🔬 Feature Importance",
                "📋 Predictions",
                "📈 History",
            ])

            # ── Performance tab ────────────────────────────────────────────────
            with _tab_perf:
                st.caption(
                    f"Model: **{_res['model_type']}**  ·  "
                    f"Trained: {_res['timestamp']}  ·  "
                    f"Train n = {_res['n_train']}, Test n = {_res['n_test']}"
                )

                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                _mc1.metric("Test R²",            f"{_res['test_r2']:.4f}")
                _mc2.metric("Test RMSE",           f"{_res['test_rmse']:.4f}")
                _mc3.metric("Test MAE",            f"{_res['test_mae']:.4f}")
                _mc4.metric(
                    f"CV R² ({_res['cv_folds']}-fold)",
                    f"{_res['cv_r2_mean']:.4f} ± {_res['cv_r2_std']:.4f}",
                )

                # Actual vs predicted scatter with perfect-fit diagonal
                _preds = _res["predictions_df"]
                _avp_lo = float(min(_preds["actual_pIC50"].min(),
                                    _preds["predicted_pIC50"].min()))
                _avp_hi = float(max(_preds["actual_pIC50"].max(),
                                    _preds["predicted_pIC50"].max()))
                _avp_fig = go.Figure()
                _avp_fig.add_scatter(
                    x=_preds["actual_pIC50"].tolist(),
                    y=_preds["predicted_pIC50"].tolist(),
                    mode="markers",
                    marker=dict(size=7, opacity=0.7, color="#1f77b4"),
                    name="Test molecules",
                    hovertemplate=(
                        "Actual: %{x:.3f}<br>Predicted: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
                _avp_fig.add_scatter(
                    x=[_avp_lo, _avp_hi],
                    y=[_avp_lo, _avp_hi],
                    mode="lines",
                    line=dict(color="red", dash="dash", width=1.5),
                    name="Perfect fit",
                    hoverinfo="skip",
                )
                _avp_fig.update_layout(
                    title="Actual vs Predicted pIC50 (test set)",
                    xaxis_title="Actual pIC50",
                    yaxis_title="Predicted pIC50",
                    height=500,
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(_avp_fig, use_container_width=True)

                # ── Residuals plot ─────────────────────────────────────────────
                _residuals = _preds["predicted_pIC50"] - _preds["actual_pIC50"]
                _resid_fig = go.Figure()
                _resid_fig.add_scatter(
                    x=_preds["predicted_pIC50"].tolist(),
                    y=_residuals.tolist(),
                    mode="markers",
                    marker=dict(size=7, opacity=0.7, color="#1f77b4"),
                    name="Residuals",
                    hovertemplate=(
                        "Predicted: %{x:.3f}<br>"
                        "Residual: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
                _resid_fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red",
                    line_width=1.5,
                )
                _resid_fig.update_layout(
                    title="Residuals Plot (Predicted − Actual)",
                    xaxis_title="Predicted pIC50",
                    yaxis_title="Residual (Predicted − Actual)",
                    height=450,
                    showlegend=False,
                )
                st.plotly_chart(_resid_fig, use_container_width=True)
                st.caption(
                    "Residuals should scatter randomly around zero with no systematic pattern. "
                    "A funnel shape suggests heteroscedasticity; a curved trend suggests the "
                    "model is missing non-linear structure."
                )

            # ── Feature importance tab ─────────────────────────────────────────
            with _tab_feat:
                if _res["feature_importances"] is not None:
                    _fi = _res["feature_importances"]
                    _top_idx = np.argsort(_fi)[::-1][:20]
                    _fi_fig = go.Figure(go.Bar(
                        x=[f"Bit {b}" for b in _top_idx],
                        y=_fi[_top_idx].tolist(),
                        marker_color="#2ca02c",
                        hovertemplate=(
                            "ECFP4 bit %{x}<br>"
                            "Importance: %{y:.6f}<extra></extra>"
                        ),
                    ))
                    _fi_fig.update_layout(
                        title=f"Top 20 ECFP4 Bit Importances — {_res['model_type']}",
                        xaxis_title="Fingerprint Bit",
                        yaxis_title="Importance (MDI)",
                        height=450,
                        xaxis=dict(tickangle=-45),
                    )
                    st.plotly_chart(_fi_fig, use_container_width=True)
                    st.caption(
                        "Importances are mean decrease in impurity (MDI). "
                        "Bit indices correspond to Morgan (ECFP4) circular "
                        "substructure features; higher = more predictive."
                    )
                else:
                    st.info(
                        "Ridge Regression does not produce feature importances. "
                        "Switch to **Random Forest** or **Gradient Boosting** "
                        "to see which ECFP4 bits drive predictions."
                    )

            # ── Predictions tab ────────────────────────────────────────────────
            with _tab_pred:
                st.caption(
                    f"Test-set predictions ({_res['n_test']} molecules), "
                    "sorted by |error| descending — worst predictions first."
                )
                _disp = _res["predictions_df"].copy()
                for _c in ("actual_pIC50", "predicted_pIC50", "error"):
                    _disp[_c] = _disp[_c].round(4)
                st.dataframe(_disp, use_container_width=True)

            # ── History tab ────────────────────────────────────────────────────
            with _tab_hist:
                if "_model_history" in st.session_state and st.session_state["_model_history"]:
                    _hist_df = pd.DataFrame(st.session_state["_model_history"])
                    st.dataframe(_hist_df, use_container_width=True)
                    st.caption(
                        f"{len(_hist_df)} run(s) recorded this session. "
                        "History resets when the working dataset changes."
                    )
                else:
                    st.info("No model runs recorded yet this session.")

            # ── Download button ────────────────────────────────────────────────
            _model_buf = io.BytesIO()
            pickle.dump(_res["model"], _model_buf)
            _safe_name = _res["model_type"].lower().replace(" ", "_")
            st.download_button(
                label="⬇️ Download trained model (.pkl)",
                data=_model_buf.getvalue(),
                file_name=f"bioactivity_{_safe_name}.pkl",
                mime="application/octet-stream",
                key="model_download_btn",
            )
