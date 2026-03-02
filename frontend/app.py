import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import streamlit as st
import plotly.express as px

from pipeline import (
    run_rag,
    get_generated_code,
    get_target_candidates,
    fetch_chembl_data,
)
from curation import curate_dataframe, CurationConfig

st.set_page_config(page_title="ChEMBL Data Curation", layout="wide")
st.title("🧪 ChEMBL Data Curation Dashboard")

# ── Phase 1: Query ────────────────────────────────────────────────────────────
st.header("1. Query")
user_query = st.text_input(
    "Enter your ChEMBL query",
    placeholder="e.g. Find all IC50 data for erbB2 in nM",
)

if st.button("🔍 Search", disabled=not user_query):
    with st.spinner("Running RAG and generating code..."):
        context = run_rag(user_query)
        st.session_state["generated_code"] = get_generated_code(user_query, context)
        st.session_state["candidates_df"] = get_target_candidates(user_query)
        st.session_state.pop("raw_df", None)

# ── Phase 2: Target selection ─────────────────────────────────────────────────
if "candidates_df" in st.session_state:
    st.header("2. Select Target")
    candidates = st.session_state["candidates_df"]

    options = {
        f"{row['pref_name']} | {row['organism']} | {row['target_type']} ({row['target_chembl_id']})": row["target_chembl_id"]
        for _, row in candidates.iterrows()
    }
    selected_label = st.selectbox("Target candidates", list(options.keys()))
    target_chembl_id = options[selected_label]

    if st.button("📥 Fetch Data"):
        with st.spinner("Fetching data from ChEMBL..."):
            try:
                st.session_state["raw_df"] = fetch_chembl_data(
                    st.session_state["generated_code"], target_chembl_id
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
    compute_pic50 = st.sidebar.checkbox("Compute pIC50", value=True)

    config = CurationConfig(
        exact_relation_only=exact_only,
        standardised_only=standardised_only,
        remove_validity_issues=remove_validity,
        compute_pic50=compute_pic50,
    )

    df_curated = curate_dataframe(raw_df, config)

    # ── Metrics row 1: pipeline counts ────────────────────────────────────────
    st.header("3. Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw compounds", len(raw_df))
    col2.metric("After curation", len(df_curated))
    col3.metric("Removed", len(raw_df) - len(df_curated))

    # ── Metrics row 2: quality overview ───────────────────────────────────────
    st.subheader("Quality Overview")
    q1, q2, q3, q4 = st.columns(4)

    unique_smiles = (
        df_curated["canonical_smiles"].nunique()
        if "canonical_smiles" in df_curated.columns else None
    )
    dup_rate = (
        (1 - unique_smiles / len(df_curated)) * 100
        if unique_smiles is not None and len(df_curated) > 0 else None
    )
    pct_valid_pic50 = (
        df_curated["pIC50"].notna().mean() * 100
        if "pIC50" in df_curated.columns else None
    )
    unique_assays = (
        df_curated["assay_chembl_id"].nunique()
        if "assay_chembl_id" in df_curated.columns else None
    )

    q1.metric("Unique compounds", f"{unique_smiles:,}" if unique_smiles is not None else "N/A")
    q2.metric("Duplication rate", f"{dup_rate:.1f}%" if dup_rate is not None else "N/A")
    q3.metric("Valid pIC50", f"{pct_valid_pic50:.1f}%" if pct_valid_pic50 is not None else "N/A")
    q4.metric("Unique assays", f"{unique_assays:,}" if unique_assays is not None else "N/A")

    # ── Metrics row 3: pIC50 statistics ───────────────────────────────────────
    if "pIC50" in df_curated.columns:
        pic50 = df_curated["pIC50"].dropna()
        if not pic50.empty:
            st.subheader("pIC50 Statistics")
            p1, p2, p3, p4 = st.columns(4)
            iqr = pic50.quantile(0.75) - pic50.quantile(0.25)
            p1.metric("Mean", f"{pic50.mean():.2f}")
            p2.metric("Std", f"{pic50.std():.2f}")
            p3.metric("Median", f"{pic50.median():.2f}")
            p4.metric("IQR", f"{iqr:.2f}")

    # ── Charts row 1: pIC50 distribution | Assay type ────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if "pIC50" in df_curated.columns:
            pic50_data = df_curated["pIC50"].dropna()
            if not pic50_data.empty:
                fig = px.histogram(
                    pic50_data,
                    nbins=40,
                    title="pIC50 Distribution",
                    labels={"value": "pIC50", "count": "Count"},
                    color_discrete_sequence=["#1f77b4"],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        if "assay_type" in df_curated.columns:
            assay_counts = df_curated["assay_type"].value_counts().reset_index()
            assay_counts.columns = ["assay_type", "count"]
            fig2 = px.bar(
                assay_counts,
                x="assay_type",
                y="count",
                title="Assay Type Breakdown",
                color_discrete_sequence=["#2ca02c"],
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2: Publications by year | Measurement type ─────────────────
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        if "document_year" in df_curated.columns:
            year_counts = df_curated["document_year"].value_counts().sort_index().reset_index()
            year_counts.columns = ["year", "count"]
            fig3 = px.bar(
                year_counts,
                x="year",
                y="count",
                title="Publications by Year",
                color_discrete_sequence=["#ff7f0e"],
            )
            st.plotly_chart(fig3, use_container_width=True)

    with row2_col2:
        if "standard_type" in df_curated.columns:
            type_counts = df_curated["standard_type"].value_counts().reset_index()
            type_counts.columns = ["standard_type", "count"]
            fig4 = px.bar(
                type_counts,
                x="standard_type",
                y="count",
                title="Measurement Type Breakdown",
                labels={"standard_type": "Type", "count": "Count"},
                color_discrete_sequence=["#9467bd"],
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── CV (Coefficient of Variation) per assay ───────────────────────────────
    if "assay_chembl_id" in df_curated.columns and "pIC50" in df_curated.columns:
        cv_df = (
            df_curated.groupby("assay_chembl_id")["pIC50"]
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
                    cv_df,
                    x="cv",
                    nbins=30,
                    title="CV Distribution Across Assays",
                    labels={"cv": "CV (%)", "count": "# Assays"},
                    color_discrete_sequence=["#e377c2"],
                )
                fig_cv1.update_layout(showlegend=False)
                st.plotly_chart(fig_cv1, use_container_width=True)
            with cv_col2:
                top10_cv = cv_df.nlargest(10, "cv")
                fig_cv2 = px.bar(
                    top10_cv,
                    x="cv",
                    y="assay_chembl_id",
                    orientation="h",
                    title="Top 10 Most Variable Assays (CV %)",
                    labels={"cv": "CV (%)", "assay_chembl_id": "Assay"},
                    color_discrete_sequence=["#d62728"],
                )
                fig_cv2.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_cv2, use_container_width=True)

    # ── Top 10 journals ───────────────────────────────────────────────────────
    if "document_journal" in df_curated.columns:
        journal_counts = (
            df_curated["document_journal"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        journal_counts.columns = ["journal", "count"]
        fig5 = px.bar(
            journal_counts,
            x="count",
            y="journal",
            orientation="h",
            title="Top 10 Journals",
            labels={"count": "Count", "journal": "Journal"},
            color_discrete_sequence=["#8c564b"],
        )
        fig5.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig5, use_container_width=True)

    # ── Data completeness ─────────────────────────────────────────────────────
    completeness = (df_curated.notna().mean() * 100).sort_values()
    completeness_df = completeness.reset_index()
    completeness_df.columns = ["column", "pct_complete"]
    fig6 = px.bar(
        completeness_df,
        x="pct_complete",
        y="column",
        orientation="h",
        title="Data Completeness (% non-null per column)",
        color="pct_complete",
        color_continuous_scale=["#d62728", "#ff7f0e", "#2ca02c"],
        range_color=[0, 100],
        labels={"pct_complete": "% Complete", "column": "Column"},
    )
    fig6.update_layout(coloraxis_showscale=False, xaxis_range=[0, 100])
    st.plotly_chart(fig6, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    st.header("4. Curated Dataset")
    st.dataframe(df_curated, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download curated CSV",
        data=df_curated.to_csv(index=False),
        file_name="chembl_curated.csv",
        mime="text/csv",
    )
