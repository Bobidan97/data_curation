"""assay_vocabulary.py — NLP-based vocabulary extraction from assay descriptions.

Surfaces recurring words and phrases the user might want to filter on. Pattern
detection highlights likely-filterable concepts (mutations, cell lines, species,
preparation method, etc.) so they stand out from generic vocabulary.

Workflow:

    1. ``extract_vocabulary(descriptions)`` →
       DataFrame of recurring n-grams with category labels and per-term row counts.

    2. User picks N terms in the UI.

    3. ``filter_by_terms(df, picked)`` →
       (kept_df, removed_df) — substring match in the description column.

No LLM call — pure string analysis, fast and deterministic.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import pandas as pd


# ── Stopwords + filtering ─────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "for", "with", "and", "or", "to", "is", "on",
    "by", "at", "as", "from", "be", "this", "that", "it", "its", "such", "are",
    "was", "were", "has", "have", "had", "but", "not", "if", "into", "over",
    "after", "before", "between", "than", "then", "when", "where", "which",
    "who", "what", "why", "how",
})


# ── Categorisation patterns ───────────────────────────────────────────────────
# Terms matching any of these patterns get tagged with the category so the user
# can scan them quickly. Patterns are lowercase substring matches unless they
# start with ``^`` (then they're treated as regex anchored at the start).

_CATEGORY_PATTERNS: dict[str, Sequence[str]] = {
    "Mutation": [
        r"^[a-z]\d{1,4}[a-z]$",      # T315I, L858R, F50A pattern
        "mutant", "mutated", "mutation",
        "knockout", "knock-out", "ko",
        "double mutant", "single mutant",
        "deletion", "truncated", "truncation",
    ],
    "Wild-type": [
        r"^wt$", "wild-type", "wildtype", "wild type",
    ],
    "Cell line": [
        # Common cell lines (case-insensitive substring match works for these)
        "hela", "hek293", "hek-293", "cho", "cos-7", "cos7",
        "a549", "mcf-7", "mcf7", "pc-3", "pc3", "k562", "jurkat",
        "hepg2", "hep-g2", "u937", "raji", "thp-1", "thp1",
        "ht-29", "ht29", "sh-sy5y", "skbr3", "sk-br-3",
        "dld-1", "dld1", "sw480", "sw620", "ls174t",
    ],
    "Species": [
        "human", "mouse", "rat", "bovine", "porcine", "rabbit",
        "canine", "feline", "murine", "ovine", "equine",
        "homo sapiens", "mus musculus", "rattus norvegicus",
    ],
    "Preparation": [
        "membrane", "purified", "recombinant", "isolated", "lysate",
        "homogenate", "microsome", "microsomal", "cytosol", "cytosolic",
        "expressed in", "expressed-in",
        "his-tagged", "his-tag", "his tag", "gst-tagged", "gst tag",
        "tagged", "fusion", "fusion protein", "fragment",
    ],
    "Assay format": [
        "fluorescence", "fluorometric", "fluorescent",
        "colorimetric", "radiometric", "radioactive",
        "luminescent", "luminescence",
        "tr-fret", "trfret", "fret", "fp",
        "elisa", "western", "hplc", "lc-ms", "lcms",
        "patch clamp", "patch-clamp",
    ],
    "Concentration": [
        r"^\d+um$", r"^\d+nm$", r"^\d+mm$",
        r"^\d+\.\d+um$", r"^\d+\.\d+nm$",
    ],
}


def _categorise(term: str) -> str:
    """Tag a term with a human-readable category, or 'Generic'."""
    t = term.lower().strip()
    for cat, patterns in _CATEGORY_PATTERNS.items():
        for p in patterns:
            if p.startswith("^"):
                if re.match(p, t):
                    return cat
            else:
                if p in t:
                    return cat
    return "Generic"


# ── Tokenisation + n-gram building ────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Pull alphanumeric tokens, lowercase, drop stopwords and 1-char tokens."""
    if not isinstance(text, str):
        return []
    # Keep mutations like T315I (mixed case + digits) — operate on raw text first
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)
    return [
        t.lower() for t in raw_tokens
        if len(t) > 1 and t.lower() not in _STOPWORDS
    ]


def _ngrams(tokens: Sequence[str], n: int) -> list[str]:
    """Build joined-by-space n-gram phrases."""
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ── Public API ────────────────────────────────────────────────────────────────

def extract_vocabulary(
    descriptions: Sequence[str],
    min_row_count: int = 2,
    include_unigrams: bool = True,
    include_bigrams:  bool = True,
    include_trigrams: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of recurring terms across the assay descriptions.

    Args:
        descriptions:     Iterable of free-text strings (one per row — duplicates
                          are expected; per-term row counts use the duplicates).
        min_row_count:    Drop terms that appear in fewer than this many rows.
        include_unigrams: Include single-word terms.
        include_bigrams:  Include two-word phrases (much more informative than
                          unigrams for assay text).
        include_trigrams: Include three-word phrases (off by default — long).

    Returns:
        DataFrame with columns:
            term · category · n_rows · n_descriptions · example_description
        Sorted: highlighted-category terms first, then by row count descending.
    """
    # Row count per unique description (so re-runs of the same assay are weighted)
    desc_row_counts = Counter(
        d.strip() for d in descriptions
        if isinstance(d, str) and d.strip()
    )
    unique_descs = list(desc_row_counts.keys())
    if not unique_descs:
        return pd.DataFrame(
            columns=["term", "category", "n_rows", "n_descriptions", "example_description"]
        )

    # Map term → set of descriptions it appears in (de-duped within a description)
    term_to_descs: dict[str, set[str]] = {}
    for desc in unique_descs:
        tokens = _tokenize(desc)
        bag: list[str] = []
        if include_unigrams: bag.extend(tokens)
        if include_bigrams:  bag.extend(_ngrams(tokens, 2))
        if include_trigrams: bag.extend(_ngrams(tokens, 3))
        for term in set(bag):
            term_to_descs.setdefault(term, set()).add(desc)

    rows = []
    for term, descs in term_to_descs.items():
        n_rows = sum(desc_row_counts[d] for d in descs)
        if n_rows < min_row_count:
            continue
        rows.append({
            "term":               term,
            "category":           _categorise(term),
            "n_rows":             int(n_rows),
            "n_descriptions":     len(descs),
            "example_description": next(iter(descs))[:180],
        })

    if not rows:
        return pd.DataFrame(
            columns=["term", "category", "n_rows", "n_descriptions", "example_description"]
        )

    out = pd.DataFrame(rows)
    # Highlighted categories sort first, then by row count desc
    out["_is_generic"] = (out["category"] == "Generic").astype(int)
    out = out.sort_values(
        ["_is_generic", "n_rows"],
        ascending=[True, False],
    ).drop(columns=["_is_generic"]).reset_index(drop=True)
    return out


def filter_by_terms(
    df: pd.DataFrame,
    terms: Sequence[str],
    description_col: str = "assay_description",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (kept, removed) by substring match of ``terms`` in description.

    The match is case-insensitive and does not require whole-word boundaries —
    selecting "mutant" will also catch "mutants" and "G12D mutant".
    """
    if description_col not in df.columns or not terms:
        return df.reset_index(drop=True), pd.DataFrame(columns=df.columns)

    # Build a single regex of all terms — use re.escape for safety
    pattern = "|".join(re.escape(t.strip().lower()) for t in terms if t.strip())
    if not pattern:
        return df.reset_index(drop=True), pd.DataFrame(columns=df.columns)

    mask = df[description_col].fillna("").str.lower().str.contains(pattern, regex=True)
    return (
        df[~mask].reset_index(drop=True),
        df[mask].reset_index(drop=True),
    )


def preview_filter_impact(
    df: pd.DataFrame,
    terms: Sequence[str],
    description_col: str = "assay_description",
) -> dict:
    """Quick statistics about what ``filter_by_terms`` would do — no DataFrame copy."""
    if description_col not in df.columns or not terms:
        return {"n_remove": 0, "n_keep": len(df), "pct_remove": 0.0}

    pattern = "|".join(re.escape(t.strip().lower()) for t in terms if t.strip())
    if not pattern:
        return {"n_remove": 0, "n_keep": len(df), "pct_remove": 0.0}

    n_remove = int(
        df[description_col].fillna("").str.lower().str.contains(pattern, regex=True).sum()
    )
    n_total = len(df)
    return {
        "n_remove":   n_remove,
        "n_keep":     n_total - n_remove,
        "pct_remove": round(n_remove / n_total * 100, 1) if n_total else 0.0,
    }
