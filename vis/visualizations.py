# 1) Produces charts for the report/presentation:
#  - Outcome distribution (counts + percents)
#  - Top outcomes overall (barh)
#  - Cases by court & outcome (stacked barh, Top N + Other)
#  - Courts Pareto (bars + cumulative % line)
#  - Opinions over time
#  - Word count histogram + boxplot by outcome
# 2) Save everything to data/outputs/

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("pipeline")

OUTDIR = "data/outputs"

# Outcome code -> label
OUTCOME_LABELS = {0: "Loss", 1: "Win", 2: "Settlement"}
OUTCOME_ORDER = [0, 1, 2]  # display order for stacked bars

# Small helpers
def _ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def _finish_fig(path: str):
    """Tight layout → save → close (avoid memory leaks)."""
    try:
        plt.tight_layout()
        try:
            plt.subplots_adjust(bottom=0.18, top=0.90)
        except Exception:
            pass
        plt.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close()

def _labelify(values):
    """Map codes to labels (returns simple list of strings)."""
    out = []
    for v in list(values):
        try:
            iv = int(v)
            out.append(OUTCOME_LABELS.get(iv, str(v)))
        except Exception:
            out.append(OUTCOME_LABELS.get(v, str(v)))
    return out

def _aggregate_top_n(series: pd.Series, top_n: int = 15, other_label: str = "Other") -> pd.Series:
    """
    Keep only top_n categories; bundle the rest into an 'Other' bucket.
    """
    counts = series.value_counts()
    if len(counts) <= top_n:
        return counts
    top = counts.head(top_n)
    tail_sum = counts.iloc[top_n:].sum()
    return pd.concat([top, pd.Series({other_label: tail_sum})])

def _barh_from_counts(counts: pd.Series, title: str, xlabel: str, ylabel: str, outname: str, figsize=(10, 6)):
    """Horizontal bar chart helper."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(counts)), counts.values)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()  # biggest on top
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finish_fig(os.path.join(OUTDIR, outname))

# Charts
def outcome_distribution(df: pd.DataFrame):
    if "outcome_code" not in df.columns:
        logger.warning("Outcome distribution: 'outcome_code' not found; skipping.")
        return
    counts = df["outcome_code"].value_counts(dropna=False).sort_index()
    labels = _labelify(counts.index)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values)
    ax.set_title("Outcome Distribution (Count & %)")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")

    total = counts.sum()
    for b, v in zip(bars, counts.values):
        pct = (v / total * 100.0) if total else 0
        ax.annotate(f"{v} ({pct:.1f}%)",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    _finish_fig(os.path.join(OUTDIR, "outcome_distribution.png"))

def top_outcomes_overall(df: pd.DataFrame):
    if "outcome_code" not in df.columns:
        logger.warning("Top outcomes: 'outcome_code' not found; skipping.")
        return
    counts = df["outcome_code"].map(lambda x: OUTCOME_LABELS.get(int(x), str(x))).value_counts()
    _barh_from_counts(
        counts=counts.sort_values(ascending=True),
        title="Top Outcomes Overall",
        xlabel="Count",
        ylabel="Outcome",
        outname="top_outcomes_overall.png",
        figsize=(8, 4.5),
    )

def cases_by_court_stacked(df: pd.DataFrame, top_n: int = 12):
    """
    Stacked horizontal bars: Top X courts + 'Other', split by outcome.
    """
    if "court" not in df.columns or "outcome_code" not in df.columns:
        logger.warning("Stacked by court: missing 'court' or 'outcome_code'; skipping.")
        return

    counts = _aggregate_top_n(df["court"], top_n=top_n, other_label="Other")
    keep_courts = list(counts.index)

    dd = df.copy()
    dd["court_limited"] = np.where(dd["court"].isin(keep_courts), dd["court"], "Other")

    pivot = (
        dd.pivot_table(index="court_limited", columns="outcome_code", values="case_id", aggfunc="count", fill_value=0)
        .reindex(keep_courts)
    )
    cols = [c for c in OUTCOME_ORDER if c in pivot.columns]
    if not cols:
        logger.warning("Stacked by court: no outcome columns present; skipping.")
        return
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(11, 7))
    left = np.zeros(len(pivot), dtype=float)
    y_pos = np.arange(len(pivot.index))

    for c in cols:
        vals = pivot[c].values
        label = OUTCOME_LABELS.get(c, str(c))
        ax.barh(y_pos, vals, left=left, label=label)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Court")
    ax.set_title(f"Cases by Court & Outcome (Top {top_n} Courts + Other)")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    _finish_fig(os.path.join(OUTDIR, "cases_by_court_stacked.png"))

def courts_pareto(df: pd.DataFrame, top_n: int = 20):
    """
    Pareto chart: sorted court counts + cumulative % line.
    """
    if "court" not in df.columns:
        logger.warning("Courts Pareto: 'court' not found; skipping.")
        return

    counts_full = df["court"].value_counts()
    if len(counts_full) > top_n:
        top = counts_full.head(top_n)
        other_sum = counts_full.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other_sum})])
    else:
        counts = counts_full

    counts = counts.sort_values(ascending=False)
    cum_pct = counts.cumsum() / counts.sum() * 100.0

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(counts.index, counts.values)
    ax1.set_xlabel("Court")
    ax1.set_ylabel("Count", color="black")
    ax1.set_title(f"Courts by Volume — Pareto (Top {top_n} + Other)")

    ax1.tick_params(axis="x", labelrotation=45)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")

    ax2 = ax1.twinx()
    ax2.plot(range(len(cum_pct)), cum_pct.values, marker="o")
    ax2.set_ylabel("Cumulative %", color="black")
    ax2.set_ylim(0, 110)

    if len(cum_pct) > 0:
        mid_idx = max(len(cum_pct) // 2, 0)
        for idx in {mid_idx, len(cum_pct) - 1}:
            ax2.annotate(f"{cum_pct.iloc[idx]:.0f}%",
                         (idx, cum_pct.iloc[idx]),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=9)

    _finish_fig(os.path.join(OUTDIR, "courts_pareto.png"))

def opinions_over_time(df: pd.DataFrame):
    if "opinion_year" not in df.columns:
        logger.warning("Opinions over time: 'opinion_year' not found; skipping.")
        return
    years = pd.to_numeric(df["opinion_year"], errors="coerce").dropna().astype(int)
    if years.empty:
        logger.warning("Opinions over time: no valid years; skipping.")
        return
    series = years.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(series.index, series.values, marker="o")
    ax.set_title("Opinions Over Time (by Year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    _finish_fig(os.path.join(OUTDIR, "opinions_over_time.png"))

def text_wordcount_hist(df: pd.DataFrame, max_bin: int = 5000):
    if "text_word_count" not in df.columns:
        logger.warning("Word count hist: 'text_word_count' not found; skipping.")
        return
    wc = pd.to_numeric(df["text_word_count"], errors="coerce").dropna()
    if wc.empty:
        logger.warning("Word count hist: no numeric values; skipping.")
        return
    wc_clipped = np.clip(wc, a_min=0, a_max=max_bin)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(wc_clipped, bins=50)
    ax.set_title(f"Opinion Text Word Count (clipped at {max_bin})")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    _finish_fig(os.path.join(OUTDIR, "text_wordcount_hist.png"))

def wordcount_by_outcome_box(df: pd.DataFrame, max_cap: int = 10000):
    if "text_word_count" not in df.columns or "outcome_code" not in df.columns:
        logger.warning("Wordcount by outcome: missing columns; skipping.")
        return
    wc = pd.to_numeric(df["text_word_count"], errors="coerce")
    dd = pd.DataFrame({"wc": wc, "y": df["outcome_code"]}).dropna()
    if dd.empty:
        logger.warning("Wordcount by outcome: no data; skipping.")
        return
    dd["wc"] = dd["wc"].clip(upper=max_cap)
    groups = [dd.loc[dd["y"] == k, "wc"].values for k in sorted(dd["y"].unique())]
    labels = _labelify(sorted(dd["y"].unique()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(f"Word Count by Outcome (cap {max_cap})")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Word Count")
    _finish_fig(os.path.join(OUTDIR, "wordcount_by_outcome_box.png"))

def generate_visualizations(df: pd.DataFrame):
    """
    Runs all charts and save to data/outputs/.
    """
    _ensure_outdir()
    try:
        outcome_distribution(df)
        top_outcomes_overall(df)
        cases_by_court_stacked(df, top_n=12)
        courts_pareto(df, top_n=20)
        opinions_over_time(df)
        text_wordcount_hist(df, max_bin=5000)
        wordcount_by_outcome_box(df, max_cap=10000)

        logger.info("Saved visualizations to %s", OUTDIR)
        print(f"Visualizations saved in {OUTDIR}")
    except Exception as e:
        logger.exception("Visualization step failed: %s", e)
        print("Visualization step failed:", e)
