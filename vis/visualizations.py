# vis/visualizations.py
# -----------------------------------------------------------------------------
# What this file does
# 1) Produce easy-to-read charts (PNG) for the report:
#    - Outcome distribution (counts + percents)
#    - Top outcomes overall (barh)
#    - Cases by court & outcome (stacked barh, Top N + Other)
#    - Courts Pareto (bars + cumulative % line)
#    - Opinions over time
#    - Word count histogram + boxplot by outcome
#
# Tool-grade upgrades baked in:
# 2) "Model Comparison" charts (baseline vs logreg vs rf) from model metrics JSON.
# 3) "Review Queue Overview" charts from needs_review + outcome_confidence.
# 4) Correct outcome label mapping to match YOUR transform codes.
#
# 5) Save everything to data/outputs/
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger("pipeline")

OUTDIR = "data/outputs"
EVAL_DIR = "data/model-eval"

# -----------------------------------------------------------------------------
# Correct outcome mapping for YOUR transform codes
# outcome_code: 0 other, 1 affirmed/dismissed, 2 changed/mixed (reversed/vacated/remanded/mixed)
# -----------------------------------------------------------------------------
OUTCOME_LABELS = {
    0: "other",
    1: "affirmed_or_dismissed",
    2: "changed_or_mixed",
}
OUTCOME_ORDER = [0, 1, 2]  # stable display order


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _finish_fig(path: str) -> None:
    try:
        plt.tight_layout()
        try:
            plt.subplots_adjust(bottom=0.18, top=0.90)
        except Exception:
            pass
        plt.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close()


def _labelify(values) -> List[str]:
    """Map numeric outcome codes to human-readable labels."""
    out: List[str] = []
    for v in list(values):
        try:
            iv = int(v)
            out.append(OUTCOME_LABELS.get(iv, str(iv)))
        except Exception:
            out.append(str(v))
    return out


def _aggregate_top_n(series: pd.Series, top_n: int = 15, other_label: str = "Other") -> pd.Series:
    """Collapse long tail into 'Other' for readability."""
    counts = series.value_counts()
    if len(counts) <= top_n:
        return counts
    top = counts.head(top_n)
    tail_sum = counts.iloc[top_n:].sum()
    return pd.concat([top, pd.Series({other_label: tail_sum})])


def _barh_from_counts(
    counts: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    outname: str,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(counts)), counts.values)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finish_fig(os.path.join(OUTDIR, outname))


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_series_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


# -----------------------------------------------------------------------------
# Core charts
# -----------------------------------------------------------------------------
def outcome_distribution(df: pd.DataFrame) -> None:
    """Outcome distribution (counts + %)."""
    if "outcome_code" not in df.columns:
        logger.warning("Outcome distribution: 'outcome_code' not found; skipping.")
        return

    y = pd.to_numeric(df["outcome_code"], errors="coerce").fillna(-1).astype(int)
    counts = y.value_counts(dropna=False).sort_index()

    labels = ["Unknown" if i == -1 else OUTCOME_LABELS.get(i, str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, counts.values)
    ax.set_title("Outcome Distribution (Coarse) — Count & %")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")

    total = counts.sum()
    for b, v in zip(bars, counts.values):
        pct = (v / total * 100.0) if total else 0
        ax.annotate(
            f"{v} ({pct:.1f}%)",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _finish_fig(os.path.join(OUTDIR, "outcome_distribution.png"))


def top_outcomes_overall(df: pd.DataFrame) -> None:
    """Horizontal bar of most common coarse outcomes."""
    if "outcome_code" not in df.columns:
        logger.warning("Top outcomes: 'outcome_code' not found; skipping.")
        return

    def _label_outcome(x):
        if pd.isna(x):
            return "Unknown"
        try:
            return OUTCOME_LABELS.get(int(x), str(int(x)))
        except Exception:
            return "Unknown"

    counts = df["outcome_code"].map(_label_outcome).value_counts()

    _barh_from_counts(
        counts=counts.sort_values(ascending=True),
        title="Top Outcomes Overall (Coarse)",
        xlabel="Count",
        ylabel="Outcome",
        outname="top_outcomes_overall.png",
        figsize=(9, 4.8),
    )


def cases_by_court_stacked(df: pd.DataFrame, top_n: int = 12) -> None:
    """Stacked barh: cases by court split by coarse outcome, top N + other."""
    if "court" not in df.columns or "outcome_code" not in df.columns:
        logger.warning("Stacked by court: missing 'court' or 'outcome_code'; skipping.")
        return

    top_counts = _aggregate_top_n(df["court"].fillna("Unknown"), top_n=top_n, other_label="Other")
    keep_courts = list(top_counts.index)

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

    fig, ax = plt.subplots(figsize=(11.5, 7))
    left = np.zeros(len(pivot), dtype=float)
    y_pos = np.arange(len(pivot.index))

    for c in cols:
        vals = pivot[c].values
        label = OUTCOME_LABELS.get(int(c), str(c))
        ax.barh(y_pos, vals, left=left, label=label)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Court")
    ax.set_title(f"Cases by Court & Outcome (Top {top_n} Courts + Other)")
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    _finish_fig(os.path.join(OUTDIR, "cases_by_court_stacked.png"))


def courts_pareto(df: pd.DataFrame, top_n: int = 20) -> None:
    """Pareto: courts by volume + cumulative %."""
    if "court" not in df.columns:
        logger.warning("Courts Pareto: 'court' not found; skipping.")
        return

    counts_full = df["court"].fillna("Unknown").value_counts()
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
    ax1.set_ylabel("Count")
    ax1.set_title(f"Courts by Volume — Pareto (Top {top_n} + Other)")

    ax1.tick_params(axis="x", labelrotation=45)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")

    ax2 = ax1.twinx()
    ax2.plot(range(len(cum_pct)), cum_pct.values, marker="o")
    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 110)

    if len(cum_pct) > 0:
        for idx in {max(len(cum_pct) // 2, 0), len(cum_pct) - 1}:
            ax2.annotate(
                f"{cum_pct.iloc[idx]:.0f}%",
                (idx, cum_pct.iloc[idx]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )

    _finish_fig(os.path.join(OUTDIR, "courts_pareto.png"))


def opinions_over_time(df: pd.DataFrame) -> None:
    """Line: opinions per year."""
    if "opinion_year" not in df.columns:
        logger.warning("Opinions over time: 'opinion_year' not found; skipping.")
        return

    years = pd.to_numeric(df["opinion_year"], errors="coerce").dropna().astype(int)
    if years.empty:
        logger.warning("Opinions over time: no valid years; skipping.")
        return

    series = years.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(series.index, series.values, marker="o")
    ax.set_title("Opinions Over Time (by Year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    _finish_fig(os.path.join(OUTDIR, "opinions_over_time.png"))


def text_wordcount_hist(df: pd.DataFrame, max_bin: int = 5000) -> None:
    """Histogram: word counts (clipped)."""
    if "text_word_count" not in df.columns:
        logger.warning("Word count hist: 'text_word_count' not found; skipping.")
        return

    wc = pd.to_numeric(df["text_word_count"], errors="coerce").dropna()
    if wc.empty:
        logger.warning("Word count hist: no numeric values; skipping.")
        return

    wc_clipped = np.clip(wc, a_min=0, a_max=max_bin)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(wc_clipped, bins=50)
    ax.set_title(f"Opinion Text Word Count (clipped at {max_bin})")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    _finish_fig(os.path.join(OUTDIR, "text_wordcount_hist.png"))


def wordcount_by_outcome_box(df: pd.DataFrame, max_cap: int = 10000) -> None:
    """Boxplot: word counts grouped by coarse outcome."""
    if "text_word_count" not in df.columns or "outcome_code" not in df.columns:
        logger.warning("Wordcount by outcome: missing columns; skipping.")
        return

    wc = pd.to_numeric(df["text_word_count"], errors="coerce")
    dd = pd.DataFrame({"wc": wc, "y": df["outcome_code"]}).dropna()
    if dd.empty:
        logger.warning("Wordcount by outcome: no data; skipping.")
        return

    dd["wc"] = dd["wc"].clip(upper=max_cap)

    ys = sorted([int(x) for x in dd["y"].unique() if pd.notna(x)])
    groups = [dd.loc[dd["y"] == k, "wc"].values for k in ys]
    labels = _labelify(ys)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(f"Word Count by Outcome (cap {max_cap})")
    ax.set_xlabel("Outcome (coarse)")
    ax.set_ylabel("Word Count")
    _finish_fig(os.path.join(OUTDIR, "wordcount_by_outcome_box.png"))


# -----------------------------------------------------------------------------
# Tool-grade additions
# -----------------------------------------------------------------------------
def review_queue_overview(df: pd.DataFrame) -> None:
    """
    Review queue charts:
      1) counts + % flagged
      2) confidence histogram split by needs_review
    """
    needed = {"needs_review", "outcome_confidence"}
    if not needed.issubset(set(df.columns)):
        missing = sorted(list(needed - set(df.columns)))
        logger.warning("Review queue overview: missing %s; skipping.", ", ".join(missing))
        return

    _ensure_outdir()

    nr = pd.to_numeric(df["needs_review"], errors="coerce").fillna(1).astype(int)
    conf = pd.to_numeric(df["outcome_confidence"], errors="coerce").fillna(0.0).astype(float)

    pct_review = float((nr == 1).mean() * 100.0)
    counts = pd.Series(
        {"auto_pass": int((nr == 0).sum()), "needs_review": int((nr == 1).sum())}
    )

    # 1) bucket chart
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(counts.index, counts.values)
    ax.set_title(f"Review Queue Overview — {pct_review:.1f}% flagged")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Count")
    for b, v in zip(bars, counts.values):
        ax.annotate(
            str(v),
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _finish_fig(os.path.join(OUTDIR, "review_queue_overview.png"))

    # 2) confidence histogram split by triage
    conf_review = conf[nr == 1].values
    conf_auto = conf[nr == 0].values

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(conf_review, bins=20, alpha=0.9, label="needs_review")
    ax.hist(conf_auto, bins=20, alpha=0.9, label="auto_pass")
    ax.set_title("Outcome Confidence Histogram (split by triage)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.legend()
    _finish_fig(os.path.join(OUTDIR, "confidence_hist_by_triage.png"))


def model_comparison_from_metrics(label: str = "coarse") -> None:
    """
    Model comparison charts from metrics JSON produced by analysis/model.py.

    Looks for:
      - metrics_baseline_most_frequent_{label}.json
      - metrics_logreg_{label}.json
      - metrics_random_forest_{label}.json
    """
    _ensure_outdir()

    expected = [
        ("baseline_most_frequent", os.path.join(EVAL_DIR, f"metrics_baseline_most_frequent_{label}.json")),
        ("logreg", os.path.join(EVAL_DIR, f"metrics_logreg_{label}.json")),
        ("random_forest", os.path.join(EVAL_DIR, f"metrics_random_forest_{label}.json")),
    ]

    rows = []
    for name, path in expected:
        if not os.path.exists(path):
            continue
        j = _read_json(path)
        if not j:
            continue
        rows.append(
            {
                "model": name,
                "accuracy": float(j.get("accuracy", 0.0)),
                "f1_macro": float(j.get("f1_macro", 0.0)),
                "f1_weighted": float(j.get("f1_weighted", 0.0)),
            }
        )

    if not rows:
        logger.warning("Model comparison: no metrics JSON files found for label=%s; skipping.", label)
        return

    mdf = pd.DataFrame(rows).set_index("model").sort_values("f1_macro", ascending=True)

    # Macro-F1 chart
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(mdf.index, mdf["f1_macro"].values)
    ax.set_title(f"Model Comparison ({label}) — Macro F1")
    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Model")
    for i, v in enumerate(mdf["f1_macro"].values):
        ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(6, -2), va="center", fontsize=9)
    _finish_fig(os.path.join(OUTDIR, f"model_comparison_f1_macro_{label}.png"))

    # Accuracy chart
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(mdf.index, mdf["accuracy"].values)
    ax.set_title(f"Model Comparison ({label}) — Accuracy")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Model")
    for i, v in enumerate(mdf["accuracy"].values):
        ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(6, -2), va="center", fontsize=9)
    _finish_fig(os.path.join(OUTDIR, f"model_comparison_accuracy_{label}.png"))


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def generate_visualizations(df: pd.DataFrame) -> None:
    """
    Run all chart functions and save PNGs to data/outputs.
    """
    _ensure_outdir()
    try:
        # Existing charts
        outcome_distribution(df)
        top_outcomes_overall(df)
        cases_by_court_stacked(df, top_n=12)
        courts_pareto(df, top_n=20)
        opinions_over_time(df)
        text_wordcount_hist(df, max_bin=5000)
        wordcount_by_outcome_box(df, max_cap=10000)

        # Tool-grade additions
        review_queue_overview(df)
        model_comparison_from_metrics(label="coarse")
        # fine chart will render if fine metrics exist (safe to call)
        model_comparison_from_metrics(label="fine")

        logger.info("Saved visualizations to %s", OUTDIR)
        print(f"Visualizations saved in {OUTDIR}")
    except Exception as e:
        logger.exception("Visualization step failed: %s", e)
        print("Visualization step failed:", e)
        raise