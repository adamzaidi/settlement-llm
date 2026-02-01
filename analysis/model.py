# analysis/model.py
# -----------------------------------------------------------------------------
# Purpose
# - Build a small feature set, train baseline classifiers, and ALWAYS write
#   evaluation artifacts (even if training is skipped).
#
# Tool-grade upgrades baked in:
# - Label names match your transform codes (coarse/fine).
# - Save label mapping artifacts so outputs are interpretable without reading code.
# - Add a baseline (majority class) for honest context.
# - Add macro-F1 + weighted-F1 + accuracy to JSON metrics + evaluation_summary.json.
# - Make label order stable (0..max_code) so confusion matrices align across runs.
# - Add a slightly richer (still lightweight) feature set beyond court-only.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import logging
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

logger = logging.getLogger("pipeline")

# Output locations
EVAL_DIR = "data/model-eval"
EVAL_SUMMARY = os.path.join(EVAL_DIR, "evaluation_summary.json")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_eval_dir() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


def _save_json(obj: dict, path: str) -> None:
    _ensure_eval_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _append_summary(entry: dict) -> None:
    """
    Append or upsert an evaluation entry into evaluation_summary.json.
    Key: '{label}_{model}'
    """
    _ensure_eval_dir()
    if os.path.exists(EVAL_SUMMARY):
        try:
            with open(EVAL_SUMMARY, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    key = f"{entry.get('label')}_{entry.get('model', 'skip')}"
    data[key] = entry
    _save_json(data, EVAL_SUMMARY)


def _save_report_dict(rep: dict, path: str) -> None:
    try:
        _ensure_eval_dir()
        pd.DataFrame(rep).to_csv(path)
    except Exception as e:
        logger.warning("Could not save classification report to %s: %s", path, e)


def _safe_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _stable_label_list(label: str) -> List[int]:
    """
    Ensure confusion matrices and reports have stable axes across runs.
    Your transform uses:
      - coarse: 0..2
      - fine:   0..5
    """
    return list(range(0, 6)) if label == "fine" else list(range(0, 3))


def _code_to_name(label: str) -> Dict[int, str]:
    """
    Human-readable names matching YOUR codes (from transform.py semantics):
      - coarse outcome_code:
          0 other/unclear
          1 affirmed_or_dismissed
          2 changed_or_mixed (reversed/vacated/remanded/mixed)
      - fine outcome_code_fine:
          0 other
          1 affirmed
          2 reversed
          3 vacated
          4 remanded
          5 dismissed
    """
    if label == "fine":
        return {
            0: "other",
            1: "affirmed",
            2: "reversed",
            3: "vacated",
            4: "remanded",
            5: "dismissed",
        }
    return {
        0: "other",
        1: "affirmed_or_dismissed",
        2: "changed_or_mixed",
    }


def _save_label_mapping(label: str, y_col: str, mapping: Dict[int, str]) -> None:
    """
    Save an explicit mapping artifact so readers can interpret codes.
    """
    out = {
        "label": label,
        "y_col": y_col,
        "code_to_name": {str(k): v for k, v in mapping.items()},
    }
    _save_json(out, os.path.join(EVAL_DIR, f"label_mapping_{label}.json"))


def _metrics_row(y_true, y_pred) -> Dict[str, float]:
    """
    Include accuracy + macro-F1 + weighted-F1 (more honest for imbalance).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _save_metrics_json(label: str, model_name: str, y_col: str, metrics: Dict[str, Any]) -> None:
    path = os.path.join(EVAL_DIR, f"metrics_{model_name}_{label}.json")
    _save_json({"label": label, "model": model_name, "y_col": y_col, **metrics}, path)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight feature set:
      - court (one-hot)
      - opinion_year (numeric)
      - text_word_count bucket (categorical)
      - triage signals (numeric), if present
    Keeps things simple + resume-credible without embeddings.
    """
    feat = pd.DataFrame(index=df.index)

    # categorical
    feat["court"] = df.get("court").fillna("Unknown")

    # numeric year
    feat["opinion_year"] = pd.to_numeric(df.get("opinion_year"), errors="coerce").fillna(0).astype(int)

    # wordcount bucket (categorical)
    wc = pd.to_numeric(df.get("text_word_count"), errors="coerce").fillna(0)
    feat["wordcount_bucket"] = pd.cut(
        wc,
        bins=[-1, 300, 800, 1500, 3000, 10_000_000],
        labels=["<=300", "301-800", "801-1500", "1501-3000", "3001+"],
    ).astype(str)

    # triage signals (if missing, default 0)
    feat["disposition_zone_found"] = pd.to_numeric(df.get("disposition_zone_found"), errors="coerce").fillna(0).astype(int)
    feat["strong_phrase"] = pd.to_numeric(df.get("evidence_contains_strong_phrase"), errors="coerce").fillna(0).astype(int)
    feat["evidence_pos"] = pd.to_numeric(df.get("evidence_match_position"), errors="coerce").fillna(0.0).astype(float)

    # one-hot categorical columns
    X = pd.get_dummies(feat, columns=["court", "wordcount_bucket"], dummy_na=False)

    # safety: ensure no inf/nan
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    return X


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_models(df: pd.DataFrame, label: str = "coarse") -> Tuple[Optional[LogisticRegression], Optional[RandomForestClassifier]]:
    """
    Train and evaluate baselines over the selected label set.

    label='coarse' -> df['outcome_code']
    label='fine'   -> df['outcome_code_fine']

    Always writes artifacts under data/model-eval/:
      - classification reports (CSV)
      - confusion matrices (CSV)
      - metrics JSON for each model (accuracy + macro-F1 + weighted-F1)
      - label_mapping_{label}.json
      - evaluation_summary.json (status/metrics per model/label)
    """
    _ensure_eval_dir()

    y_col = "outcome_code_fine" if label == "fine" else "outcome_code"

    # Guardrails
    if "court" not in df.columns:
        raise ValueError("Expected column 'court' not found in processed DataFrame.")
    if y_col not in df.columns:
        raise ValueError(f"Expected target column '{y_col}' not found in processed DataFrame.")

    # Label mapping + stable label axis
    mapping = _code_to_name(label)
    labels_sorted = _stable_label_list(label)
    names_sorted = [mapping.get(i, str(i)) for i in labels_sorted]
    _save_label_mapping(label, y_col, mapping)

    # Features
    X = _build_features(df)

    # Target as int
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)

    # If only one class present, skip training but still write artifacts
    if y.nunique() < 2:
        msg = f"Only one class present in {y_col}: skipping model training."
        logger.warning(msg)
        class_counts = y.value_counts(dropna=False).to_dict()
        entry = {
            "status": "skipped",
            "reason": "only_one_class",
            "label": label,
            "y_col": y_col,
            "n_rows": int(len(y)),
            "class_counts": {str(_safe_int(k)): int(v) for k, v in class_counts.items()},
            "labels": labels_sorted,
            "label_names": names_sorted,
        }
        _append_summary({**entry, "model": "both"})
        _save_metrics_json(label, "both_skipped", y_col, entry)
        return None, None

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Baseline: majority class (most_frequent)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    preds_base = baseline.predict(X_test)

    base_metrics = _metrics_row(y_test, preds_base)
    _save_metrics_json(label, "baseline_most_frequent", y_col, {
        **base_metrics,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
    })
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "baseline_most_frequent",
        "y_col": y_col,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
        **base_metrics,
    })

    # Models (balanced to hedge imbalance)
    log_model = LogisticRegression(max_iter=1500, class_weight="balanced")
    rf_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        n_estimators=300,
        min_samples_leaf=2,
    )

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    preds_log = log_model.predict(X_test)
    preds_rf = rf_model.predict(X_test)

    # Reports with stable label order
    rep_log = classification_report(
        y_test, preds_log,
        labels=labels_sorted,
        target_names=names_sorted,
        zero_division=0,
        output_dict=True,
    )
    rep_rf = classification_report(
        y_test, preds_rf,
        labels=labels_sorted,
        target_names=names_sorted,
        zero_division=0,
        output_dict=True,
    )

    _save_report_dict(rep_log, f"{EVAL_DIR}/classification_report_logreg_{label}.csv")
    _save_report_dict(rep_rf,  f"{EVAL_DIR}/classification_report_rf_{label}.csv")

    # Confusion matrices with stable axes
    cm_log = confusion_matrix(y_test, preds_log, labels=labels_sorted)
    cm_rf  = confusion_matrix(y_test, preds_rf,  labels=labels_sorted)

    pd.DataFrame(cm_log, index=names_sorted, columns=names_sorted).to_csv(f"{EVAL_DIR}/confusion_logreg_{label}.csv")
    pd.DataFrame(cm_rf,  index=names_sorted, columns=names_sorted).to_csv(f"{EVAL_DIR}/confusion_rf_{label}.csv")

    # Metrics
    log_metrics = _metrics_row(y_test, preds_log)
    rf_metrics  = _metrics_row(y_test, preds_rf)

    logger.info("LogReg metrics (%s): acc=%.4f f1_macro=%.4f f1_w=%.4f", label, log_metrics["accuracy"], log_metrics["f1_macro"], log_metrics["f1_weighted"])
    logger.info("RF metrics (%s):     acc=%.4f f1_macro=%.4f f1_w=%.4f", label, rf_metrics["accuracy"], rf_metrics["f1_macro"], rf_metrics["f1_weighted"])
    logger.info("Baseline (%s):       acc=%.4f f1_macro=%.4f f1_w=%.4f", label, base_metrics["accuracy"], base_metrics["f1_macro"], base_metrics["f1_weighted"])

    _save_metrics_json(label, "logreg", y_col, {
        **log_metrics,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
    })
    _save_metrics_json(label, "random_forest", y_col, {
        **rf_metrics,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
    })

    # Summary JSON
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "logreg",
        "y_col": y_col,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
        **log_metrics,
    })
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "random_forest",
        "y_col": y_col,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
        **rf_metrics,
    })

    return log_model, rf_model