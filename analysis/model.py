# analysis/model.py
# -----------------------------------------------------------------------------
# I build a minimal feature set, train two baseline classifiers, and always
# write evaluation artifacts (even if training is skipped).
#
# CHANGE (tool-grade upgrades):
# - Fix label names to match your actual transform codes (coarse/fine).
# - Save a label mapping artifact so outputs are interpretable without reading code.
# - Add a simple baseline (majority class) for honest context.
# - Add macro-F1 + weighted-F1 + accuracy to JSON/CSV metrics.
# - Make label order stable (always 0..max_code) so confusion matrices align.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import logging
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

logger = logging.getLogger("pipeline")

# Output locations for evaluation artifacts
EVAL_DIR = "data/model-eval"
EVAL_SUMMARY = os.path.join(EVAL_DIR, "evaluation_summary.json")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_eval_dir() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


def _save_json(obj: dict, path: str) -> None:
    _ensure_eval_dir()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _append_summary(entry: dict) -> None:
    """
    Append or upsert an evaluation entry into evaluation_summary.json.
    Key: '{label}_{model}'
    """
    _ensure_eval_dir()
    if os.path.exists(EVAL_SUMMARY):
        try:
            with open(EVAL_SUMMARY, "r") as f:
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


def _stable_label_list(df: pd.DataFrame, y_col: str, label: str) -> List[int]:
    """
    CHANGE: ensure confusion matrices have stable axes across runs.
    If the data doesn't contain all codes, we still keep the expected range.
    """
    if label == "fine":
        # transform.py uses 0..5 for fine (other, affirmed, reversed, vacated, remanded, dismissed)
        return list(range(0, 6))
    # coarse uses 0..2
    return list(range(0, 3))


def _infer_code_to_name(df: pd.DataFrame, y_col: str, label: str) -> Dict[int, str]:
    """
    CHANGE: derive human-readable names that match YOUR codes.
    - coarse: outcome_code (0 other/unknown, 1 affirmed/dismissed, 2 changed (reversed/vacated/remanded/mixed))
    - fine: outcome_code_fine (0 other, 1 affirmed, 2 reversed, 3 vacated, 4 remanded, 5 dismissed)
    """
    if label == "fine":
        default = {
            0: "other",
            1: "affirmed",
            2: "reversed",
            3: "vacated",
            4: "remanded",
            5: "dismissed",
        }
        return default

    # coarse
    default = {
        0: "other",
        1: "affirmed_or_dismissed",
        2: "changed_or_mixed",
    }
    return default


def _save_label_mapping(label: str, y_col: str, mapping: Dict[int, str]) -> None:
    """
    CHANGE: save an explicit mapping artifact so graders can interpret codes.
    """
    _ensure_eval_dir()
    out = {
        "label": label,
        "y_col": y_col,
        "code_to_name": {str(k): v for k, v in mapping.items()},
    }
    _save_json(out, os.path.join(EVAL_DIR, f"label_mapping_{label}.json"))


def _metrics_row(y_true, y_pred) -> Dict[str, float]:
    """
    CHANGE: include macro-F1 + weighted-F1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _save_metrics_json(label: str, model_name: str, y_col: str, metrics: Dict[str, Any]) -> None:
    _ensure_eval_dir()
    path = os.path.join(EVAL_DIR, f"metrics_{model_name}_{label}.json")
    _save_json({"label": label, "model": model_name, "y_col": y_col, **metrics}, path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_models(df: pd.DataFrame, label: str = "coarse") -> Tuple[Optional[LogisticRegression], Optional[RandomForestClassifier]]:
    """
    Train and evaluate two baseline classifiers over the selected label set.

    label='coarse' -> df['outcome_code'] (0 other, 1 affirmed/dismissed, 2 changed/mixed)
    label='fine'   -> df['outcome_code_fine'] (0 other, 1 affirmed, 2 reversed, 3 vacated, 4 remanded, 5 dismissed)

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

    # CHANGE: proper label names + saved mapping
    code_to_name = _infer_code_to_name(df, y_col, label)
    _save_label_mapping(label, y_col, code_to_name)

    # Minimal, deterministic features: one-hot encode 'court'
    X = pd.get_dummies(df[["court"]].fillna("Unknown"), dummy_na=False)

    # Force y to int-ish where possible (your transform already does this)
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)

    # Stable label list (even if some classes are absent)
    labels_sorted = _stable_label_list(df, y_col, label)
    names_sorted = [code_to_name.get(i, str(i)) for i in labels_sorted]

    # If there's only one class present, skip training but still record an artifact.
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

    # Standard train/test split. Stratify on y to preserve class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # CHANGE: baseline majority-class model (important for credibility)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    preds_base = baseline.predict(X_test)

    base_metrics = _metrics_row(y_test, preds_base)
    _save_metrics_json(label, "baseline_most_frequent", y_col, {
        **base_metrics,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted
    })
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "baseline_most_frequent",
        "y_col": y_col,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
        **base_metrics
    })

    # Baseline models with class_weight='balanced' to hedge against imbalance.
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

    # Build reports (keep stable label order)
    rep_log = classification_report(
        y_test, preds_log, labels=labels_sorted, target_names=names_sorted,
        zero_division=0, output_dict=True
    )
    rep_rf = classification_report(
        y_test, preds_rf, labels=labels_sorted, target_names=names_sorted,
        zero_division=0, output_dict=True
    )

    # Persist reports + confusion matrices
    _save_report_dict(rep_log, f"{EVAL_DIR}/classification_report_logreg_{label}.csv")
    _save_report_dict(rep_rf,  f"{EVAL_DIR}/classification_report_rf_{label}.csv")

    cm_log = confusion_matrix(y_test, preds_log, labels=labels_sorted)
    cm_rf  = confusion_matrix(y_test, preds_rf,  labels=labels_sorted)

    pd.DataFrame(cm_log, index=names_sorted, columns=names_sorted).to_csv(f"{EVAL_DIR}/confusion_logreg_{label}.csv")
    pd.DataFrame(cm_rf,  index=names_sorted, columns=names_sorted).to_csv(f"{EVAL_DIR}/confusion_rf_{label}.csv")

    # CHANGE: metrics that actually matter for imbalanced classes
    log_metrics = _metrics_row(y_test, preds_log)
    rf_metrics  = _metrics_row(y_test, preds_rf)

    logger.info("LogReg metrics (%s): acc=%.4f f1_macro=%.4f", label, log_metrics["accuracy"], log_metrics["f1_macro"])
    logger.info("RF metrics (%s):     acc=%.4f f1_macro=%.4f", label, rf_metrics["accuracy"], rf_metrics["f1_macro"])
    logger.info("Baseline (%s):       acc=%.4f f1_macro=%.4f", label, base_metrics["accuracy"], base_metrics["f1_macro"])

    _save_metrics_json(label, "logreg", y_col, {
        **log_metrics,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted
    })
    _save_metrics_json(label, "random_forest", y_col, {
        **rf_metrics,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted
    })

    # Record in cumulative summary JSON
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "logreg",
        "y_col": y_col,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
        **log_metrics
    })
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "random_forest",
        "y_col": y_col,
        "n_test": int(len(y_test)),
        "labels": labels_sorted,
        "label_names": names_sorted,
        **rf_metrics
    })

    return log_model, rf_model