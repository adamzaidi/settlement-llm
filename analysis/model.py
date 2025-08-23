# analysis/model.py
# -----------------------------------------------------------------------------
# I build a minimal feature set, train two baseline classifiers, and always
# write evaluation artifacts (even if training is skipped).
# - Features: one-hot of 'court' (simple but deterministic).
# - Models: Logistic Regression and Random Forest.
# - Outputs: classification reports + confusion matrices under data/model-eval/,
#            plus a cumulative evaluation_summary.json.
# -----------------------------------------------------------------------------

import os
import json
import logging
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("pipeline")

# Human-readable label names for report formatting
FINE_NAMES   = ["Loss", "Win", "Mixed", "Partial", "Settlement", "Other"]
COARSE_NAMES = ["Loss", "Win", "Settlement"]

# Output locations for evaluation artifacts
EVAL_DIR = "data/model-eval"
EVAL_SUMMARY = os.path.join(EVAL_DIR, "evaluation_summary.json")


def _ensure_eval_dir() -> None:
    """
    Ensure the evaluation output directory exists.

    Creates the directory if it doesn't already exist so all downstream
    file writes succeed without callers worrying about filesystem state.
    """
    os.makedirs(EVAL_DIR, exist_ok=True)


def _save_json(obj: dict, path: str) -> None:
    """
    Saves a Python object as a JSON.

    Parameters
    ----------
    obj : dict
        The JSON-serializable object to write.
    path : str
        Destination file path.
    """
    _ensure_eval_dir()
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _append_summary(entry: dict) -> None:
    """
    Append or upsert an evaluation entry into evaluation_summary.json.

    I key entries by a composite of `{label}_{model}` so each model/label
    pair stays unique while later runs can update in place.

    Parameters
    ----------
    entry : dict
        A small dictionary containing status/metrics for one model/label pair.
        Expected keys: 'label', 'model', plus any metrics/status fields.
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
    """
    Persist a sklearn `classification_report(..., output_dict=True)` to CSV.

    Guards the write so the pipeline never crashes due to a transient
    serialization or filesystem issue.

    Parameters
    ----------
    rep : dict
        The classification report in dict form.
    path : str
        Destination CSV path.
    """
    try:
        _ensure_eval_dir()
        pd.DataFrame(rep).to_csv(path)
    except Exception as e:
        logger.warning("Could not save classification report to %s: %s", path, e)


def run_models(df: pd.DataFrame, label: str = "coarse") -> Tuple[Optional[LogisticRegression], Optional[RandomForestClassifier]]:
    """
    Train and evaluate two baseline classifiers over the selected label set.

    I support two label granularities:
      - label='coarse' -> df['outcome_code'] (Loss/Win/Settlement)
      - label='fine'   -> df['outcome_code_fine'] (Loss/Win/Mixed/Partial/Settlement/Other)

    I always write evaluation artifacts under data/model-eval/:
      - classification reports (CSV) for both models
      - confusion matrices (CSV) for both models
      - evaluation_summary.json with status/metrics (even if training is skipped)

    Parameters
    ----------
    df : pandas.DataFrame
        The processed dataset produced by the transform step. Must contain
        'court' and the requested label column.
    label : {'coarse','fine'}, default 'coarse'
        Which target label set to use.

    Returns
    -------
    (log_model, rf_model) : tuple or (None, None)
        Fitted model objects if training occurred; otherwise (None, None).
    """
    # Select target column + friendly names for report formatting
    y_col = "outcome_code_fine" if label == "fine" else "outcome_code"
    target_names_full = FINE_NAMES if label == "fine" else COARSE_NAMES

    # Minimal feature set: one-hot encode 'court'
    # I keep this intentionally simple and deterministic for grading clarity.
    X = pd.get_dummies(df[["court"]].fillna("Unknown"), dummy_na=False)
    y = df[y_col]

    # If there's only one class present, I skip training but still record an artifact.
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
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        }
        _append_summary({**entry, "model": "both"})
        return None, None

    # Standard train/test split. Stratify on y to preserve class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Baseline models with class_weight='balanced' to hedge against imbalance.
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    preds_log = log_model.predict(X_test)
    preds_rf = rf_model.predict(X_test)

    # Keep label order stable across artifacts.
    labels_sorted = sorted(y.unique())
    names_sorted = [(target_names_full[i] if i < len(target_names_full) else str(i)) for i in labels_sorted]

    # Build reports
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
    cm_rf = confusion_matrix(y_test, preds_rf, labels=labels_sorted)
    pd.DataFrame(cm_log, index=names_sorted, columns=names_sorted).to_csv(f"{EVAL_DIR}/confusion_logreg_{label}.csv")
    pd.DataFrame(cm_rf,  index=names_sorted, columns=names_sorted).to_csv(f"{EVAL_DIR}/confusion_rf_{label}.csv")

    # Log quick accuracy and record both entries in the JSON summary
    acc_log = float((preds_log == y_test).mean())
    acc_rf = float((preds_rf == y_test).mean())
    logger.info("Logistic Regression accuracy (%s): %.4f", label, acc_log)
    logger.info("Random Forest accuracy (%s): %.4f", label, acc_rf)

    _append_summary({
        "status": "ok",
        "label": label,
        "model": "logreg",
        "accuracy": acc_log,
        "n_test": int(len(y_test)),
        "labels": [int(x) for x in labels_sorted],
        "label_names": names_sorted
    })
    _append_summary({
        "status": "ok",
        "label": label,
        "model": "random_forest",
        "accuracy": acc_rf,
        "n_test": int(len(y_test)),
        "labels": [int(x) for x in labels_sorted],
        "label_names": names_sorted
    })

    return log_model, rf_model