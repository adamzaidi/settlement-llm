# What this file does
# 1) Build a simple feature set (one-hot of 'court' for now).
# 2) Train Logistic Regression + Random Forest.
# 3) Save evaluation reports + confusion matrices to data/model-eval/.
# 4) Works for both coarse and fine label sets.
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger('pipeline')

FINE_NAMES   = ["Loss","Win","Mixed","Partial","Settlement","Other"]
COARSE_NAMES = ["Loss","Win","Settlement"]

def _save_report_dict(rep: dict, path: str):
    #Writes a classification_report (dict) to CSV.
    try:
        pd.DataFrame(rep).to_csv(path)
    except Exception as e:
        logger.warning("Could not save classification report to %s: %s", path, e)

def run_models(df: pd.DataFrame, label: str = "coarse"):
    """
    Train/eval two classifiers on either:
      - 'coarse' -> df['outcome_code'] (Loss/Win/Settlement)
      - 'fine'   -> df['outcome_code_fine'] (Loss/Win/Mixed/Partial/Settlement/Other)
    Saves classification reports and confusion matrices to data/model-eval/.
    """
    # 1) Chooses label column + friendly names
    y_col = "outcome_code_fine" if label == "fine" else "outcome_code"
    target_names_full = FINE_NAMES if label == "fine" else COARSE_NAMES

    # 2) Simple features (for now)
    X = pd.get_dummies(df[["court"]].fillna("Unknown"), dummy_na=False)
    y = df[y_col]

    if y.nunique() < 2:
        logger.warning("Only one class present in %s: skipping model training.", y_col)
        return None, None

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4) Build + fit models (class_weight='balanced' helps with imbalanced data)
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    rf_model  = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    preds_log = log_model.predict(X_test)
    preds_rf  = rf_model.predict(X_test)

    # 5) Saves classification reports + confusion matrices
    labels_sorted = sorted(y.unique())
    names_sorted = [(target_names_full[i] if i < len(target_names_full) else str(i)) for i in labels_sorted]

    rep_log = classification_report(y_test, preds_log, labels=labels_sorted, target_names=names_sorted, zero_division=0, output_dict=True)
    rep_rf  = classification_report(y_test, preds_rf,  labels=labels_sorted, target_names=names_sorted, zero_division=0, output_dict=True)

    os.makedirs("data/model-eval", exist_ok=True)
    _save_report_dict(rep_log, f"data/model-eval/classification_report_logreg_{label}.csv")
    _save_report_dict(rep_rf,  f"data/model-eval/classification_report_rf_{label}.csv")

    cm_log = confusion_matrix(y_test, preds_log, labels=labels_sorted)
    cm_rf  = confusion_matrix(y_test, preds_rf,  labels=labels_sorted)
    pd.DataFrame(cm_log, index=names_sorted, columns=names_sorted).to_csv(f"data/model-eval/confusion_logreg_{label}.csv")
    pd.DataFrame(cm_rf,  index=names_sorted, columns=names_sorted).to_csv(f"data/model-eval/confusion_rf_{label}.csv")

    # 6) Quick accuracy to console/logs
    acc_log = (preds_log == y_test).mean()
    acc_rf  = (preds_rf  == y_test).mean()
    logger.info("Logistic Regression accuracy (%s): %.4f", label, acc_log)
    logger.info("Random Forest accuracy (%s): %.4f", label, acc_rf)

    return log_model, rf_model