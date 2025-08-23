# evaluate.py
# -----------------------------------------------------------------------------
# Purpose:
#   Helpers I use to evaluate models and save both metrics and simple charts.
#   - For classification: accuracy, weighted precision/recall/F1, confusion
#     matrix, and ROC/AUC (if binary and probabilities are passed).
#   - For regression: RMSE, MAE, R^2, plus scatter and residual plots.
#   Everything saves into data/model-eval/ for later inspection.
# -----------------------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_json(path, obj):
    """Save a Python object as JSON (pretty formatted)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def evaluate_classification(y_true, y_pred, y_proba=None, tag="baseline"):
    """
    Evaluate a classifier and write results to data/model-eval/.

    Args:
        y_true: true labels
        y_pred: predicted labels
        y_proba: optional probabilities (only useful for ROC/AUC in binary case)
        tag: string tag to distinguish runs (e.g., 'logreg_coarse')

    What I save:
      - metrics_<tag>.json/.csv : accuracy, weighted precision/recall/F1 (+ roc_auc if possible)
      - confusion_<tag>.png     : confusion matrix with counts overlaid
      - roc_<tag>.png           : ROC curve (binary only if probabilities are given)
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        RocCurveDisplay,
    )

    outdir = Path("data/model-eval")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Metrics ---
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics.update({"precision_w": p, "recall_w": r, "f1_w": f1})

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix ({tag})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overlay counts
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / f"confusion_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- ROC curve (if binary and proba passed) ---
    if y_proba is not None and len(set(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            RocCurveDisplay.from_predictions(y_true, y_proba)
            plt.title(f"ROC Curve ({tag})")
            plt.tight_layout()
            plt.savefig(outdir / f"roc_{tag}.png", bbox_inches="tight", dpi=150)
            plt.close()
        except Exception:
            # If probs are wrong shape/type, I skip ROC without erroring
            pass

    # --- Save metrics ---
    _save_json(outdir / f"metrics_{tag}.json", metrics)
    pd.DataFrame([metrics]).to_csv(outdir / f"metrics_{tag}.csv", index=False)
    return metrics


def evaluate_regression(y_true, y_pred, tag="baseline"):
    """
    Evaluate a regressor and write results to data/model-eval/.

    Args:
        y_true: true values
        y_pred: predicted values
        tag: string tag to distinguish runs (e.g., 'rf_regressor')

    What I save:
      - metrics_<tag>.json/.csv : rmse, mae, r2
      - ytrue_ypred_<tag>.png   : scatter plot of predictions vs. truth
      - residuals_<tag>.png     : histogram of errors (y_true - y_pred)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    outdir = Path("data/model-eval")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Metrics ---
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    # --- Scatter plot y_true vs y_pred ---
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=8)
    ax.set_title(f"y_true vs y_pred ({tag})")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi])  # y = x reference line
    plt.tight_layout()
    plt.savefig(outdir / f"ytrue_ypred_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Residuals histogram ---
    res = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.hist(res, bins=30)
    ax.set_title(f"Residuals ({tag})")
    ax.set_xlabel("Error (y_true - y_pred)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / f"residuals_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Save metrics ---
    _save_json(outdir / f"metrics_{tag}.json", metrics)
    pd.DataFrame([metrics]).to_csv(outdir / f"metrics_{tag}.csv", index=False)
    return metrics