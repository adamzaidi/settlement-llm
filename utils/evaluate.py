# Quick helpers to evaluate models and save both numbers and simple charts.
# 1) Classification: accuracy, precision/recall/F1 (weighted), confusion matrix, optional ROC/AUC (binary only if you pass probabilities).
# 2) Regression: RMSE, MAE, R^2, plus a couple of helpful plots.
# Outputs go to: data/model-eval/

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_json(path, obj):
    """Save a Python dict/list as a JSON (make folder if missing)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def evaluate_classification(y_true, y_pred, y_proba=None, tag="baseline"):
    """
    Evaluate a classifier and write results to data/model-eval/.

    What it saves:
      - metrics_<tag>.json/.csv : accuracy, weighted precision/recall/F1 (+ roc_auc if possible)
      - confusion_<tag>.png     : confusion matrix heatmap with counts
      - roc_<tag>.png           : ROC curve (only for binary if y_proba is given)
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        RocCurveDisplay,
    )

    outdir = Path("data/model-eval"); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Basic numbers
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics.update({"precision_w": p, "recall_w": r, "f1_w": f1})

    # 2) Confusion matrix (with counts drawn on top)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix ({tag})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / f"confusion_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 3) ROC (only if binary + passing probabilities)
    if y_proba is not None and len(set(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            RocCurveDisplay.from_predictions(y_true, y_proba)
            plt.title(f"ROC Curve ({tag})")
            plt.tight_layout()
            plt.savefig(outdir / f"roc_{tag}.png", bbox_inches="tight", dpi=150)
            plt.close()
        except Exception:
            # If shapes/types are off, it skips ROC without crashing
            pass

    # 4) Saves numbers
    _save_json(outdir / f"metrics_{tag}.json", metrics)
    pd.DataFrame([metrics]).to_csv(outdir / f"metrics_{tag}.csv", index=False)
    return metrics


def evaluate_regression(y_true, y_pred, tag="baseline"):
    """
    Evaluate a regressor and write results to data/model-eval/.

    What it saves:
      - metrics_<tag>.json/.csv : rmse, mae, r2
      - ytrue_ypred_<tag>.png   : scatter plot of predictions vs. truth
      - residuals_<tag>.png     : histogram of errors (y_true - y_pred)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    outdir = Path("data/model-eval"); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Basic numbers
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    # 2) y_true vs y_pred (NOTE: good models should cluster near y = x)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=8)
    ax.set_title(f"y_true vs y_pred ({tag})")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi]) # the y = x reference
    plt.tight_layout()
    plt.savefig(outdir / f"ytrue_ypred_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 3) Residuals (errors) histogram
    res = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.hist(res, bins=30)
    ax.set_title(f"Residuals ({tag})")
    ax.set_xlabel("Error (y_true - y_pred)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / f"residuals_{tag}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 4) Save numbers
    _save_json(outdir / f"metrics_{tag}.json", metrics)
    pd.DataFrame([metrics]).to_csv(outdir / f"metrics_{tag}.csv", index=False)
    return metrics