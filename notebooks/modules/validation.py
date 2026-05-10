"""
Splitters + metrics. Зеркало source/validation/{splitters,metrics}.py.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------
def subject_independent_splits(df: pd.DataFrame) -> Iterator:
    """LOSO: test = один участник, train = все остальные. 32 фолда."""
    for test_pid in df["participant"].unique():
        test_mask = (df["participant"] == test_pid).to_numpy()
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def subject_dependent_splits(df: pd.DataFrame, y: np.ndarray, n_folds: int = 5) -> Iterator:
    """StratifiedKFold по окнам — участники смешиваются. Верхняя граница качества."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
    for tr, te in skf.split(df, y):
        yield tr, te


def video_independent_splits(df: pd.DataFrame) -> Iterator:
    """LOVO: test = одно видео, train = остальные 7."""
    for test_video in df["video"].unique():
        test_mask = (df["video"] == test_video).to_numpy()
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """accuracy, precision_macro, recall_macro, f1_macro, auc_roc."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                proba = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
                out["auc_roc"] = float(roc_auc_score(y_true, proba))
            else:
                out["auc_roc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            out["auc_roc"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
    return out


def compute_regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, R²."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
