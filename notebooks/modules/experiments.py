"""
Общие helpers для exp01-exp06.

Объединяет:
  - source/experiments/exp01_baseline.py  — фабрики + run_cv;
  - source/experiments/exp02_per_subject.py — per-subject z-norm + per-subject target;
  - source/experiments/exp03_ssq_block.py — block-level dataset + LOO + block-XGB;
  - source/experiments/exp04_ablation.py  — channel/modality/stat ablations;
  - source/experiments/exp05_shap.py      — XGBoost для SHAP (тот же что exp03);
  - source/experiments/exp06_deep.py      — LSTM/BiLSTM, build_sequences.

Все гиперпараметры идентичны соответствующим source-файлам.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .validation import (
    compute_classification_metrics, subject_independent_splits,
)


# Метаданные/таргеты — не признаки
NON_FEATURE_COLS = {
    "participant", "video", "window_idx", "t_start", "t_end",
    "arousal_mean", "valence_mean",
    "ssq_post_total", "ssq_pre_total", "ssq_delta",
    "arousal_class_bin", "arousal_class_tri", "ssq_high",
}


# ---------------------------------------------------------------------------
# Per-subject преобразования (exp02)
# ---------------------------------------------------------------------------
def add_per_subject_target(df: pd.DataFrame) -> pd.DataFrame:
    """y = 1 если arousal_mean > личной медианы участника. Сбалансировано ~50/50."""
    out = df.copy()
    median_per_subj = out.groupby("participant")["arousal_mean"].transform("median")
    out["arousal_class_persubj"] = (out["arousal_mean"] > median_per_subj).astype(int)
    return out


def per_subject_zscore(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Каждый признак приведён к 0-mean / 1-std внутри одного участника."""
    out = df[feature_cols].copy()
    out = out.groupby(df["participant"]).transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() > 1e-9 else 1.0)
    )
    return out.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Window-level фабрики моделей (exp01 = 200 деревьев; exp02 = 300)
# ---------------------------------------------------------------------------
def make_logistic_regression_window() -> Pipeline:
    """LogReg для оконной задачи (exp01/exp02)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_SEED, n_jobs=-1,
        )),
    ])


def make_random_forest_exp01() -> Pipeline:
    """RF из exp01 (200 деревьев)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=config.RANDOM_SEED,
        )),
    ])


def make_random_forest_exp02() -> Pipeline:
    """RF из exp02 (300 деревьев) — также используется в exp04."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=None,
            n_jobs=-1, random_state=config.RANDOM_SEED,
        )),
    ])


def make_xgboost_window() -> Pipeline:
    """XGBoost для оконной задачи (exp02). Требует xgboost."""
    from xgboost import XGBClassifier
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1, random_state=config.RANDOM_SEED, verbosity=0,
        )),
    ])


def make_lightgbm_window() -> Pipeline:
    """LightGBM для оконной задачи (exp02). Требует lightgbm."""
    from lightgbm import LGBMClassifier
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(
            n_estimators=400, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, objective="binary",
            n_jobs=-1, random_state=config.RANDOM_SEED, verbosity=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Block-level фабрики (exp03 + exp05 одинаковые гиперпараметры)
# ---------------------------------------------------------------------------
def make_logistic_regression_block() -> Pipeline:
    """LogReg для block-level (32 точки) — сильная L2-регуляризация."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2", C=0.5, max_iter=2000,
            random_state=config.RANDOM_SEED,
        )),
    ])


def make_random_forest_block() -> Pipeline:
    """RF для block-level — ограниченная глубина против overfit."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=2,
            n_jobs=-1, random_state=config.RANDOM_SEED,
        )),
    ])


def make_xgboost_block() -> Pipeline:
    """XGBoost для block-level (главный результат диплома + SHAP в exp05)."""
    from xgboost import XGBClassifier
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1, random_state=config.RANDOM_SEED, verbosity=0,
        )),
    ])


# ---------------------------------------------------------------------------
# Прогон CV (window-level)
# ---------------------------------------------------------------------------
def run_cv(model_factory, X, y, splits, strategy_name, model_name) -> list[dict]:
    """Обучить модель на каждом train-фолде, собрать метрики по фолдам."""
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        m = model_factory()
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])
        try:
            y_proba = m.predict_proba(X[test_idx])
        except AttributeError:
            y_proba = None
        metrics = compute_classification_metrics(y[test_idx], y_pred, y_proba)
        rows.append({
            "model": model_name, "strategy": strategy_name, "fold": fold_i,
            "n_train": len(train_idx), "n_test": len(test_idx), **metrics,
        })
    return rows


def fold_summary_str(rows: list[dict]) -> str:
    acc = np.array([r["accuracy"] for r in rows])
    f1 = np.array([r["f1_macro"] for r in rows])
    auc = np.array([r["auc_roc"] for r in rows])
    return (f"acc={acc.mean():.4f}±{acc.std():.4f}  "
            f"f1={f1.mean():.4f}±{f1.std():.4f}  "
            f"auc={np.nanmean(auc):.4f}±{np.nanstd(auc):.4f}")


# ---------------------------------------------------------------------------
# Block-level dataset (exp03)
# ---------------------------------------------------------------------------
def build_block_level_dataset(windows: pd.DataFrame) -> pd.DataFrame:
    """
    Один участник — одна строка. Для каждого window-признака — _mean и _std
    по всем окнам сессии. Плюс таргеты SSQ.
    """
    feature_cols = [c for c in windows.columns if c not in NON_FEATURE_COLS]
    agg = windows.groupby("participant")[feature_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    targets = (
        windows.groupby("participant")[["ssq_post_total", "ssq_pre_total",
                                        "ssq_delta", "ssq_high"]]
        .first().reset_index()
    )
    return agg.merge(targets, on="participant", how="left")


def run_block_loso(X, y, model_factory, model_name) -> tuple[dict, np.ndarray, np.ndarray]:
    """LeaveOneOut на 32 точках. Вернуть метрики + предсказания + вероятности."""
    y_pred = np.zeros(len(y), dtype=int)
    y_proba = np.zeros(len(y), dtype=float)
    for tr, te in LeaveOneOut().split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
        try:
            y_proba[te] = m.predict_proba(X[te])[:, 1]
        except AttributeError:
            y_proba[te] = float("nan")
    metrics = compute_classification_metrics(y, y_pred, y_proba)
    metrics["model"] = model_name
    metrics["strategy"] = "LOSO"
    metrics["n_samples"] = int(len(y))
    return metrics, y_pred, y_proba


# ---------------------------------------------------------------------------
# Ablation helpers (exp04)
# ---------------------------------------------------------------------------
def classify_feature(col: str) -> tuple[str, str]:
    """Канал/модальность для одной колонки."""
    for prefix in ("left_pupil", "right_pupil", "pupil_diff"):
        if col.startswith(prefix):
            return "eye", "pupil"
    for prefix in ("gaze_pitch", "gaze_yaw", "gaze_speed"):
        if col.startswith(prefix):
            return "eye", "gaze"
    for prefix in ("head_pitch", "head_yaw", "head_speed"):
        if col.startswith(prefix):
            return "eye", "head"
    if col.startswith("eda"):
        return "physio", "eda"
    if col.startswith("hr_"):
        return "physio", "hr"
    if col.startswith("skt"):
        return "physio", "skt"
    if col.startswith("bvp"):
        return "physio", "bvp"
    if col.startswith("acc_mag"):
        return "physio", "acc"
    if col.startswith("hrv"):
        return "physio", "hrv"
    return "unknown", "unknown"


def select_columns(all_cols: list[str], modality=None, drop_channel=None,
                   keep_stat=None) -> list[str]:
    """Отфильтровать колонки по модальности/каналу/типу статистики."""
    out = []
    for col in all_cols:
        mod, ch = classify_feature(col)
        if modality is not None and mod != modality:
            continue
        if drop_channel is not None and ch == drop_channel:
            continue
        if keep_stat == "mean" and not col.endswith("_mean"):
            continue
        if keep_stat == "mean_std" and not (col.endswith("_mean") or col.endswith("_std")):
            continue
        out.append(col)
    return out


def run_window_loso_ablation(df, feature_cols) -> dict:
    """Window-level LOSO-метрики для подмножества признаков (exp04)."""
    X = per_subject_zscore(df, feature_cols)
    y = df["arousal_class_persubj"].to_numpy(dtype=int)
    splits = list(subject_independent_splits(df))
    acc, f1, auc = [], [], []
    for tr, te in splits:
        m = make_random_forest_exp02()
        m.fit(X[tr], y[tr])
        y_pred = m.predict(X[te])
        y_proba = m.predict_proba(X[te])
        mt = compute_classification_metrics(y[te], y_pred, y_proba)
        acc.append(mt["accuracy"])
        f1.append(mt["f1_macro"])
        auc.append(mt["auc_roc"])
    return {
        "n_features": len(feature_cols),
        "accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc)),
        "f1_mean": float(np.mean(f1)),         "f1_std": float(np.std(f1)),
        "auc_mean": float(np.nanmean(auc)),    "auc_std": float(np.nanstd(auc)),
    }


def run_block_loso_ablation(block, feature_cols) -> dict:
    """Block-level LOO XGBoost для подмножества признаков (exp04)."""
    X = block[feature_cols].to_numpy(dtype=float)
    y = block["ssq_high"].to_numpy(dtype=int)
    y_pred = np.zeros(len(y), dtype=int)
    y_proba = np.zeros(len(y), dtype=float)
    for tr, te in LeaveOneOut().split(X):
        m = make_xgboost_block()
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
        y_proba[te] = m.predict_proba(X[te])[:, 1]
    mt = compute_classification_metrics(y, y_pred, y_proba)
    return {
        "n_features": len(feature_cols),
        "accuracy_mean": mt["accuracy"], "accuracy_std": 0.0,
        "f1_mean": mt["f1_macro"],       "f1_std": 0.0,
        "auc_mean": mt["auc_roc"],       "auc_std": 0.0,
    }
