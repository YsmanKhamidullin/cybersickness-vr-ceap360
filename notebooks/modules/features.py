"""
Извлечение признаков из CEAP-360VR.

Объединяет:
  - source/features/windowing.py    — нарезка на окна;
  - source/features/statistical.py  — универсальные статистики (9 моментов);
  - source/features/oculomotor.py   — pupil/gaze/head;
  - source/features/physio.py       — EDA/BVP/HR/SKT/ACC + HRV;
  - source/features/build_dataset.py — сборка feature_table.csv.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .loader import (
    load_annotations, load_behavior, load_ibi, load_physio, load_ssq_scores,
)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def iter_windows(
    df: pd.DataFrame,
    window_sec: float = config.WINDOW_SIZE_SEC,
    overlap: float = config.WINDOW_OVERLAP,
    time_col: str = "t_sec",
    group_cols: tuple = ("participant", "video"),
) -> Iterator[dict]:
    """Окна фиксированной длины с перекрытием в пределах одной (participant, video)."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap должен быть в [0.0, 1.0), получено {overlap}")
    step = window_sec * (1.0 - overlap)
    for keys, group_df in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        group_df = group_df.sort_values(time_col).reset_index(drop=True)
        t_max = float(group_df[time_col].iloc[-1])
        t_start = float(group_df[time_col].iloc[0])
        window_idx = 0
        while t_start + window_sec <= t_max + 1e-9:
            t_end = t_start + window_sec
            mask = (group_df[time_col] >= t_start) & (group_df[time_col] < t_end)
            window_df = group_df.loc[mask]
            if len(window_df) > 0:
                yield {**meta, "window_idx": window_idx,
                       "t_start": round(t_start, 3), "t_end": round(t_end, 3),
                       "data": window_df}
            window_idx += 1
            t_start += step


def count_windows(df, **kw) -> int:
    return sum(1 for _ in iter_windows(df, **kw))


# ---------------------------------------------------------------------------
# Универсальные статистики
# ---------------------------------------------------------------------------
def summarize_signal(arr, prefix: str) -> dict:
    """9 моментов: mean/std/min/max/median/range/skew/kurt/rms."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {f"{prefix}_{k}": np.nan for k in
                ["mean", "std", "min", "max", "median", "range", "skew", "kurt", "rms"]}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if arr.size >= 3 and std > 1e-10:
        skew = float(stats.skew(arr, bias=False))
        kurt = float(stats.kurtosis(arr, bias=False))
    else:
        skew = 0.0
        kurt = 0.0
    return {
        f"{prefix}_mean": mean, f"{prefix}_std": std,
        f"{prefix}_min": amin, f"{prefix}_max": amax,
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_range": amax - amin,
        f"{prefix}_skew": skew, f"{prefix}_kurt": kurt,
        f"{prefix}_rms": float(np.sqrt(np.mean(arr ** 2))),
    }


def slope(arr) -> float:
    """Линейный наклон сигнала (для медленных EDA/SKT)."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    x = np.arange(arr.size)
    a, _b = np.polyfit(x, arr, 1)
    return float(a)


# ---------------------------------------------------------------------------
# Окуломоторные признаки
# ---------------------------------------------------------------------------
def _angular_speed(pitch, yaw, dt):
    d_pitch = np.diff(pitch) / dt
    d_yaw = np.diff(yaw) / dt
    return np.sqrt(d_pitch ** 2 + d_yaw ** 2)


def extract_oculomotor_features(window_df: pd.DataFrame) -> dict:
    """Pupil + gaze + head ~40 чисел."""
    feats: dict = {}
    dt = 1.0 / config.FRAME_RATE_HZ

    lp = window_df["left_pupil"].to_numpy()
    rp = window_df["right_pupil"].to_numpy()
    feats.update(summarize_signal(lp, "left_pupil"))
    feats.update(summarize_signal(rp, "right_pupil"))
    feats.update(summarize_signal(lp - rp, "pupil_diff"))

    if len(lp) > 1:
        feats.update(summarize_signal(np.diff(lp) / dt, "left_pupil_deriv"))
    else:
        feats.update(summarize_signal([], "left_pupil_deriv"))

    gp = window_df["gaze_pitch"].to_numpy()
    gy = window_df["gaze_yaw"].to_numpy()
    feats.update(summarize_signal(gp, "gaze_pitch"))
    feats.update(summarize_signal(gy, "gaze_yaw"))
    if len(gp) > 1:
        feats.update(summarize_signal(_angular_speed(gp, gy, dt), "gaze_speed"))
    else:
        feats.update(summarize_signal([], "gaze_speed"))

    hp = window_df["head_pitch"].to_numpy()
    hy = window_df["head_yaw"].to_numpy()
    feats.update(summarize_signal(hp, "head_pitch"))
    feats.update(summarize_signal(hy, "head_yaw"))
    if len(hp) > 1:
        feats.update(summarize_signal(_angular_speed(hp, hy, dt), "head_speed"))
    else:
        feats.update(summarize_signal([], "head_speed"))

    return feats


# ---------------------------------------------------------------------------
# Физиологические признаки + HRV
# ---------------------------------------------------------------------------
def extract_physio_features(window_df: pd.DataFrame) -> dict:
    """EDA + HR + SKT + BVP + ACC."""
    feats: dict = {}

    eda = window_df["eda"].to_numpy()
    feats.update(summarize_signal(eda, "eda"))
    feats["eda_slope"] = slope(eda)
    if len(eda) >= 2 and not np.isnan(eda[0]) and not np.isnan(eda[-1]):
        feats["eda_delta"] = float(eda[-1] - eda[0])
    else:
        feats["eda_delta"] = np.nan

    hr = window_df["hr"].to_numpy()
    feats.update(summarize_signal(hr, "hr"))

    skt = window_df["skt"].to_numpy()
    feats.update(summarize_signal(skt, "skt"))
    feats["skt_slope"] = slope(skt)

    bvp = window_df["bvp"].to_numpy()
    feats.update(summarize_signal(bvp, "bvp"))

    ax = window_df["acc_x"].to_numpy()
    ay = window_df["acc_y"].to_numpy()
    az = window_df["acc_z"].to_numpy()
    feats.update(summarize_signal(np.sqrt(ax ** 2 + ay ** 2 + az ** 2), "acc_mag"))
    return feats


def extract_hrv_features(ibi_df: pd.DataFrame, t_start: float, t_end: float) -> dict:
    """SDNN, RMSSD, pNN50, mean_ibi, n_beats."""
    mask = (ibi_df["t_sec"] >= t_start) & (ibi_df["t_sec"] < t_end)
    ibi_ms = ibi_df.loc[mask, "ibi_sec"].to_numpy() * 1000.0
    feats = {"hrv_n_beats": int(len(ibi_ms)), "hrv_mean_ibi": np.nan,
             "hrv_sdnn": np.nan, "hrv_rmssd": np.nan, "hrv_pnn50": np.nan}
    if len(ibi_ms) < 3:
        return feats
    feats["hrv_mean_ibi"] = float(np.mean(ibi_ms))
    feats["hrv_sdnn"] = float(np.std(ibi_ms, ddof=1))
    diffs = np.diff(ibi_ms)
    feats["hrv_rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
    feats["hrv_pnn50"] = float(np.sum(np.abs(diffs) > 50) / len(diffs))
    return feats


# ---------------------------------------------------------------------------
# Сборка датасета
# ---------------------------------------------------------------------------
def _merge_on_time(beh_df, phy_df, ann_df) -> pd.DataFrame:
    """Объединить три потока (behavior/physio/annotations) по (participant, video, t_sec)."""
    merged = beh_df.merge(phy_df, on=["participant", "video", "t_sec"], how="inner")
    merged = merged.merge(
        ann_df[["participant", "video", "t_sec", "arousal", "valence"]],
        on=["participant", "video", "t_sec"], how="inner",
    )
    return merged


def extract_all_features_for_participant(participant_id: str) -> list[dict]:
    """Все окна одного участника — окуломоторика + физиология + HRV + targets."""
    beh = load_behavior(participant_id)
    phy = load_physio(participant_id)
    ann = load_annotations(participant_id)
    ibi = load_ibi(participant_id)
    ssq = load_ssq_scores(participant_id)

    ssq_post_total = float(ssq.loc[ssq["phase"] == "post", "TotalScore"].iloc[0])
    ssq_pre_total = float(ssq.loc[ssq["phase"] == "pre", "TotalScore"].iloc[0])
    merged = _merge_on_time(beh, phy, ann)

    rows = []
    for win in iter_windows(merged):
        data = win["data"]
        feats = {
            "participant": win["participant"], "video": win["video"],
            "window_idx": win["window_idx"],
            "t_start": win["t_start"], "t_end": win["t_end"],
        }
        feats.update(extract_oculomotor_features(data))
        feats.update(extract_physio_features(data))
        feats.update(extract_hrv_features(ibi, win["t_start"], win["t_end"]))
        feats["arousal_mean"] = float(data["arousal"].mean())
        feats["valence_mean"] = float(data["valence"].mean())
        feats["ssq_post_total"] = ssq_post_total
        feats["ssq_pre_total"] = ssq_pre_total
        feats["ssq_delta"] = ssq_post_total - ssq_pre_total
        rows.append(feats)
    return rows


def add_global_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Бинарный/тернарный arousal class (по глобальным процентилям) + ssq_high."""
    df = df.copy()
    median_arousal = df["arousal_mean"].median()
    df["arousal_class_bin"] = (df["arousal_mean"] > median_arousal).astype(int)
    q33 = df["arousal_mean"].quantile(0.33)
    q67 = df["arousal_mean"].quantile(0.67)
    df["arousal_class_tri"] = np.where(
        df["arousal_mean"] <= q33, 0,
        np.where(df["arousal_mean"] <= q67, 1, 2),
    ).astype(int)
    df["ssq_high"] = (df["ssq_post_total"] > config.SSQ_THRESHOLD_HIGH).astype(int)
    return df
