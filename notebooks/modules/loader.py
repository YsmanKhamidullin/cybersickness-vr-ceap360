"""
Загрузка данных CEAP-360VR + анализ качества данных по участнику.

Зеркало `source/data/loader.py` + `source/data/quality.py`, объединено для удобства.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Опросники (SSQ)
# ---------------------------------------------------------------------------
def _parse_ssq_string(ssq_str: str) -> list[int]:
    values = [int(x) for x in ssq_str.split()]
    if len(values) != 16:
        raise ValueError(f"Ожидалось 16 SSQ-ответов, получено {len(values)}: {ssq_str!r}")
    return values


def _ssq_total_score(ssq_raw: list[int]) -> dict:
    zero_based = [v - 1 for v in ssq_raw]  # 0..3
    n_sum = sum(zero_based[i] for i in config.SSQ_ITEMS_NAUSEA)
    o_sum = sum(zero_based[i] for i in config.SSQ_ITEMS_OCULOMOTOR)
    d_sum = sum(zero_based[i] for i in config.SSQ_ITEMS_DISORIENTATION)
    return {
        "Nausea": n_sum * config.SSQ_WEIGHT_NAUSEA,
        "Oculomotor": o_sum * config.SSQ_WEIGHT_OCULOMOTOR,
        "Disorientation": d_sum * config.SSQ_WEIGHT_DISORIENTATION,
        "TotalScore": (n_sum + o_sum + d_sum) * config.SSQ_WEIGHT_TOTAL,
    }


def load_ssq_scores(participant_id: str) -> pd.DataFrame:
    path = config.QUESTIONNAIRE_DIR / f"{participant_id}_Questionnaire_Data.json"
    raw = _read_json(path)
    q = raw["QuestionnaireData"][0]
    rows = []
    for phase, key in [("pre", "SSQ 0"), ("mid", "SSQ 1"), ("post", "SSQ 2")]:
        scores = _ssq_total_score(_parse_ssq_string(q[key]))
        rows.append({"participant": participant_id, "phase": phase, **scores})
    return pd.DataFrame(rows)


def load_demographics(participant_id: str) -> dict:
    path = config.QUESTIONNAIRE_DIR / f"{participant_id}_Questionnaire_Data.json"
    q = _read_json(path)["QuestionnaireData"][0]
    return {
        "participant": participant_id,
        "age": q.get("Age"),
        "gender": q.get("Gender"),
        "vr_experience": q.get("VR Experience"),
        "profession": q.get("Profession"),
        "video_order": q.get("VideoOrder", "").strip(),
    }


# ---------------------------------------------------------------------------
# 2. Аннотации (valence/arousal)
# ---------------------------------------------------------------------------
def load_annotations(participant_id: str) -> pd.DataFrame:
    path = config.ANNOTATION_FRAME_DIR / f"{participant_id}_Annotation_FrameData.json"
    raw = _read_json(path)
    blocks = raw["ContinuousAnnotation_FrameData"][0]["Video_Annotation_FrameData"]
    rows = []
    for blk in blocks:
        video = blk["VideoID"]
        for s in blk["TimeStamp_Valence_Arousal"]:
            rows.append({
                "participant": participant_id, "video": video,
                "t_sec": s["TimeStamp"], "valence": s["Valence"], "arousal": s["Arousal"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Поведение (айтрекинг + голова)
# ---------------------------------------------------------------------------
def load_behavior(participant_id: str) -> pd.DataFrame:
    path = config.BEHAVIOR_FRAME_DIR / f"{participant_id}_Behavior_FrameData.json"
    raw = _read_json(path)
    blocks = raw["Behavior_FrameData"][0]["Video_Behavior_FrameData"]

    frames = []
    for blk in blocks:
        video = blk["VideoID"]
        hm, em, lem, rem = blk["HM"], blk["EM"], blk["LEM"], blk["REM"]
        lpd, rpd = blk["LPD"], blk["RPD"]
        n = len(hm)
        df = pd.DataFrame({
            "participant": participant_id, "video": video,
            "t_sec":           [hm[i]["TimeStamp"] for i in range(n)],
            "head_pitch":      [hm[i]["Pitch"] for i in range(n)],
            "head_yaw":        [hm[i]["Yaw"] for i in range(n)],
            "gaze_pitch":      [em[i]["Pitch"] for i in range(n)],
            "gaze_yaw":        [em[i]["Yaw"] for i in range(n)],
            "left_eye_pitch":  [lem[i]["Pitch"] for i in range(n)],
            "left_eye_yaw":    [lem[i]["Yaw"] for i in range(n)],
            "right_eye_pitch": [rem[i]["Pitch"] for i in range(n)],
            "right_eye_yaw":   [rem[i]["Yaw"] for i in range(n)],
            "left_pupil":      [lpd[i]["PD"] for i in range(n)],
            "right_pupil":     [rpd[i]["PD"] for i in range(n)],
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Физиология (Empatica E4)
# ---------------------------------------------------------------------------
def load_physio(participant_id: str) -> pd.DataFrame:
    path = config.PHYSIO_FRAME_DIR / f"{participant_id}_Physio_FrameData.json"
    raw = _read_json(path)
    blocks = raw["Physio_FrameData"][0]["Video_Physio_FrameData"]

    frames = []
    for blk in blocks:
        video = blk["VideoID"]
        acc, skt = blk["ACC_FrameData"], blk["SKT_FrameData"]
        eda, bvp, hr = blk["EDA_FrameData"], blk["BVP_FrameData"], blk["HR_FrameData"]
        n = len(eda)
        df = pd.DataFrame({
            "participant": participant_id, "video": video,
            "t_sec": [eda[i]["TimeStamp"] for i in range(n)],
            "eda":   [eda[i]["EDA"] for i in range(n)],
            "bvp":   [bvp[i]["BVP"] for i in range(n)],
            "hr":    [hr[i]["HR"] for i in range(n)],
            "skt":   [skt[i]["SKT"] for i in range(n)],
            "acc_x": [acc[i]["ACC_X"] for i in range(n)],
            "acc_y": [acc[i]["ACC_Y"] for i in range(n)],
            "acc_z": [acc[i]["ACC_Z"] for i in range(n)],
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ibi(participant_id: str) -> pd.DataFrame:
    path = config.PHYSIO_FRAME_DIR / f"{participant_id}_Physio_FrameData.json"
    raw = _read_json(path)
    blocks = raw["Physio_FrameData"][0]["Video_Physio_FrameData"]
    rows = []
    for blk in blocks:
        video = blk["VideoID"]
        for s in blk["IBI_FrameData"]:
            rows.append({"participant": participant_id, "video": video,
                         "t_sec": s["TimeStamp"], "ibi_sec": s["IBI"]})
    return pd.DataFrame(rows)


def load_participant(participant_id: str) -> dict:
    return {
        "demographics": load_demographics(participant_id),
        "ssq": load_ssq_scores(participant_id),
        "annotations": load_annotations(participant_id),
        "behavior": load_behavior(participant_id),
        "physio": load_physio(participant_id),
        "ibi": load_ibi(participant_id),
    }


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------
def _invalid_ratio(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 1.0
    invalid = series.isna().sum() + (series <= 0).sum()
    return float(invalid) / n


def _nan_ratio(series: pd.Series) -> float:
    n = len(series)
    return 0.0 if n == 0 else float(series.isna().sum()) / n


def analyze_participant(participant_id: str) -> dict:
    """Сводка по одному участнику: возраст/гендер, доли невалидных сэмплов, SSQ."""
    demo = load_demographics(participant_id)
    ssq = load_ssq_scores(participant_id)
    beh = load_behavior(participant_id)
    phy = load_physio(participant_id)
    ibi = load_ibi(participant_id)
    ssq_by_phase = ssq.set_index("phase")["TotalScore"]
    return {
        "participant": participant_id,
        "age": demo["age"], "gender": demo["gender"],
        "vr_experience": demo["vr_experience"],
        "n_videos_behavior": beh["video"].nunique(),
        "n_samples_behavior": len(beh),
        "invalid_left_pupil": round(_invalid_ratio(beh["left_pupil"]), 4),
        "invalid_right_pupil": round(_invalid_ratio(beh["right_pupil"]), 4),
        "nan_gaze": round(_nan_ratio(beh["gaze_pitch"]), 4),
        "n_samples_physio": len(phy),
        "nan_eda": round(_nan_ratio(phy["eda"]), 4),
        "nan_hr": round(_nan_ratio(phy["hr"]), 4),
        "n_ibi": len(ibi),
        "ssq_pre_total": round(float(ssq_by_phase.get("pre", np.nan)), 2),
        "ssq_mid_total": round(float(ssq_by_phase.get("mid", np.nan)), 2),
        "ssq_post_total": round(float(ssq_by_phase.get("post", np.nan)), 2),
        "ssq_delta_post_pre": round(
            float(ssq_by_phase.get("post", np.nan) - ssq_by_phase.get("pre", np.nan)), 2,
        ),
    }
