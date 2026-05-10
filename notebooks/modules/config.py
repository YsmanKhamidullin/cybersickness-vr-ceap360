"""
Единая точка настроек проекта (модуль ноутбуков).

Зеркало `source/config.py`. Используется всеми ноутбуками через
`from modules import config` или `from modules.config import ...`.

Структура датасета CEAP-360VR (уровень Frame, 25 Гц):
  2_QuestionnaireData/PX_Questionnaire_Data.json   — SSQ/IPQ/NASA-TLX
  3_AnnotationData/Frame/PX_Annotation_FrameData.json — valence/arousal
  4_BehaviorData/Frame/PX_Behavior_FrameData.json  — айтрекинг + движения головы
  5_PhysioData/Frame/PX_Physio_FrameData.json      — EDA, BVP, HR, SKT, ACC, IBI
"""

from pathlib import Path

# --- Пути ---
# notebooks/modules/config.py → корень проекта на 2 уровня выше
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_ROOT = PROJECT_ROOT / "CEAP-360VR-Dataset-master" / "CEAP-360VR"
QUESTIONNAIRE_DIR = DATASET_ROOT / "2_QuestionnaireData"
ANNOTATION_FRAME_DIR = DATASET_ROOT / "3_AnnotationData" / "Frame"
BEHAVIOR_FRAME_DIR = DATASET_ROOT / "4_BehaviorData" / "Frame"
PHYSIO_FRAME_DIR = DATASET_ROOT / "5_PhysioData" / "Frame"

RESULTS_DIR = PROJECT_ROOT / "source" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Участники ---
PARTICIPANT_IDS = [f"P{i}" for i in range(1, 33)]

# --- Видео ---
VIDEO_IDS = [f"V{i}" for i in range(1, 9)]
VIDEO_DURATION_SEC = 60

# --- Частота сигналов (фактически 25 Гц, не 30 как в описании датасета) ---
FRAME_RATE_HZ = 25

# --- SSQ (Simulator Sickness Questionnaire) ---
SSQ_ITEMS_NAUSEA = [0, 5, 6, 7, 8, 9, 14]
SSQ_ITEMS_OCULOMOTOR = [0, 1, 2, 3, 4, 8, 9]
SSQ_ITEMS_DISORIENTATION = [4, 10, 11, 12, 13, 14, 15]

SSQ_WEIGHT_NAUSEA = 9.54
SSQ_WEIGHT_OCULOMOTOR = 7.58
SSQ_WEIGHT_DISORIENTATION = 13.92
SSQ_WEIGHT_TOTAL = 3.74

SSQ_THRESHOLD_HIGH = 15

# --- Окна ---
WINDOW_SIZE_SEC = 2
WINDOW_OVERLAP = 0.5

# --- Фильтрация окон ---
MAX_INVALID_EYE_RATIO = 0.30

# --- Случайность ---
RANDOM_SEED = 42
