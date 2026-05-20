# VR Cybersickness Prediction — Notebooks

Воспроизводимый цикл анализа для дипломной работы по предсказанию уровня
киберболезни (cybersickness) у пользователей VR на основе окуломоторных
и физиологических показателей.

## Цель работы

Исследование и разработка методов **повышения точности предсказания уровня
киберболезни** на основе окуломоторных и физиологических показателей с
применением алгоритмов машинного обучения, при использовании шкалы Simulator
Sickness Questionnaire (SSQ) в качестве целевой метрики.

## Главный результат

**XGBoost block-level — LOSO accuracy 0,7188** (F1 0,7046, AUC 0,7698) при
предсказании клинически значимого SSQ Total Score > 15.

## Датасет

[CEAP-360VR](https://github.com/cwi-dis/CEAP-360VR-Dataset) (Xue et al., 2023):

- 32 участника, 8 × 60 с 360°-видео;
- айтрекинг Tobii Pro (встроенный в HTC Vive Pro Eye, 120 Гц → ресемпл 25 Гц);
- физиология Empatica E4 (BVP 64 Гц, EDA 4 Гц, SKT 4 Гц, ACC 32 Гц,
  HR 1 Гц, IBI);
- непрерывные аннотации valence/arousal (Joy-Con, 10 Гц);
- опросники SSQ, IPQ, NASA-TLX (pre/mid/post).

## Методология

Сравнение моделей машинного обучения в двух постановках:

1. **Window-level** — бинарная классификация arousal на 2-секундных окнах
   с 50%-перекрытием (15 104 окна × 145 признаков), валидация LOSO
   (Leave-One-Subject-Out, 32 фолда).
2. **Block-level** — бинарная классификация SSQ Total > 15 на одного
   участника как одной точки (32 × 286 признаков, среднее и стандартное
   отклонение оконных признаков), валидация LeaveOneOut.

Используются классические ML-алгоритмы (Logistic Regression, Random Forest,
XGBoost, LightGBM) и глубокое обучение (LSTM, BiLSTM). Применяется
персональная Z-нормализация физиологических каналов и персональная
бинаризация arousal по индивидуальной медиане для устранения межсубъектной
вариативности.

## Структура каталога

```
notebooks/
├── modules/                    # общая Python-библиотека (импортируется ноутбуками)
│   ├── config.py               # пути, частоты, пороги SSQ, размеры окон, RANDOM_SEED
│   ├── loader.py               # загрузка JSON CEAP-360VR + контроль качества
│   ├── features.py             # окна, статистики, окуломоторика, физиология, HRV, сборка датасета
│   ├── validation.py           # сплиттеры (LOSO/SD/LOVO) + метрики классификации
│   ├── experiments.py          # фабрики моделей (RF/XGB/LGBM/LR) + per-subject + block + ablation
│   └── deep.py                 # LSTM/BiLSTM, build_sequences, train_fold
├── 01_data_loading.ipynb       # загрузка датасета, отчёт о качестве (раздел 3.1)
├── 02_feature_extraction.ipynb # окна 2с/50%, 145 признаков (раздел 3.1)
├── 03_eda_targets.ipynb        # EDA SSQ и arousal (раздел 3.1)
├── 04_exp01_baseline.ipynb     # baseline LR/RF, SD vs LOSO (раздел 3.2)
├── 05_exp02_per_subject.ipynb  # per-subject + XGB/LGBM (раздел 3.3)
├── 06_exp03_ssq_block.ipynb    # block-level SSQ — главный результат (раздел 3.4)
├── 07_exp04_ablations.ipynb    # модальности / каналы / типы статистик (раздел 3.5)
├── 08_exp05_shap.ipynb         # SHAP-интерпретация (раздел 3.6.1)
├── 09_exp06_deep.ipynb         # LSTM/BiLSTM (раздел 3.6.2)
└── 10_stat_tests.ipynb         # Wilcoxon + McNemar + Holm (раздел 3.6.4)
```

## Запуск

```bash
# 1. Установить зависимости
pip install pandas numpy scipy scikit-learn xgboost lightgbm torch matplotlib seaborn statsmodels

# 2. Скачать датасет CEAP-360VR в корень проекта (на уровне с notebooks/):
#    CEAP-360VR-Dataset-master/CEAP-360VR/
#    └─ 2_QuestionnaireData/   3_AnnotationData/Frame/
#       4_BehaviorData/Frame/  5_PhysioData/Frame/

# 3. Запустить Jupyter из корня notebooks/
cd notebooks
jupyter lab
```

## Импорты в ноутбуках

Каждый ноутбук в первой ячейке добавляет каталог `notebooks/` в `sys.path`,
после чего импортирует общую библиотеку:

```python
import sys
from pathlib import Path
NB_ROOT = Path.cwd()
if str(NB_ROOT) not in sys.path:
    sys.path.insert(0, str(NB_ROOT))

from modules import config
from modules.experiments import make_xgboost_block, build_block_level_dataset
```

Зафиксированный `RANDOM_SEED = 42` для всех моделей и сплитов.

## Технологии

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn, XGBoost, LightGBM
- PyTorch (для LSTM/BiLSTM)
- SHAP через нативный механизм XGBoost `Booster.predict(pred_contribs=True)`
- statsmodels (McNemar exact test для блочного уровня в ноутбуке 10)
- matplotlib, seaborn

## Лицензия

Исходный код — MIT. Датасет CEAP-360VR доступен под лицензией авторов
(см. [репозиторий датасета](https://github.com/cwi-dis/CEAP-360VR-Dataset)).
