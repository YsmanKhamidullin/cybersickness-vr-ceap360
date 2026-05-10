"""
Deep learning helpers (exp06): LSTM/BiLSTM, последовательности.

Зеркало source/experiments/exp06_deep.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .validation import compute_classification_metrics


# Гиперпараметры
K_DEFAULT = 10
HIDDEN = 128
N_LAYERS = 1
DROPOUT = 0.3
BATCH = 256
LR = 1e-3
EPOCHS = 8


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sequences(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, k: int = K_DEFAULT
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Собрать последовательности длины k в пределах одной (participant, video).
    На границах участника/видео окна НЕ перемешиваются.

    Возвращает:
      seq_X   — (N_seq, k, n_features);
      seq_y   — (N_seq,) бинарный класс последнего окна;
      seq_pid — (N_seq,) participant последнего окна.
    """
    seqs_X, seqs_y, seqs_pid = [], [], []
    participants = df["participant"].to_numpy()
    videos = df["video"].to_numpy()

    for pid in df["participant"].unique():
        for vid in df[df["participant"] == pid]["video"].unique():
            mask = (participants == pid) & (videos == vid)
            idx = np.where(mask)[0]
            if len(idx) < k:
                continue
            for i in range(k - 1, len(idx)):
                window_indices = idx[i - k + 1: i + 1]
                seqs_X.append(X[window_indices])
                seqs_y.append(y[idx[i]])
                seqs_pid.append(pid)

    return (np.array(seqs_X, dtype=np.float32),
            np.array(seqs_y, dtype=np.int64),
            np.array(seqs_pid))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN,
                 n_layers: int = N_LAYERS, bidirectional: bool = False,
                 dropout: float = DROPOUT, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


def train_fold(X_tr, y_tr, X_te, y_te, bidirectional: bool, n_features: int) -> dict:
    """Обучить LSTM/BiLSTM на одном LOSO-фолде, вернуть метрики."""
    set_seed(config.RANDOM_SEED)
    model = LSTMClassifier(n_features=n_features, bidirectional=bidirectional)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()
    return compute_classification_metrics(y_te, preds, probs)
