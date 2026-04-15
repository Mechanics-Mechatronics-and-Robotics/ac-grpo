from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auroc(labels: Iterable[float], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float32)
    s = np.asarray(list(scores), dtype=np.float32)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, s))


def trajectory_auroc(successes: Iterable[float], mean_certainties: Iterable[float]) -> float:
    return safe_auroc(successes, mean_certainties)


def timestep_auroc(certainties: Iterable[float], timesteps: Iterable[int], episode_lengths: Iterable[int]) -> float:
    labels = []
    scores = []
    for c, t, length in zip(certainties, timesteps, episode_lengths):
        labels.append(float(t > 0.8 * length))
        scores.append(1.0 - float(c))
    return safe_auroc(labels, scores)
