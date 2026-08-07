from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class MetricPack:
    auc: float
    auprc: float
    brier: float
    sep_prn: float
    recall_k: float


def compute_metric_pack(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask_p: np.ndarray,
    mask_rn: np.ndarray,
) -> MetricPack:
    auc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))

    p_mean = float(y_score[mask_p].mean()) if mask_p.any() else 0.0
    rn_mean = float(y_score[mask_rn].mean()) if mask_rn.any() else 0.0
    sep_prn = p_mean - rn_mean

    k = int(mask_p.sum())
    if k > 0:
        top_k_idx = np.argsort(y_score)[::-1][:k]
        recall_k = float(y_true[top_k_idx].sum() / k)
    else:
        recall_k = 0.0

    return MetricPack(auc=auc, auprc=auprc, brier=brier, sep_prn=float(sep_prn), recall_k=recall_k)

