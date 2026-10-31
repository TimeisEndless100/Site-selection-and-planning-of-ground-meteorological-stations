from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class MetricPack:
    auc: float
    auc_pu: float  # 候选域 P∪U P-vs-U AUC：跨模型公平比较主口径（2026-08-12）
    auprc: float
    brier: float
    ece: float     # 全域 ECE（P=1，RN+U=0 的替代标签下；含 RN 易负样本偏倚）
    ece_pu: float  # 候选域 ECE（P∪U，剔除 RN）：主口径对应的校准度量
    sep_prn: float
    recall_k: float


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """标准分箱 ECE：加权平均 |置信度 − 准确率|。

    用于 PU 设定时以替代标签（P=1，其余=0）度量校准——U 含真实正例，ECE 是
    代理标签口径下的保守估计；跨模型同口径可比。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2 or len(y_score) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(y_score, bins=edges)
    total = max(int(counts.sum()), 1)
    ece = 0.0
    for b in range(n_bins):
        if b < n_bins - 1:
            m = (y_score >= edges[b]) & (y_score < edges[b + 1])
        else:
            m = (y_score >= edges[b]) & (y_score <= edges[b + 1])  # 末箱含 1.0
        n = int(m.sum())
        if n == 0:
            continue
        conf = float(y_score[m].mean())
        acc = float(y_true[m].mean())
        ece += (n / total) * abs(conf - acc)
    return float(ece)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_metric_pack(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask_p: np.ndarray,
    mask_rn: np.ndarray,
) -> MetricPack:
    auc = _safe_auc(y_true, y_score)
    auprc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    ece = expected_calibration_error(y_true, y_score)

    mask_pu = ~mask_rn
    if mask_pu.sum() > 1:
        auc_pu = _safe_auc(y_true[mask_pu], y_score[mask_pu])
        ece_pu = expected_calibration_error(y_true[mask_pu], y_score[mask_pu])
    else:
        auc_pu, ece_pu = float("nan"), float("nan")

    p_mean = float(y_score[mask_p].mean()) if mask_p.any() else 0.0
    rn_mean = float(y_score[mask_rn].mean()) if mask_rn.any() else 0.0
    sep_prn = p_mean - rn_mean

    k = int(mask_p.sum())
    if k > 0:
        top_k_idx = np.argsort(y_score)[::-1][:k]
        recall_k = float(y_true[top_k_idx].sum() / k)
    else:
        recall_k = 0.0

    return MetricPack(
        auc=auc,
        auc_pu=auc_pu,
        auprc=auprc,
        brier=brier,
        ece=ece,
        ece_pu=ece_pu,
        sep_prn=float(sep_prn),
        recall_k=recall_k,
    )
