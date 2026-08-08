"""统计检验单一实现：所有报告共用的假设检验 / 效应量 / 重采样工具。

设计原则：全项目只有这一份 Wilcoxon / Holm / Cohen's d / bootstrap。
各实验报告（reporting/）只调这里，禁止在脚本里各自实现统计。

方法约定（与论文写作一致）：
- 配对比较：scipy.stats.wilcoxon（符号秩检验，n≥15 每配置）
- 多重比较校正：Holm–Bonferroni（p_adj）
- 效应量：Cohen's d（配对版，分母 = 差值的合并标准差）
- 置信区间：bootstrap 百分位法（B=1000，种子固定，可复现）
"""
from __future__ import annotations

import numpy as np
import scipy.stats as st


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """配对 Wilcoxon 符号秩检验 p 值（a 与 b 同长、一一配对）。

    与历史 run_closure_aggregate 语义一致：差值近 0（np.allclose 默认容差）→ 返回 1.0；
    scipy 退化报 ValueError（如秩全同号）→ 返回 np.nan。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired_wilcoxon: 两样本长度不一致")
    diff = a - b
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        stat, p = st.wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return np.nan


def holm_bonferroni(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm–Bonferroni 多重比较校正，返回 p_adj（与输入顺序一一对应）。

    标准做法：最小 p 乘 m，次小乘 m-1，……，然后从大到小强制单调不减。
    """
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    p_adj = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(order):
        # Holm：p_adj(k) = max_{j<=k} (m-j+1)·p_(j)，即正向累计最大值
        running = max(running, p[i] * (m - rank))
        p_adj[i] = min(1.0, running)
    return p_adj


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """配对 Cohen's d：d = mean(a-b) / std(a-b)（差值标准差作分母，配对设计标准做法）。

    与历史 run_closure_aggregate 的 paired_cohen_d 守卫一致：样本 <2 或
    sd < 1e-12（含退化 0）→ 返回 nan（效应量无定义），保证聚合输出逐位一致。
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    if len(diff) < 2:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return np.nan
    return float(np.mean(diff) / sd)


def bootstrap_ci(samples: np.ndarray, stat: callable, n_boot: int = 1000,
                 seed: int = 42, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 百分位置信区间。

    samples 为每配置的观测数组，stat 为要估计的统计量函数（如 np.mean）。
    固定种子保证可复现。

    注意：分位输入取 round(alpha, 10)，避免 (1-ci)/2 在 float64 下得到
    0.025000000000000022（1 ulp 偏差）而偏离历史 np.percentile(x, 2.5)/97.5
    的逐位结果。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(samples, float)
    if len(x) == 0:
        return np.nan, np.nan
    boots = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    alpha = round((1 - ci) / 2, 10)
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


if __name__ == "__main__":
    # 冒烟：构造两列带差异的数据，检验三条输出都正常
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 15)
    b = a + rng.normal(0.02, 0.01, 15)
    print("p (paired wilcoxon):", paired_wilcoxon(a, b))
    print("p_adj (holm):", holm_bonferroni([0.01, 0.03, 0.2, 0.8]))
    print("cohen's d:", cohens_d_paired(a, b))
    print("bootstrap ci(mean):", bootstrap_ci(a, np.mean))
