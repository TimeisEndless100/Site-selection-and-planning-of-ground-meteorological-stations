"""实验注册表：语义化实验名 → {分类, 指标桶, 论文章节, 旧编号, 状态}。

2026-08-07 起废弃 E1–E19 模糊编号，改用语义化命名（权威映射见
`docs/实验地图与论文对照.md` §2）。本文件是代码侧的单一事实来源：
聚合（analysis/aggregate.py）与报告（reporting/）据此自动获得每个实验的
分类、指标桶与论文章节，保证三桶永不混比、分类口径唯一。
"""
from __future__ import annotations

from dataclasses import dataclass

# --- 指标桶（不可混比） -----------------------------------------------------
BUCKET_CNN = "cnn"            # ① CNN 阶段：AUC/AUPRC/Brier/Recall@K
BUCKET_GNN = "gnn"            # ② GNN 阶段：同上
BUCKET_DECISION = "decision"  # ③ 下游决策：rmclp_gain（与 AUC 不可比）

# --- 实验分类（正式科研写作术语） -------------------------------------------
CAT_COMPARISON = "comparison"    # 对比方法 / 基线方法
CAT_ABLATION = "ablation"        # 消融
CAT_SENSITIVITY = "sensitivity"  # 敏感性
CAT_PROTOCOL = "protocol"        # 评估协议稳健性
CAT_APPLICATION = "application"  # 应用展示
CAT_DIAGNOSTIC = "diagnostic"    # 诊断

STATUS_HAVE = "have"        # 已有（历史产物在 outputs/exp_closure_20260315/）
STATUS_NEEDS_FIX = "fix"    # 已有但需修（预算/口径/改造）
STATUS_PLANNED = "planned"  # 待补实验
STATUS_REVOKED = "revoked"  # 已撤销（代码删除，结果留档作历史证据）

# --- 论文章节（标准中文论文结构） -------------------------------------------
SEC_MAIN = "4.2 主结果"
SEC_COMPARISON = "对比方法"
SEC_ABLATION = "4.3 消融实验"
SEC_SENSITIVITY = "4.4 敏感性分析"
SEC_PROTOCOL = "4.5 评估协议"
SEC_APPLICATION = "4.6 应用展示"
SEC_DISCUSSION = "5 讨论与局限"


@dataclass(frozen=True)
class Experiment:
    name: str            # 语义化实验名（唯一）
    category: str        # CAT_*
    bucket: str | None   # BUCKET_* 或 None（诊断）
    section: str         # 论文章节
    legacy_id: str       # 旧编号（历史输出目录交叉引用）
    status: str          # STATUS_*
    description: str = ""


_REGISTRY: dict[str, Experiment] = {}


def _reg(name, category, bucket, section, legacy_id, status, description=""):
    exp = Experiment(name, category, bucket, section, legacy_id, status, description)
    _REGISTRY[name] = exp


# --- 注册全部实验（与 docs/实验地图与论文对照.md §2 一致） -------------------

# 对比方法
_reg("cmp_main_cnn", CAT_COMPARISON, BUCKET_CNN, SEC_MAIN, "E5(CNN)",
     STATUS_HAVE, "主方法 PURN-CNN vs PN / TaylorPU（CNN 阶段主对比）")
_reg("cmp_main_gnn", CAT_COMPARISON, BUCKET_GNN, SEC_MAIN, "E5(GNN)",
     STATUS_HAVE, "主方法 SAGE vs GCN / GAT（GNN 阶段主对比）")
_reg("cmp_pu_baseline", CAT_COMPARISON, BUCKET_CNN, SEC_COMPARISON, "E6",
     STATUS_NEEDS_FIX, "单阶段 PU 基线 uPU / ElkanNoto / BaggingPU（待并入主表同显著性框架）")
_reg("cmp_graph_baseline", CAT_COMPARISON, BUCKET_GNN, SEC_COMPARISON, "E6",
     STATUS_NEEDS_FIX, "图域基线 LP / APPNP（待并入主表同显著性框架）")

# 消融
_reg("abla_rn_source", CAT_ABLATION, BUCKET_CNN, SEC_ABLATION, "E1",
     STATUS_HAVE, "RN 来源策略 physics/random/none/noisy10/20/30（noisy 仅单向，勿写强）")
_reg("abla_objective", CAT_ABLATION, BUCKET_CNN, SEC_ABLATION, "E13",
     STATUS_NEEDS_FIX, "目标组件 warmup×nnPU×RN penalty（修预算统一 20 + 补全监督 P-vs-RN 格）")
_reg("abla_graph", CAT_ABLATION, BUCKET_GNN, SEC_ABLATION, "E14",
     STATUS_NEEDS_FIX, "图传播 NodeMLP / SAGE 全图 / SAGE 自环图（补同架构无传播对照）")
_reg("abla_structure", CAT_ABLATION, BUCKET_CNN, SEC_ABLATION, "E11/E12",
     STATUS_REVOKED, "结构模块 CBAM/CoordConv，已按实证删除，结果留档作历史证据")

# 敏感性
_reg("sens_rn_threshold", CAT_SENSITIVITY, BUCKET_CNN, SEC_SENSITIVITY, "E2",
     STATUS_NEEDS_FIX, "RN 阈值 slope×water（改造：每档附 RN/U/P 计数，暴露数据构成混杂）")
_reg("sens_pi", CAT_SENSITIVITY, BUCKET_CNN, SEC_SENSITIVITY, "E3",
     STATUS_NEEDS_FIX, "类先验 π（改造：预设网格 + ECE/Brier 校准类指标）")
_reg("sens_lambda", CAT_SENSITIVITY, BUCKET_CNN, SEC_SENSITIVITY, "E4",
     STATUS_NEEDS_FIX, "RN 惩罚强度 λ_rn（扩对数网格；λ=0 崩塌是最扎实机制证据）")
_reg("sens_graph_k", CAT_SENSITIVITY, BUCKET_GNN, SEC_SENSITIVITY, "E16",
     STATUS_PLANNED, "图近邻 k ∈ {4,8,12,16}（待补）")
_reg("sens_fold_grid", CAT_SENSITIVITY, BUCKET_GNN, SEC_SENSITIVITY, "E17",
     STATUS_PLANNED, "分折粒度 grid ∈ {5,10,20} / buffer ∈ {0,1,2}（待补）")
_reg("sens_feature", CAT_SENSITIVITY, BUCKET_CNN, SEC_SENSITIVITY, "E18",
     STATUS_PLANNED, "特征通道（去 DEM / 去 LC / 仅 Near 尺度）（待补）")

# 评估协议稳健性
_reg("proto_trans_ind_buffer", CAT_PROTOCOL, BUCKET_GNN, SEC_PROTOCOL, "E7",
     STATUS_NEEDS_FIX, "transductive / inductive / buffered（修 trans/ind 评测口径；目的是稳健性非选优）")

# 应用展示
_reg("app_infer", CAT_APPLICATION, BUCKET_DECISION, SEC_APPLICATION, "E9",
     STATUS_HAVE, "整域推理（CNN 概率图），为 app_rmclp_gain 提供候选源，本身不单独成表")
_reg("app_rmclp_gain", CAT_APPLICATION, BUCKET_DECISION, SEC_APPLICATION, "E10",
     STATUS_NEEDS_FIX, "RMCLP 选址收益 random/gis/cnn/purn/purn_gnn（决策链未闭合，需校准或降级）")
_reg("app_rmclp_vs_mclp", CAT_APPLICATION, BUCKET_DECISION, SEC_APPLICATION, "E15",
     STATUS_NEEDS_FIX, "RMCLP vs MCLP × rn_on/off（上游退化全 0，先修上游）")

# 诊断
_reg("diag_fold4", CAT_DIAGNOSTIC, None, SEC_DISCUSSION, "E8",
     STATUS_HAVE, "Fold4 误差诊断（feature_shift 支撑分折脆弱性讨论）")


# --- 查询接口 ---------------------------------------------------------------

def all_experiments() -> dict[str, Experiment]:
    """全部实验：语义名 → Experiment（排序稳定）。"""
    return dict(_REGISTRY)


def get(name: str) -> Experiment:
    return _REGISTRY[name]


def by_category(category: str) -> list[Experiment]:
    return [e for e in _REGISTRY.values() if e.category == category]


def by_bucket(bucket: str) -> list[Experiment]:
    return [e for e in _REGISTRY.values() if e.bucket == bucket]


def by_legacy(legacy_id: str) -> list[Experiment]:
    """按旧编号反查（E 编号可能对应多个实验，如 E5/E6/E11/E12）。"""
    return [e for e in _REGISTRY.values() if legacy_id in e.legacy_id.split("/")]
