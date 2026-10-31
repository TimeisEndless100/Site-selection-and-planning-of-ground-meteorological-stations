# -*- coding: utf-8 -*-
"""论文表格：只消费 analysis/aggregate.py + analysis/prob_map.py 的输出。

数据来源（不依赖不存在的 results_by_config.csv）：
- <out_root>/master_results.csv（逐配置逐折逐种子）
- <out_root>/master_summary.csv（stage × model 的 mean ± std）
- <out_root>/stats_results.csv（组内配对 Wilcoxon + Holm p_adj，仅 ad-hoc，不再进论文表）
- prob_map_summary.csv（analysis/prob_map 产出，应用表）

统计口径（2026-08-12 决策）：单 seed×5 折，n=5；折间共享训练数据 → 空间伪重复，
论文表只报描述性统计（mean±std + Δ），不做显著性推断、不打 p 值。

子命令：
    python -m reporting.tables main cnn|gnn   [--out_root …] [--stage …] [--ref …]
    python -m reporting.tables ablation       [--out_root …] [--stage …] [--ref …]
    python -m reporting.tables sensitivity    [--out_root …] [--stage …]
    python -m reporting.tables protocol       [--out_root …] [--stage …]
    python -m reporting.tables application    [--out_root …] [--prob_map_summary …]

stage 默认从 registry（experiments/registry.py）按分类/比较组自动发现；数据中若只有旧
编号（E5/E7/E13/E14）则用 --stage 显式指定。参照配置（主方法 / Full）默认从配置
（experiments/config.py）解析，旧数据用 --ref 指定。

输出：<out_root>/tables/paper_tables.md（## 段落 upsert，幂等）+ 每子命令 <cmd>.csv。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd

from experiments.registry import (
    CAT_ABLATION,
    CAT_COMPARISON,
    CAT_PROTOCOL,
    CAT_SENSITIVITY,
    STATUS_REVOKED,
    by_category,
)
from experiments.config import load_experiment

# 指标列 → 表头（顺序与 docs/实验地图与论文对照.md §5 一致）
# auc_pu：候选域 P∪U P-vs-U AUC，跨模型公平比较主口径（2026-08-12）
# ece_pu：候选域 ECE 校准误差（sens_pi 校准论证，2026-08-12 补）
METRICS = [
    ("auc", "AUC"),
    ("auc_pu", "AUC(P∪U)"),
    ("auprc", "AUPRC"),
    ("brier", "Brier"),
    ("ece_pu", "ECE(P∪U)"),
    ("recall_k", "Recall@K"),
]

# ================= 标准科研命名（审稿人一眼可读） =================
# 原始 label → 学术标准显示名。优先用学术界既有标准词汇（PU learning / uPU /
# Elkan–Noto / BaggingPU / TaylorPU / GraphSAGE / GCN / GAT / LP / APPNP / GBDT），
# 并标注信息体制（无标签监督 / 全监督 / 无学习），使不了解具体知识的审稿人
# 也能直接看出「比的是谁、差在哪」。聚合层保持原始 label 稳定，仅渲染层映射。
MODEL_DISPLAY = {
    # 主方法（本文方法）
    "PURN-CNN": "PURN-CNN（本文方法：物理感知 PU 学习）",
    "SAGE": "GraphSAGE（本文方法：空间细化）",
    # 同信息体制 PU 学习方法（无标签监督）
    "TaylorPU": "TaylorPU（泰勒展开近似 PU）",
    "uPU": "uPU（无偏估计 PU）",
    "ElkanNoto": "Elkan–Noto（PU 估计器）",
    "BaggingPU": "BaggingPU（装袋集成 PU）",
    # 全监督参照（使用物理规则导出的 RN 标签）
    "PN": "PN（全监督 P-vs-RN）",
    "GBDT": "GBDT（梯度提升决策树，表格监督基线）",
    # 无学习参照
    "Physics-only": "物理规则参照（无学习）",
    # GNN 对比方法
    "GCN": "GCN（图卷积网络）",
    "GAT": "GAT（图注意力网络）",
    # 图域基线
    "LabelPropagation": "LP（标签传播，图域基线）",
    "APPNP": "APPNP（个性化传播网络，图域基线）",
    # 消融：目标组件（abla_objective）
    "PURN-full": "PURN-CNN 完整模型（本文方法）",
    "Supervised": "全监督 P-vs-RN（丢弃 U）",
    "No-RN": "去 RN 惩罚（λ=0）",
    "No-warmup": "去 warmup 预热",
    "nnPU-only": "仅 nnPU（去 RN 惩罚与 warmup）",
    # 消融：RN 来源（abla_rn_source）
    "physics": "物理规则",
    "random": "随机",
    "none": "无 RN",
    "noisy10": "噪声 10%",
    "noisy20": "噪声 20%",
    "noisy30": "噪声 30%",
    # 消融：图传播（abla_graph）
    "SAGE-full": "SAGE 全图传播",
    "SAGE-selfloop": "SAGE 自环图",
    "NodeMLP": "NodeMLP（无传播对照）",
    # 评估协议
    "transductive": "转导式（transductive）",
    "inductive": "归纳式（inductive）",
}

# 主表行序与「方法族」分组（跨 stage 合并后稳定排序）
MAIN_ORDER = {
    "cnn": ["PURN-CNN", "TaylorPU", "uPU", "ElkanNoto", "BaggingPU", "PN", "GBDT", "Physics-only"],
    "gnn": ["SAGE", "GCN", "GAT", "LabelPropagation", "APPNP"],
}
MAIN_FAMILY = {
    "PURN-CNN": "本文方法",
    "SAGE": "本文方法",
    "TaylorPU": "PU 学习（无标签监督）",
    "uPU": "PU 学习（无标签监督）",
    "ElkanNoto": "PU 学习（无标签监督）",
    "BaggingPU": "PU 学习（无标签监督）",
    "PN": "全监督（使用 RN 标签）",
    "GBDT": "全监督（使用 RN 标签）",
    "Physics-only": "无学习参照",
    "GCN": "图神经网络",
    "GAT": "图神经网络",
    "LabelPropagation": "图域基线",
    "APPNP": "图域基线",
}


def _display_model(label: str) -> str:
    """原始 label → 标准科研显示名；参数化标签（λ/π/RN 阈值）按正则转换。"""
    label = str(label)
    if label in MODEL_DISPLAY:
        return MODEL_DISPLAY[label]
    m = re.fullmatch(r"lambda_([\d.]+)", label)
    if m:
        return f"λ = {m.group(1)}"
    m = re.fullmatch(r"pi([\d.]+)", label)
    if m:
        return f"π = {m.group(1)}"
    m = re.fullmatch(r"s(\d+)_w([\d.]+)", label)
    if m:
        return f"slope {m.group(1)}° / water {m.group(2)}"
    return label


# ================= 数据加载 =================

def _load(out_root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(out_root)
    def _read(name: str) -> pd.DataFrame:
        p = out / name
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    return _read("master_results.csv"), _read("master_summary.csv"), _read("stats_results.csv")


def _require(master: pd.DataFrame, what: str) -> None:
    if len(master) == 0:
        raise SystemExit(f"{what}: master_results.csv 为空/缺失（先跑 analysis.aggregate --out_root …）")


def _stage_set(master: pd.DataFrame) -> set[str]:
    return set(master["stage"].astype(str).dropna())


def _pick_stage(stage_arg: str, candidates: list[str], master: pd.DataFrame, what: str) -> str:
    present = _stage_set(master)
    if stage_arg:
        if stage_arg not in present:
            raise SystemExit(f"{what}: stage '{stage_arg}' 不在 master_results.csv（可选: {sorted(present)}）")
        return stage_arg
    for cand in candidates:
        if cand in present:
            return cand
    raise SystemExit(
        f"{what}: 数据中无候选 stage {candidates}（现有: {sorted(present)}）。"
        f"可 --stage 指定（旧编号数据如 E5/E13），或先按语义名重跑 plan+aggregate"
    )


def _safe_cfg(stage: str) -> dict | None:
    """加载语义 stage 的配置；旧编号/未注册 stage 返回 None。"""
    if not stage or not re.fullmatch(r"[a-z]+_[a-z0-9_]+", stage):
        return None
    try:
        return load_experiment(stage)
    except Exception:
        return None


# ================= 格式化工具 =================

def _fmt_ms(mean, std) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ================= paper_tables.md 段落 upsert（幂等） =================

def _read_sections(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    sections: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for ln in path.read_text(encoding="utf-8").split("\n"):
        if ln.startswith("## "):
            if cur is not None:
                sections[cur] = "\n".join(buf).rstrip()
            cur = ln[3:].strip()
            buf = []
        else:
            buf.append(ln)
    if cur is not None:
        sections[cur] = "\n".join(buf).rstrip()
    return sections


def _write_section(path: Path, name: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = _read_sections(path)
    sections[name] = content.rstrip()
    with open(path, "w", encoding="utf-8") as f:
        for k, v in sections.items():
            f.write(f"## {k}\n{v}\n\n")


# ================= 行 / 参照解析 =================

def _ordered_models(master: pd.DataFrame, stage: str, exp_cfg: dict | None) -> list[str]:
    present = set(master.loc[master["stage"].astype(str) == stage, "model"].dropna().astype(str))
    order: list[str] = []
    if exp_cfg and isinstance(exp_cfg.get("cells"), list):
        for cell in exp_cfg["cells"]:
            label = cell.get("label")
            if label is not None and str(label) in present and str(label) not in order:
                order.append(str(label))
    for m in sorted(present):
        if m not in order:
            order.append(m)
    return order


def _summary_of(summary: pd.DataFrame, stage: str, model: str) -> pd.Series:
    if len(summary) == 0:
        return pd.Series(dtype=float)
    s = summary[(summary["stage"].astype(str) == stage) & (summary["model"].astype(str) == model)]
    return s.iloc[0] if len(s) else pd.Series(dtype=float)


def _pair_delta(stats: pd.DataFrame, stage: str, metric: str, a: str, b: str) -> tuple[float, float]:
    """stats 里 a−b 的 (delta_mean, p_adj)；无正向行则取 b−a 取负。"""
    if len(stats) == 0:
        return np.nan, np.nan
    m = (stats["stage"].astype(str) == stage) & (stats["metric"] == metric)
    fwd = stats[m & (stats["model_a"].astype(str) == a) & (stats["model_b"].astype(str) == b)]
    if len(fwd):
        r = fwd.iloc[0]
        return float(r["delta_mean"]), float(r["p_adj"])
    rev = stats[m & (stats["model_a"].astype(str) == b) & (stats["model_b"].astype(str) == a)]
    if len(rev):
        r = rev.iloc[0]
        return -float(r["delta_mean"]), float(r["p_adj"])
    return np.nan, np.nan


def _ref_from_cfg(stage: str, exp_cfg: dict | None, models: list[str]) -> str | None:
    """参照配置（主方法 / Full）label：按实验语义从配置 cells 解析。"""
    cells = (exp_cfg or {}).get("cells")
    if not isinstance(cells, list):
        return None
    for cell in cells:
        label = cell.get("label")
        if label is None:
            continue
        if stage == "cmp_main_cnn" and cell.get("method") == "purn":
            return str(label)
        if stage == "cmp_main_gnn" and (cell.get("gnn_type") == "sage" or cell.get("arch") == "sage"):
            return str(label)
        if stage == "abla_rn_source" and cell.get("rn_strategy") == "physics":
            return str(label)
        if stage == "abla_objective" and str(label) == "PURN-full":
            return str(label)
        if stage == "abla_graph" and cell.get("arch") == "sage" and cell.get("graph") == "full":
            return str(label)
    if cells and cells[0].get("label") is not None:
        return str(cells[0]["label"])
    return None


def _resolve_ref(stage: str, exp_cfg: dict | None, bucket: str, models: list[str], ref_arg: str) -> str | None:
    if ref_arg:
        return ref_arg
    ref = _ref_from_cfg(stage, exp_cfg, models)
    if ref:
        return ref
    for m in models:  # 旧数据启发式：按比较组关键词兜底
        up = m.upper()
        if bucket == "gnn" and "SAGE" in up:
            return m
        if bucket == "cnn" and "PURN" in up:
            return m
    return None


# ================= 指标行构建（主表 / 消融表共用） =================

def _metric_rows(master, summary, stats, stage, models, ref) -> tuple[list[dict], list[list[str]]]:
    rows: list[dict] = []
    for m in models:
        s = _summary_of(summary, stage, m)
        if m == ref:
            delta, p = 0.0, np.nan
        else:
            # 主口径 = 候选域 auc_pu（2026-08-12 决策）；stats 中 delta_mean = model_a − model_b
            delta, p = _pair_delta(stats, stage, "auc_pu", m, ref) if ref else (np.nan, np.nan)
        rows.append(
            {
                "stage": stage,
                "model": m,
                "ref": "" if ref is None or m != ref else ref,
                "auc_mean": s.get("auc_mean", np.nan),
                "auc_std": s.get("auc_std", np.nan),
                "auc_pu_mean": s.get("auc_pu_mean", np.nan),
                "auc_pu_std": s.get("auc_pu_std", np.nan),
                "auprc_mean": s.get("auprc_mean", np.nan),
                "auprc_std": s.get("auprc_std", np.nan),
                "brier_mean": s.get("brier_mean", np.nan),
                "brier_std": s.get("brier_std", np.nan),
                "ece_pu_mean": s.get("ece_pu_mean", np.nan),
                "ece_pu_std": s.get("ece_pu_std", np.nan),
                "recall_k_mean": s.get("recall_k_mean", np.nan),
                "recall_k_std": s.get("recall_k_std", np.nan),
                "delta_auc_vs_ref": delta,
                "p_adj": p,
                "n": s.get("n", np.nan),
            }
        )
    grid = [
        [
            _display_model(m),
            _fmt_ms(r["auc_mean"], r["auc_std"]),
            _fmt_ms(r["auc_pu_mean"], r["auc_pu_std"]),
            _fmt_ms(r["auprc_mean"], r["auprc_std"]),
            _fmt_ms(r["brier_mean"], r["brier_std"]),
            _fmt_ms(r["ece_pu_mean"], r["ece_pu_std"]),
            _fmt_ms(r["recall_k_mean"], r["recall_k_std"]),
            "—" if m == ref or np.isnan(r["delta_auc_vs_ref"]) else f"{r['delta_auc_vs_ref']:+.4f}",
        ]
        for m, r in zip(models, rows)
    ]
    return rows, grid


# ================= 子命令 =================

def _model_stage_map(master: pd.DataFrame) -> dict[str, str]:
    """model → stage（保持首次出现顺序；同一 model 只取第一个 stage）。"""
    d: dict[str, str] = {}
    for _, r in master.iterrows():
        m = str(r["model"])
        if m not in d:
            d[m] = str(r["stage"])
    return d


def cmd_main(args: argparse.Namespace) -> None:
    """主结果表：跨 stage 合并（语义 stage + E6 外部基线），方法族分组，Δ 用主口径 auc_pu。"""
    master, summary, stats = _load(args.out_root)
    _require(master, "主结果表")
    bucket = args.bucket

    # 主表数据来源（跨 stage 合并；E6 外部基线 stage 名为 "E6"，按方法族归属过滤）
    if bucket == "cnn":
        stage_sources = ["cmp_main_cnn", "cmp_pu_baseline", "E6"]
        ext_pu = {"uPU", "ElkanNoto", "BaggingPU"}    # E6 中属 PU 学习方法
        ext_graph = set()                              # 图域基线归 GNN 桶
        ref_default = "PURN-CNN"
        bucket_name = "① CNN 阶段"
    else:
        stage_sources = ["cmp_main_gnn", "E6"]
        ext_pu = set()
        ext_graph = {"LabelPropagation", "APPNP"}     # E6 中属图域基线
        ref_default = "SAGE"
        bucket_name = "② GNN 阶段"

    present = _stage_set(master)
    for st in stage_sources:
        if st not in present:
            raise SystemExit(
                f"主结果表（{bucket_name}）：stage '{st}' 不在 master_results.csv（现有: {sorted(present)}）。"
                f"先跑 analysis.aggregate，或用 --stage 指定"
            )

    # 行序：标准排序（MAIN_ORDER）优先，其余（含未注册的外部基线）追加
    all_models = set(master["model"].astype(str))
    models = [m for m in MAIN_ORDER[bucket] if m in all_models]
    for m in sorted(all_models):
        if m not in models:
            models.append(m)

    # 过滤：只保留本桶语义 stage 的模型 + E6 中属本桶方法族的模型
    m2s = _model_stage_map(master)
    sem_stages = {"cnn": {"cmp_main_cnn", "cmp_pu_baseline"}, "gnn": {"cmp_main_gnn"}}[bucket]
    models = [
        m for m in models
        if m2s.get(m) in sem_stages
        or (m2s.get(m) == "E6" and (m in ext_pu or m in ext_graph))
    ]

    ref = args.ref or ref_default
    if ref not in models:
        raise SystemExit(f"主结果表（{bucket_name}）：参照 '{ref}' 不在数据中（可选: {models}），请 --ref 指定")

    # 逐模型取 summary 行（stage 可能不同，用 model→stage 映射）
    rows: list[dict] = []
    grid: list[list[str]] = []
    ref_pu = _summary_of(summary, m2s.get(ref, ""), ref).get("auc_pu_mean", np.nan)
    for m in models:
        stg = m2s.get(m, "")
        s = _summary_of(summary, stg, m)
        pu = s.get("auc_pu_mean", np.nan)
        delta = (pu - ref_pu) if (m != ref and not pd.isna(ref_pu)) else np.nan
        rows.append(
            {
                "bucket": bucket,
                "family": MAIN_FAMILY.get(m, "其他"),
                "model": m,
                "model_display": _display_model(m),
                "ref": "" if m != ref else ref,
                "stage": stg,
                "auc_mean": s.get("auc_mean", np.nan),
                "auc_std": s.get("auc_std", np.nan),
                "auc_pu_mean": s.get("auc_pu_mean", np.nan),
                "auc_pu_std": s.get("auc_pu_std", np.nan),
                "auprc_mean": s.get("auprc_mean", np.nan),
                "auprc_std": s.get("auprc_std", np.nan),
                "brier_mean": s.get("brier_mean", np.nan),
                "brier_std": s.get("brier_std", np.nan),
                "ece_pu_mean": s.get("ece_pu_mean", np.nan),
                "ece_pu_std": s.get("ece_pu_std", np.nan),
                "recall_k_mean": s.get("recall_k_mean", np.nan),
                "recall_k_std": s.get("recall_k_std", np.nan),
                "delta_auc_pu_vs_ref": delta,
                "n": s.get("n", np.nan),
            }
        )
        grid.append(
            [
                MAIN_FAMILY.get(m, "其他"),
                _display_model(m),
                _fmt_ms(s.get("auc_mean"), s.get("auc_std")),
                _fmt_ms(s.get("auc_pu_mean"), s.get("auc_pu_std")),
                _fmt_ms(s.get("auprc_mean"), s.get("auprc_std")),
                _fmt_ms(s.get("brier_mean"), s.get("brier_std")),
                _fmt_ms(s.get("ece_pu_mean"), s.get("ece_pu_std")),
                _fmt_ms(s.get("recall_k_mean"), s.get("recall_k_std")),
                "—" if m == ref or np.isnan(delta) else f"{delta:+.4f}",
            ]
        )

    headers = ["方法族", "模型", "AUC", "AUC(P∪U)", "AUPRC", "Brier", "ECE(P∪U)", "Recall@K", f"ΔAUC(P∪U) vs {_display_model(ref)}"]
    title = f"主结果表（{bucket_name}，参照 = {_display_model(ref)}）"
    content = (
        f"> 来源：{args.out_root}；单 seed×5 折（n=5），空间伪重复下仅描述性统计，不做显著性推断；"
        f"ΔAUC(P∪U) = 模型 − 参照；AUC = 全域口径（负样本含 RN，含易负样本偏倚），"
        f"AUC(P∪U) = 候选域 P-vs-U 口径（跨模型公平比较主口径），ECE(P∪U) = 候选域校准误差（越低越好）。\n\n"
        + _md_table(headers, grid)
    )
    _write_section(_md_path(args), title, content)
    _write_csv(args, f"main_{bucket}", pd.DataFrame(rows))
    print(f"[tables] {title}：{len(rows)} 行 → {_md_path(args)} / {_csv_path(args, f'main_{bucket}')}")


def cmd_ablation(args: argparse.Namespace) -> None:
    master, summary, stats = _load(args.out_root)
    _require(master, "消融表")
    candidates = [e.name for e in by_category(CAT_ABLATION) if e.status != STATUS_REVOKED]
    if args.stage:
        stages = [args.stage]
    else:
        present = _stage_set(master)
        stages = [s for s in candidates if s in present]
    if not stages:
        raise SystemExit(f"消融表：数据中无候选 stage {candidates}（现有: {sorted(_stage_set(master))}），可 --stage 指定")
    all_rows: list[dict] = []
    for stage in stages:
        exp_cfg = _safe_cfg(stage)
        bucket = (exp_cfg or {}).get("bucket") or ("gnn" if "graph" in stage else "cnn")
        models = _ordered_models(master, stage, exp_cfg)
        ref = _resolve_ref(stage, exp_cfg, bucket, models, args.ref)
        if ref is None:
            print(f"[tables] 消融表 {stage}：参照配置未解析，请 --ref <label>；该段跳过")
            continue
        rows, grid = _metric_rows(master, summary, stats, stage, models, ref)
        headers = ["配置", "AUC", "AUC(P∪U)", "AUPRC", "Brier", "ECE(P∪U)", "Recall@K", f"ΔAUC(P∪U) vs Full({_display_model(ref)})"]
        title = f"消融表：{stage}（参照 = {_display_model(ref)}）"
        content = f"> 来源：{args.out_root}；每行相对参照配置只差一个组件；ΔAUC(P∪U) = 该配置 − Full（主口径候选域 auc_pu）；AUC(P∪U)=候选域主口径，ECE(P∪U)=候选域校准误差；单 seed×5 折（n=5），空间伪重复下仅描述性统计，不做显著性推断\n\n{_md_table(headers, grid)}"
        _write_section(_md_path(args), title, content)
        all_rows.extend(rows)
        print(f"[tables] {title}：{len(rows)} 行")
    if all_rows:
        _write_csv(args, "ablation", pd.DataFrame(all_rows))


def cmd_sensitivity(args: argparse.Namespace) -> None:
    master, summary, stats = _load(args.out_root)
    _require(master, "敏感性表")
    candidates = [e.name for e in by_category(CAT_SENSITIVITY) if e.status != STATUS_REVOKED]
    if args.stage:
        stages = [args.stage]
    else:
        present = _stage_set(master)
        stages = [s for s in candidates if s in present]
    if not stages:
        raise SystemExit(f"敏感性表：数据中无候选 stage {candidates}（现有: {sorted(_stage_set(master))}），可 --stage 指定")
    all_rows: list[dict] = []
    for stage in stages:
        exp_cfg = _safe_cfg(stage)
        models = _ordered_models(master, stage, exp_cfg)
        params = ""
        cells = (exp_cfg or {}).get("cells")
        if isinstance(cells, dict):
            params = " × ".join(cells.keys())
        note = str((exp_cfg or {}).get("report") or (exp_cfg or {}).get("note") or "").replace("\n", " ")
        rows = []
        grid = []
        for m in models:
            s = _summary_of(summary, stage, m)
            rows.append({"stage": stage, "model": m, "param": params, "n": s.get("n", np.nan),
                         "auc_mean": s.get("auc_mean", np.nan), "auc_std": s.get("auc_std", np.nan),
                         "auc_pu_mean": s.get("auc_pu_mean", np.nan), "auc_pu_std": s.get("auc_pu_std", np.nan),
                         "auprc_mean": s.get("auprc_mean", np.nan), "auprc_std": s.get("auprc_std", np.nan),
                         "brier_mean": s.get("brier_mean", np.nan), "brier_std": s.get("brier_std", np.nan),
                         "ece_pu_mean": s.get("ece_pu_mean", np.nan), "ece_pu_std": s.get("ece_pu_std", np.nan),
                         "recall_k_mean": s.get("recall_k_mean", np.nan), "recall_k_std": s.get("recall_k_std", np.nan)})
            grid.append([_display_model(m), str(s.get("n", "—")),
                         _fmt_ms(s.get("auc_mean"), s.get("auc_std")),
                         _fmt_ms(s.get("auc_pu_mean"), s.get("auc_pu_std")),
                         _fmt_ms(s.get("auprc_mean"), s.get("auprc_std")),
                         _fmt_ms(s.get("brier_mean"), s.get("brier_std")),
                         _fmt_ms(s.get("ece_pu_mean"), s.get("ece_pu_std")),
                         _fmt_ms(s.get("recall_k_mean"), s.get("recall_k_std"))])
        title = f"敏感性表：{stage}" + (f"（参数：{params}）" if params else "")
        content = f"> 来源：{args.out_root}；结论排序保持/构成变化见配置 report：{note or '—'}；AUC(P∪U)=候选域主口径，ECE(P∪U)=候选域校准误差（sens_pi 校准论证用）；单 seed×5 折（n=5），空间伪重复下仅描述性统计，不做显著性推断\n\n{_md_table(['配置（扫描值）', 'n', 'AUC', 'AUC(P∪U)', 'AUPRC', 'Brier', 'ECE(P∪U)', 'Recall@K'], grid)}"
        _write_section(_md_path(args), title, content)
        all_rows.extend(rows)
        print(f"[tables] {title}：{len(rows)} 行")
    if all_rows:
        _write_csv(args, "sensitivity", pd.DataFrame(all_rows))


def cmd_protocol(args: argparse.Namespace) -> None:
    master, summary, stats = _load(args.out_root)
    _require(master, "协议表")
    candidates = [e.name for e in by_category(CAT_PROTOCOL)]
    stage = _pick_stage(args.stage, candidates, master, "协议表")
    models = _ordered_models(master, stage, _safe_cfg(stage))
    rows = []
    grid = []
    for m in models:
        s = _summary_of(summary, stage, m)
        rows.append({"stage": stage, "model": m, "n": s.get("n", np.nan),
                     "auc_mean": s.get("auc_mean", np.nan), "auc_std": s.get("auc_std", np.nan),
                     "auc_pu_mean": s.get("auc_pu_mean", np.nan), "auc_pu_std": s.get("auc_pu_std", np.nan),
                     "auprc_mean": s.get("auprc_mean", np.nan), "auprc_std": s.get("auprc_std", np.nan),
                     "brier_mean": s.get("brier_mean", np.nan), "brier_std": s.get("brier_std", np.nan),
                     "ece_pu_mean": s.get("ece_pu_mean", np.nan), "ece_pu_std": s.get("ece_pu_std", np.nan),
                     "recall_k_mean": s.get("recall_k_mean", np.nan), "recall_k_std": s.get("recall_k_std", np.nan)})
        grid.append([_display_model(m), _fmt_ms(s.get("auc_mean"), s.get("auc_std")),
                     _fmt_ms(s.get("auc_pu_mean"), s.get("auc_pu_std")),
                     _fmt_ms(s.get("auprc_mean"), s.get("auprc_std")),
                     _fmt_ms(s.get("brier_mean"), s.get("brier_std")),
                     _fmt_ms(s.get("ece_pu_mean"), s.get("ece_pu_std")),
                     _fmt_ms(s.get("recall_k_mean"), s.get("recall_k_std"))])
    title = f"协议表：{stage}"
    content = (
        f"> 来源：{args.out_root}；同一方法（SAGE）在不同评估协议下的表现；"
        f"SAGE>GAT>GCN 排序保持性结合主结果表判断；AUC(P∪U)=候选域主口径，ECE(P∪U)=候选域校准误差；"
        f"单 seed×5 折（n=5），空间伪重复下仅描述性统计，不做显著性推断\n\n"
        + _md_table(["协议", "AUC", "AUC(P∪U)", "AUPRC", "Brier", "ECE(P∪U)", "Recall@K"], grid)
    )
    _write_section(_md_path(args), title, content)
    _write_csv(args, "protocol", pd.DataFrame(rows))
    print(f"[tables] {title}：{len(rows)} 行")


def cmd_application(args: argparse.Namespace) -> None:
    summary_path = Path(args.prob_map_summary) if args.prob_map_summary else Path(args.out_root) / "app_prob_map" / "prob_map_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"应用表：prob_map_summary.csv 不存在：{summary_path}（先跑 analysis.prob_map）")
    df = pd.read_csv(summary_path)
    by = {}
    for _, r in df.iterrows():
        fkey = str(r["feature"]).strip()
        if fkey in ("nan", "None", ""):
            fkey = "all"
        by[(str(r["metric"]), fkey)] = r

    def _fmt_pval(p) -> str:
        if p is None:
            return "—"
        if pd.isna(p):
            return "—"
        if p < 1e-4:
            return "<1e-4"
        return f"{p:.4f}"

    # 1. Spearman 相关
    spear = []
    for feat in ["slope_max", "slope_mean", "water_ratio", "dem_mean"]:
        r = by.get(("spearman_rho", feat))
        p = by.get(("spearman_p", feat))
        spear.append([feat,
                      f"{r['value']:.3f}" if r is not None else "—",
                      _fmt_pval(p["value"] if p is not None else None),
                      str(int(r["n"])) if r is not None else "—"])
    spear_md = _md_table(["地形特征", "Spearman ρ", "p", "n"], spear)

    # 2. RN 物理吻合
    rn_rows = []
    for key in [("prob_mean", "rn_cell"), ("prob_mean", "non_rn_cell")]:
        r = by.get(key)
        rn_rows.append([key[1], f"{r['value']:.4f}" if r is not None else "—", str(int(r["n"])) if r is not None else "—"])
    diff = by.get(("prob_diff_non_rn_minus_rn", "all"))
    frac = by.get(("rn_frac", "all"))
    diff_txt = f"{diff['value']:+.4f}" if diff is not None else "—"
    frac_txt = f"{frac['value']:.3f}" if frac is not None else "—"
    rn_md = (
        "| 单元类型 | prob_mean 均值 | n |\n|---|---|---|\n"
        + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rn_rows)
        + f"\n\n- prob_diff（非RN − RN）: {diff_txt}，RN 占比: {frac_txt}"
    )

    # 3. Top-10% 物理可行率
    feas = by.get(("feasible_frac", "top10"))
    top_p = by.get(("prob_mean", "top10"))
    feas_txt = f"{feas['value']:.4f}" if feas is not None else "—"
    top_p_txt = f"{top_p['value']:.4f}" if top_p is not None else "—"
    top_md = f"- 高概率单元（top 10%）物理可行率: {feas_txt}，其 prob_mean 均值: {top_p_txt}"

    title = "应用表：概率地图 vs 地形地图差异"
    content = (
        f"> 来源：{summary_path}\n\n"
        f"### 概率 ↔ 地形 Spearman 相关\n{spear_md}\n\n"
        f"### RN 物理吻合\n{rn_md}\n\n"
        f"### 高概率单元物理可行率\n{top_md}"
    )
    _write_section(_md_path(args), title, content)
    _write_csv(args, "application", df)
    print(f"[tables] 应用表：{len(df)} 行 summary → {_md_path(args)}")


# ================= CLI =================

def _md_path(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "tables" / "paper_tables.md"


def _csv_path(args: argparse.Namespace, name: str) -> Path:
    return Path(args.out_root) / "tables" / f"{name}.csv"


def _write_csv(args: argparse.Namespace, name: str, df: pd.DataFrame) -> None:
    p = _csv_path(args, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="论文表格（消费 aggregate + prob_map 输出）")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out_root", default="outputs/exp_closure_20260315", help="聚合输出根目录")
    common.add_argument("--stage", default="", help="显式指定 stage（旧编号数据如 E5/E7/E13/E14 需指定）")
    common.add_argument("--ref", default="", help="参照配置 label（主方法 / Full）；默认从配置解析")

    p = sub.add_parser("main", parents=[common], help="主结果表（组① CNN / 组② GNN）")
    p.add_argument("bucket", choices=["cnn", "gnn"], help="比较组")

    p = sub.add_parser("ablation", parents=[common], help="消融表（abla_* 各一段）")
    p = sub.add_parser("sensitivity", parents=[common], help="敏感性表（sens_* 各一段）")
    p = sub.add_parser("protocol", parents=[common], help="协议表（proto_trans_ind_buffer）")
    p = sub.add_parser("application", parents=[common], help="应用表（prob_map_summary.csv）")
    p.add_argument("--prob_map_summary", default="", help="prob_map_summary.csv 路径（默认 <out_root>/app_prob_map/）")

    args = parser.parse_args()
    {
        "main": cmd_main,
        "ablation": cmd_ablation,
        "sensitivity": cmd_sensitivity,
        "protocol": cmd_protocol,
        "application": cmd_application,
    }[args.command](args)


if __name__ == "__main__":
    main()
