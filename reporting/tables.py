# -*- coding: utf-8 -*-
"""论文表格：只消费 analysis/aggregate.py + analysis/prob_map.py 的输出。

数据来源（不依赖不存在的 results_by_config.csv）：
- <out_root>/master_results.csv（逐格逐折逐种子）
- <out_root>/master_summary.csv（stage × model 的 mean ± std）
- <out_root>/stats_results.csv（组内配对 Wilcoxon + Holm p_adj）
- prob_map_summary.csv（analysis/prob_map 产出，应用表）

子命令：
    python -m reporting.tables main cnn|gnn   [--out_root …] [--stage …] [--ref …]
    python -m reporting.tables ablation       [--out_root …] [--stage …] [--ref …]
    python -m reporting.tables sensitivity    [--out_root …] [--stage …]
    python -m reporting.tables protocol       [--out_root …] [--stage …]
    python -m reporting.tables application    [--out_root …] [--prob_map_summary …]

stage 默认从 registry（experiments/registry.py）按分类/桶自动发现；数据中若只有旧
编号（E5/E7/E13/E14）则用 --stage 显式指定。参照格（主方法 / Full）默认从配置
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
METRICS = [
    ("auc", "AUC"),
    ("auprc", "AUPRC"),
    ("brier", "Brier"),
    ("recall_k", "Recall@K"),
]


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


def _fmt_p(p) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.05:
        return f"{p:.3f}*"
    return f"{p:.3f}"


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
    """参照格（主方法 / Full）label：按实验语义从配置 cells 解析。"""
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
    for m in models:  # 旧数据启发式：按桶关键词兜底
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
            delta, p = _pair_delta(stats, stage, "auc", m, ref) if ref else (np.nan, np.nan)
        rows.append(
            {
                "stage": stage,
                "model": m,
                "ref": "" if ref is None or m != ref else ref,
                "auc_mean": s.get("auc_mean", np.nan),
                "auc_std": s.get("auc_std", np.nan),
                "auprc_mean": s.get("auprc_mean", np.nan),
                "auprc_std": s.get("auprc_std", np.nan),
                "brier_mean": s.get("brier_mean", np.nan),
                "brier_std": s.get("brier_std", np.nan),
                "recall_k_mean": s.get("recall_k_mean", np.nan),
                "recall_k_std": s.get("recall_k_std", np.nan),
                "delta_auc_vs_ref": delta,
                "p_adj": p,
                "n": s.get("n", np.nan),
            }
        )
    grid = [
        [
            m,
            _fmt_ms(r["auc_mean"], r["auc_std"]),
            _fmt_ms(r["auprc_mean"], r["auprc_std"]),
            _fmt_ms(r["brier_mean"], r["brier_std"]),
            _fmt_ms(r["recall_k_mean"], r["recall_k_std"]),
            "—" if m == ref or np.isnan(r["delta_auc_vs_ref"]) else f"{r['delta_auc_vs_ref']:+.4f}",
            "—" if m == ref else _fmt_p(r["p_adj"]),
        ]
        for m, r in zip(models, rows)
    ]
    return rows, grid


# ================= 子命令 =================

def cmd_main(args: argparse.Namespace) -> None:
    master, summary, stats = _load(args.out_root)
    _require(master, "主结果表")
    bucket = args.bucket
    candidates = [e.name for e in by_category(CAT_COMPARISON) if e.bucket == bucket and e.status != STATUS_REVOKED]
    stage = _pick_stage(args.stage, candidates, master, f"主结果表（桶{'CNN' if bucket == 'cnn' else 'GNN'}）")
    exp_cfg = _safe_cfg(stage)
    models = _ordered_models(master, stage, exp_cfg)
    ref = _resolve_ref(stage, exp_cfg, bucket, models, args.ref)
    if ref is None:
        raise SystemExit(f"主结果表（{stage}）：无法确定参照格，请 --ref <label>（如 SAGE / PURN-CNN）")

    rows, grid = _metric_rows(master, summary, stats, stage, models, ref)
    headers = ["模型", "AUC", "AUPRC", "Brier", "Recall@K", f"ΔAUC vs {ref}", "p_adj*"]
    title = f"主结果表（桶{'① CNN' if bucket == 'cnn' else '② GNN'}，stage={stage}，参照={ref}）"
    content = f"> 来源：{args.out_root}；ΔAUC = 模型 − 参照格；p_adj 为 Holm 校正；* 表示 <0.05\n\n{_md_table(headers, grid)}"
    _write_section(_md_path(args), title, content)
    _write_csv(args, "main", pd.DataFrame(rows))
    print(f"[tables] {title}：{len(rows)} 行 → {_md_path(args)} / {_csv_path(args, 'main')}")


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
            print(f"[tables] 消融表 {stage}：参照格未解析，请 --ref <label>；该段跳过")
            continue
        rows, grid = _metric_rows(master, summary, stats, stage, models, ref)
        headers = ["格", "AUC", "AUPRC", "Brier", "Recall@K", f"ΔAUC vs Full({ref})", "p_adj*"]
        title = f"消融表：{stage}（参照=Full {ref}）"
        content = f"> 来源：{args.out_root}；每行相对参照格只差一个组件；ΔAUC = 该格 − Full；* 表示 <0.05\n\n{_md_table(headers, grid)}"
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
                         "auprc_mean": s.get("auprc_mean", np.nan), "auprc_std": s.get("auprc_std", np.nan),
                         "brier_mean": s.get("brier_mean", np.nan), "brier_std": s.get("brier_std", np.nan),
                         "recall_k_mean": s.get("recall_k_mean", np.nan), "recall_k_std": s.get("recall_k_std", np.nan)})
            grid.append([m, str(s.get("n", "—")),
                         _fmt_ms(s.get("auc_mean"), s.get("auc_std")),
                         _fmt_ms(s.get("auprc_mean"), s.get("auprc_std")),
                         _fmt_ms(s.get("brier_mean"), s.get("brier_std")),
                         _fmt_ms(s.get("recall_k_mean"), s.get("recall_k_std"))])
        title = f"敏感性表：{stage}" + (f"（参数：{params}）" if params else "")
        content = f"> 来源：{args.out_root}；结论排序保持/计数漂移见配置 report：{note or '—'}\n\n{_md_table(['格（扫描值）', 'n', 'AUC', 'AUPRC', 'Brier', 'Recall@K'], grid)}"
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
                     "auprc_mean": s.get("auprc_mean", np.nan), "auprc_std": s.get("auprc_std", np.nan),
                     "brier_mean": s.get("brier_mean", np.nan), "brier_std": s.get("brier_std", np.nan),
                     "recall_k_mean": s.get("recall_k_mean", np.nan), "recall_k_std": s.get("recall_k_std", np.nan)})
        grid.append([m, _fmt_ms(s.get("auc_mean"), s.get("auc_std")),
                     _fmt_ms(s.get("auprc_mean"), s.get("auprc_std")),
                     _fmt_ms(s.get("brier_mean"), s.get("brier_std")),
                     _fmt_ms(s.get("recall_k_mean"), s.get("recall_k_std"))])
    title = f"协议表：{stage}"
    content = (
        f"> 来源：{args.out_root}；同一方法（SAGE）在不同评估协议下的表现；"
        f"SAGE>GAT>GCN 排序保持性结合主结果表判断\n\n"
        + _md_table(["协议", "AUC", "AUPRC", "Brier", "Recall@K"], grid)
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
        "| 格类 | prob_mean 均值 | n |\n|---|---|---|\n"
        + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rn_rows)
        + f"\n\n- prob_diff（非RN − RN）: {diff_txt}，RN 占比: {frac_txt}"
    )

    # 3. Top-10% 物理可行率
    feas = by.get(("feasible_frac", "top10"))
    top_p = by.get(("prob_mean", "top10"))
    feas_txt = f"{feas['value']:.4f}" if feas is not None else "—"
    top_p_txt = f"{top_p['value']:.4f}" if top_p is not None else "—"
    top_md = f"- 高概率格（top 10%）物理可行率: {feas_txt}，其 prob_mean 均值: {top_p_txt}"

    title = "应用表：概率地图 vs 地形地图差异"
    content = (
        f"> 来源：{summary_path}\n\n"
        f"### 概率 ↔ 地形 Spearman 相关\n{spear_md}\n\n"
        f"### RN 物理吻合\n{rn_md}\n\n"
        f"### 高概率格物理可行率\n{top_md}"
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
    common.add_argument("--ref", default="", help="参照格 label（主方法 / Full）；默认从配置解析")

    p = sub.add_parser("main", parents=[common], help="主结果表（桶① CNN / 桶② GNN）")
    p.add_argument("bucket", choices=["cnn", "gnn"], help="指标桶")

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
