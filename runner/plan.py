# -*- coding: utf-8 -*-
"""配置驱动 E1–E15 作业构建（← scripts/run_closure_plan.py 的 build_jobs + add_*_job）。

数据源：
- experiments/registry.py —— 语义实验名、legacy_id、status；
- experiments/configs/*.yaml —— 每个实验的 cells 格子（list[dict] 原样 / dict 笛卡尔积）；
- base.yaml —— 全局训练超参默认（run_closure_plan.py 的 argparse 默认值单一数据源）。

CLI 标志名是 runner/train 的实现细节，**不进 YAML**：cell 语义键 → CLI 标志的
映射集中在 _CELL_FLAG_MAP（一处、可测试）。训练/推理目录统一走
`_train_out_dir(out_root, exp_name, cell_label, seed)` 解析规则，取代旧 E 布局。

作业族：
- 训练作业：`_render_cnn` / `_render_gnn`（stage=cnn/gnn 的实验）；
- 外部基线：`cmp_pu_baseline` + `cmp_graph_baseline` 合并为一个 baselines 作业（同原 E6）；
- 闭环节点：`app_infer` / `app_prob_map` / `diag_fold4`
  由管线渲染器生成，消费下游训练输出（经 _train_out_dir 解析）。

用法：
    python -m runner.plan --exp sens_lambda sens_rn_threshold --seeds 62        # dry-run（默认）
    python -m runner.plan --subset E2 E4 --seeds 62                             # legacy 编号
    python -m runner.plan --execute --seeds 42,52,62 --out_root outputs/...     # 执行全部
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from experiments.config import load_family
from experiments.registry import STATUS_REVOKED, all_experiments, by_legacy, get
from src.core.matrix import merge_matrices

ROOT_DIR = Path(__file__).resolve().parents[1]

# 家族配置文件（加载顺序 = 合并优先级链中的同层并列）
_FAMILY_FILES = ["main", "ablations", "sensitivity", "protocol", "application"]

# 实验 → 家族（config.load_experiment 的 prefix 推断不含 diag_*）
_FAMILY_OF = {
    "cmp_main_cnn": "main",
    "cmp_main_gnn": "main",
    "cmp_pu_baseline": "main",
    "cmp_graph_baseline": "main",
    "abla_rn_source": "ablations",
    "abla_objective": "ablations",
    "abla_graph": "ablations",
    "sens_rn_threshold": "sensitivity",
    "sens_pi": "sensitivity",
    "sens_lambda": "sensitivity",
    "sens_graph_k": "sensitivity",
    "sens_fold_grid": "sensitivity",
    "sens_feature": "sensitivity",
    "proto_trans_ind_buffer": "protocol",
    "app_infer": "application",
    "app_prob_map": "application",
}

# 闭环节点：无独立 cells，由管线渲染器生成（消费下游训练输出）
# （Phase 2 加入 app_prob_map，随 render_app_prob_map 一并提交）
_PIPELINE_EXPS = ["app_infer", "diag_fold4"]

# 外部基线：合并为一个 baselines 作业（脚本一次训练全部方法，同原 E6 行为）
_EXTERNAL_EXPS = ["cmp_pu_baseline", "cmp_graph_baseline"]

# cell 语义键 → CLI 标志（CLI 标志名是 runner/train 的实现细节，集中在此唯一数据源）
_CELL_FLAG_MAP = {
    "mode": "--mode",
    "lambda_rn": "--lambda_rn",
    "warm_epochs": "--warm_epochs",
    "nnpu_epochs": "--nnpu_epochs",
    "rn_strategy": "--rn_strategy",
    "noise_rate": "--rn_noise_rate",
    "slope": "--rn_slope_thresh",
    "water": "--rn_water_thresh",
    "pi": "--pi_prior",
}

# YAML 架构名 → train_gnn 的 --gnn_type（nodemlp = 无图传播的 MLP，同架构对照）
_ARCH_TO_GNN_TYPE = {"sage": "sage", "gcn": "gcn", "gat": "gat", "nodemlp": "mlp", "mlp": "mlp"}

# dict 格子的 cell 标签短键：单字符键直接拼接（s35），多字符键用 `short_<v>`（lambda_0.4）
_LABEL_KEY = {
    "slope": "s",
    "water": "w",
    "lambda_rn": "lambda",
    "pi": "pi",
    "mode": "mode",
    "arch": "arch",
    "graph": "graph",
    "method": "method",
    "gnn_type": "arch",
    "candidate_source": "src",
    "model": "model",
    "rn": "rn",
    "noise_rate": "noise",
}


def _expand_cells(cells: Any) -> list[dict]:
    """统一格子形状：list[dict] 原样返回；dict{参数: [值,...]} 展开为笛卡尔积。"""
    if cells is None:
        return []
    if isinstance(cells, list):
        return [dict(c) for c in cells]
    if isinstance(cells, dict):
        keys = list(cells)
        if not keys:
            return []
        return [dict(zip(keys, value)) for value in itertools.product(*(cells[k] for k in keys))]
    raise TypeError(f"cells 结构不支持: {type(cells)}")


def _cell_label(cell: dict) -> str:
    """list[dict] 用 label 键；dict 格子 = k{v:g} 拼接（如 s35_w0.3、lambda_0.4）。"""
    if "label" in cell:
        return str(cell["label"])
    if len(cell) == 1:  # 单键且值为字符串：直接用值（proto 的 mode 格）
        (k, v) = next(iter(cell.items()))
        if isinstance(v, str):
            return v
    parts: list[str] = []
    for k, v in cell.items():
        short = _LABEL_KEY.get(k, k)
        vs = f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        parts.append(f"{short}{vs}" if len(short) <= 2 else f"{short}_{vs}")
    return "_".join(parts)


def _train_out_dir(out_root: Path, exp_name: str, cell_label: str, seed: int) -> Path:
    """训练/推理输出目录统一解析规则（取代旧 E 布局；聚合读 matrix 的 result_file，不依赖布局）。"""
    return out_root / exp_name / f"{cell_label}_s{seed}"


class Planner:
    """从 registry + YAML 渲染作业清单。"""

    def __init__(
        self,
        out_root: str | Path,
        seeds: Iterable[int],
        external_seeds: Iterable[int],
        overrides: dict | None = None,
        fast_2fold: bool = False,
    ) -> None:
        self.out_root = Path(out_root)
        self.seeds = list(seeds)
        self.external_seeds = list(external_seeds)
        self.overrides = overrides or {}
        self.fast_2fold = fast_2fold
        self.py = sys.executable
        self.jobs: list[dict] = []
        self._external_rendered = False
        self.cfgs: dict[str, Any] = {}
        for family in _FAMILY_FILES:
            self.cfgs.update(load_family(family))

    # ---- 配置访问 ------------------------------------------------------------

    def cfg(self, exp_name: str) -> dict:
        return self.cfgs[exp_name]

    def val(self, exp_name: str, key: str, default: Any = None) -> Any:
        """合并默认值 + 该实验覆盖 + CLI 覆盖。"""
        if key in self.overrides:
            return self.overrides[key]
        return self.cfg(exp_name).get(key, default)

    # ---- 作业注册 ------------------------------------------------------------

    def _emit(
        self,
        exp_name: str,
        label: str,
        seed: int,
        cmd: list[str],
        out_dir: Path,
        result_type: str,
        result_file: Path,
        negative_file: str = "",
        *,
        stage: str | None = None,
        model: str | None = None,
        exp_id: str | None = None,
    ) -> None:
        self.jobs.append(
            {
                "exp_id": exp_id or f"{exp_name}_{label}_s{seed}",
                "stage": stage or exp_name,
                "model": model or label,
                "seed": seed,
                "command": cmd,
                "out_dir": str(out_dir),
                "result_type": result_type,
                "result_file": str(result_file),
                "negative_file": negative_file,
            }
        )

    def _resolve_cell(self, exp_name: str, predicate: Callable[[dict], bool]) -> tuple[dict, str]:
        for cell in _expand_cells(self.cfg(exp_name).get("cells")):
            if predicate(cell):
                return cell, _cell_label(cell)
        raise RuntimeError(f"无法解析 {exp_name} 的源格子（predicate 未匹配任何 cell）")

    # ---- CNN 训练渲染（stage=cnn） -------------------------------------------

    def _render_cnn(self, exp_name: str) -> None:
        cfg = self.cfg(exp_name)
        for cell in _expand_cells(cfg.get("cells")):
            label = _cell_label(cell)
            mode = cell.get("mode", "pu")
            warm_default = self.val(exp_name, "cnn_warm_epochs", self.val(exp_name, "warm_epochs", 5))
            warm = int(cell.get("warm_epochs", warm_default))
            # P1 预算统一：warm_epochs==0 时无 warm 阶段，nnpu 补满总预算（budget_epochs）
            nnpu = int(cell.get("nnpu_epochs", self.val(exp_name, "cnn_nnpu_epochs", 15)))
            if warm == 0:
                nnpu = int(self.val(exp_name, "budget_epochs", 20))
            for seed in self.seeds:
                out_dir = _train_out_dir(self.out_root, exp_name, label, seed)
                cmd = [
                    self.py,
                    "-m",
                    "runner.train.train_cnn",
                    "--processed_dir",
                    str(self.val(exp_name, "processed_dir", "data/processed")),
                    "--out_dir",
                    str(out_dir),
                    "--mode",
                    str(mode),
                    "--folds",
                    str(self.val(exp_name, "folds", 5)),
                    "--spatial_grid_size",
                    str(self.val(exp_name, "grid_size", 10)),
                    "--batch_size",
                    str(self.val(exp_name, "cnn_batch_size", 128)),
                    "--warm_epochs",
                    str(warm),
                    "--nnpu_epochs",
                    str(nnpu),
                    "--lr",
                    str(self.val(exp_name, "cnn_lr", 1e-4)),
                    "--seed",
                    str(seed),
                ]
                if self.fast_2fold:
                    cmd += ["--fold_ids", "0,1"]
                buffer_cells = int(self.val(exp_name, "buffer_cells", 0))
                if buffer_cells > 0:
                    cmd += ["--buffer_cells", str(buffer_cells)]
                # RN 锚点 + cell 派生标志（顺序与旧 run_closure_plan 一致）
                cmd += ["--rn_strategy", str(cell.get("rn_strategy", self.val(exp_name, "rn_strategy", "physics")))]
                if "noise_rate" in cell:
                    cmd += ["--rn_noise_rate", str(cell["noise_rate"])]
                cmd += ["--rn_slope_thresh", str(cell.get("slope", self.val(exp_name, "rn_slope_thresh", 45.0)))]
                cmd += ["--rn_water_thresh", str(cell.get("water", self.val(exp_name, "rn_water_thresh", 0.5)))]
                if "lambda_rn" in cell:  # 仅显式操纵 λ 的实验才传（train_cnn 默认 1.0，行为等价）
                    cmd += ["--lambda_rn", str(cell["lambda_rn"])]
                if "pi" in cell:
                    cmd += ["--pi_prior", str(cell["pi"])]
                if warm == 0:
                    cmd += ["--disable_warmup"]
                self._emit(exp_name, label, seed, cmd, out_dir, "cnn_results", out_dir / "results.csv")

    # ---- GNN 训练渲染（stage=gnn） -------------------------------------------

    def _render_gnn(self, exp_name: str) -> None:
        cfg = self.cfg(exp_name)
        for cell in _expand_cells(cfg.get("cells")):
            label = _cell_label(cell)
            gnn_type = _ARCH_TO_GNN_TYPE.get(
                str(cell.get("gnn_type") or cell.get("arch") or cfg.get("gnn_type", "sage")), "sage"
            )
            eval_mode = str(cell.get("mode", "transductive"))
            extra: list[str] = []
            if str(cell.get("graph")) == "self_loop":
                extra += ["--self_loop_only"]  # 同架构无传播对照
            if eval_mode == "buffered":
                buffer_cells = int(self.val(exp_name, "buffer_cells", 0))
                if buffer_cells <= 0:
                    print(f"[plan] skip {exp_name}/{label}: buffered 协议需 buffer_cells>0（当前为 0）")
                    continue
                eval_mode = "inductive"
                extra += ["--buffer_cells", str(buffer_cells)]
            for seed in self.seeds:
                out_dir = _train_out_dir(self.out_root, exp_name, label, seed)
                cmd = [
                    self.py,
                    "-m",
                    "runner.train.train_gnn",
                    "--gnn_data_dir",
                    str(self.val(exp_name, "gnn_data_dir", "data/gnn_ready")),
                    "--out_dir",
                    str(out_dir),
                    "--gnn_type",
                    gnn_type,
                    "--eval_mode",
                    eval_mode,
                    "--folds",
                    str(self.val(exp_name, "folds", 5)),
                    "--spatial_grid_size",
                    str(self.val(exp_name, "grid_size", 10)),
                    "--epochs",
                    str(self.val(exp_name, "gnn_epochs", 200)),
                    "--lr",
                    str(self.val(exp_name, "gnn_lr", 0.005)),
                    "--seed",
                    str(seed),
                    "--rn_slope_thresh",
                    str(self.val(exp_name, "rn_slope_thresh", 45.0)),
                    "--rn_water_thresh",
                    str(self.val(exp_name, "rn_water_thresh", 0.5)),
                    "--lambda_rn",
                    str(self.val(exp_name, "lambda_rn", 1.0)),
                ]
                if self.fast_2fold:
                    cmd += ["--fold_ids", "0,1"]
                if extra:
                    cmd += extra
                # train_gnn 内部把模型写入 out_dir/<gnn_type>/，result_file 跟随该约定
                self._emit(exp_name, label, seed, cmd, out_dir, "gnn_results", out_dir / gnn_type / "results.csv")

    # ---- 外部基线渲染（E6） --------------------------------------------------

    def _render_external(self) -> None:
        if self._external_rendered:
            return
        self._external_rendered = True
        exp_name = "cmp_pu_baseline"  # E6 共用同一批默认值，任取一个即可
        out_dir = self.out_root / "E6_external"
        cmd = [
            self.py,
            "-m",
            "runner.train.baselines",
            "--gnn_data_dir",
            str(self.val(exp_name, "gnn_data_dir", "data/gnn_ready")),
            "--out_dir",
            str(out_dir),
            "--folds",
            str(self.val(exp_name, "folds", 5)),
            "--grid_size",
            str(self.val(exp_name, "grid_size", 10)),
            "--seeds",
            ",".join(str(s) for s in self.external_seeds),
            "--rn_slope_thresh",
            str(self.val(exp_name, "rn_slope_thresh", 45.0)),
            "--rn_water_thresh",
            str(self.val(exp_name, "rn_water_thresh", 0.5)),
            "--upu_epochs",
            str(self.val(exp_name, "upu_epochs", 120)),
            "--appnp_epochs",
            str(self.val(exp_name, "appnp_epochs", 150)),
            "--lp_max_iter",
            str(self.val(exp_name, "lp_max_iter", 1000)),
            "--device",
            str(self.val(exp_name, "device", "cuda")),
        ]
        self._emit(
            "cmp_pu_baseline+cmp_graph_baseline",
            "external",
            -1,
            cmd,
            out_dir,
            "external_results",
            out_dir / "results_external.csv",
            negative_file=out_dir / "negative_external.csv",
            stage="E6",
            model="uPU+ElkanNoto+BaggingPU+LP+APPNP",
            exp_id="E6_external",
        )

    # ---- 管线渲染器 ----------------------------------------------------------

    def _app_source_dirs(self, seed: int) -> tuple[Path, Path, Path]:
        """解析下游推理的源训练目录：PN / PURN-CNN（CNN）、SAGE（GNN）。"""
        _, pn_label = self._resolve_cell("cmp_main_cnn", lambda c: c.get("mode") == "pn")
        _, purn_label = self._resolve_cell("cmp_main_cnn", lambda c: c.get("mode") == "pu")
        _, sage_label = self._resolve_cell("cmp_main_gnn", lambda c: c.get("gnn_type") == "sage")
        return (
            _train_out_dir(self.out_root, "cmp_main_cnn", pn_label, seed),
            _train_out_dir(self.out_root, "cmp_main_cnn", purn_label, seed),
            _train_out_dir(self.out_root, "cmp_main_gnn", sage_label, seed),
        )

    def render_app_infer(self) -> None:
        seed = self.seeds[0]  # 整域推理只需主 seed
        pn_dir, purn_dir, sage_dir = self._app_source_dirs(seed)
        processed = str(self.val("cmp_main_cnn", "processed_dir", "data/processed"))

        cmd = [self.py, "-m", "runner.train.infer", "--processed_dir", processed,
               "--exp_dir", str(pn_dir), "--out_name", "inference_results.csv"]
        self._emit("app_infer", "cnn", seed, cmd, pn_dir, "inference", pn_dir / "inference_results.csv",
                   model="CNN-infer", exp_id="app_infer_cnn")

        cmd = [self.py, "-m", "runner.train.infer", "--processed_dir", processed,
               "--exp_dir", str(purn_dir), "--out_name", "inference_results.csv"]
        self._emit("app_infer", "purn", seed, cmd, purn_dir, "inference", purn_dir / "inference_results.csv",
                   model="PURN-infer", exp_id="app_infer_purn")

        archs = [str(c.get("gnn_type")) for c in _expand_cells(self.cfg("cmp_main_gnn").get("cells")) if c.get("gnn_type")]
        cmd = [self.py, "-m", "runner.train.infer_gnn_all",
               "--gnn_data_dir", str(self.val("cmp_main_gnn", "gnn_data_dir", "data/gnn_ready")),
               "--demand_csv", str(self.val("cmp_main_gnn", "demand_csv", "data/processed/demand_per_patch.csv")),
               "--base_exp_dir", str(sage_dir), "--archs", *archs,
               "--p_add", "100", "--prob_thresh", "0.7"]
        self._emit("app_infer", "gnn", seed, cmd, sage_dir, "inference", sage_dir / "inference_results_sage.csv",
                   model="PURN+GNN-infer", exp_id="app_infer_gnn")

    def render_diag_fold4(self) -> None:
        seed = self.seeds[0]
        arch = str(self.val("proto_trans_ind_buffer", "gnn_type", "sage"))
        trans_dir = _train_out_dir(self.out_root, "proto_trans_ind_buffer", "transductive", seed)
        ind_dir = _train_out_dir(self.out_root, "proto_trans_ind_buffer", "inductive", seed)
        gnn_data = str(self.val("proto_trans_ind_buffer", "gnn_data_dir", "data/gnn_ready"))
        demand = str(self.val("proto_trans_ind_buffer", "demand_csv", "data/processed/demand_per_patch.csv"))

        for tag, base_dir in [("trans", trans_dir), ("ind", ind_dir)]:
            cmd = [self.py, "-m", "runner.train.infer_gnn_all", "--gnn_data_dir", gnn_data,
                   "--demand_csv", demand, "--base_exp_dir", str(base_dir), "--archs", arch,
                   "--p_add", "100", "--prob_thresh", "0.7"]
            self._emit("diag_fold4", f"infer_{tag}", seed, cmd, base_dir, "inference",
                       base_dir / f"inference_results_{arch}.csv",
                       model=f"SAGE-{tag}-infer", exp_id=f"diag_fold4_infer_{tag}")

        out_dir = self.out_root / "diag_fold4"
        cmd = [self.py, "-m", "analysis.fold4",
               "--gnn_data_dir", gnn_data, "--out_dir", str(out_dir),
               "--folds", str(self.val("proto_trans_ind_buffer", "folds", 5)),
               "--grid_size", str(self.val("proto_trans_ind_buffer", "grid_size", 10)),
               "--seed", str(seed), "--target_fold", "4",
               "--infer_csv", f"trans={trans_dir / f'inference_results_{arch}.csv'}",
               "--infer_csv", f"ind={ind_dir / f'inference_results_{arch}.csv'}"]
        self._emit("diag_fold4", "fold4", seed, cmd, out_dir, "fold_diag",
                   out_dir / "fold_diagnostics.csv", model="Fold4Diagnostic", exp_id="diag_fold4_analysis")

    # ---- 作业装配 ------------------------------------------------------------

    def build(self, exp_names: Iterable[str]) -> list[dict]:
        """按语义实验名渲染作业（过滤 revoked / 无 cells 的实验）。"""
        for name in exp_names:
            meta = get(name)
            if meta.status == STATUS_REVOKED:
                print(f"[plan] skip {name}: 已撤销（abla_structure）")
                continue
            if name in _EXTERNAL_EXPS:
                self._render_external()
                continue
            if name in _PIPELINE_EXPS:
                getattr(self, f"render_{name}")()
                continue
            cfg = self.cfg(name)
            if not cfg.get("cells"):
                print(f"[plan] skip {name}: cells 为空（待补实验）")
                continue
            if cfg.get("stage") == "gnn":
                self._render_gnn(name)
            else:
                self._render_cnn(name)
        return self.jobs


def resolve_exp_names(exp: list[str], subset: list[str]) -> list[str]:
    """--exp 语义名（primary）或 --subset legacy 编号（经 registry.by_legacy 解析）。"""
    if exp:
        for n in exp:
            if n not in all_experiments():
                raise SystemExit(f"未知实验名: {n}（见 experiments/registry.py）")
        return list(exp)
    if subset:
        names: list[str] = []
        for token in subset:
            found = by_legacy(token)
            if not found:
                raise SystemExit(f"legacy 编号未注册: {token}")
            for e in found:
                if e.name not in names:
                    names.append(e.name)
        return names
    return list(all_experiments())


def matrix_rows(jobs: list[dict], run_label: str, run_started_at: str, log_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for i, j in enumerate(jobs):
        cmd_str = subprocess.list2cmdline(j["command"])
        rows.append(
            {
                "job_id": f"job_{i:04d}",
                "exp_id": j["exp_id"],
                "stage": j["stage"],
                "model": j["model"],
                "seed": j["seed"],
                "command": cmd_str,
                "out_dir": j["out_dir"],
                "result_type": j["result_type"],
                "result_file": j["result_file"],
                "negative_file": j.get("negative_file", ""),
                "status": "planned",
                "return_code": "",
                "runtime_min": "",
                "log_file": str(log_dir / f"job_{i:04d}.log"),
                "run_label": run_label,
                "run_started_at": run_started_at,
                "job_key": j["result_file"] or cmd_str,
            }
        )
    return pd.DataFrame(rows)


def save_plan_outputs(df: pd.DataFrame, out_root: Path, run_label: str) -> pd.DataFrame:
    """写矩阵 + 快照 + 命令清单（合并复用 src/core/matrix.py，断点状态不被 planned 行覆盖）。"""
    out_root.mkdir(parents=True, exist_ok=True)
    matrix_csv = out_root / "experiment_matrix.csv"
    snapshot = out_root / f"experiment_matrix_{run_label}.csv"
    cmd_file = out_root / "run_commands.ps1"
    cmd_snapshot = out_root / f"run_commands_{run_label}.ps1"

    root_df = merge_matrices(pd.read_csv(matrix_csv), df) if matrix_csv.exists() else df.copy()
    root_df.to_csv(matrix_csv, index=False)
    df.to_csv(snapshot, index=False)
    cmd_file.write_text("\n".join(root_df["command"].astype(str).tolist()) + "\n", encoding="utf-8")
    cmd_snapshot.write_text("\n".join(df["command"].astype(str).tolist()) + "\n", encoding="utf-8")
    print(f"Saved matrix: {matrix_csv}")
    print(f"Saved snapshot: {snapshot}")
    print(f"Saved commands: {cmd_file}")
    return root_df


def parse_seeds(text: str, default: str) -> list[int]:
    return [int(x) for x in (text if text else default).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="配置驱动 E1–E15 作业构建（registry + YAML）")
    parser.add_argument("--exp", nargs="*", default=[], help="语义实验名（experiments/registry.py）")
    parser.add_argument("--subset", nargs="*", default=[], help="legacy E 编号（经 registry.by_legacy 解析）")
    parser.add_argument("--seeds", default="", help="逗号分隔 seed 列表；空 = base.yaml 默认")
    parser.add_argument("--external_seeds", default="", help="外部基线 seeds（空 = base.yaml 默认）")
    parser.add_argument("--out_root", default="outputs/exp_closure_20260315")
    parser.add_argument("--execute", action="store_true", help="渲染后立即执行（默认 dry-run）")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--fast_2fold", action="store_true")
    # base.yaml 超参默认的 CLI 覆盖（与 run_closure_plan.py argparse 同名）
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--buffer_cells", type=int, default=None)
    parser.add_argument("--cnn_batch_size", type=int, default=None)
    parser.add_argument("--cnn_warm_epochs", type=int, default=None)
    parser.add_argument("--cnn_nnpu_epochs", type=int, default=None)
    parser.add_argument("--cnn_lr", type=float, default=None)
    parser.add_argument("--gnn_epochs", type=int, default=None)
    parser.add_argument("--gnn_lr", type=float, default=None)
    parser.add_argument("--upu_epochs", type=int, default=None)
    parser.add_argument("--appnp_epochs", type=int, default=None)
    parser.add_argument("--lp_max_iter", type=int, default=None)
    parser.add_argument("--rn_slope_thresh", type=float, default=None)
    parser.add_argument("--rn_water_thresh", type=float, default=None)
    parser.add_argument("--lambda_rn", type=float, default=None)
    parser.add_argument("--pi_ref", type=float, default=None)
    ns = parser.parse_args()

    overridable = {
        "folds", "grid_size", "device", "buffer_cells", "cnn_batch_size", "cnn_warm_epochs",
        "cnn_nnpu_epochs", "cnn_lr", "gnn_epochs", "gnn_lr", "upu_epochs", "appnp_epochs",
        "lp_max_iter", "rn_slope_thresh", "rn_water_thresh", "lambda_rn", "pi_ref",
    }
    overrides = {k: v for k, v in vars(ns).items() if v is not None and k in overridable}

    out_root = Path(ns.out_root).resolve()
    seeds = parse_seeds(ns.seeds, "42,52,62")
    external_seeds = parse_seeds(ns.external_seeds, "42,52,62")
    exp_names = resolve_exp_names(ns.exp, ns.subset)

    planner = Planner(out_root, seeds, external_seeds, overrides, fast_2fold=ns.fast_2fold)
    jobs = planner.build(exp_names)
    if not jobs:
        raise SystemExit("无可用作业。检查 --exp / --subset 过滤条件。")

    run_label = f"{'_'.join(exp_names) if len(exp_names) <= 4 else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_started_at = datetime.now().isoformat()
    log_dir = out_root / f"logs_{run_label}"
    log_dir.mkdir(parents=True, exist_ok=True)

    df = matrix_rows(jobs, run_label, run_started_at, log_dir)
    save_plan_outputs(df, out_root, run_label)

    # 按实验汇总（dry-run 概览）
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j["stage"]] = counts.get(j["stage"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"Planned {len(jobs)} jobs: {summary}")

    if not ns.execute:
        print("Dry run mode. Add --execute to run all jobs.")
        return

    from runner.execute import run_jobs

    run_jobs(df, out_root, run_label, stop_on_error=ns.stop_on_error)


if __name__ == "__main__":
    main()
