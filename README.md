# 复杂地形气象站选址适宜性建模（重庆）

本项目面向复杂山地地形（重庆），在仅有少量已建站点作为正样本的条件下，学习全域气象站选址适宜性：

- **PURN-CNN**：物理感知的 PU 学习（正样本 P + 未标注 U + 物理不可行 RN），从多尺度地形/土地覆盖/降水栅格中学习适宜性；
- **GraphSAGE**：在空间邻接图上对 CNN 概率场做 refinement，提升空间连贯性；
- **应用展示**：将预测概率重聚合为概率地图，与地形地图对比，分析高适宜区与地形约束的关系。

候选站址被划分为三类：**P**（已建站）、**RN**（物理不可行：坡度 > 45° 或水域占比 > 0.5）、**U**（其余未标注）。跨模型比较的主口径为 **AUC(P∪U)**（在可建候选域内做 P vs U 排序，剔除 RN）。

## 目录结构

```
src/           核心实现（数据处理、模型、损失、图构建、空间划分、指标）
runner/        作业执行层（YAML → 作业计划；训练/推理脚本；断点续跑）
experiments/   实验注册表 + 配置（main / ablations / sensitivity / protocol / application）
analysis/      结果聚合与下游分析（概率地图、空间连贯性、物理基线、统计）
reporting/     论文表格与数据图生成
data/          原始数据与预处理产物（不入库）
outputs/       实验产物（不入库）
```

## 环境

- Python 3.12，依赖由 [uv](https://docs.astral.sh/uv/) 管理
- PyTorch 2.7 + CUDA 12.6（见 `pyproject.toml` 的 `pytorch-cu126` 索引）

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> 无 GPU 时需将 `pyproject.toml` 中的 torch 源改为 CPU 版本，并相应调整 `uv.lock`。

## 数据准备

仓库不包含原始数据和实验产物，需自行放入 `data/`：

```
data/
├── raw/            # DEM、坡度、土地覆盖、降雨 CV 栅格、行政边界 geojson、站点
├── processed/      # patches_info.csv、demand_per_patch.csv 等
└── gnn_ready/      # 图结构相关产物
```

生成预处理产物：

```bash
# 多尺度切片 + 特征提取 + 物理指标（路径在脚本 Config 中配置）
uv run python -m src.features.process

# 按 patch 聚合降雨需求
uv run python -m src.features.build_rain_demand
```

## 快速开始

```bash
# 1) 生成作业计划（默认 dry-run，只打印不执行）
uv run python -m runner.plan --seeds 42 --out_root outputs/exp_main

# 2) 执行（可中断后 --resume 续跑）
uv run python -m runner.plan --seeds 42 --out_root outputs/exp_main --execute
uv run python -m runner.execute --out_root outputs/exp_main --resume

# 3) 聚合为 master_results.csv / stats_results.csv
uv run python -m analysis.aggregate --out_root outputs/exp_main

# 4) 论文表格与数据图
uv run python -m reporting.tables main cnn --out_root outputs/exp_main
uv run python -m reporting.tables main gnn --out_root outputs/exp_main
uv run python -m reporting.figures probmap --merged_csv outputs/exp_main/app_prob_map/prob_map_merged.csv
```

单条模型链的数据流：

```
src.features.process
  → runner.train.train_cnn          # 分折训练 PURN-CNN
  → runner.train.extract_features   # 按折提取节点特征（避免特征泄露）
  → runner.train.train_gnn          # GraphSAGE / GCN / GAT
  → runner.train.infer(_gnn_all)    # 全域推理
  → analysis.prob_map               # 概率地图 vs 地形地图
  → analysis.spatial_coherence      # 空间连贯性 / Moran's I
  → analysis.aggregate → reporting  # 聚合、表格、图
```

各实验族的语义名与配置见 `experiments/registry.py` 与 `experiments/configs/*.yaml`。

## 说明

- `data/`、`outputs/` 与内部文档 `docs/` 不纳入版本库，仅保留 `data/.gitkeep`、`outputs/.gitkeep` 作为目录占位。
- 所有结果可由上述命令在本地重跑生成。
