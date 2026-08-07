"""执行层（论文无关）：把 experiments/configs/*.yaml 展开成作业并执行。

- plan.py    读 YAML → 作业清单（exp × cell × seed × fold），每作业自带完整配置 JSON
- execute.py 串/并行执行 + 断点续跑 + 重试
- train/     Phase 2 从 scripts/ 平移各阶段入口（train_cnn / train_gnn / ...）

训练/推理核心逻辑留在 runner/train/*.py，实验定义归 experiments/，互不渗透。
"""
