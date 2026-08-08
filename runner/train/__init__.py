"""runner/train：训练/推理/基线下游各阶段入口。

迁移路线（来自 docs/代码重构与分区计划.md §2，每个 ← 对应一个旧脚本）：
- train_cnn.py      ← run_train.py
- train_gnn.py      ← run_train_gnn.py
- train_ssl.py      ← run_train_ssl.py
- extract_features.py ← extract_features.py
- infer.py          ← run_infer.py
- infer_gnn_all.py  ← run_infer_gnn_all.py
- baselines.py      ← run_external_baselines.py
- rmclp.py          ← run_rmclp.py + run_rmclp_curve.py（合并，子命令 single|curve）

全部以 `python -m runner.train.*` 方式调用（Phase E 删除 scripts/main.py 后成为唯一入口）。
"""
