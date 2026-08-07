"""Phase 2 平移目标：各训练/推理阶段入口。

迁移路线（来自 docs/代码重构与分区计划.md §2，每个 ← 对应一个现存文件）：
- train_cnn.py      ← run_train.py（加验证集模型选择 P5）
- train_gnn.py      ← run_train_gnn.py（协议口径修复 P2 + --self_loop_only）
- train_ssl.py      ← run_train_ssl.py
- extract_features.py ← extract_features.py
- infer.py / infer_gnn_all.py ← run_infer.py / run_infer_gnn_all.py
- baselines.py      ← run_external_baselines.py

尚未迁移（Phase 2 未启动），本目录只占位。现有入口仍走 scripts/main.py 转发。
"""
