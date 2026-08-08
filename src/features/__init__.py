# -*- coding: utf-8 -*-
"""数据层：原始栅格/矢量 → 特征（← 顶层 process.py + scripts/build_rain_demand.py）。

- process.py: 多尺度切片、空间对齐、特征提取（patch .npz + patches_info.csv）
- build_rain_demand.py: 降雨需求映射到每个 patch（demand_per_patch.csv）
"""
