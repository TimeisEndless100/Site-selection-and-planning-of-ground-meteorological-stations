# -*- coding: utf-8 -*-
"""论文流程示意图（← scripts/generate_paper_schematics.py）。

应用链重定向（2026-08-08）：fig1/fig6/fig7 的"选址/布局"框架改为
"概率地图 vs 地形地图对比"叙事（RMCLP 决策链已撤销，应用展示 = 概率重聚合
为地图并用物理约束校验），fig6/fig7 文件名相应更名。

用法：
    python -m reporting.schematics [--figure fig1_overall_framework_ai …]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# 相对 repo 根：reporting/schematics.py -> parents[1] = repo 根
OUT_DIR = ROOT_DIR / "docs" / "figures_generated"

COLORS = {
    "navy": "#163A5F",
    "blue": "#2F6B99",
    "teal": "#2A9D8F",
    "orange": "#E07A5F",
    "yellow": "#E9C46A",
    "graphite": "#4A4A4A",
    "light": "#D9E1E8",
    "bg": "#F7F7F4",
    "grid": "#E7ECEF",
    "muted": "#B8C4CF",
    "soft_teal": "#D8EEE9",
    "soft_orange": "#F8DDD4",
    "soft_blue": "#DCE8F2",
    "soft_yellow": "#F8EEC5",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "savefig.facecolor": COLORS["bg"],
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "axes.edgecolor": COLORS["graphite"],
            "text.color": COLORS["graphite"],
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def rounded_box(ax, x, y, w, h, title, body, fc, ec=None, title_color=None) -> None:
    ec = ec or COLORS["blue"]
    title_color = title_color or COLORS["navy"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + 0.018, y + h - 0.05, title, fontsize=13, fontweight="bold", color=title_color)
    ax.text(x + 0.018, y + h - 0.095, body, va="top", fontsize=10.2, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, mutation=16) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color or COLORS["blue"],
            shrinkA=6,
            shrinkB=6,
        )
    )


def draw_grid_panel(ax, x, y, w, h, title=None, show_nodes=False, block_split=None, buffer=False):
    rows, cols = 5, 7
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            fc = "white"
            if block_split is not None:
                split_col = block_split
                if c < split_col:
                    fc = COLORS["soft_blue"]
                else:
                    fc = "#F4F7FA"
                if buffer and c in (split_col - 1, split_col):
                    fc = COLORS["soft_yellow"]
            rect = Rectangle(
                (x + c * cell_w, y + r * cell_h),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor=COLORS["grid"],
                linewidth=0.9,
            )
            ax.add_patch(rect)
    if show_nodes:
        centers = []
        for r in range(rows):
            for c in range(cols):
                cx = x + (c + 0.5) * cell_w
                cy = y + (r + 0.5) * cell_h
                centers.append((cx, cy))
                ax.add_patch(Circle((cx, cy), min(cell_w, cell_h) * 0.11, facecolor=COLORS["blue"], edgecolor="white", lw=0.5))
        for i, (cx, cy) in enumerate(centers):
            if (i + 1) % cols != 0:
                nx, ny = centers[i + 1]
                ax.plot([cx, nx], [cy, ny], color=COLORS["muted"], lw=0.8, zorder=0)
            if i + cols < len(centers):
                nx, ny = centers[i + cols]
                ax.plot([cx, nx], [cy, ny], color=COLORS["muted"], lw=0.8, zorder=0)
    if title:
        ax.text(x, y + h + 0.02, title, fontsize=11.5, fontweight="bold", color=COLORS["navy"])


def figure1_overall_framework() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.94, "Overall Framework of Physics-aware PU Learning and Spatial Refinement", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(0.04, 0.905, "Two-stage suitability modeling with reliable negatives, graph propagation, and probability-map vs terrain comparison", fontsize=11.5, color=COLORS["graphite"])

    rounded_box(
        ax,
        0.04,
        0.58,
        0.13,
        0.24,
        "Input Space",
        "Gridded terrain cells\nEnvironmental patches\nObserved stations",
        COLORS["soft_blue"],
    )
    gx, gy, gw, gh = 0.055, 0.62, 0.10, 0.13
    draw_grid_panel(ax, gx, gy, gw, gh)
    ax.add_patch(Circle((gx + 0.022, gy + 0.09), 0.009, facecolor=COLORS["teal"], edgecolor="white", lw=0.8))
    ax.add_patch(Circle((gx + 0.062, gy + 0.045), 0.009, facecolor=COLORS["orange"], edgecolor="white", lw=0.8))
    ax.add_patch(Circle((gx + 0.088, gy + 0.09), 0.009, facecolor=COLORS["muted"], edgecolor="white", lw=0.8))
    ax.text(0.055, 0.595, "P", color=COLORS["teal"], fontweight="bold")
    ax.text(0.075, 0.595, "RN", color=COLORS["orange"], fontweight="bold")
    ax.text(0.112, 0.595, "U", color=COLORS["graphite"], fontweight="bold")

    rounded_box(
        ax,
        0.20,
        0.58,
        0.15,
        0.24,
        "Physics-aware RN",
        "Slope threshold\nWater-ratio filter\nTerrain infeasibility\nRN <- physics rules",
        COLORS["soft_orange"],
        ec=COLORS["orange"],
    )
    ax.plot([0.225, 0.26], [0.655, 0.725], color=COLORS["orange"], lw=2.0)
    ax.plot([0.235, 0.285], [0.655, 0.655], color=COLORS["orange"], lw=2.0)
    ax.text(0.285, 0.72, "slope", fontsize=9.5)
    water = Circle((0.28, 0.625), 0.018, facecolor=COLORS["soft_blue"], edgecolor=COLORS["blue"], lw=1.2)
    ax.add_patch(water)
    ax.text(0.272, 0.62, "≈", fontsize=12, color=COLORS["blue"])

    rounded_box(
        ax,
        0.38,
        0.54,
        0.20,
        0.30,
        "Stage 1: PURN-GeoConvNeXt",
        "Local patch encoder\nPositive-unlabeled risk\nReliable-negative penalty\nL = L_PU + λ_RN L_RN",
        COLORS["soft_teal"],
        ec=COLORS["teal"],
        title_color=COLORS["teal"],
    )
    for i in range(3):
        rect = Rectangle((0.405 + i * 0.018, 0.64 - i * 0.015), 0.06, 0.06, facecolor="white", edgecolor=COLORS["blue"], lw=1.1)
        ax.add_patch(rect)
    ax.text(0.49, 0.66, "patch stack", fontsize=10, color=COLORS["graphite"])
    ax.add_patch(FancyBboxPatch((0.47, 0.60), 0.085, 0.07, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#FFFFFF", edgecolor=COLORS["teal"], linewidth=1.4))
    ax.text(0.482, 0.635, "GeoConvNeXt", fontsize=10.5, color=COLORS["teal"], fontweight="bold")
    ax.text(0.40, 0.565, "Initial suitability score", fontsize=10, color=COLORS["navy"])

    rounded_box(
        ax,
        0.61,
        0.54,
        0.18,
        0.30,
        "Stage 2: Spatial Graph Refinement",
        "Node construction\nNeighborhood edges\nGraphSAGE aggregation\nh' = AGG(h, N(v))",
        "#E6F0F7",
    )
    draw_grid_panel(ax, 0.635, 0.61, 0.11, 0.12, show_nodes=True)
    ax.text(0.635, 0.585, "Context propagation across nearby cells", fontsize=9.7, color=COLORS["graphite"])

    rounded_box(
        ax,
        0.82,
        0.58,
        0.14,
        0.24,
        "Probability Map",
        "Re-aggregate cells\nSpatial probability surface\nP(cell) over the domain",
        COLORS["soft_yellow"],
        ec=COLORS["yellow"],
        title_color="#9A6A00",
    )
    draw_grid_panel(ax, 0.845, 0.62, 0.09, 0.10)
    for i in range(5):
        ax.add_patch(Circle((0.855 + i * 0.012, 0.65 + 0.02 * math.sin(i)), 0.006, facecolor=COLORS["yellow"], edgecolor="white", lw=0.5))

    rounded_box(
        ax,
        0.69,
        0.16,
        0.27,
        0.24,
        "Terrain Map Comparison",
        "Slope / water / DEM\nOverlay + mask infeasible cells\nSpearman ρ, RN agreement",
        "#F5F8FB",
    )
    draw_grid_panel(ax, 0.715, 0.205, 0.13, 0.11)
    for pt, color in [((0.735, 0.255), COLORS["teal"]), ((0.775, 0.235), COLORS["yellow"]), ((0.82, 0.275), COLORS["orange"])]:
        ax.add_patch(Circle(pt, 0.008, facecolor=color, edgecolor="white", lw=0.8))
    ax.text(0.85, 0.285, "feasible", color=COLORS["teal"], fontsize=9.5)
    ax.text(0.85, 0.255, "probability", color=COLORS["yellow"], fontsize=9.5)
    ax.text(0.85, 0.225, "infeasible", color=COLORS["orange"], fontsize=9.5)

    arrow(ax, 0.17, 0.70, 0.20, 0.70)
    arrow(ax, 0.35, 0.69, 0.38, 0.69)
    arrow(ax, 0.58, 0.69, 0.61, 0.69)
    arrow(ax, 0.79, 0.69, 0.82, 0.69)
    arrow(ax, 0.89, 0.58, 0.82, 0.40)

    ax.text(0.42, 0.46, "Stage I: local suitability inference under incomplete labels", fontsize=10.2, color=COLORS["teal"], style="italic")
    ax.text(0.62, 0.46, "Stage II: graph-based spatial refinement", fontsize=10.2, color=COLORS["blue"], style="italic")

    save_figure(fig, "fig1_overall_framework_ai")


def figure2_pu_definition() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Positive-Unlabeled Task Definition with Physics-derived Reliable Negatives", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(0.04, 0.895, "Observed stations are incomplete positives; reliable negatives are derived from physical exclusion rules; the remaining space stays unlabeled", fontsize=11.3)

    rounded_box(ax, 0.05, 0.20, 0.34, 0.60, "Spatial Universe", "All terrain cells in the study area", "#F5F8FB")
    draw_grid_panel(ax, 0.09, 0.28, 0.26, 0.42)
    points = [
        (0.12, 0.60, COLORS["teal"]), (0.18, 0.48, COLORS["teal"]), (0.30, 0.64, COLORS["teal"]),
        (0.15, 0.36, COLORS["orange"]), (0.27, 0.50, COLORS["orange"]), (0.32, 0.40, COLORS["orange"]),
        (0.22, 0.61, COLORS["muted"]), (0.25, 0.32, COLORS["muted"]), (0.10, 0.46, COLORS["muted"])
    ]
    for x, y, c in points:
        ax.add_patch(Circle((x, y), 0.012, facecolor=c, edgecolor="white", lw=0.9))

    rounded_box(ax, 0.46, 0.53, 0.20, 0.25, "Observed Positives (P)", "Existing station cells\nDirect supervision signal", COLORS["soft_teal"], ec=COLORS["teal"], title_color=COLORS["teal"])
    rounded_box(ax, 0.46, 0.22, 0.20, 0.25, "Reliable Negatives (RN)", "Steep slopes\nHigh water ratio\nPhysically implausible siting", COLORS["soft_orange"], ec=COLORS["orange"], title_color=COLORS["orange"])
    rounded_box(ax, 0.72, 0.36, 0.22, 0.30, "Unlabeled Cells (U)", "Large unresolved search space\nMay include both suitable and unsuitable cells", "#EEF3F7")

    ax.text(0.53, 0.64, "P", fontsize=22, fontweight="bold", color=COLORS["teal"])
    ax.text(0.53, 0.33, "RN", fontsize=20, fontweight="bold", color=COLORS["orange"])
    ax.text(0.81, 0.49, "U", fontsize=22, fontweight="bold", color=COLORS["blue"])
    ax.text(0.79, 0.40, "Unknown label state", fontsize=10.5, color=COLORS["graphite"])

    arrow(ax, 0.39, 0.60, 0.46, 0.64, color=COLORS["teal"])
    arrow(ax, 0.39, 0.37, 0.46, 0.35, color=COLORS["orange"])
    arrow(ax, 0.39, 0.50, 0.72, 0.50, color=COLORS["blue"])

    ax.text(0.46, 0.12, "Learning objective: rank unlabeled cells using positive anchors and physics-derived reliable negatives", fontsize=11, color=COLORS["navy"])

    save_figure(fig, "fig2_pu_problem_definition_ai")


def figure3_protocols() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Spatial Validation Protocols for Graph-based Suitability Modeling", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(0.04, 0.895, "The same spatial partition can be evaluated under increasingly strict graph access assumptions", fontsize=11.3)

    panels = [
        (0.05, "Transductive", "Train and validation nodes share the full graph.\nOnly validation labels remain hidden.", False),
        (0.37, "Inductive", "Validation subgraph is held out from message passing.\nNo edges from validation nodes during training.", False),
        (0.69, "Buffered", "A buffer zone is removed between train and validation regions\nto reduce boundary interaction.", True),
    ]

    for x, title, body, is_buffer in panels:
        rounded_box(ax, x, 0.22, 0.26, 0.56, title, body, "#F5F8FB")
        if title == "Transductive":
            draw_grid_panel(ax, x + 0.03, 0.39, 0.20, 0.22, title=None, show_nodes=True, block_split=4, buffer=False)
            ax.text(x + 0.03, 0.34, "Full graph available during refinement", fontsize=9.7, color=COLORS["graphite"])
        elif title == "Inductive":
            draw_grid_panel(ax, x + 0.03, 0.39, 0.20, 0.22, title=None, show_nodes=False, block_split=4, buffer=False)
            draw_grid_panel(ax, x + 0.03, 0.39, 0.11, 0.22, title=None, show_nodes=True)
            draw_grid_panel(ax, x + 0.16, 0.39, 0.07, 0.22, title=None, show_nodes=True)
            ax.plot([x + 0.145, x + 0.145], [0.39, 0.61], color=COLORS["orange"], lw=2.2, linestyle=(0, (4, 3)))
            ax.text(x + 0.03, 0.34, "Graph is split across train/validation regions", fontsize=9.7, color=COLORS["graphite"])
        else:
            draw_grid_panel(ax, x + 0.03, 0.39, 0.20, 0.22, title=None, show_nodes=True, block_split=4, buffer=is_buffer)
            ax.text(x + 0.03, 0.34, "Yellow cells indicate removed buffer region", fontsize=9.7, color=COLORS["graphite"])

        ax.add_patch(Rectangle((x + 0.03, 0.65), 0.025, 0.018, facecolor=COLORS["soft_blue"], edgecolor="none"))
        ax.text(x + 0.06, 0.659, "Train region", fontsize=9.5)
        ax.add_patch(Rectangle((x + 0.13, 0.65), 0.025, 0.018, facecolor="#F4F7FA", edgecolor="none"))
        ax.text(x + 0.16, 0.659, "Validation region", fontsize=9.5)
        if is_buffer:
            ax.add_patch(Rectangle((x + 0.03, 0.62), 0.025, 0.018, facecolor=COLORS["soft_yellow"], edgecolor="none"))
            ax.text(x + 0.06, 0.629, "Buffer", fontsize=9.5)

    save_figure(fig, "fig3_spatial_protocols_ai")


def figure5_node_mlp_vs_graph() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Node-only MLP versus GraphSAGE under Spatial Protocols", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(0.04, 0.895, "NodeMLP isolates feature transformation only, whereas GraphSAGE adds neighborhood aggregation and spatial context propagation", fontsize=11.3)

    rounded_box(ax, 0.05, 0.30, 0.26, 0.48, "NodeMLP", "Input node features\nNo graph edges in the forward pass\nIndependent node classification", COLORS["soft_blue"])
    for i in range(4):
        y = 0.63 - i * 0.07
        ax.add_patch(Circle((0.11, y), 0.012, facecolor=COLORS["blue"], edgecolor="white", lw=0.8))
        ax.add_patch(Circle((0.18, y), 0.012, facecolor=COLORS["blue"], edgecolor="white", lw=0.8))
        ax.add_patch(Circle((0.25, y), 0.012, facecolor=COLORS["blue"], edgecolor="white", lw=0.8))
        ax.plot([0.11, 0.18], [y, y], color=COLORS["light"], lw=1.1)
        ax.plot([0.18, 0.25], [y, y], color=COLORS["light"], lw=1.1)
    ax.text(0.085, 0.35, "Feature-only transformation", fontsize=10.2)

    rounded_box(ax, 0.37, 0.30, 0.26, 0.48, "GraphSAGE", "Input node features\nNeighborhood graph\nMessage passing and aggregation", COLORS["soft_teal"], ec=COLORS["teal"], title_color=COLORS["teal"])
    graph_pts = [(0.43, 0.61), (0.50, 0.66), (0.56, 0.58), (0.48, 0.51), (0.58, 0.47), (0.41, 0.44)]
    edges = [(0, 1), (0, 3), (1, 2), (2, 4), (3, 4), (3, 5)]
    for i, j in edges:
        x1, y1 = graph_pts[i]
        x2, y2 = graph_pts[j]
        ax.plot([x1, x2], [y1, y2], color=COLORS["muted"], lw=1.3)
    for x, y in graph_pts:
        ax.add_patch(Circle((x, y), 0.014, facecolor=COLORS["teal"], edgecolor="white", lw=0.8))
    for x, y in graph_pts[:3]:
        ax.add_patch(Circle((x, y), 0.032, facecolor="none", edgecolor=COLORS["yellow"], lw=1.1, linestyle=(0, (3, 2))))
    ax.text(0.425, 0.35, "Neighborhood-aware refinement", fontsize=10.2)

    rounded_box(ax, 0.69, 0.20, 0.26, 0.58, "Protocol Awareness", "Graph models should be interpreted under clear spatial access assumptions", "#F5F8FB")
    for idx, (name, desc, y, fc) in enumerate(
        [
            ("Transductive", "Validation nodes remain in the graph", 0.62, COLORS["soft_blue"]),
            ("Inductive", "Validation subgraph is disconnected", 0.49, "#EEF3F7"),
            ("Buffered", "Boundary cells are removed as a buffer", 0.36, COLORS["soft_yellow"]),
        ]
    ):
        panel = FancyBboxPatch((0.72, y), 0.20, 0.09, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=fc, edgecolor=COLORS["light"], linewidth=1.0)
        ax.add_patch(panel)
        ax.text(0.735, y + 0.055, name, fontsize=11, fontweight="bold", color=COLORS["navy"])
        ax.text(0.735, y + 0.022, desc, fontsize=9.3, color=COLORS["graphite"])

    arrow(ax, 0.31, 0.54, 0.37, 0.54)
    arrow(ax, 0.63, 0.54, 0.69, 0.54)

    save_figure(fig, "fig5_node_mlp_graph_protocols")


def figure6_prob_map_workflow() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Application Workflow: Probability Map versus Terrain Map", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(0.04, 0.895, "Re-aggregate predicted probabilities into a spatial surface and validate it against physical terrain constraints", fontsize=11.3)

    blocks = [
        (0.04, "Suitability Surface", "Per-cell predicted probability\nfrom the final model", COLORS["soft_blue"]),
        (0.25, "Re-aggregate to Map", "Pivot per-cell scores\nonto a spatial probability grid", COLORS["soft_teal"]),
        (0.46, "Terrain Constraints", "Slope / water / DEM rules\nphysics-based infeasibility", COLORS["soft_orange"]),
        (0.67, "Compare & Validate", "Overlay + mask infeasible cells\nSpearman ρ, RN agreement, top-10% rate", COLORS["soft_yellow"]),
    ]
    for x, title, body, fc in blocks:
        rounded_box(ax, x, 0.58, 0.17, 0.22, title, body, fc)

    draw_grid_panel(ax, 0.055, 0.61, 0.12, 0.12)
    for i in range(7):
        ax.add_patch(Circle((0.07 + i * 0.014, 0.64 + 0.025 * math.sin(i)), 0.006, facecolor=COLORS["yellow"], edgecolor="white", lw=0.5))
    ax.text(0.07, 0.595, "per-cell predicted scores", fontsize=9.2)

    # 重聚合：网格分箱 → 平滑概率面
    draw_grid_panel(ax, 0.29, 0.61, 0.09, 0.10)
    for i in range(5):
        ax.add_patch(Circle((0.30 + i * 0.013, 0.645 + 0.018 * math.sin(i)), 0.006, facecolor=COLORS["teal"], edgecolor="white", lw=0.5))
    ax.text(0.29, 0.595, "binned probability surface", fontsize=9.2)

    # 地形约束：坡度 / 水域 规则
    slope = Polygon([(0.50, 0.66), (0.565, 0.66), (0.535, 0.61)], facecolor=COLORS["orange"], edgecolor="white", lw=0.8)
    ax.add_patch(slope)
    water = Circle((0.53, 0.64), 0.016, facecolor=COLORS["soft_blue"], edgecolor=COLORS["blue"], lw=1.0)
    ax.add_patch(water)
    ax.text(0.475, 0.66, "slope", fontsize=9.5, color=COLORS["orange"])
    ax.text(0.55, 0.66, "water", fontsize=9.5, color=COLORS["blue"])

    # 对比与校验：概率面 + 不可行掩膜
    ax.add_patch(Rectangle((0.70, 0.615), 0.11, 0.10, facecolor="white", edgecolor=COLORS["yellow"], linewidth=1.3))
    for i in range(3):
        ax.add_patch(Circle((0.72 + i * 0.03, 0.655 + 0.02 * math.sin(i)), 0.008, facecolor=COLORS["orange"], edgecolor="white", lw=0.7))
    ax.text(0.705, 0.603, "mask infeasible (RN)", fontsize=9.2, color="#9A6A00")

    rounded_box(ax, 0.29, 0.18, 0.42, 0.22, "Agreement Summary", "Correlation between probability and terrain features, separation between RN and non-RN cells, and the feasible fraction of high-probability cells together validate the learned suitability surface.", "#F5F8FB")
    draw_grid_panel(ax, 0.33, 0.225, 0.22, 0.10)
    for pt, color in [((0.355, 0.275), COLORS["teal"]), ((0.420, 0.255), COLORS["yellow"]), ((0.495, 0.285), COLORS["orange"])]:
        ax.add_patch(Circle(pt, 0.008, facecolor=color, edgecolor="white", lw=0.8))
    ax.text(0.57, 0.285, "ρ < 0 (steeper → lower)", color=COLORS["teal"], fontsize=9.5)
    ax.text(0.57, 0.255, "RN separated", color=COLORS["yellow"], fontsize=9.5)
    ax.text(0.57, 0.225, "top-10% feasible", color=COLORS["orange"], fontsize=9.5)

    arrow(ax, 0.21, 0.69, 0.25, 0.69)
    arrow(ax, 0.42, 0.69, 0.46, 0.69)
    arrow(ax, 0.63, 0.69, 0.67, 0.69)
    arrow(ax, 0.76, 0.58, 0.65, 0.40)
    arrow(ax, 0.35, 0.40, 0.35, 0.58)

    save_figure(fig, "fig6_prob_map_terrain_workflow_ai")


def mini_map(ax, x, y, w, h, title, mode):
    """示意性地图面板：概率面 / 坡度 / 水域 / 掩膜对比（fig7 用，示意非数据图）。"""
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#F5F8FB", edgecolor=COLORS["light"], linewidth=1.0)
    ax.add_patch(panel)
    ax.text(x + 0.015, y + h - 0.04, title, fontsize=11, fontweight="bold", color=COLORS["navy"])
    draw_grid_panel(ax, x + 0.02, y + 0.05, w - 0.04, h - 0.12)
    if mode == "prob":  # 高概率点（黄）散布
        for i in range(4):
            ax.add_patch(Circle((x + 0.06 + i * 0.05, y + 0.16 + 0.03 * ((i * 7) % 3)), 0.008, facecolor=COLORS["yellow"], edgecolor="white", lw=0.7))
    elif mode == "slope":  # 陡坡三角（橙）
        for i in range(3):
            ax.add_patch(Polygon(
                [(x + 0.06 + i * 0.07, y + 0.24), (x + 0.075 + i * 0.07, y + 0.24), (x + 0.0675 + i * 0.07, y + 0.16)],
                facecolor=COLORS["orange"], edgecolor="white", lw=0.6))
    elif mode == "water":  # 水域圆（蓝）
        for i in range(3):
            ax.add_patch(Circle((x + 0.08 + i * 0.06, y + 0.17), 0.013, facecolor=COLORS["soft_blue"], edgecolor=COLORS["blue"], lw=0.9))
    else:  # mask：概率 + 掩膜区域
        for i in range(3):
            ax.add_patch(Circle((x + 0.06 + i * 0.05, y + 0.17 + 0.02 * ((i * 5) % 3)), 0.008, facecolor=COLORS["yellow"], edgecolor="white", lw=0.7))
        ax.add_patch(Rectangle((x + 0.16, y + 0.07), 0.08, 0.07, facecolor=COLORS["orange"], edgecolor="none", alpha=0.35))


def figure7_terrain_prob_comparison() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.93, "Probability Map versus Terrain Maps (Illustrative)", fontsize=18, fontweight="bold", color=COLORS["navy"])
    ax.text(0.04, 0.895, "The learned suitability surface re-aggregated to a spatial map, compared against physical terrain constraints (slope, water) and their joint mask", fontsize=11.3)

    mini_map(ax, 0.05, 0.52, 0.27, 0.28, "Suitability probability map", "prob")
    mini_map(ax, 0.37, 0.52, 0.27, 0.28, "Slope (max) terrain map", "slope")
    mini_map(ax, 0.69, 0.52, 0.27, 0.28, "Water ratio terrain map", "water")
    mini_map(ax, 0.21, 0.15, 0.27, 0.28, "Probability under terrain mask", "mask")

    legend = FancyBboxPatch((0.56, 0.16), 0.30, 0.20, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#F5F8FB", edgecolor=COLORS["light"], linewidth=1.0)
    ax.add_patch(legend)
    ax.text(0.58, 0.32, "Legend", fontsize=11.5, fontweight="bold", color=COLORS["navy"])
    for idx, (label, color) in enumerate(
        [("high probability / feasible", COLORS["teal"]), ("moderate / steep / wet", COLORS["yellow"]), ("masked infeasible (RN)", COLORS["orange"])]
    ):
        yy = 0.28 - idx * 0.05
        ax.add_patch(Circle((0.60, yy), 0.008, facecolor=color, edgecolor="white", lw=0.7))
        ax.text(0.62, yy - 0.006, label, fontsize=9.8)

    arrow(ax, 0.49, 0.52, 0.35, 0.43)
    arrow(ax, 0.82, 0.52, 0.44, 0.43)
    arrow(ax, 0.19, 0.52, 0.29, 0.43)

    save_figure(fig, "fig7_terrain_prob_comparison")


# 全部示意图（输出文件 stem 见各函数内部 save_figure）
_ALL_FIGURES = [
    figure1_overall_framework,
    figure2_pu_definition,
    figure3_protocols,
    figure5_node_mlp_vs_graph,
    figure6_prob_map_workflow,
    figure7_terrain_prob_comparison,
]
# 外部别名（图文件 stem）→ 绘制函数
_FIGURE_ALIASES = {
    "fig1_overall_framework_ai": figure1_overall_framework,
    "fig2_pu_problem_definition_ai": figure2_pu_definition,
    "fig3_spatial_protocols_ai": figure3_protocols,
    "fig5_node_mlp_graph_protocols": figure5_node_mlp_vs_graph,
    "fig6_prob_map_terrain_workflow_ai": figure6_prob_map_workflow,
    "fig7_terrain_prob_comparison": figure7_terrain_prob_comparison,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="论文流程示意图（默认全部生成）")
    parser.add_argument(
        "--figure",
        nargs="*",
        help="只生成指定图：" + " ".join(sorted(_FIGURE_ALIASES)),
    )
    args = parser.parse_args()

    configure_style()
    if args.figure:
        funcs = []
        for name in args.figure:
            fn = _FIGURE_ALIASES.get(name)
            if fn is None:
                raise SystemExit(f"Unknown figure: {name}（可选：{' '.join(sorted(_FIGURE_ALIASES))}）")
            funcs.append(fn)
    else:
        funcs = list(_ALL_FIGURES)
    for fn in funcs:
        fn()
    print(f"Generated {len(funcs)} figure(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
