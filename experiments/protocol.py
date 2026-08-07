"""评估协议稳健性（论文 §4.5）：换推理方式，结论是否保持。

写作口径：协议实验的价值是稳健性展示，不是"选最好的协议"（选择偏置陷阱）。
先修 transductive/inductive 评测口径（历史缺陷 P2：二者都用诱导子图评测），
再谈各协议下方法排序是否保持。
"""
from __future__ import annotations

from typing import Any

from experiments.config import load_family


def plan() -> dict[str, dict[str, Any]]:
    """返回该族全部实验的合并配置：{proto_trans_ind_buffer}。"""
    return load_family("protocol")


def modes() -> list[str]:
    """三种推理模式：transductive / inductive / buffered。"""
    return plan()["proto_trans_ind_buffer"]["cells"]["mode"]


if __name__ == "__main__":
    from pprint import pprint

    for name, cfg in plan().items():
        print(f"\n=== {name} (legacy {cfg['legacy_id']}, bucket={cfg['bucket']}) ===")
        pprint(cfg["cells"])
