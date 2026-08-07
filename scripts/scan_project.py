# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def has_placeholder_logic(text: str) -> bool:
    text_lower = text.lower()
    has_pass_stmt = re.search(r"^\s*pass(\s+#.*)?\s*$", text, flags=re.MULTILINE) is not None
    has_placeholder_hint = (
        "todo" in text_lower or "placeholder" in text_lower or "stub" in text_lower
    )
    return has_pass_stmt and has_placeholder_hint


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan script redundancy and structure quality.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="outputs/project_scan_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scripts_dir = root / "scripts"
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    py_files = sorted([p for p in scripts_dir.glob("*.py") if p.is_file()])

    hash_map: dict[str, list[str]] = {}
    placeholders: list[str] = []
    for path in py_files:
        hash_map.setdefault(file_hash(path), []).append(path.name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        if has_placeholder_logic(text):
            placeholders.append(path.name)

    duplicate_groups = [group for group in hash_map.values() if len(group) > 1]

    lines: list[str] = []
    lines.append("# Project Script Scan Report")
    lines.append("")
    lines.append(f"- script_count: {len(py_files)}")
    lines.append(f"- exact_duplicate_groups: {len(duplicate_groups)}")
    lines.append(f"- placeholder_candidates: {len(placeholders)}")
    lines.append("")

    lines.append("## Script List")
    for path in py_files:
        lines.append(f"- {path.name}")
    lines.append("")

    if duplicate_groups:
        lines.append("## Duplicate Groups")
        for group in duplicate_groups:
            lines.append(f"- {', '.join(sorted(group))}")
        lines.append("")

    if placeholders:
        lines.append("## Placeholder Candidates")
        for name in sorted(placeholders):
            lines.append(f"- {name}")
        lines.append("")

    lines.append("## Auto Suggestions")
    lines.append("- Single entrypoint: route commands through `scripts/main.py`.")
    lines.append("- Shared logic: centralize RN/metrics/RMCLP in `src/core`.")
    lines.append("- Keep CLI aliases for backward compatibility.")
    lines.append("- Preserve data split behavior via `src/spatial_split.py`.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
