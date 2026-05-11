#!/usr/bin/env python3
"""Print requirement lines present in `pip freeze` output but absent from `requirements.txt`.

Normalizes distribution names (case, underscore vs hyphen) for comparison.

Usage:
  python3 tools/diff_freeze_vs_lock.py pip_freeze_llm_end.txt requirements.txt
"""

from __future__ import annotations

import sys
from pathlib import Path


def norm_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def parse_eq_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("-r "):
        return None
    if line.startswith("-e ") or " @ " in line:
        return ("__url__", line)
    if "==" not in line:
        return None
    name, ver = line.split("==", 1)
    name, ver = name.strip(), ver.strip()
    if not name:
        return None
    return (name, ver)


def load_lock_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        p = parse_eq_line(raw)
        if not p:
            continue
        if p[0] == "__url__":
            continue
        names.add(norm_name(p[0]))
    return names


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: diff_freeze_vs_lock.py <pip-freeze.txt> <requirements.txt>")
    freeze_p = Path(sys.argv[1])
    lock_p = Path(sys.argv[2])
    lock_names = load_lock_names(lock_p)
    extras: list[str] = []
    for raw in freeze_p.read_text(encoding="utf-8").splitlines():
        p = parse_eq_line(raw)
        if not p:
            continue
        if p[0] == "__url__":
            extras.append(p[1])
            continue
        name, ver = p
        if norm_name(name) not in lock_names:
            extras.append(f"{name}=={ver}")
    for line in sorted(set(extras)):
        print(line)


if __name__ == "__main__":
    main()
