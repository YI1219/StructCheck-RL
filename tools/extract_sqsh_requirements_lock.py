#!/usr/bin/env python3
"""Regenerate root `requirements.txt` from conda env `llm` inside structcheck-rl.sqsh.

Reads each *.dist-info/METADATA (Name + Version). Deduplicates by normalized name; if multiple
versions appear, keeps the highest.

**Overwrites** `requirements.txt`: keeps a fixed comment header, then writes all Name==Version pins.
Back up the file first if you have hand-edited pins you need to keep.

Usage:
  python3 tools/extract_sqsh_requirements_lock.py /path/to/structcheck-rl.sqsh
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


# Everything before the first non-comment requirement line in requirements.txt
REQ_HEADER = """# =============================================================================
# StructCheck-RL — Python 依赖（唯一文件，可直接 pip install -r）
# =============================================================================
#
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
#
# 需要 Python 3.10+，Linux + NVIDIA（CUDA 12.8 / cu128 轮子）。数据与模型自备。
#
# -----------------------------------------------------------------------------
# 依赖在代码里的角色（摘要，便于阅读）
# -----------------------------------------------------------------------------
#
# 【深度学习】torch, torchvision, triton, nvidia-* — PyTorch 与 CUDA 运行时
# 【模型与训练】transformers, peft, accelerate, trl, tokenizers, safetensors
# 【多模态 / 加速】xformers, bitsandbytes, unsloth, unsloth_zoo, diffusers
# 【数据与工具】datasets, pandas, pyarrow, pillow, PyYAML, tqdm, requests, …
#
# 下列为全量 pin（与参考集群 conda env `llm` 一致）。维护者可用 sqsh 重算：
#   python3 tools/extract_sqsh_requirements_lock.py /path/to/structcheck-rl.sqsh
# 若实验中发现 pip freeze 多出包，可 diff 后把行追加进本文件再提交。
#
# =============================================================================

"""


def _parse_version_tuple(v: str) -> tuple:
    parts = re.split(r"(\d+)", v)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        elif p:
            key.append(p)
    return tuple(key)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(
            "usage: extract_sqsh_requirements_lock.py /path/to/structcheck-rl.sqsh\n"
            "Overwrites requirements.txt in the repo root."
        )
    sqsh = Path(sys.argv[1])
    if not sqsh.is_file():
        sys.exit(f"missing sqsh: {sqsh}")

    listing = subprocess.run(
        ["unsquashfs", "-l", str(sqsh)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    rel_paths: set[str] = set()
    for line in listing.splitlines():
        line = line.strip()
        if "opt/conda/envs/llm/lib/python3.10/site-packages/" not in line:
            continue
        if not line.endswith(".dist-info"):
            continue
        if "squashfs-root/" in line:
            line = line.split("squashfs-root/", 1)[1]
        rel_paths.add(line)

    by_name: dict[str, str] = {}
    for rel in sorted(rel_paths):
        key = f"{rel}/METADATA"
        proc = subprocess.run(
            ["sqfscat", str(sqsh), key],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"# MISSING METADATA: {key}", file=sys.stderr)
            continue
        name = version = None
        for ln in proc.stdout.splitlines():
            if ln.startswith("Name: "):
                name = ln[6:].strip()
            elif ln.startswith("Version: "):
                version = ln[9:].strip()
            if name and version:
                break
        if not name or not version:
            continue
        prev = by_name.get(name)
        if prev is None or _parse_version_tuple(version) > _parse_version_tuple(prev):
            by_name[name] = version

    pin_lines = [f"{n}=={by_name[n]}\n" for n in sorted(by_name)]
    out = Path(__file__).resolve().parent.parent / "requirements.txt"
    out.write_text(REQ_HEADER + "".join(pin_lines), encoding="utf-8")
    print(f"Wrote {len(pin_lines)} packages to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
