#!/usr/bin/env python3
"""Build a supplemental train JSONL from *eval* samples the model got wrong.

Joins `predictions*.jsonl` to rows in `eval.jsonl` via `images[0]` == `image_path`.

**Leakage warning:** using these rows in SFT mixes held-out eval into training. Use only for
internal iteration / error analysis, or regenerate splits so “hard” IDs live in train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from structcheck.eval.evaluate_predictions import extract_completion_reason, norm_label


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _image_key(path_str: str) -> str:
    if not path_str:
        return ""
    try:
        return str(Path(path_str).resolve())
    except Exception:
        return str(path_str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-jsonl", required=True, help="Same eval file used for inference.")
    ap.add_argument("--predictions-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Append each mined row this many times (after the first copy). Default: 2.",
    )
    ap.add_argument(
        "--gold-in",
        default="",
        help="Optional comma list: only mine when gold completion is one of these "
        '(e.g. "Completed,In Progress").',
    )
    args = ap.parse_args()

    gold_filter = {norm_label(x.strip()) for x in args.gold_in.split(",") if x.strip()}
    if gold_filter:
        gold_filter.discard("")

    eval_path = Path(args.eval_jsonl).expanduser().resolve()
    pred_path = Path(args.predictions_jsonl).expanduser().resolve()
    out_path = Path(args.output_jsonl).expanduser().resolve()

    by_img: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(eval_path):
        imgs = row.get("images") or []
        if not imgs:
            continue
        by_img[_image_key(str(imgs[0]))] = row

    preds = load_jsonl(pred_path)
    mined: List[Dict[str, Any]] = []
    for pred in preds:
        tgt = pred.get("target") or {}
        gold = norm_label(tgt.get("completion") or tgt.get("Completion"))
        p_comp, _, _ = extract_completion_reason(pred)
        if not gold or not p_comp:
            continue
        if gold == p_comp:
            continue
        if gold_filter and gold not in gold_filter:
            continue
        key = _image_key(str(pred.get("image_path") or ""))
        base = by_img.get(key)
        if base is None:
            continue
        mined.append(base)
        for _ in range(max(0, int(args.repeats) - 1)):
            mined.append(base)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for row in mined:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "eval_rows_indexed": len(by_img),
                "predictions_scanned": len(preds),
                "supplement_lines_written": len(mined),
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
