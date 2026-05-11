#!/usr/bin/env python3
"""Validate StructCheck JSONL rows (schema, labels, image paths)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ALLOWED = frozenset({"Not Started", "In Progress", "Completed", "Uncertain"})


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: JSON decode error: {e}") from e
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="Dataset JSONL to check.")
    ap.add_argument(
        "--require-image",
        action="store_true",
        help="Require images[0] to exist on disk (same as SFT --require-image).",
    )
    ap.add_argument("--check-reason-min-words", type=int, default=0, help="Warn if reason has fewer words.")
    args = ap.parse_args()

    path = Path(args.jsonl).expanduser().resolve()
    rows = load_jsonl(path)

    n = len(rows)
    missing_target = 0
    bad_completion = 0
    missing_meta = 0
    missing_pcs = 0
    missing_metrics = 0
    short_reason = 0
    no_image = 0
    missing_file = 0

    comp_ctr: Counter[str] = Counter()
    element_ids: Counter[str] = Counter()

    for r in rows:
        tid = str(r.get("element_id") or "").strip()
        if tid:
            element_ids[tid] += 1

        tgt = r.get("target")
        if not isinstance(tgt, dict):
            missing_target += 1
            continue
        c = str(tgt.get("completion") or "").strip()
        if c not in ALLOWED:
            bad_completion += 1
        else:
            comp_ctr[c] += 1

        reason = str(tgt.get("reason") or "").strip()
        if args.check_reason_min_words > 0 and len(reason.split()) < args.check_reason_min_words:
            short_reason += 1

        if not isinstance(r.get("meta"), dict):
            missing_meta += 1

        pcs = r.get("point_cloud_summary")
        if not isinstance(pcs, dict):
            missing_pcs += 1
        else:
            m = pcs.get("metrics")
            if not isinstance(m, dict) or not m:
                missing_metrics += 1

        imgs = r.get("images") or []
        if not imgs or not str(imgs[0]).strip():
            no_image += 1
        elif args.require_image and not Path(str(imgs[0])).is_file():
            missing_file += 1

    report = {
        "path": str(path),
        "rows": n,
        "unique_element_id": len(element_ids),
        "completion_counts": dict(comp_ctr),
        "issues": {
            "missing_or_bad_target": missing_target,
            "invalid_completion_string": bad_completion,
            "missing_meta_object": missing_meta,
            "missing_point_cloud_summary": missing_pcs,
            "point_cloud_summary_without_metrics_dict": missing_metrics,
            "no_images_0": no_image,
            "images_0_missing_on_disk": missing_file if args.require_image else "(skipped)",
            "reason_below_min_words": short_reason if args.check_reason_min_words else "(skipped)",
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
