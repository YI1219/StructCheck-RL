#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_label(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        x = json.dumps(x, ensure_ascii=False)
    s = str(x).strip()
    if not s:
        return ""
    low = s.lower().strip()
    mapping = {
        "not started": "Not Started",
        "not start": "Not Started",
        "unstarted": "Not Started",
        "in progress": "In Progress",
        "in-progress": "In Progress",
        "ongoing": "In Progress",
        "completed": "Completed",
        "complete": "Completed",
        "done": "Completed",
        "uncertain": "Uncertain",
        "unknown": "Uncertain",
        "unsure": "Uncertain",
        "incomplete": "Not Started",
        "...": "",
    }
    return mapping.get(low, s)


def extract_completion_reason(pred_row: Dict[str, Any]) -> Tuple[str, str, bool]:
    pj = pred_row.get("prediction_json")
    if isinstance(pj, dict):
        completion = norm_label(pj.get("completion") or pj.get("Completion"))
        raw_r = pj.get("reason") or pj.get("Reason")
        if isinstance(raw_r, dict):
            reason = norm_label(json.dumps(raw_r, ensure_ascii=False))[:2000]
        else:
            reason = norm_label(raw_r)
        return completion, reason, True

    text = str(pred_row.get("prediction_text") or "")
    m_json = re.search(r"\"completion\"\s*:\s*\"([^\"]+)\"", text, flags=re.IGNORECASE)
    if m_json:
        completion = norm_label(m_json.group(1))
    else:
        m_plain = re.search(r"\bcompletion\b\s*[:=]\s*([A-Za-z][A-Za-z ]+)", text, flags=re.IGNORECASE)
        completion = norm_label(m_plain.group(1)) if m_plain else ""

    mr = re.search(r"\"reason\"\s*:\s*\"([^\"]+)\"", text, flags=re.IGNORECASE)
    reason = norm_label(mr.group(1)) if mr else ""
    return completion, reason, False


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def per_class_prf(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(y_true) | set(y_pred))
    out: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": float(sum(1 for t in y_true if t == lab))}
    return out


def confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    cm: Dict[str, Dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in cm:
            cm[t] = {pp: 0 for pp in labels}
        if p not in cm[t]:
            cm[t][p] = 0
        cm[t][p] += 1
    return cm


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate StructCheck predictions jsonl.")
    ap.add_argument("--predictions-jsonl", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.predictions_jsonl))
    y_true: List[str] = []
    y_pred: List[str] = []

    json_ok = 0
    reason_nonempty = 0
    completion_nonempty = 0

    for r in rows:
        tgt = r.get("target") or {}
        t_completion = norm_label(tgt.get("completion") or tgt.get("Completion"))
        p_completion, p_reason, ok = extract_completion_reason(r)
        if ok:
            json_ok += 1
        if p_completion:
            completion_nonempty += 1
        if p_reason:
            reason_nonempty += 1
        if t_completion:
            y_true.append(t_completion)
            y_pred.append(p_completion)

    total = len(rows)
    labeled = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / labeled if labeled else 0.0
    mf1 = macro_f1(y_true, y_pred) if labeled else 0.0

    out = {
        "total_samples": total,
        "labeled_samples_for_completion": labeled,
        "completion_accuracy": acc,
        "completion_macro_f1": mf1,
        "completion_per_class": per_class_prf(y_true, y_pred) if labeled else {},
        "completion_confusion_matrix": confusion_matrix(y_true, y_pred) if labeled else {},
        "reason_json_parse_rate": json_ok / total if total else 0.0,
        "pred_completion_nonempty_rate": completion_nonempty / total if total else 0.0,
        "pred_reason_nonempty_rate": reason_nonempty / total if total else 0.0,
        "completion_label_distribution_true": dict(Counter(y_true)),
        "completion_label_distribution_pred": dict(Counter(y_pred)),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
