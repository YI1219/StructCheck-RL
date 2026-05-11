import json
from typing import Dict, List, Optional


def _safe(v) -> str:
    return "" if v is None else str(v).strip()


def build_bim_text(meta: Dict) -> str:
    return (
        f'Name="{_safe(meta.get("Name"))}"; '
        f'IfcType="{_safe(meta.get("IfcType"))}"; '
        f'Description="{_safe(meta.get("说明", meta.get("Description", "")))}"; '
        f'Height="{_safe(meta.get("Height"))}"; '
        f'Width="{_safe(meta.get("Width"))}"; '
        f'Length="{_safe(meta.get("Length"))}"'
    )


def build_temporal_text(history: List[Dict]) -> str:
    if not history:
        return "No historical progress_decision records are available."
    items = []
    for h in history[-5:]:
        items.append(f'{h.get("date", "")}: Completion={h.get("completion", "")}')
    return " | ".join(items)


def build_target_json(completion: str, reason: str) -> str:
    """Single flat JSON line for SFT targets (must match inference schema)."""
    comp = completion if completion else "Uncertain"
    r = (reason or "").strip()
    if len(r) > 1200:
        r = r[:1200] + "…"
    payload = {"completion": comp, "reason": r}
    return json.dumps(payload, ensure_ascii=False)


def sft_json_tail_reminder() -> str:
    """Inserted immediately before ``Expected JSON:`` in SFT. With **left** truncation, the head of the
    rubric may drop but this line + JSON target stay in context."""
    return (
        "Respond with one JSON object only. "
        '"completion" must be exactly one of: Not Started | In Progress | Completed | Uncertain.'
    )


def completion_rubric() -> str:
    return (
        "Choose exactly one completion. "
        "Not Started: the element is absent / not installed yet (early shell, exposed framing, or no visible element). "
        "In Progress: partial install for THIS target (starts of install, scaffolding, temporary works, or clearly unfinished "
        "surfaces—not the finished installed element yet). "
        "Completed: the finished element is clearly installed and looks operational in view (closed-up, finished surfaces, "
        "no longer mid-construction for this trade), with visual evidence that matches BIM. "
        "When PointCloudMetrics show strong BIM–scan alignment and the image shows a finished install matching the BIM element type, "
        "prefer Completed over Not Started. "
        "Do NOT use Completed when only framing or rough-in is visible—prefer In Progress or Not Started. "
        "Uncertain: evidence is missing/occluded/conflicting; do not guess when unsure. "
        'If you cannot justify Not Started / In Progress / Completed, choose "Uncertain".'
    )


def build_instruction_compact(row: Dict) -> str:
    """Inference-only: drop long lines so ``Expected JSON:`` survives processor truncation."""
    full = build_instruction(row)
    keep: List[str] = []
    for ln in full.splitlines():
        if ln.startswith("PointCloudSummary:"):
            continue
        if ln.startswith("Use PointCloudMetrics as evidence:"):
            continue
        if ln.startswith("PointCloudMetrics heuristic"):
            continue
        keep.append(ln)
    return "\n".join(keep).strip()


def build_instruction(row: Dict) -> str:
    tgt = row.get("target") or {}
    stage = _safe(tgt.get("stage_label"))
    stage_line = f"Stage / work package (if known): {stage}\n" if stage else ""

    pcs = row.get("point_cloud_summary") or {}
    metrics = pcs.get("metrics") if isinstance(pcs, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    def _get_float(key: str) -> Optional[float]:
        v = metrics.get(key)
        try:
            return float(v)
        except Exception:
            return None

    cov = _get_float("coverage")
    cov_hit = _get_float("coverage_hit")
    vox_rec = _get_float("voxel_recall")
    vox_iou = _get_float("voxel_iou")
    pc_metrics_line = ""
    if any(x is not None for x in (cov, cov_hit, vox_rec, vox_iou)):
        parts = []
        if cov is not None:
            parts.append(f"coverage={cov:.3f}")
        if cov_hit is not None:
            parts.append(f"coverage_hit={cov_hit:.3f}")
        if vox_rec is not None:
            parts.append(f"voxel_recall={vox_rec:.3f}")
        if vox_iou is not None:
            parts.append(f"voxel_iou={vox_iou:.3f}")
        pc_metrics_line = "PointCloudMetrics: " + ", ".join(parts) + "\n"

    # Keep this compact to avoid processor() truncation desyncing image placeholder tokens.
    return (
        "Task: assess construction completion for the target BIM element.\n"
        f"{completion_rubric()}\n\n"
        f"{stage_line}"
        f"BIM: {build_bim_text(row.get('meta', {}))}\n"
        f"{pc_metrics_line}"
        f"PointCloudSummary: {_safe((row.get('point_cloud_summary') or {}).get('summary_line'))}\n"
        f"Temporal: {build_temporal_text(row.get('temporal_history', []))}\n"
        "PointCloudMetrics heuristic (rough): <0.20 -> Not Started; 0.20-0.70 -> In Progress; >0.70 -> Completed (if image also confirms), else Uncertain.\n"
        "Use Completed when the image shows a finished, BIM-consistent install; strong metrics support that label but still require visible matching finish (not rough-in alone).\n"
        "If an image is provided, focus on the target region and whether the element/stage is present.\n"
    ).strip()


def build_response_schema_hint() -> str:
    return (
        "Respond with JSON only (one object, no markdown). Keys: completion, reason.\n"
        '"completion" must be exactly one of: Not Started | In Progress | Completed | Uncertain.\n'
        '"reason" must be a non-empty short English paragraph (plain string, not an object), at least 10 words.\n'
        "Stop immediately after the closing '}' of the JSON object.\n"
        "Example: {\"completion\": \"Not Started\", \"reason\": \"...\"}"
    )
