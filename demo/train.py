import os
import re
import tempfile
import zipfile
from datetime import datetime
from typing import Optional

# Disable Unsloth fast generation to avoid tensor shape mismatch
# during GRPO generation in your current dependency combination.
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

import unsloth  # must be imported first
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel

DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------
# Compatibility patch:
# `trl`'s GRPOTrainer imports `llm_blender`, which (in some versions)
# expects `transformers.utils.hub.TRANSFORMERS_CACHE` to exist.
# Newer transformers versions may remove/rename this symbol, which causes
# an ImportError before training even starts.
# ---------------------------------------------------------------------
try:
    import transformers.utils.hub as _hf_hub

    if not hasattr(_hf_hub, "TRANSFORMERS_CACHE"):
        # Best-effort default path; llm_blender mainly needs the symbol.
        _hf_hub.TRANSFORMERS_CACHE = os.environ.get(
            "TRANSFORMERS_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"),
        )
except Exception:
    pass

from trl import GRPOConfig, GRPOTrainer


def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe(v: Any) -> str:
    return "" if v is None else str(v).strip()


def build_bim_semantics(meta_path: Path) -> Tuple[str, str]:
    meta = read_json(meta_path)
    target_element_text = f"""
Name: "{safe(meta.get('Name'))}";
Description: "{safe(meta.get('说明'))}";
IfcType: "{safe(meta.get('IfcType'))}";
Height: "{safe(meta.get('Height'))}";
Width: "{safe(meta.get('Width'))}";
Length: "{safe(meta.get('Length'))}";
""".strip()

    bim_semantics = f"""
This element is identified as "{safe(meta.get('Name'))}", with IFC type "{safe(meta.get('IfcType'))}".
It has approximate dimensions of height {safe(meta.get('Height'))}, width {safe(meta.get('Width'))}, and length {safe(meta.get('Length'))}.
Additional description: "{safe(meta.get('说明'))}".
""".strip()
    return target_element_text, bim_semantics


def build_pointcloud_summary(point_cloud_summary_path: Path) -> str:
    pc_summary = read_json(point_cloud_summary_path)
    return f'SummaryLine: "{safe(pc_summary.get("summary_line"))}";'


def build_temporal_text(
    element_dir: Path,
    run_date: str,
    history_json_name: str = "progress_decision.json",
) -> str:
    all_dirs: List[Path] = []
    for d in element_dir.iterdir():
        if d.is_dir():
            name = d.name
            if len(name) == 10 and name[4] == "-" and name[7] == "-" and name < run_date:
                all_dirs.append(d)

    all_dirs = sorted(all_dirs, key=lambda x: x.name)
    timeline_records: List[str] = []
    for d in all_dirs:
        json_path = d / history_json_name
        if not json_path.exists():
            continue
        try:
            data = read_json(json_path)
            stage_label = safe(data.get("Stage_label"))
            completion = safe(data.get("Completion"))
            timeline_records.append(f"{d.name}: Stage_label={stage_label}, Completion={completion}")
        except Exception:
            continue

    if not timeline_records:
        return """
TemporalHistory: "No historical records available before the current date.";
TemporalRule: "No earlier progress records are available for this element.";
""".strip()

    history_str = "; ".join(timeline_records)
    return f"""
TemporalHistory: "{history_str}";
TemporalRule: "Earlier records describe the historical construction state progression of the same element.";
""".strip()


def collect_local_image_paths(run_dir: Path, contour_json_path: Path, max_images: int = 4) -> List[str]:
    """Collect existing annotated local images for multimodal input."""
    if not contour_json_path.exists():
        return []

    contour_data = read_json(contour_json_path)
    element_name = contour_data.get("element_name", "unknown_element")
    items: List[Dict[str, Any]] = [x for x in contour_data.get("items", []) if x.get("num_contours", 0) > 0]
    items = sorted(
        items,
        key=lambda x: (float(x.get("mask_area_ratio", 0.0)), int(x.get("projected_points_unique_px", 0))),
        reverse=True,
    )
    paths: List[str] = []
    for item in items:
        base = Path(item["image_name"]).stem
        png_path = run_dir / f"{element_name}_{base}_{item['side']}_.png"
        if png_path.exists():
            paths.append(str(png_path))
        if len(paths) >= max_images:
            break
    return paths


def build_local_image_text(run_dir: Path, contour_json_path: Path) -> Tuple[str, int]:
    if not contour_json_path.exists():
        local_image_text = """
NumValidViews: "0";
NumAnnotatedImages: "0";
ViewInformation: "Local image information is unavailable because bim_projection_contours.json is missing or no valid annotated image could be loaded.";
LocalImageRule: "No local image evidence is available for this element at the current time step.";
ViewPolygons: "";
""".strip()
        return local_image_text, 0

    contour_data = read_json(contour_json_path)
    element_name = contour_data.get("element_name", "unknown_element")
    items: List[Dict[str, Any]] = [x for x in contour_data.get("items", []) if x.get("num_contours", 0) > 0]
    items = sorted(
        items,
        key=lambda x: (float(x.get("mask_area_ratio", 0.0)), int(x.get("projected_points_unique_px", 0))),
        reverse=True,
    )

    annotated_count = 0
    view_info_lines: List[str] = []
    view_polygon_lines: List[str] = []
    num_views = len(items)
    for i, item in enumerate(items, start=1):
        base = Path(item["image_name"]).stem
        png_path = run_dir / f"{element_name}_{base}_{item['side']}_.png"
        if png_path.exists():
            annotated_count += 1

        proj_pts = int(item.get("projected_points_unique_px", 0))
        side = item.get("side", "")
        img_name = item.get("image_name", "")
        contours = item.get("contours", [])
        num_contours = int(item.get("num_contours", 0))
        view_info_lines.append(
            f"View {i}: side={side}, image={img_name}, projected_points_unique_px={proj_pts}, num_contours={num_contours}."
        )
        view_polygon_lines.append(
            f"View {i}: side={side}, image={img_name}, contours={json.dumps(contours, ensure_ascii=False)}"
        )

    local_rule = (
        "For each valid view, the target region is defined by the contour polygons in bim_projection_contours.json. "
        "The model should focus primarily on the contour-enclosed region."
    )
    local_image_text = f"""
NumValidViews: "{num_views}";
NumAnnotatedImages: "{annotated_count}";
ViewInformation: "{safe(chr(10).join(view_info_lines))}";
LocalImageRule: "{safe(local_rule)}";
ViewPolygons: "{safe(chr(10).join(view_polygon_lines))}";
""".strip()
    return local_image_text, num_views


def build_input_completeness(has_bim: bool, has_pc: bool, has_temporal: bool, has_local_image: bool) -> Tuple[int, str]:
    missing = []
    if not has_bim:
        missing.append("BIM semantic information")
    if not has_pc:
        missing.append("point cloud summary information")
    if not has_temporal:
        missing.append("temporal context information")
    if not has_local_image:
        missing.append("local image information")

    n = 4 - len(missing)
    if not missing:
        availability_text = "no input information is missing"
    elif len(missing) == 1:
        availability_text = f"{missing[0]} is unavailable"
    elif len(missing) == 2:
        availability_text = f"{missing[0]} and {missing[1]} are unavailable"
    else:
        availability_text = ", ".join(missing[:-1]) + f", and {missing[-1]} are unavailable"
    return n, availability_text


def build_final_prompt(
    target_element_text: str,
    bim_semantics: str,
    pointcloud_summary_text: str,
    temporal_text: str,
    local_image_text: str,
    n_types: int,
    missing_text: str,
) -> str:
    return f"""
You are a construction progress assessment assistant.

Your task is to determine:
1. Stage_label
2. Completion
3. Reason

========================
1. BIM SEMANTICS
========================
{bim_semantics}

========================
2. POINT CLOUD SUMMARY
========================
{pointcloud_summary_text}

========================
3. TEMPORAL CONTEXT
========================
{temporal_text}

========================
4. LOCAL IMAGE INFORMATION
========================
{local_image_text}

The target element is described in meta.json as:
{target_element_text}

Return JSON only, with exactly these keys:
{{
  "Stage_label": "...",
  "Completion": "...",
  "Reason": "..."
}}

Reason must mention that {n_types} input types were considered and that {missing_text}.
""".strip()


def discover_element_dirs_with_meta(data_root: Path) -> List[Path]:
    """All directories under data_root that contain meta.json (same layout as readme / case1)."""
    element_dirs: List[Path] = []
    if (data_root / "meta.json").exists():
        element_dirs.append(data_root)
    element_dirs.extend([p for p in data_root.rglob("*") if p.is_dir() and (p / "meta.json").exists()])
    return list(dict.fromkeys(element_dirs))


def index_ifc_global_id_to_element_dir(element_dirs: List[Path]) -> Dict[str, Path]:
    """Map IfcGlobalId (and folder suffix after '__') to element directory."""
    idx: Dict[str, Path] = {}
    for d in element_dirs:
        meta_path = d / "meta.json"
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        gid = safe(meta.get("IfcGlobalId"))
        if gid:
            idx[gid] = d
        m = re.search(r"__(.+)$", d.name)
        if m:
            suf = m.group(1).strip()
            if suf:
                idx.setdefault(suf, d)
    return idx


def extract_zip_dataset_root(zip_path: Path) -> Path:
    """Extract zip to a temp directory and return the inner dataset root (single top folder if any)."""
    td = Path(tempfile.mkdtemp(prefix="structcheck_data_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(td)
    children = [c for c in td.iterdir() if not c.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return td


def _norm_col_name(s: Any) -> str:
    return re.sub(r"\s+", "_", str(s).strip().lower())


def _find_df_column(columns: List[Any], aliases: List[str]) -> Optional[str]:
    norm_to_orig = {_norm_col_name(c): c for c in columns}
    for a in aliases:
        key = _norm_col_name(a)
        if key in norm_to_orig:
            return str(norm_to_orig[key])
    for c in columns:
        cl = _norm_col_name(c)
        for a in aliases:
            if _norm_col_name(a) in cl or cl in _norm_col_name(a):
                return str(c)
    return None


def _format_run_date_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return ""
    if hasattr(v, "strftime") and not isinstance(v, str):
        try:
            return v.strftime("%Y-%m-%d")
        except Exception:
            pass
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return s


def _parse_run_date_from_wide_completion_col(col_name: str) -> Optional[str]:
    """
    Parse wide-label column like '10/2/2025_Completion' to 'YYYY-MM-DD'.
    Prefer day-first format for this project.
    """
    s = safe(col_name)
    m = re.match(r"^\s*(\d{1,2}/\d{1,2}/\d{4})\s*_completion\s*$", s, flags=re.IGNORECASE)
    if not m:
        return None
    d = m.group(1)
    for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def make_prompt_row_for_element_run(
    element_dir: Path,
    run_date: str,
    extra: Optional[Dict[str, str]] = None,
    use_vision_input: bool = False,
    max_vision_images: int = 4,
) -> Optional[Dict[str, Any]]:
    """One training row: same file logic as case1 (meta + run_date/point_cloud_summary)."""
    meta_path = element_dir / "meta.json"
    run_dir = element_dir / run_date
    point_cloud_summary_path = run_dir / "point_cloud_summary.json"
    contour_json_path = run_dir / "bim_projection_contours.json"

    if not meta_path.exists() or not point_cloud_summary_path.exists():
        return None

    target_element_text, bim_semantics = build_bim_semantics(meta_path)
    pointcloud_summary_text = build_pointcloud_summary(point_cloud_summary_path)
    temporal_text = build_temporal_text(element_dir, run_date)
    local_image_text, num_views = build_local_image_text(run_dir, contour_json_path)

    has_temporal = "No historical records available" not in temporal_text
    has_local_image = num_views > 0
    n_types, missing_text = build_input_completeness(
        has_bim=True, has_pc=True, has_temporal=has_temporal, has_local_image=has_local_image
    )
    prompt = build_final_prompt(
        target_element_text=target_element_text,
        bim_semantics=bim_semantics,
        pointcloud_summary_text=pointcloud_summary_text,
        temporal_text=temporal_text,
        local_image_text=local_image_text,
        n_types=n_types,
        missing_text=missing_text,
    )
    row: Dict[str, Any] = {"prompt": prompt}
    if use_vision_input:
        image_paths = collect_local_image_paths(run_dir, contour_json_path, max_images=max_vision_images)
        if image_paths:
            image_tokens = "\n".join([f"<image_{i + 1}>: {p}" for i, p in enumerate(image_paths)])
            row["prompt"] = f"{prompt}\n\nVisualInputs:\n{image_tokens}"
            row["image_paths"] = image_paths
            row["image_count"] = len(image_paths)
    if extra:
        row.update({k: v for k, v in extra.items() if v is not None})
    return row


def build_dataset_from_data_use(
    data_root: Path,
    run_date: str,
    max_samples: int,
    use_vision_input: bool = False,
    max_vision_images: int = 4,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    element_dirs = discover_element_dirs_with_meta(data_root)

    for element_dir in element_dirs:
        rec = make_prompt_row_for_element_run(
            element_dir,
            run_date,
            use_vision_input=use_vision_input,
            max_vision_images=max_vision_images,
        )
        if rec is None:
            continue
        rows.append(rec)
        if len(rows) >= max_samples:
            break

    if not rows:
        raise RuntimeError(
            f"No samples found under {data_root} for run_date={run_date}. "
            "Expected element_dir/meta.json and element_dir/<run_date>/point_cloud_summary.json."
        )
    return Dataset.from_list(rows)


def build_dataset_from_xlsx(
    data_root: Path,
    xlsx_path: Path,
    max_samples: int,
    use_vision_input: bool = False,
    max_vision_images: int = 4,
) -> Dataset:
    """
    Load the full labeled set: each Excel row picks (element, run_date) like case1.
    Matches element via IfcGlobalId index or optional folder path column (paths.folder_rel style).
    Requires: pip install pandas openpyxl
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("Reading labels requires pandas: pip install pandas") from e

    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except ImportError as e:
        raise RuntimeError("Reading .xlsx requires openpyxl: pip install openpyxl") from e

    cols = list(df.columns)
    col_gid = _find_df_column(
        cols,
        ["IfcGlobalId", "ifcglobalid", "global_id", "guid", "GlobalId", "Ifc_Global_Id", "构件ID", "全局ID"],
    )
    col_date = _find_df_column(
        cols,
        ["run_date", "RunDate", "date", "Date", "observation_date", "snapshot_date", "采集日期", "日期", "时间"],
    )
    col_folder = _find_df_column(
        cols,
        ["folder_rel", "folder", "relative_path", "path", "paths", "folder_path", "相对路径"],
    )
    col_stage = _find_df_column(cols, ["Stage_label", "stage_label", "stage", "阶段", "Stage"])
    col_completion = _find_df_column(cols, ["Completion", "completion", "完成度", "完成"])
    col_json_rel = _find_df_column(cols, ["JSON_RelPath", "json_relpath", "json_rel", "meta_path"])

    if not col_gid and not col_folder and not col_json_rel:
        raise RuntimeError(
            f"Could not find IfcGlobalId/folder/json path column in {xlsx_path}. "
            "Expected IfcGlobalId / folder_rel / JSON_RelPath."
        )

    element_dirs = discover_element_dirs_with_meta(data_root)
    gid_index = index_ifc_global_id_to_element_dir(element_dirs)
    wide_completion_cols: List[Tuple[str, str]] = []
    if not col_date:
        for c in cols:
            run_date = _parse_run_date_from_wide_completion_col(str(c))
            if run_date:
                wide_completion_cols.append((str(c), run_date))
        if not wide_completion_cols:
            raise RuntimeError(
                f"Could not find a date column in {xlsx_path}. "
                "Expected run_date/Date/日期 or wide columns like 10/2/2025_Completion."
            )

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for _, r in df.iterrows():
        if len(rows) >= max_samples:
            break
        element_dir: Optional[Path] = None
        if col_gid:
            gid = safe(r[col_gid])
            if gid:
                element_dir = gid_index.get(gid)
        if element_dir is None and col_folder:
            rel = safe(r[col_folder]).replace("\\", "/").strip()
            if rel:
                cand = data_root / rel
                if (cand / "meta.json").exists():
                    element_dir = cand
        if element_dir is None and col_json_rel:
            rel_json = safe(r[col_json_rel]).replace("\\", "/").strip()
            if rel_json:
                cand_meta = data_root / rel_json
                if cand_meta.name.lower() == "meta.json" and cand_meta.exists():
                    element_dir = cand_meta.parent

        if element_dir is None:
            skipped += 1
            continue

        if col_date:
            run_dates: List[Tuple[str, Optional[str]]] = [(_format_run_date_cell(r[col_date]), col_completion)]
        else:
            run_dates = [(rd, c_name) for c_name, rd in wide_completion_cols]

        for run_date, completion_col_for_row in run_dates:
            if len(rows) >= max_samples:
                break
            if not run_date:
                skipped += 1
                continue

            extra: Dict[str, str] = {}
            if col_stage:
                extra["label_stage"] = safe(r[col_stage])
            if completion_col_for_row:
                extra["label_completion"] = safe(r[completion_col_for_row])
            elif col_completion:
                extra["label_completion"] = safe(r[col_completion])

            if safe(extra.get("label_completion", "")) == "":
                continue

            rec = make_prompt_row_for_element_run(
                element_dir,
                run_date,
                extra=extra or None,
                use_vision_input=use_vision_input,
                max_vision_images=max_vision_images,
            )
            if rec is None:
                skipped += 1
                continue
            rows.append(rec)

    if not rows:
        raise RuntimeError(
            f"No samples built from {xlsx_path} under {data_root}. "
            f"Check IfcGlobalId/folder paths and that each row's date folder has point_cloud_summary.json. "
            f"(skipped_rows={skipped})"
        )
    print(f"Built {len(rows)} samples from labels; skipped {skipped} rows (no match or missing files).")
    return Dataset.from_list(rows)


def _normalize_text_label(v: Any) -> str:
    s = safe(v).lower()
    s = re.sub(r"[\s_\-]+", "", s)
    return s


def _extract_json_candidate(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Best-effort: parse the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _parse_completion_numeric(v: Any) -> Optional[float]:
    s = safe(v)
    if not s:
        return None
    # Accept formats like "0.62", "62%", "Completion=0.62"
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except Exception:
        return None
    # If value is likely percent, normalize to 0~1.
    if "%" in s or val > 1.0:
        val = val / 100.0
    if val < 0:
        val = 0.0
    if val > 1:
        val = 1.0
    return val


def reward_fn(completions, **kwargs):
    scores = []
    gt_stages = kwargs.get("label_stage", [])
    gt_completions = kwargs.get("label_completion", [])

    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = safe(completion[0].get("content") if completion else "")
        else:
            text = safe(completion)

        score = 0.0
        pred_obj = _extract_json_candidate(text)
        if pred_obj is None:
            # Strong penalty: output is not valid JSON as required by prompt.
            scores.append(-0.5)
            continue

        # Format reward: has required keys.
        has_stage = "Stage_label" in pred_obj
        has_completion = "Completion" in pred_obj
        has_reason = "Reason" in pred_obj
        score += 0.2 * float(has_stage) + 0.2 * float(has_completion) + 0.1 * float(has_reason)

        gt_stage = gt_stages[i] if i < len(gt_stages) else ""
        gt_comp = gt_completions[i] if i < len(gt_completions) else ""

        # 1) Stage label exact/normalized match is the primary objective.
        if has_stage and safe(gt_stage):
            pred_stage_n = _normalize_text_label(pred_obj.get("Stage_label"))
            gt_stage_n = _normalize_text_label(gt_stage)
            if pred_stage_n == gt_stage_n:
                score += 1.0
            elif pred_stage_n and gt_stage_n and (pred_stage_n in gt_stage_n or gt_stage_n in pred_stage_n):
                score += 0.4
            else:
                score -= 0.2

        # 2) Completion closeness reward (if numeric parse succeeds).
        if has_completion and safe(gt_comp):
            pred_c = _parse_completion_numeric(pred_obj.get("Completion"))
            gt_c = _parse_completion_numeric(gt_comp)
            if pred_c is not None and gt_c is not None:
                # Linear closeness in [0, 1], weighted to 0.6 max reward.
                closeness = max(0.0, 1.0 - abs(pred_c - gt_c))
                score += 0.6 * closeness
            elif _normalize_text_label(pred_obj.get("Completion")) == _normalize_text_label(gt_comp):
                # Fallback textual equality for categorical completion labels.
                score += 0.3

        scores.append(score)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name or local model path for FastLanguageModel.from_pretrained.",
    )
    parser.add_argument("--data-root", type=str, default="", help="Extracted dataset root (readme / case1 layout).")
    parser.add_argument(
        "--zip",
        type=str,
        default="",
        help="Optional: path to L3-5d.zip; extracted to a temp dir and used as data root.",
    )
    parser.add_argument(
        "--labels-xlsx",
        type=str,
        default="",
        help="Construction_status_label_v2.xlsx: each row = one (IfcGlobalId or folder path) + date; loads full labeled set.",
    )
    parser.add_argument(
        "--no-vision-input",
        action="store_true",
        help="Disable visual input fields and fall back to text-only rows.",
    )
    parser.add_argument(
        "--max-vision-images",
        type=int,
        default=4,
        help="Maximum number of local annotated images attached per sample when vision input is enabled.",
    )
    parser.add_argument("--run-date", type=str, default="2025-08-01", help="Run date when not using --labels-xlsx.")
    parser.add_argument("--max-samples", type=int, default=200, help="Max training rows (cap for xlsx scan or single-date walk).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Per-device train batch size (lower to reduce VRAM).")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num-generations", type=int, default=2, help="GRPO generations per prompt (major VRAM factor).")
    parser.add_argument("--max-completion-length", type=int, default=96, help="Max generated tokens for each completion.")
    args = parser.parse_args()

    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        max_seq_length=1024,
        dtype=torch.float16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # trl>=某些版本在 GRPOTrainer 初始化时会写入 `model.warnings_issued[...]`，
    # 但当前 unsloth/peft 返回的模型对象可能不带该属性。
    # 这里做一个最小兼容，避免在真正开始训练前就崩溃。
    if not hasattr(model, "warnings_issued"):
        try:
            model.warnings_issued = {}
        except Exception:
            pass

    # GRPO 在生成/打分阶段会调用 `model.generate(...)`。
    # 当前 unsloth/peft 组合在缓存路径上出现了张量 shape 不匹配，
    # 先禁用 use_cache 让训练流程更容易跑通。
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.generation_config.use_cache = False  # type: ignore[attr-defined]
    except Exception:
        pass

    data_root_path: Optional[Path] = None
    if args.zip:
        zp = Path(args.zip)
        if not zp.is_file():
            raise RuntimeError(f"--zip not found: {zp}")
        data_root_path = extract_zip_dataset_root(zp)
        print(f"Using extracted dataset root from zip: {data_root_path}")
    elif args.data_root:
        data_root_path = Path(args.data_root)

    if args.labels_xlsx and data_root_path is None:
        raise RuntimeError("--labels-xlsx requires --data-root (extracted tree) or --zip (L3-5d.zip).")

    print("Loading dataset...")
    use_vision_input = not args.no_vision_input

    if data_root_path and args.labels_xlsx:
        lx = Path(args.labels_xlsx)
        if not lx.is_file():
            raise RuntimeError(f"--labels-xlsx not found: {lx}")
        dataset = build_dataset_from_xlsx(
            data_root_path,
            lx,
            args.max_samples,
            use_vision_input=use_vision_input,
            max_vision_images=args.max_vision_images,
        )
        print(f"Loaded {len(dataset)} samples from labels xlsx + dataset tree.")
    elif data_root_path:
        dataset = build_dataset_from_data_use(
            data_root_path,
            args.run_date,
            args.max_samples,
            use_vision_input=use_vision_input,
            max_vision_images=args.max_vision_images,
        )
        print(f"Loaded {len(dataset)} samples from local Data-use layout (single run_date={args.run_date}).")
    else:
        dataset = load_dataset("trl-lib/ultrafeedback-prompt", split=f"train[:{args.max_samples}]")
        print(f"Loaded {len(dataset)} samples from trl-lib/ultrafeedback-prompt.")

    if use_vision_input:
        print("Vision input enabled: dataset rows include image_paths and image_count for multimodal training pipelines.")
    else:
        print("Vision input disabled: text-only dataset rows.")

    training_args = GRPOConfig(
        output_dir="/app/grpo_output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=50,
        fp16=True,
    )

    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
