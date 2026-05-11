#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from structcheck.data.formatting import (
    build_instruction,
    build_instruction_compact,
    build_response_schema_hint,
    sft_json_tail_reminder,
)
from structcheck.train.compat import patch_transformers_for_minicpm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    # Parse the *first* valid JSON object by brace matching.
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = s[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    # continue searching for the next JSON object start
                    next_start = s.find("{", i + 1)
                    if next_start == -1:
                        return None
                    start = next_start
                    depth = 0
    return None


def extract_first_json_snippet(text: str) -> str:
    """
    Return the first brace-matched JSON object substring.
    If none found, return the original text (trimmed).
    """
    s = (text or "").strip()
    if not s:
        return ""
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()
    return s


ALLOWED_COMPLETIONS = ["Not Started", "In Progress", "Completed", "Uncertain"]

def _norm_completion(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    low = s.lower()
    mapping = {
        "not started": "Not Started",
        "not start": "Not Started",
        "in progress": "In Progress",
        "in-progress": "In Progress",
        "completed": "Completed",
        "complete": "Completed",
        "done": "Completed",
        "uncertain": "Uncertain",
        "unknown": "Uncertain",
        "unsure": "Uncertain",
    }
    return mapping.get(low, s)


def _extract_completion_from_text(text: str) -> str:
    t = (text or "")
    # JSON field
    for key in ("completion", "Completion"):
        p = f'"{key}"'
        if p in t:
            pj = try_parse_json(t)
            if isinstance(pj, dict) and (key in pj or key.lower() in pj):
                return _norm_completion(pj.get(key) or pj.get(key.lower()))
    # fallback keyword search
    for lab in ALLOWED_COMPLETIONS:
        if lab.lower() in t.lower():
            return lab
    return ""


def _reason_is_nonempty(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    # keep it simple: require a few words (low threshold so stage-2 repair can fill short reasons)
    return len(s.split()) >= 4


def _tensor_to_device(x, device_fn):
    if isinstance(x, torch.Tensor):
        return device_fn(x)
    if isinstance(x, list):
        return [_tensor_to_device(y, device_fn) for y in x]
    return x


def _fix_minicpm_processor_batch(model_inputs: Dict[str, Any]) -> None:
    for key in ("pixel_values", "image_bound", "image_sizes"):
        if key not in model_inputs:
            continue
        v = model_inputs[key]
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], torch.Tensor)):
            model_inputs[key] = [v]

    if "image_bound" in model_inputs and isinstance(model_inputs["image_bound"], list):
        fixed_bounds = []
        for b in model_inputs["image_bound"]:
            if isinstance(b, torch.Tensor):
                fixed_bounds.append(b.tolist())
                continue
            if isinstance(b, list):
                new_b = []
                for x in b:
                    if isinstance(x, torch.Tensor):
                        new_b.extend(x.tolist() if x.dim() == 2 else [x.tolist()])
                    else:
                        new_b.append(x)
                fixed_bounds.append(new_b)
                continue
            fixed_bounds.append(b)
        model_inputs["image_bound"] = fixed_bounds

    if (
        "image_bound" in model_inputs
        and "pixel_values" in model_inputs
        and isinstance(model_inputs["image_bound"], list)
        and isinstance(model_inputs["pixel_values"], list)
        and model_inputs["pixel_values"]
        and isinstance(model_inputs["pixel_values"][0], list)
    ):
        new_bounds_batch = []
        for bounds, pv in zip(model_inputs["image_bound"], model_inputs["pixel_values"]):
            if not isinstance(bounds, list):
                new_bounds_batch.append(bounds)
                continue
            slice_n = len(pv)
            diffs = [r[1] - r[0] for r in bounds if isinstance(r, (list, tuple)) and len(r) == 2]
            keep_len = 64 if 64 in diffs else (max(diffs) if diffs else None)
            if keep_len is None:
                new_bounds_batch.append(bounds)
                continue
            kept = [
                r
                for r in bounds
                if isinstance(r, (list, tuple)) and len(r) == 2 and (r[1] - r[0]) == keep_len
            ]
            if slice_n and len(kept) > slice_n:
                kept = kept[-slice_n:]
            new_bounds_batch.append(kept)
        model_inputs["image_bound"] = new_bounds_batch

    if "pixel_values" in model_inputs and isinstance(model_inputs["pixel_values"], list):
        pv_list = model_inputs["pixel_values"]
        if pv_list and isinstance(pv_list[0], list):
            fixed_batch = []
            for pv in pv_list:
                fixed_slices = []
                for t in pv:
                    if isinstance(t, torch.Tensor):
                        if t.dim() == 2:
                            t = t.unsqueeze(0)
                        elif t.dim() == 1:
                            t = t.unsqueeze(0).unsqueeze(0)
                    fixed_slices.append(t)
                fixed_batch.append(fixed_slices)
            model_inputs["pixel_values"] = fixed_batch


def _majority_completion(labels: List[str]) -> str:
    valid = [x for x in labels if x in ALLOWED_COMPLETIONS]
    if not valid:
        return "Uncertain"
    ctr = Counter(valid)
    best_n = max(ctr.values())
    tied = sorted([lab for lab, n in ctr.items() if n == best_n])
    return tied[0]


def _parse_stage1_json(raw_text: str, completion_first: bool) -> Tuple[str, str, str, str]:
    """Return (completion, reason, extracted_json_snippet, raw_text)."""
    text = extract_first_json_snippet(raw_text)
    parsed = try_parse_json(text) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    completion = _norm_completion(parsed.get("completion") or parsed.get("Completion") or "")
    if completion not in ALLOWED_COMPLETIONS:
        completion = _extract_completion_from_text(text)
    if completion not in ALLOWED_COMPLETIONS:
        completion = "Uncertain"
    reason = ""
    if not completion_first:
        reason = parsed.get("reason") or parsed.get("Reason") or ""
    return completion, str(reason).strip(), text, raw_text


def _generate_minicpm(
    model,
    processor,
    model_inputs: Dict[str, Any],
    max_new_tokens: int,
    *,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    pixel_values = model_inputs.get("pixel_values")
    tgt_sizes = model_inputs.get("tgt_sizes", [])
    image_bound = model_inputs.get("image_bound", [])

    if pixel_values == []:
        pixel_values = [[]]
    if image_bound == []:
        image_bound = [[]]

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        tgt_sizes=tgt_sizes,
        image_bound=image_bound,
        tokenizer=processor.tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
    )
    if do_sample:
        kwargs["temperature"] = float(temperature)

    with torch.no_grad():
        # Prefer remote-code text decoding when available; then strip the prompt by marker.
        try:
            out_text = model.generate(**kwargs, decode_text=True)
            if isinstance(out_text, list) and out_text and isinstance(out_text[0], str):
                full = out_text[0]
            elif isinstance(out_text, str):
                full = out_text
            else:
                full = str(out_text)
            # The prompt ends with "Expected JSON:"; keep only what comes after it.
            if "Expected JSON:" in full:
                full = full.split("Expected JSON:")[-1]
            return (full or "").strip()
        except Exception:
            pass

        # Fallback: token IDs path.
        try:
            out = model.generate(**kwargs, decode_text=False)
        except TypeError:
            out = model.generate(**kwargs)

    if isinstance(out, torch.Tensor):
        gen_ids = out
    elif isinstance(out, list) and out and isinstance(out[0], torch.Tensor):
        gen_ids = out[0]
    elif hasattr(out, "sequences") and isinstance(out.sequences, torch.Tensor):
        gen_ids = out.sequences
    else:
        return str(out)

    # gen_ids may be either:
    # - full sequences including prompt: shape (bs, prompt+new)
    # - only generated tokens: shape (bs, new)
    if gen_ids.ndim == 2:
        seq = gen_ids[0]
    else:
        seq = gen_ids

    prompt_len = int(input_ids.shape[1])
    if seq.numel() > prompt_len:
        tail = seq[prompt_len:]
        text = processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
        if text:
            return text
    # If the model returned "generated-only" ids, seq may be shorter than the prompt.
    # Decode as-is in that case.
    if seq.numel() > 0 and seq.numel() <= prompt_len:
        text = processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
        if text:
            return text
    # Otherwise: no usable generated text.
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference for StructCheck MiniCPM-V SFT model.")
    ap.add_argument("--base-model-path", required=True, help="Base MiniCPM-V-2_6 directory")
    ap.add_argument("--adapter-path", required=True, help="SFT output dir (LoRA adapter)")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--completion-first",
        action="store_true",
        help="Stage1 predicts completion only; Stage2 writes reason with completion fixed.",
    )
    ap.add_argument(
        "--require-image",
        action="store_true",
        help="Skip rows with no images[0] or missing file (aligns with SFT --require-image).",
    )
    ap.add_argument(
        "--compact-prompt",
        dest="compact_prompt",
        action="store_true",
        default=True,
        help="Drop long instruction lines so Expected JSON survives truncation (default: on).",
    )
    ap.add_argument(
        "--no-compact-prompt",
        dest="compact_prompt",
        action="store_false",
        help="Use full training-style instruction (may lose Expected JSON at short --max-length).",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only run first N samples.")
    ap.add_argument(
        "--completion-votes",
        type=int,
        default=1,
        help="Stage-1 samples for completion self-consistency; majority vote (default 1 = greedy).",
    )
    ap.add_argument(
        "--completion-vote-temp",
        type=float,
        default=0.7,
        help="Sampling temperature when --completion-votes > 1.",
    )
    ap.add_argument(
        "--legacy-sft-prompt",
        action="store_true",
        help="Omit schema hint + sft_json_tail_reminder before Expected JSON (older adapters, e.g. pre–tail-alignment SFT).",
    )
    args = ap.parse_args()

    patch_transformers_for_minicpm()

    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter_path)
    except Exception:
        model = base

    model.eval()

    rows = load_jsonl(Path(args.input_jsonl))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Two-stage inference:
    # 1) get a stable completion (optionally with a short reason)
    # 2) if reason is empty, ask the model to write reason only (completion fixed)
    schema_stage1 = (
        "Output JSON only.\n"
        '"completion" must be exactly one of: Not Started | In Progress | Completed | Uncertain.\n'
        "Pick the single best-supported label; use Uncertain only when evidence is genuinely missing or conflicting.\n"
        "Do not output multiple JSON objects."
    )
    schema_stage2 = (
        "Write ONLY the JSON object (no markdown). Keys: completion, reason.\n"
        '"completion" is fixed; do NOT change it.\n'
        '"reason" must be a non-empty short English paragraph (1-3 sentences).\n'
        "Do not output multiple JSON objects."
    )

    schema_sft = build_response_schema_hint()
    tail = sft_json_tail_reminder()

    with out_path.open("w", encoding="utf-8") as w:
        for row in rows:
            instruction = (
                build_instruction_compact(row) if args.compact_prompt else build_instruction(row)
            )
            if args.completion_first:
                if args.legacy_sft_prompt:
                    user_text_stage1 = (
                        f"{instruction}\n\n"
                        f"{schema_stage1}\n"
                        'Return JSON with key "completion" only.\n\n'
                        "Expected JSON:\n"
                        '{"completion": "Not Started"}'
                    )
                else:
                    user_text_stage1 = (
                        f"{instruction}\n\n"
                        f"{schema_sft}\n{tail}\n\n"
                        f"{schema_stage1}\n"
                        'Return JSON with key "completion" only.\n\n'
                        "Expected JSON:\n"
                        '{"completion": "Not Started"}'
                    )
            else:
                if args.legacy_sft_prompt:
                    # Matches early SFT / "compact" eval rows: instruction then Expected JSON only.
                    user_text_stage1 = f"{instruction}\n\nExpected JSON:\n"
                else:
                    # Match current SFT layout: instruction + schema + tail reminder + Expected JSON (see sft_train).
                    user_text_stage1 = (
                        f"{instruction}\n\n{schema_sft}\n{tail}\n\nExpected JSON:\n"
                    )
            # Leave headroom so generation can actually emit tokens.
            # If we truncate the prompt to the exact model context length, some runs
            # end up generating an empty string (effectively "no room to answer").
            prompt_max_len = max(64, int(args.max_length) - int(args.max_new_tokens) - 8)
            images = row.get("images") or []
            image_path = images[0] if images else None
            if args.require_image and (not image_path or not Path(image_path).exists()):
                continue

            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                prompt = f"<image>0/</image>\n{user_text_stage1}"
                try:
                    model_inputs = processor(
                        text=[prompt],
                        images=[image],
                        return_tensors="pt",
                        truncation=True,
                        max_length=prompt_max_len,
                        max_slice_nums=1,
                    )
                    _fix_minicpm_processor_batch(model_inputs)
                except Exception:
                    # Some samples still trigger image_bound/token mismatch in processor() (often due to truncation).
                    # Fall back to text-only for robustness.
                    image_path = None
                    prompt = user_text_stage1
                    model_inputs = processor.tokenizer(
                        text=[prompt],
                        return_tensors="pt",
                        truncation=True,
                        max_length=prompt_max_len,
                    )
                    model_inputs["pixel_values"] = []
                    model_inputs["image_sizes"] = []
                    model_inputs["image_bound"] = []
                    model_inputs["tgt_sizes"] = []
            else:
                prompt = user_text_stage1
                model_inputs = processor.tokenizer(
                    text=[prompt],
                    return_tensors="pt",
                    truncation=True,
                    max_length=prompt_max_len,
                )
                model_inputs["pixel_values"] = []
                model_inputs["image_sizes"] = []
                model_inputs["image_bound"] = []
                model_inputs["tgt_sizes"] = []

            if torch.cuda.is_available():

                def to_dev(t):
                    return t.cuda()

                model_inputs = {k: _tensor_to_device(v, to_dev) for k, v in model_inputs.items()}

            # Stage 1 (JSON): optional multi-sample completion vote
            votes = max(1, int(args.completion_votes))
            sample_rows: List[Tuple[str, str, str, str]] = []
            for vi in range(votes):
                if votes > 1:
                    seed = 91011 + vi * 9973
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    raw_try = _generate_minicpm(
                        model,
                        processor,
                        model_inputs,
                        args.max_new_tokens,
                        do_sample=True,
                        temperature=float(args.completion_vote_temp),
                    )
                else:
                    raw_try = _generate_minicpm(model, processor, model_inputs, args.max_new_tokens)
                sample_rows.append(_parse_stage1_json(raw_try, args.completion_first))

            completion = (
                sample_rows[0][0]
                if votes == 1
                else _majority_completion([r[0] for r in sample_rows])
            )
            reason = ""
            if not args.completion_first:
                chosen = next(
                    (r for r in sample_rows if r[0] == completion and _reason_is_nonempty(r[1])),
                    None,
                )
                if chosen is not None:
                    reason = chosen[1]
                else:
                    fallback = next((r for r in sample_rows if r[0] == completion), sample_rows[0])
                    reason = fallback[1]
            text = next((r[2] for r in sample_rows if r[0] == completion), sample_rows[0][2])
            raw_text = "\n---vote---\n".join(r[3] for r in sample_rows) if votes > 1 else sample_rows[0][3]
            # Stage 2: reason repair (text-only; keep completion fixed)
            if not _reason_is_nonempty(reason):
                r_tail = "" if args.legacy_sft_prompt else f"{sft_json_tail_reminder()}\n\n"
                repair_user = (
                    f"{instruction}\n\n"
                    f'Completion is fixed to: "{completion}"\n'
                    f"{schema_stage2}\n{r_tail}Expected JSON:\n"
                    f'{{\"completion\": \"{completion}\", \"reason\": \"\"}}'
                )
                # Prefer image-aware repair if we still have an image available.
                repair_inputs = None
                if image_path and Path(image_path).exists():
                    try:
                        repair_image = Image.open(image_path).convert("RGB")
                        repair_prompt = f"<image>0/</image>\n{repair_user}"
                        repair_inputs = processor(
                            text=[repair_prompt],
                            images=[repair_image],
                            return_tensors="pt",
                            truncation=True,
                            max_length=prompt_max_len,
                            max_slice_nums=1,
                        )
                        _fix_minicpm_processor_batch(repair_inputs)
                    except Exception:
                        repair_inputs = None
                if repair_inputs is None:
                    repair_inputs = processor.tokenizer(
                        text=[repair_user],
                        return_tensors="pt",
                        truncation=True,
                        max_length=prompt_max_len,
                    )
                    repair_inputs["pixel_values"] = []
                    repair_inputs["image_sizes"] = []
                    repair_inputs["image_bound"] = []
                    repair_inputs["tgt_sizes"] = []
                if torch.cuda.is_available():
                    repair_inputs = {k: _tensor_to_device(v, to_dev) for k, v in repair_inputs.items()}
                repair_text = _generate_minicpm(
                    model, processor, repair_inputs, max_new_tokens=min(256, max(192, int(args.max_new_tokens)))
                )
                repair_json = try_parse_json(repair_text)
                if isinstance(repair_json, dict):
                    r2 = repair_json.get("reason") or repair_json.get("Reason")
                    if _reason_is_nonempty(r2):
                        reason = str(r2).strip()

            parsed = {"completion": completion, "reason": str(reason).strip()}

            pred = {
                "element_id": row.get("element_id"),
                "timestamp": row.get("timestamp"),
                "image_path": image_path,
                "prediction_text": text,
                "prediction_text_raw": raw_text,
                "prediction_json": parsed,
                "target": row.get("target", {}),
            }
            w.write(json.dumps(pred, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
