#!/usr/bin/env python3
import argparse
import json
import warnings
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments

from structcheck.data.formatting import (
    build_instruction,
    build_instruction_compact,
    build_response_schema_hint,
    build_target_json,
    sft_json_tail_reminder,
)
from structcheck.text_truncation import truncation_side_left
from structcheck.train.compat import patch_transformers_for_minicpm


class MiniCPMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(data=inputs)
        loss = None
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            loss = outputs[0]

        if loss is None or isinstance(loss, dict) or not torch.is_tensor(loss):
            if isinstance(outputs, dict) and "logits" in outputs and "labels" in inputs:
                logits = outputs["logits"]
                labels = inputs["labels"]
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = labels[..., 1:].contiguous().long()
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                raise RuntimeError(
                    f"Unable to compute loss from outputs keys="
                    f"{list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}"
                )
        return (loss, outputs) if return_outputs else loss


def load_jsonl(path: Path, require_image: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if require_image:
                    imgs = row.get("images") or []
                    if not imgs:
                        continue
                    if not Path(imgs[0]).exists():
                        continue
                rows.append(row)
    return rows


def _completion_bucket(row: Dict[str, Any]) -> str:
    c = (row.get("target") or {}).get("completion") or ""
    s = str(c).strip()
    return s if s else "Uncertain"


def balance_rows_by_completion(rows: List[Dict[str, Any]], max_factor: int = 10) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[_completion_bucket(r)].append(r)
    if not buckets:
        return rows
    max_n = max(len(v) for v in buckets.values())
    out: List[Dict[str, Any]] = []
    for items in buckets.values():
        if not items:
            continue
        out.extend(items)
        # Cap oversampling so a tiny minority class doesn't dominate.
        target_n = min(max_n, len(items) * max(1, int(max_factor)))
        need = target_n - len(items)
        for j in range(need):
            out.append(items[j % len(items)])
    return out


def apply_completion_extra_copies(rows: List[Dict[str, Any]], spec: str) -> List[Dict[str, Any]]:
    """
    After balancing, duplicate rows for named completion labels to increase their gradient mass.

    ``spec`` format: ``Completed:1,In Progress:1`` — each integer is **extra** copies appended per matching row
    (``1`` ⇒ row appears twice in total for that label).
    """
    spec = (spec or "").strip()
    if not spec:
        return rows
    out: List[Dict[str, Any]] = list(rows)
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        label, n_s = part.rsplit(":", 1)
        label = label.strip()
        try:
            n_extra = max(0, int(n_s.strip()))
        except ValueError:
            continue
        if n_extra == 0:
            continue
        for r in rows:
            if _completion_bucket(r) == label:
                for _ in range(n_extra):
                    out.append(r)
    return out


def _sft_multimodal_text(user_text: str, target: str) -> str:
    """Full training line: image tag added separately by caller; same tail as `_sft_plain_text`."""
    tail = sft_json_tail_reminder()
    return f"{user_text}\n{tail}\n\nExpected JSON:\n{target}"


def _sft_plain_text(user_text: str, target: str) -> str:
    """Text-only SFT (must match multimodal suffix structure)."""
    return _sft_multimodal_text(user_text, target)


def _pixel_batch_nonempty(model_inputs: Dict[str, Any]) -> bool:
    pv = model_inputs.get("pixel_values")
    if not isinstance(pv, list) or len(pv) == 0:
        return False
    first = pv[0]
    if isinstance(first, list):
        return len(first) > 0
    return isinstance(first, torch.Tensor)


def _mask_labels_instruction_tokens(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    full_text: str,
) -> None:
    """Mask prompt tokens so loss applies only to the JSON target (text-only paths)."""
    marker = "\n\nExpected JSON:\n"
    pos = full_text.rfind(marker)
    if pos < 0:
        return
    assist_char = pos + len(marker)
    enc = tokenizer(
        full_text,
        truncation_side="left",
        truncation=True,
        max_length=labels.shape[1],
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    om = enc.get("offset_mapping")
    if not om:
        return
    offsets = om[0] if isinstance(om[0], list) else om
    first: Optional[int] = None
    for i, span in enumerate(offsets):
        if not span or len(span) < 2:
            continue
        a, b = int(span[0]), int(span[1])
        if a == 0 and b == 0:
            continue
        if a >= assist_char:
            first = i
            break
        if a < assist_char < b:
            first = i
            break
    if first is None or first <= 0:
        return
    labels[:, :first] = -100
    labels[attention_mask == 0] = -100


_multimodal_mask_miss_warned = False


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return None


def _mask_labels_instruction_tokens_multimodal(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    input_ids: torch.Tensor,
    target_json: str,
) -> None:
    """Mask labels before the JSON target. MiniCPM-V expands ``<image>`` into long placeholders in
    ``input_ids``; char offset mapping on the raw user string does not align, so we locate ``target_json``
    by token subsequence (last match)."""
    row = input_ids[0].tolist()
    candidates = []
    t = (target_json or "").strip()
    if t:
        candidates.append(t)
    if target_json and target_json not in candidates:
        candidates.append(target_json)

    start_idx: Optional[int] = None
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 0:
            continue
        pos = _find_last_subsequence(row, ids)
        if pos is not None:
            start_idx = pos
            break

    if start_idx is None:
        global _multimodal_mask_miss_warned
        if not _multimodal_mask_miss_warned:
            _multimodal_mask_miss_warned = True
            warnings.warn(
                "mask-instruction-labels: could not find target JSON token span in at least one sample; "
                "those batches use full-sequence loss. See structcheck.train.sft_train.",
                RuntimeWarning,
                stacklevel=2,
            )
        labels[attention_mask == 0] = -100
        return

    labels[:, :start_idx] = -100
    labels[attention_mask == 0] = -100


class StructCheckSFTDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        processor,
        max_length: int,
        *,
        compact_instruction: bool = True,
        mask_instruction_labels: bool = False,
        train_left_truncate: bool = True,
    ):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.compact_instruction = compact_instruction
        self.mask_instruction_labels = mask_instruction_labels
        self.train_left_truncate = train_left_truncate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        schema = build_response_schema_hint()
        instruction = (
            build_instruction_compact(row) if self.compact_instruction else build_instruction(row)
        )
        user_text = f"{instruction}\n\n{schema}"
        target = build_target_json(
            completion=row.get("target", {}).get("completion", ""),
            reason=row.get("target", {}).get("reason", ""),
        )

        images = row.get("images") or []
        image_path = images[0] if images else None
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
            # MiniCPM-V: explicit <image> tags + processor() are more robust than chat_template
            # when sequences are long (truncation can otherwise desync vision placeholders).
            body = _sft_multimodal_text(user_text, target)
            text = f"<image>0/</image>\n{body}"
            try:
                ctx = (
                    truncation_side_left(self.processor.tokenizer)
                    if self.train_left_truncate
                    else nullcontext()
                )
                with ctx:
                    model_inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        max_slice_nums=1,
                    )
            except Exception:
                text = _sft_plain_text(user_text, target)
                ctx = (
                    truncation_side_left(self.processor.tokenizer)
                    if self.train_left_truncate
                    else nullcontext()
                )
                with ctx:
                    model_inputs = self.processor.tokenizer(
                        text=[text],
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    )
                model_inputs["pixel_values"] = []
                model_inputs["image_sizes"] = []
                model_inputs["image_bound"] = []
                model_inputs["tgt_sizes"] = []
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

            # Final safety: if vision placeholders/bounds are inconsistent, fall back to text-only.
            try:
                pv_slices = (
                    len(model_inputs.get("pixel_values", [[]])[0])
                    if isinstance(model_inputs.get("pixel_values"), list) and model_inputs.get("pixel_values")
                    else 0
                )
                bounds0 = model_inputs.get("image_bound", [[]])[0] if isinstance(model_inputs.get("image_bound"), list) else []
                b_n = len(bounds0) if isinstance(bounds0, list) else 0
                if pv_slices and b_n and b_n != pv_slices:
                    raise ValueError(f"vision mismatch slices={pv_slices} bounds={b_n}")
            except Exception:
                text = _sft_plain_text(user_text, target)
                ctx = (
                    truncation_side_left(self.processor.tokenizer)
                    if self.train_left_truncate
                    else nullcontext()
                )
                with ctx:
                    model_inputs = self.processor.tokenizer(
                        text=[text],
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    )
                model_inputs["pixel_values"] = []
                model_inputs["image_sizes"] = []
                model_inputs["image_bound"] = []
                model_inputs["tgt_sizes"] = []
        else:
            text = _sft_plain_text(user_text, target)
            ctx = (
                truncation_side_left(self.processor.tokenizer)
                if self.train_left_truncate
                else nullcontext()
            )
            with ctx:
                model_inputs = self.processor.tokenizer(
                    text=[text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
            model_inputs["pixel_values"] = []
            model_inputs["image_sizes"] = []
            model_inputs["image_bound"] = []
            model_inputs["tgt_sizes"] = []

        out: Dict[str, Any] = {k: v for k, v in model_inputs.items()}
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.mask_instruction_labels:
            if _pixel_batch_nonempty(model_inputs):
                _mask_labels_instruction_tokens_multimodal(
                    labels, attention_mask, self.processor.tokenizer, input_ids, target
                )
            else:
                _mask_labels_instruction_tokens(labels, attention_mask, self.processor.tokenizer, text)
        out["labels"] = labels
        if "position_ids" not in out:
            bsz, seqlen = input_ids.shape
            out["position_ids"] = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) == 1:
        return batch[0]

    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ("input_ids", "attention_mask", "labels"):
            out[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch],
                batch_first=True,
                padding_value=0 if key != "labels" else -100,
            )
        elif torch.is_tensor(batch[0][key]):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            out[key] = [b[key] for b in batch]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for StructCheck multimodal completion+reason.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If >0, stop after this many steps (for smoke tests).",
    )
    parser.add_argument(
        "--no-require-image",
        dest="require_image",
        action="store_false",
        help="Keep text-only rows when the first image path is missing (default: drop them).",
    )
    parser.set_defaults(require_image=True)
    parser.add_argument(
        "--no-balance-classes",
        dest="balance_classes",
        action="store_false",
        help="Disable per-completion oversampling (default: balance enabled).",
    )
    parser.set_defaults(balance_classes=True)
    parser.add_argument(
        "--balance-max-factor",
        type=int,
        default=10,
        help="Cap oversampling: each class is expanded to at most (its_count * factor).",
    )
    parser.add_argument(
        "--extra-train-jsonl",
        default="",
        help="Optional JSONL merged after class balancing (e.g. structcheck.data.mine_eval_errors output).",
    )
    parser.add_argument(
        "--no-compact-instruction",
        dest="compact_instruction",
        action="store_false",
        help="Use full instruction text in SFT (longer; may lose JSON tail if max_length is tight).",
    )
    parser.set_defaults(compact_instruction=True)
    parser.add_argument(
        "--mask-instruction-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask prompt tokens; loss only on Expected JSON target (multimodal + text). Use --no-mask-instruction-labels to disable.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (default: 32).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=0,
        help="LoRA alpha; 0 means 2 * --lora-r.",
    )
    parser.add_argument(
        "--train-right-truncate",
        dest="train_left_truncate",
        action="store_false",
        help="Use default (right) truncation in SFT; only safe if --max-length fits full sequences.",
    )
    parser.set_defaults(train_left_truncate=True)
    parser.add_argument(
        "--completion-extra-copies",
        default="",
        help='After balancing, append extra row copies per label, e.g. "Completed:1,In Progress:1".',
    )
    args = parser.parse_args()

    patch_transformers_for_minicpm()
    model_path = str(Path(args.model_path).expanduser().resolve())

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_alpha = int(args.lora_alpha) if int(args.lora_alpha) > 0 else int(args.lora_r) * 2
    lora_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_rows = load_jsonl(Path(args.train_jsonl), require_image=args.require_image)
    eval_rows = load_jsonl(Path(args.eval_jsonl), require_image=args.require_image)
    if args.balance_classes:
        train_rows = balance_rows_by_completion(train_rows, max_factor=args.balance_max_factor)
    if args.completion_extra_copies:
        train_rows = apply_completion_extra_copies(train_rows, args.completion_extra_copies)
    if args.extra_train_jsonl:
        extra_p = Path(args.extra_train_jsonl).expanduser().resolve()
        if extra_p.is_file():
            train_rows = train_rows + load_jsonl(extra_p, require_image=args.require_image)

    ds_kw = dict(
        compact_instruction=args.compact_instruction,
        mask_instruction_labels=args.mask_instruction_labels,
        train_left_truncate=args.train_left_truncate,
    )
    train_ds = StructCheckSFTDataset(train_rows, processor, args.max_length, **ds_kw)
    eval_ds = StructCheckSFTDataset(eval_rows, processor, args.max_length, **ds_kw)

    save_steps = 200
    eval_steps = 200
    if args.max_steps and args.max_steps > 0:
        save_steps = max(1, min(200, args.max_steps // 2))
        eval_steps = save_steps

    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=[],
    )
    if args.max_steps and args.max_steps > 0:
        ta_kwargs["max_steps"] = args.max_steps
    training_args = TrainingArguments(**ta_kwargs)

    trainer = MiniCPMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
