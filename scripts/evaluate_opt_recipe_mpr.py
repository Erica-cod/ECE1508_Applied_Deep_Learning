#!/usr/bin/env python
"""
Evaluate a fine-tuned OPT model on the Recipe-MPR dataset using the benchmark taxonomy.

Computes overall accuracy and per-query-type accuracies (Specific, Commonsense, Negated,
Analogical, Temporal) by selecting the answer with the highest log-likelihood.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
PROMPT_TEMPLATE = """### Question:
{question}

### Choices:
{choices}

### Answer:
"""
CATEGORY_NAMES = ["Specific", "Commonsense", "Negated", "Analogical", "Temporal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned OPT model on Recipe-MPR.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (e.g. runs/opt-mpr/checkpoint-* or final save).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional tokenizer directory. Defaults to --model-path.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional config directory. Defaults to --model-path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/500QA.json",
        help="Path to Recipe-MPR JSON benchmark file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to score concurrently (1 recommended to avoid padding issues).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
        help="Maximum token length for prompt + completion.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation (e.g. cuda, cuda:0, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 weights during evaluation (requires CUDA).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 weights during evaluation (requires CUDA or MPS).",
    )
    return parser.parse_args()


def detect_device(explicit_device: Optional[str]) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(dataset_path: str) -> List[Dict[str, object]]:
    data_file = Path(dataset_path).expanduser()
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file {data_file} not found.")

    with data_file.open("r", encoding="utf-8") as handle:
        raw_examples: List[Dict[str, object]] = json.load(handle)

    processed = []
    for entry in raw_examples:
        options: Dict[str, str] = entry["options"]
        option_items = list(options.items())
        if len(option_items) > len(CHOICE_LABELS):
            raise ValueError(
                "Encountered more choices than supported labels. "
                "Increase CHOICE_LABELS to cover all options."
            )

        labeled_choices: List[Tuple[str, str, str]] = []
        correct_label: Optional[str] = None
        for idx, (option_id, option_text) in enumerate(option_items):
            label = CHOICE_LABELS[idx]
            labeled_choices.append((label, option_id, option_text))
            if option_id == entry["answer"]:
                correct_label = label

        if correct_label is None:
            raise ValueError(f"Answer id {entry['answer']} not found among options.")

        prompt = PROMPT_TEMPLATE.format(
            question=entry["query"].strip(),
            choices="\n".join(f"{label}. {text}" for label, _, text in labeled_choices),
        )

        processed.append(
            {
                "prompt": prompt,
                "choices": labeled_choices,
                "answer_id": entry["answer"],
                "correct_label": correct_label,
                "query_type": entry.get("query_type", {}),
            }
        )
    return processed


def build_inputs(
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        raise ValueError(
            f"Prompt + completion length {len(input_ids)} exceeds max_length={max_length}.\n"
            f"Prompt sample: {prompt[:200]!r}\nCompletion sample: {completion[:200]!r}"
        )
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + completion_ids

    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([attention_mask], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


def score_choice(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    device: torch.device,
) -> float:
    input_ids, attention_mask, labels = build_inputs(tokenizer, prompt, completion, max_length)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        labels_count = (labels != -100).sum().item()
        neg_log_likelihood = outputs.loss.item() * labels_count
        return -neg_log_likelihood


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict[str, object]],
    max_length: int,
    device: torch.device,
) -> Dict[str, float]:
    overall_total = len(dataset)
    overall_correct = 0

    category_totals = {name: 0 for name in CATEGORY_NAMES}
    category_correct = {name: 0 for name in CATEGORY_NAMES}

    for example in tqdm(dataset, desc="Evaluating", leave=False):
        prompt: str = example["prompt"]  # type: ignore[assignment]
        choices: List[Tuple[str, str, str]] = example["choices"]  # type: ignore[assignment]

        best_label: Optional[str] = None
        best_score: float = -math.inf

        for label, _, text in choices:
            completion = f"{label}. {text}\n"
            score = score_choice(model, tokenizer, prompt, completion, max_length, device)
            if score > best_score:
                best_score = score
                best_label = label

        is_correct = best_label == example["correct_label"]
        overall_correct += int(is_correct)

        query_type: Dict[str, int] = example.get("query_type", {})  # type: ignore[assignment]
        for category in CATEGORY_NAMES:
            if query_type.get(category, 0):
                category_totals[category] += 1
                if is_correct:
                    category_correct[category] += 1

    metrics = {
        "accuracy": overall_correct / overall_total if overall_total else 0.0,
    }
    for category in CATEGORY_NAMES:
        total = category_totals[category]
        if total:
            metrics[f"accuracy_{category.lower()}"] = category_correct[category] / total
        else:
            metrics[f"accuracy_{category.lower()}"] = float("nan")

    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    device = detect_device(args.device)
    logging.info("Using device: %s", device)

    dtype = None
    if args.fp16:
        if device.type != "cuda":
            raise EnvironmentError("--fp16 evaluation requires a CUDA device.")
        dtype = torch.float16
    elif args.bf16:
        if device.type not in {"cuda", "mps"}:
            raise EnvironmentError("--bf16 evaluation requires CUDA or MPS hardware.")
        dtype = torch.bfloat16

    model_dir = Path(args.model_path).expanduser()
    tokenizer_dir = Path(args.tokenizer_path).expanduser() if args.tokenizer_path else model_dir
    config_dir = Path(args.config_path).expanduser() if args.config_path else model_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(config_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, torch_dtype=dtype)
    model.to(device)
    model.eval()

    dataset = load_dataset(args.dataset_path)
    logging.info("Loaded %d evaluation examples.", len(dataset))

    metrics = evaluate_model(model, tokenizer, dataset, args.max_length, device)
    logging.info("Benchmark results:")
    for name, value in metrics.items():
        logging.info("  %s: %.4f", name, value)


if __name__ == "__main__":
    main()
