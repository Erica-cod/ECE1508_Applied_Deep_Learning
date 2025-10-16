#!/usr/bin/env python
"""
Fine-tune a local GPT-2 model on the Recipe-MPR (500QA) dataset.

This mirrors the OPT training pipeline but defaults to GPT-2 checkpoints under
~/models/hub/models--gpt2 and stores outputs in runs/gpt2-recipe-mpr.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
PROMPT_TEMPLATE = """### Question:
{question}

### Choices:
{choices}

### Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on the Recipe-MPR dataset.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/models/hub/models--gpt2",
        help="Path to the local GPT-2 checkpoint directory (a snapshot or root cache folder).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional path to tokenizer assets. Defaults to --model-path if not provided.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/500QA.json",
        help="Path to the Recipe-MPR JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/gpt2-recipe-mpr",
        help="Directory where checkpoints and logs will be stored.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples reserved for evaluation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps to simulate larger batches.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency in steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluation frequency in steps when using 'steps' eval strategy.",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="steps",
        choices=("no", "steps", "epoch"),
        help="Checkpointing strategy (ignored if unsupported by installed Transformers).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint save frequency in steps (when using 'steps' strategy).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Limit the total number of checkpoints retained.",
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default=None,
        choices=("no", "steps", "epoch"),
        help="Preferred evaluation strategy argument for newer Transformers releases.",
    )
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default=None,
        choices=("no", "steps", "epoch"),
        help="Legacy evaluation strategy flag retained for backwards compatibility.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Optional device map passed to from_pretrained (e.g. 'auto').",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision training (requires CUDA).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision training (requires CUDA or Apple MPS).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        help="Track eval metrics and reload the best checkpoint when supported.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained model to the Hugging Face Hub (requires credentials).",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Optional Hub repository id to push to.",
    )
    args = parser.parse_args()

    if args.eval_ratio <= 0 or args.eval_ratio >= 1:
        parser.error("--eval-ratio must be in the (0, 1) interval.")

    if args.eval_strategy and args.evaluation_strategy:
        parser.error("Specify only one of --eval-strategy or --evaluation-strategy.")

    chosen_eval_strategy = args.eval_strategy or args.evaluation_strategy
    args.eval_strategy = chosen_eval_strategy
    args.evaluation_strategy = chosen_eval_strategy

    if args.no_eval and chosen_eval_strategy and chosen_eval_strategy != "no":
        parser.error("--no-eval conflicts with --eval-strategy/--evaluation-strategy.")

    if args.fp16 and args.bf16:
        parser.error("Enable either --fp16 or --bf16, not both.")

    return args


def validate_precision_flags(args: argparse.Namespace) -> None:
    cuda_available = torch.cuda.is_available()
    mps_available = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_available and torch.backends.mps.is_available())
    if args.fp16 and not cuda_available:
        raise EnvironmentError(
            "FP16 training requires a CUDA-capable GPU. Rerun without --fp16 or enable CUDA."
        )
    if args.bf16 and not (cuda_available or mps_available):
        raise EnvironmentError(
            "BF16 training requires CUDA or Apple MPS hardware. Rerun without --bf16 or enable supported hardware."
        )


def resolve_model_and_tokenizer_paths(
    model_path: str, tokenizer_path: Optional[str] = None
) -> tuple[str, str, str]:
    """
    Expand provided paths and locate snapshot folders that contain model weights,
    config, and tokenizer assets. Reused from the OPT training script to keep
    local cache handling consistent.
    """

    def _has_any(path: Path, filenames: List[str]) -> bool:
        return path.exists() and any((path / name).exists() for name in filenames)

    def _select_snapshot(base: Path, required_files: List[str]) -> Optional[Path]:
        if _has_any(base, required_files):
            return base
        snapshots = base / "snapshots"
        if snapshots.is_dir():
            snapshot_dirs = sorted(
                (d for d in snapshots.iterdir() if d.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for snapshot in snapshot_dirs:
                if _has_any(snapshot, required_files):
                    return snapshot
        return None

    expanded_model = Path(model_path).expanduser()
    if not expanded_model.exists():
        raise FileNotFoundError(f"Model path {expanded_model} does not exist.")

    model_snapshot = _select_snapshot(expanded_model, ["pytorch_model.bin", "model.safetensors"])
    if model_snapshot is None:
        raise FileNotFoundError(
            f"Could not locate model weights under {expanded_model}. Ensure the directory contains pytorch_model.bin or model.safetensors."
        )

    config_snapshot = _select_snapshot(expanded_model, ["config.json"])

    if tokenizer_path is not None:
        expanded_tokenizer = Path(tokenizer_path).expanduser()
        if not expanded_tokenizer.exists():
            raise FileNotFoundError(f"Tokenizer path {expanded_tokenizer} does not exist.")
        tokenizer_snapshot = _select_snapshot(
            expanded_tokenizer,
            ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        )
        if config_snapshot is None:
            config_snapshot = _select_snapshot(expanded_tokenizer, ["config.json"])
    else:
        tokenizer_snapshot = _select_snapshot(
            expanded_model,
            ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
        )
        if config_snapshot is None:
            config_snapshot = _select_snapshot(expanded_model, ["config.json"])

    if tokenizer_snapshot is None:
        raise FileNotFoundError(
            "Unable to locate tokenizer assets. Provide --tokenizer-path explicitly or ensure the cache folder includes tokenizer files."
        )
    if config_snapshot is None:
        raise FileNotFoundError(
            "Unable to locate config.json for the model. Ensure the cache directory includes configuration files."
        )

    if tokenizer_snapshot != model_snapshot:
        logging.info("Resolved model snapshot: %s", model_snapshot)
        logging.info("Resolved tokenizer snapshot: %s", tokenizer_snapshot)
    else:
        logging.info("Resolved model/tokenizer snapshot: %s", model_snapshot)

    if config_snapshot not in (model_snapshot, tokenizer_snapshot):
        logging.info("Resolved config snapshot: %s", config_snapshot)

    return str(model_snapshot), str(tokenizer_snapshot), str(config_snapshot)


def load_recipe_mpr(dataset_path: str) -> List[Dict[str, str]]:
    dataset_file = Path(dataset_path).expanduser()
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found.")

    with dataset_file.open("r", encoding="utf-8") as handle:
        raw_examples = json.load(handle)

    processed_examples: List[Dict[str, str]] = []
    for entry in raw_examples:
        option_items = list(entry["options"].items())
        if len(option_items) > len(CHOICE_LABELS):
            raise ValueError(
                "Encountered more choices than supported labels. Update CHOICE_LABELS to include additional letters."
            )

        choice_lines: List[str] = []
        answer_label: Optional[str] = None
        answer_text: Optional[str] = None

        for idx, (option_id, option_text) in enumerate(option_items):
            label = CHOICE_LABELS[idx]
            choice_lines.append(f"{label}. {option_text}")
            if option_id == entry["answer"]:
                answer_label = label
                answer_text = option_text

        if answer_label is None or answer_text is None:
            raise ValueError(f"Answer id {entry['answer']} not found among options.")

        prompt = PROMPT_TEMPLATE.format(
            question=entry["query"].strip(),
            choices="\n".join(choice_lines),
        )
        completion = f"{answer_label}. {answer_text}\n"
        processed_examples.append({"prompt": prompt, "completion": completion})

    return processed_examples


def train_val_split(examples: List[Dict[str, str]], eval_ratio: float, seed: int) -> DatasetDict:
    random.Random(seed).shuffle(examples)
    train_size = max(1, int((1.0 - eval_ratio) * len(examples)))
    train_examples = examples[:train_size]
    eval_examples = examples[train_size:]

    dataset_dict = DatasetDict()
    dataset_dict["train"] = Dataset.from_list(train_examples)
    if eval_examples:
        dataset_dict["validation"] = Dataset.from_list(eval_examples)
    return dataset_dict


@dataclass
class CompletionDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        label_seqs = [f["labels"] for f in features]
        max_seq_len = max(len(labels) for labels in label_seqs)
        if self.pad_to_multiple_of is not None and max_seq_len % self.pad_to_multiple_of != 0:
            max_seq_len = ((max_seq_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        padded_labels = []
        for labels in label_seqs:
            padding_length = max_seq_len - len(labels)
            padded_labels.append(labels + [self.label_pad_token_id] * padding_length)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def tokenize_examples(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    eos_token = tokenizer.eos_token or ""

    def _tokenize(example: Dict[str, str]) -> Dict[str, List[int]]:
        prompt = example["prompt"]
        completion = example["completion"] + eos_token
        full_text = prompt + completion

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        tokenized = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        labels = tokenized["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        tokenized["labels"] = labels
        return tokenized

    columns_to_remove = set(dataset["train"].column_names)
    tokenized_dataset = dataset.map(
        _tokenize,
        batched=False,
        remove_columns=list(columns_to_remove),
    )
    return tokenized_dataset


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    validate_precision_flags(args)
    set_seed(args.seed)

    model_path, tokenizer_path, config_path = resolve_model_and_tokenizer_paths(
        args.model_path, args.tokenizer_path
    )
    dataset = load_recipe_mpr(args.dataset_path)
    dataset_dict = train_val_split(dataset, args.eval_ratio, args.seed)
    logging.info(
        "Prepared dataset splits: %d train / %d eval examples",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]) if "validation" in dataset_dict else 0,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_datasets = tokenize_examples(dataset_dict, tokenizer, args.max_length)
    config = AutoConfig.from_pretrained(config_path)

    logging.info("Loading model weights from %s", model_path)
    model_kwargs: Dict[str, object] = {}
    if args.device_map is not None:
        model_kwargs["device_map"] = args.device_map
    if args.fp16 and torch.cuda.is_available():
        model_kwargs["dtype"] = torch.float16
    elif args.bf16 and (
        torch.cuda.is_available()
        or getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    has_validation = "validation" in tokenized_datasets and len(tokenized_datasets["validation"]) > 0
    default_eval_strategy = "no" if args.no_eval or not has_validation else "epoch"
    effective_eval_strategy = args.eval_strategy or args.evaluation_strategy or default_eval_strategy
    if args.no_eval or not has_validation:
        effective_eval_strategy = "no"

    eval_steps = args.eval_steps or args.logging_steps if effective_eval_strategy == "steps" else None
    save_steps = args.save_steps if args.save_strategy == "steps" else None

    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_steps": args.warmup_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "report_to": ["tensorboard"],
        "save_strategy": args.save_strategy,
        "eval_strategy": effective_eval_strategy,
        "evaluation_strategy": effective_eval_strategy,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    if eval_steps is not None:
        training_kwargs["eval_steps"] = eval_steps
    if save_steps is not None:
        training_kwargs["save_steps"] = save_steps

    try:
        signature_params = inspect.signature(TrainingArguments.__init__).parameters
    except (TypeError, ValueError):
        signature_params = {}

    valid_params = set(signature_params.keys())
    eval_strategy_supported = "eval_strategy" in valid_params
    evaluation_strategy_supported = "evaluation_strategy" in valid_params

    if not eval_strategy_supported:
        training_kwargs.pop("eval_strategy", None)

    if evaluation_strategy_supported:
        strategy_for_load_best = training_kwargs["evaluation_strategy"]
    else:
        strategy_for_load_best = effective_eval_strategy
        removed_strategy = training_kwargs.pop("evaluation_strategy", None)
        if removed_strategy and removed_strategy != "no":
            logging.info(
                "TrainingArguments lacks evaluation_strategy; enabling legacy evaluation hooks."
            )
            if "evaluate_during_training" in valid_params:
                training_kwargs["evaluate_during_training"] = True
            if eval_steps is not None and "eval_steps" in valid_params:
                training_kwargs["eval_steps"] = eval_steps

    if "save_strategy" not in valid_params:
        training_kwargs.pop("save_strategy", None)
        if save_steps is not None and "save_steps" not in valid_params:
            training_kwargs.pop("save_steps", None)

    if "save_steps" not in valid_params:
        training_kwargs.pop("save_steps", None)
    if "eval_steps" not in valid_params:
        training_kwargs.pop("eval_steps", None)

    if "load_best_model_at_end" not in valid_params:
        if args.load_best_model_at_end:
            logging.info(
                "TrainingArguments does not support load_best_model_at_end; disabling."
            )
        training_kwargs.pop("load_best_model_at_end", None)
        training_kwargs.pop("metric_for_best_model", None)
        training_kwargs.pop("greater_is_better", None)
    elif strategy_for_load_best == "no" and training_kwargs.get("load_best_model_at_end"):
        logging.info(
            "Disabling load_best_model_at_end because evaluation strategy is 'no'."
        )
        training_kwargs["load_best_model_at_end"] = False
        training_kwargs.pop("metric_for_best_model", None)
        training_kwargs.pop("greater_is_better", None)

    for optional_key in ("report_to", "push_to_hub", "hub_model_id", "bf16"):
        if optional_key not in valid_params:
            training_kwargs.pop(optional_key, None)

    filtered_training_kwargs = {
        key: value for key, value in training_kwargs.items() if key in valid_params
    }
    dropped_keys = sorted(set(training_kwargs.keys()) - set(filtered_training_kwargs.keys()))
    if dropped_keys:
        logging.info(
            "TrainingArguments dropping unsupported parameters: %s",
            ", ".join(dropped_keys),
        )

    if "do_eval" in valid_params:
        filtered_training_kwargs.setdefault("do_eval", effective_eval_strategy != "no")

    training_args = TrainingArguments(**filtered_training_kwargs)

    collator = CompletionDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        data_collator=collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    if "train_loss" in metrics:
        metrics["train_perplexity"] = float(torch.exp(torch.tensor(metrics["train_loss"])))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if effective_eval_strategy != "no" and trainer.eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            eval_metrics["perplexity"] = float(torch.exp(torch.tensor(eval_metrics["eval_loss"])))
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
