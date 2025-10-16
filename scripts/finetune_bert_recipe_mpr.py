#!/usr/bin/env python
"""
Fine-tune a BERT-style encoder on the Recipe-MPR 500QA dataset as a multiple-choice task.

The resulting weights and tokenizer are stored under ~/models/hub/bert-finetuned by default
so they live alongside other cached models.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForMultipleChoice,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT family model on the Recipe-MPR multiple-choice dataset."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="bert-base-uncased",
        help="Pretrained identifier or local path for the base BERT model.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(Path("~/models/hub").expanduser()),
        help="Optional Hugging Face cache directory to reuse downloaded assets.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/500QA.json",
        help="Path to the Recipe-MPR JSON dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("~/models/hub/bert-finetuned").expanduser()),
        help="Directory where the fine-tuned model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of the dataset reserved for evaluation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
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
        default=2e-5,
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
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Gradient clipping norm; if omitted no clipping is applied.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Number of steps between logging outputs.",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=("no", "steps", "epoch"),
        help="Checkpoint save strategy passed to TrainingArguments.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Limit the total number of saved checkpoints (delete older ones).",
    )
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default="epoch",
        choices=("no", "steps", "epoch"),
        help="Evaluation strategy passed to TrainingArguments (deprecated in HF>=4.41).",
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default=None,
        choices=("no", "steps", "epoch"),
        help="Preferred evaluation strategy argument for newer Transformers releases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision training.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 mixed precision training (AMP).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the model.",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        help="Track best eval metric and load it at the end of training.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the fine-tuned model to the Hugging Face Hub (requires credentials).",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Optional model id to use when pushing to the Hub.",
    )
    args = parser.parse_args()

    if args.eval_ratio <= 0 or args.eval_ratio >= 1:
        parser.error("--eval-ratio must be in the (0, 1) interval.")

    if args.eval_strategy is not None:
        args.evaluation_strategy = args.eval_strategy

    return args


@dataclass
class RecipeExample:
    """Container for a single multiple-choice entry."""

    question: str
    choices: Sequence[str]
    label: int


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_recipe_examples(path: Path) -> List[RecipeExample]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)

    examples: List[RecipeExample] = []
    for idx, item in enumerate(raw_items):
        options_items = list(item["options"].items())
        option_texts = [text for _, text in options_items]

        try:
            label = next(i for i, (option_id, _) in enumerate(options_items) if option_id == item["answer"])
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not locate answer id for example {idx}") from exc

        question = item["query"].strip()
        examples.append(RecipeExample(question=question, choices=option_texts, label=label))

    LOGGER.info("Loaded %d examples from %s", len(examples), path)
    return examples


def build_dataset(examples: Sequence[RecipeExample], eval_ratio: float, seed: int) -> DatasetDict:
    dataset = Dataset.from_list(
        [
            {
                "question": example.question,
                "choices": list(example.choices),
                "label": example.label,
            }
            for example in examples
        ]
    )

    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    LOGGER.info(
        "Dataset split sizes -> train: %d | eval: %d",
        split["train"].num_rows,
        split["test"].num_rows,
    )
    return DatasetDict(train=split["train"], eval=split["test"])


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, List]:
    num_choices = [len(choice_set) for choice_set in examples["choices"]]

    first_sentences: List[str] = []
    second_sentences: List[str] = []
    for question, choice_set in zip(examples["question"], examples["choices"]):
        first_sentences.extend([question] * len(choice_set))
        second_sentences.extend(choice_set)

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    formatted: Dict[str, List[List[int]]] = {key: [] for key in tokenized.keys()}
    offset = 0
    for choice_count in num_choices:
        for key, values in tokenized.items():
            formatted[key].append(values[offset : offset + choice_count])
        offset += choice_count

    formatted["labels"] = examples["label"]
    return formatted


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = Path(args.model_name_or_path).expanduser()
    if model_source.exists():
        if model_source.is_dir() and (model_source / "snapshots").is_dir():
            snapshots = sorted(
                (p for p in (model_source / "snapshots").iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not snapshots:
                raise FileNotFoundError(
                    f"No snapshots found under {model_source}. Specify a snapshot folder explicitly."
                )
            model_source = snapshots[0]
            LOGGER.info("Resolved snapshot %s", model_source)
        model_path = str(model_source)
        LOGGER.info("Loading base model from local path: %s", model_path)
    else:
        model_path = args.model_name_or_path
        LOGGER.info("Loading base model from pretrained identifier: %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_path,
        config=config,
        cache_dir=args.cache_dir,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = build_dataset(
        load_recipe_examples(Path(args.dataset_path)),
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    preprocess = lambda batch: preprocess_function(batch, tokenizer, args.max_length)  # noqa: E731
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    logging_steps = max(1, args.logging_steps)
    eval_strategy = getattr(args, "evaluation_strategy", "no")
    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "eval_strategy": eval_strategy,
        "evaluation_strategy": eval_strategy,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "logging_steps": logging_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "report_to": ["none"],
    }
    if args.max_grad_norm is None:
        training_kwargs.pop("max_grad_norm")
    if not args.load_best_model_at_end:
        training_kwargs.pop("metric_for_best_model")
        training_kwargs.pop("greater_is_better")

    training_signature = inspect.signature(TrainingArguments.__init__)
    valid_params = set(training_signature.parameters.keys())

    eval_strategy_supported = "eval_strategy" in valid_params
    evaluation_strategy_supported = "evaluation_strategy" in valid_params

    if not eval_strategy_supported:
        training_kwargs.pop("eval_strategy", None)

    if not evaluation_strategy_supported:
        strategy = training_kwargs.pop("evaluation_strategy", "no")
        if strategy != "no" and "evaluate_during_training" in valid_params:
            training_kwargs["evaluate_during_training"] = True
    else:
        strategy = training_kwargs["evaluation_strategy"]

    if "save_strategy" not in valid_params:
        training_kwargs.pop("save_strategy", None)

    if "load_best_model_at_end" not in valid_params:
        training_kwargs.pop("load_best_model_at_end", None)
        training_kwargs.pop("metric_for_best_model", None)
        training_kwargs.pop("greater_is_better", None)
    elif not (eval_strategy_supported or evaluation_strategy_supported) and training_kwargs.get("load_best_model_at_end"):
        LOGGER.info(
            "TrainingArguments does not support evaluation_strategy; disabling load_best_model_at_end."
        )
        training_kwargs["load_best_model_at_end"] = False
        training_kwargs.pop("metric_for_best_model", None)
        training_kwargs.pop("greater_is_better", None)

    for optional_key in ("report_to", "push_to_hub", "hub_model_id", "bf16", "save_total_limit", "max_grad_norm"):
        if optional_key not in valid_params:
            training_kwargs.pop(optional_key, None)

    filtered_kwargs = {key: value for key, value in training_kwargs.items() if key in valid_params}
    dropped_keys = sorted(set(training_kwargs.keys()) - set(filtered_kwargs.keys()))
    if dropped_keys:
        LOGGER.info("TrainingArguments dropping unsupported parameters: %s", ", ".join(dropped_keys))

    training_args = TrainingArguments(**filtered_kwargs)

    data_collator = DataCollatorForMultipleChoice(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.evaluation_strategy != "no" else None,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if args.evaluation_strategy != "no":
        metrics = trainer.evaluate(tokenized_dataset["eval"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    LOGGER.info("Saving fine-tuned model to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    LOGGER.info("All done.")


if __name__ == "__main__":
    main()
