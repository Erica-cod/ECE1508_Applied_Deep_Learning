#!/usr/bin/env python
"""
K-Fold Cross-Validation for BERT variants on Recipe-MPR.

Proper methodology:
1. Split data into K folds
2. For each fold: use 1 fold as test, 1 as val, rest as train
3. Augment ONLY training portion with negation oversampling
4. Train and evaluate
5. Report mean ± std accuracy across folds
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class RecipeExample:
    question: str
    choices: List[str]
    label: int
    query_types: Dict


def parse_args():
    parser = argparse.ArgumentParser(description="K-Fold CV for BERT on Recipe-MPR")

    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--dataset-path", type=str, default="../data/500QA.json")
    parser.add_argument("--output-dir", type=str, default="~/models/hub/kfold-cv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--negation-multiplier", type=int, default=3)

    # Training hyperparameters
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-train-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir).expanduser()
    return args


def load_raw_data(path: str) -> List[Dict]:
    """Load raw JSON data."""
    with open(path, 'r') as f:
        return json.load(f)


def process_example(item: Dict, rng: np.random.RandomState, shuffle: bool = True) -> RecipeExample:
    """Process a single example with optional choice shuffling."""
    options_items = list(item["options"].items())
    option_texts = [text for _, text in options_items]

    answer_id = item["answer"]
    correct_idx = next(
        i for i, (option_id, _) in enumerate(options_items)
        if option_id == answer_id
    )

    if shuffle:
        shuffled_indices = rng.permutation(len(option_texts))
        shuffled_option_texts = [option_texts[i] for i in shuffled_indices]
        shuffled_label = int(np.where(shuffled_indices == correct_idx)[0][0])
    else:
        shuffled_option_texts = option_texts
        shuffled_label = correct_idx

    return RecipeExample(
        question=item["query"].strip(),
        choices=shuffled_option_texts,
        label=shuffled_label,
        query_types=item.get("query_type", {})
    )


def augment_training_data(examples: List[RecipeExample], multiplier: int = 3) -> List[RecipeExample]:
    """Augment by oversampling negated examples."""
    augmented = []
    for ex in examples:
        augmented.append(ex)
        if ex.query_types.get("Negated", 0) == 1:
            for _ in range(multiplier - 1):
                augmented.append(ex)
    return augmented


def examples_to_dataset(examples: List[RecipeExample]) -> Dataset:
    return Dataset.from_list([
        {"question": ex.question, "choices": ex.choices, "label": ex.label}
        for ex in examples
    ])


def preprocess_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    first_sentences = []
    second_sentences = []

    for question, choices in zip(examples["question"], examples["choices"]):
        first_sentences.extend([question] * len(choices))
        second_sentences.extend(choices)

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    num_choices = [len(choices) for choices in examples["choices"]]
    result = {key: [] for key in tokenized.keys()}

    offset = 0
    for n_choices in num_choices:
        for key in tokenized.keys():
            result[key].append(tokenized[key][offset:offset + n_choices])
        offset += n_choices

    result["labels"] = examples["label"]
    return result


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_fold(
    fold_idx: int,
    train_examples: List[RecipeExample],
    val_examples: List[RecipeExample],
    test_examples: List[RecipeExample],
    args,
    tokenizer,
) -> Tuple[float, float, Dict]:
    """Train and evaluate a single fold."""

    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx + 1}/{args.n_folds}")
    logger.info(f"{'='*60}")

    # Augment training data
    train_augmented = augment_training_data(train_examples, args.negation_multiplier)

    logger.info(f"Train: {len(train_examples)} -> {len(train_augmented)} (after augmentation)")
    logger.info(f"Val: {len(val_examples)}, Test: {len(test_examples)}")

    # Create datasets
    dataset = DatasetDict({
        "train": examples_to_dataset(train_augmented),
        "eval": examples_to_dataset(val_examples),
        "test": examples_to_dataset(test_examples),
    })

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Load fresh model for each fold
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name,
        cache_dir=Path("~/models/hub").expanduser(),
    )

    # Training arguments
    fold_output_dir = args.output_dir / f"fold_{fold_idx}"
    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        label_smoothing_factor=args.label_smoothing,
        seed=args.seed,
        logging_steps=50,
        report_to="none",  # Disable wandb etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate on validation
    val_metrics = trainer.evaluate()
    val_acc = val_metrics["eval_accuracy"]

    # Evaluate on test
    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    test_acc = test_metrics["eval_accuracy"]

    # Detailed test evaluation by query type
    test_preds = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(test_preds.predictions, axis=-1)

    type_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for i, ex in enumerate(test_examples):
        is_correct = predictions[i] == ex.label
        for qtype, val in ex.query_types.items():
            if val == 1:
                type_results[qtype]["total"] += 1
                if is_correct:
                    type_results[qtype]["correct"] += 1

    logger.info(f"Fold {fold_idx + 1} Results:")
    logger.info(f"  Val Accuracy:  {val_acc:.2%}")
    logger.info(f"  Test Accuracy: {test_acc:.2%}")

    return val_acc, test_acc, dict(type_results)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model_name}")
    logger.info(f"K-Fold CV with {args.n_folds} folds")
    logger.info(f"Negation augmentation: {args.negation_multiplier}x")

    # Load data
    raw_data = load_raw_data(args.dataset_path)
    logger.info(f"Loaded {len(raw_data)} examples")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=Path("~/models/hub").expanduser(),
    )

    # K-Fold split
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    indices = np.arange(len(raw_data))

    fold_results = []
    all_type_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        # Further split train_val into train and val (90/10)
        rng = np.random.RandomState(args.seed + fold_idx)
        rng.shuffle(train_val_idx)
        val_size = len(train_val_idx) // 10
        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]

        # Process examples with fold-specific shuffling
        fold_rng = np.random.RandomState(args.seed + fold_idx)

        train_examples = [process_example(raw_data[i], fold_rng) for i in train_idx]
        val_examples = [process_example(raw_data[i], fold_rng) for i in val_idx]
        test_examples = [process_example(raw_data[i], fold_rng) for i in test_idx]

        val_acc, test_acc, type_results = train_fold(
            fold_idx, train_examples, val_examples, test_examples, args, tokenizer
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "type_results": type_results,
        })

        # Aggregate type results
        for qtype, res in type_results.items():
            all_type_results[qtype]["correct"] += res["correct"]
            all_type_results[qtype]["total"] += res["total"]

    # Summary
    val_accs = [r["val_accuracy"] for r in fold_results]
    test_accs = [r["test_accuracy"] for r in fold_results]

    logger.info("\n" + "="*60)
    logger.info("K-FOLD CROSS-VALIDATION RESULTS")
    logger.info("="*60)

    logger.info(f"\nPer-Fold Results:")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']}: Val={r['val_accuracy']:.2%}, Test={r['test_accuracy']:.2%}")

    logger.info(f"\nOverall Results:")
    logger.info(f"  Validation: {np.mean(val_accs):.2%} ± {np.std(val_accs):.2%}")
    logger.info(f"  Test:       {np.mean(test_accs):.2%} ± {np.std(test_accs):.2%}")

    logger.info(f"\nAccuracy by Query Type (aggregated across folds):")
    for qtype, res in sorted(all_type_results.items()):
        if res["total"] > 0:
            acc = res["correct"] / res["total"]
            logger.info(f"  {qtype:15s}: {acc:.2%} ({res['correct']}/{res['total']})")

    # Save results
    results_summary = {
        "model": args.model_name,
        "n_folds": args.n_folds,
        "negation_multiplier": args.negation_multiplier,
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "val_accuracy_mean": float(np.mean(val_accs)),
        "val_accuracy_std": float(np.std(val_accs)),
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "fold_results": fold_results,
        "type_results": {k: dict(v) for k, v in all_type_results.items()},
    }

    with open(args.output_dir / "cv_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir / 'cv_results.json'}")


if __name__ == "__main__":
    main()
