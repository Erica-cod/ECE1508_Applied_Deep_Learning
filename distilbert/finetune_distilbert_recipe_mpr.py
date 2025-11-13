#!/usr/bin/env python
"""
Fine-tune DistilBERT on the Recipe-MPR 500QA dataset as a multiple-choice task.

This script loads the Recipe-MPR dataset, formats it for multiple-choice classification,
and fine-tunes a DistilBERT model using the HuggingFace Transformers library.

Usage:
    python finetune_distilbert_recipe_mpr.py --num-train-epochs 3 --per-device-train-batch-size 8
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["USE_TF"] = "0"  # Disable TensorFlow backend

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
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
    """Container for a single multiple-choice example."""
    question: str
    choices: List[str]
    label: int


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on Recipe-MPR multiple-choice dataset"
    )

    # Model and data arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name or path (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/500QA.json",
        help="Path to the Recipe-MPR JSON dataset (default: data/500QA.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/models/hub/distilbert-finetuned-recipe-mpr",
        help="Directory to save the fine-tuned model (default: ~/models/hub/distilbert-finetuned-recipe-mpr)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/models/hub",
        help="Cache directory for pretrained models (default: ~/models/hub)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (default: 0.1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length after tokenization (default: 256)",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Training batch size per device (default: 8)",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Evaluation batch size per device (default: 8)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Training options
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (fp16)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log every N steps (default: 50)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)",
    )

    args = parser.parse_args()

    # Expand paths
    args.output_dir = Path(args.output_dir).expanduser()
    args.cache_dir = Path(args.cache_dir).expanduser()
    args.dataset_path = Path(args.dataset_path)

    return args


def load_recipe_dataset(dataset_path: Path) -> List[RecipeExample]:
    """
    Load Recipe-MPR dataset from JSON file.

    Args:
        dataset_path: Path to the JSON dataset file

    Returns:
        List of RecipeExample objects
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    examples = []
    for idx, item in enumerate(raw_data):
        # Extract options as ordered list
        options_items = list(item["options"].items())
        option_texts = [text for _, text in options_items]

        # Find the index of the correct answer
        answer_id = item["answer"]
        try:
            label = next(
                i for i, (option_id, _) in enumerate(options_items)
                if option_id == answer_id
            )
        except StopIteration:
            raise ValueError(f"Answer ID '{answer_id}' not found in options for example {idx}")

        examples.append(
            RecipeExample(
                question=item["query"].strip(),
                choices=option_texts,
                label=label,
            )
        )

    logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
    return examples


def create_dataset_splits(
    examples: List[RecipeExample],
    eval_ratio: float,
    seed: int
) -> DatasetDict:
    """
    Create train/eval splits from examples.

    Args:
        examples: List of RecipeExample objects
        eval_ratio: Fraction to use for evaluation
        seed: Random seed for shuffling

    Returns:
        DatasetDict with 'train' and 'eval' splits
    """
    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_list([
        {
            "question": ex.question,
            "choices": ex.choices,
            "label": ex.label,
        }
        for ex in examples
    ])

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)

    logger.info(f"Train size: {len(split['train'])}, Eval size: {len(split['test'])}")

    return DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })


def preprocess_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int) -> Dict:
    """
    Preprocess examples for multiple-choice task.

    For each question with N choices, creates N input sequences of the form:
    [CLS] question [SEP] choice [SEP]

    Args:
        examples: Batch of examples with 'question', 'choices', 'label' keys
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs ready for the model
    """
    # Flatten questions and choices
    first_sentences = []
    second_sentences = []

    for question, choices in zip(examples["question"], examples["choices"]):
        # Repeat question for each choice
        first_sentences.extend([question] * len(choices))
        second_sentences.extend(choices)

    # Tokenize all pairs
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # Reshape back to [num_examples, num_choices, seq_length]
    num_choices = [len(choices) for choices in examples["choices"]]

    # Group by examples
    result = {key: [] for key in tokenized.keys()}
    offset = 0
    for n_choices in num_choices:
        for key in tokenized.keys():
            result[key].append(tokenized[key][offset : offset + n_choices])
        offset += n_choices

    # Add labels
    result["labels"] = examples["label"]

    return result


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute accuracy metric.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary with accuracy metric
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()

    return {"accuracy": accuracy}


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        use_fast=True,
    )

    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )

    # Load and split dataset
    examples = load_recipe_dataset(args.dataset_path)
    dataset = create_dataset_splits(examples, args.eval_ratio, args.seed)

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log and save training metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save final model
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    logger.info("Training complete!")
    logger.info(f"Final evaluation accuracy: {metrics['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()
