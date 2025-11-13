#!/usr/bin/env python
"""
Fine-tune Qwen2.5-7B-Instruct on Recipe-MPR using LoRA.

This script uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to adapt
Qwen2.5 for multiple-choice recipe selection tasks.

Usage:
    python finetune_qwen_recipe_mpr.py --num-train-epochs 5
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5 on Recipe-MPR with LoRA"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Qwen model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/500QA.json",
        help="Path to Recipe-MPR dataset (default: data/500QA.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/models/hub/qwen2.5-7b-recipe-mpr-lora",
        help="Output directory for model (default: ~/models/hub/qwen2.5-7b-recipe-mpr-lora)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/models/hub",
        help="Cache directory (default: ~/models/hub)",
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Eval split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4)",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Eval batch size per device (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps (default: 100)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging steps (default: 10)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluation steps (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 mixed precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 mixed precision",
    )

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir).expanduser()
    args.cache_dir = Path(args.cache_dir).expanduser()
    args.dataset_path = Path(args.dataset_path)

    return args


def load_recipe_dataset(dataset_path: Path) -> List[Dict]:
    """Load Recipe-MPR dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def format_prompt(question: str, choices: List[str]) -> str:
    """
    Format question and choices as a prompt for Qwen.

    Args:
        question: The recipe query
        choices: List of 5 answer options

    Returns:
        Formatted prompt string
    """
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    options_text = "\n".join([
        f"{letter}) {choice}"
        for letter, choice in zip(choice_letters, choices)
    ])

    prompt = f"""Given the following recipe question and options, select the best answer.

Question: {question}

Options:
{options_text}

Answer:"""

    return prompt


def create_training_example(item: Dict) -> Dict:
    """
    Create a training example from a dataset item.

    Args:
        item: Raw dataset item

    Returns:
        Dictionary with 'text' key containing full prompt + answer
    """
    # Extract options in order
    options_items = list(item["options"].items())
    option_texts = [text for _, text in options_items]

    # Find correct answer
    answer_id = item["answer"]
    answer_idx = next(
        i for i, (option_id, _) in enumerate(options_items)
        if option_id == answer_id
    )

    # Format prompt
    prompt = format_prompt(item["query"].strip(), option_texts)

    # Add answer (A, B, C, D, or E)
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    answer_letter = choice_letters[answer_idx]

    full_text = f"{prompt} {answer_letter}"

    return {
        "text": full_text,
        "question": item["query"].strip(),
        "answer_idx": answer_idx,
    }


def create_dataset_splits(
    raw_data: List[Dict],
    eval_ratio: float,
    seed: int
) -> DatasetDict:
    """Create train/eval splits."""
    # Convert to training format
    examples = [create_training_example(item) for item in raw_data]

    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle(seed=seed)

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)

    logger.info(f"Train size: {len(split['train'])}, Eval size: {len(split['test'])}")

    return DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })


def tokenize_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Tokenize examples for training."""
    # Tokenize the full text (prompt + answer)
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels are the same as input_ids
    outputs["labels"] = outputs["input_ids"].copy()

    return outputs


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using LoRA with r={args.lora_r}, alpha={args.lora_alpha}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    logger.info("Loading dataset...")
    raw_data = load_recipe_dataset(args.dataset_path)
    dataset = create_dataset_splits(raw_data, args.eval_ratio, args.seed)

    # Tokenize
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=args.bf16,
        fp16=args.fp16,
        save_total_limit=2,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save model and tokenizer
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    logger.info("Training complete!")
    logger.info(f"Final eval loss: {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
