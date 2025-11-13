#!/usr/bin/env python
"""
Evaluate a fine-tuned DistilBERT model on the Recipe-MPR dataset.

This script loads a fine-tuned model and evaluates it on the full 500QA dataset,
providing overall accuracy, per-query-type breakdown, and example predictions.

Usage:
    python evaluate_distilbert_recipe_mpr.py --model-path ~/models/hub/distilbert-finetuned-recipe-mpr
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned DistilBERT on Recipe-MPR dataset"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/500QA.json",
        help="Path to the Recipe-MPR JSON dataset (default: data/500QA.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of correct/incorrect examples to show (default: 5)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Optional path to save detailed results as JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for example selection (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available, else cpu)",
    )

    args = parser.parse_args()
    args.model_path = Path(args.model_path).expanduser()
    args.dataset_path = Path(args.dataset_path)

    return args


def load_dataset(dataset_path: Path) -> List[Dict]:
    """Load Recipe-MPR dataset from JSON file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def preprocess_example(
    question: str,
    choices: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict:
    """
    Preprocess a single example for the model.

    Args:
        question: The query string
        choices: List of answer choice strings
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs
    """
    # Create pairs of (question, choice) for each choice
    first_sentences = [question] * len(choices)
    second_sentences = choices

    # Tokenize all pairs
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    return tokenized


def evaluate_model(
    model: AutoModelForMultipleChoice,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    args: argparse.Namespace,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate model on dataset.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        dataset: List of examples
        args: Command-line arguments

    Returns:
        Tuple of (metrics dict, detailed predictions list)
    """
    model.eval()
    model.to(args.device)

    all_predictions = []
    correct_total = 0

    # Track accuracy by query type
    query_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    logger.info(f"Evaluating on {len(dataset)} examples...")

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating"):
            # Extract data
            question = item["query"].strip()
            options_items = list(item["options"].items())
            option_texts = [text for _, text in options_items]

            # Find correct answer index
            answer_id = item["answer"]
            correct_idx = next(
                i for i, (option_id, _) in enumerate(options_items)
                if option_id == answer_id
            )

            # Preprocess and get model prediction
            inputs = preprocess_example(question, option_texts, tokenizer, args.max_length)
            inputs = {k: v.unsqueeze(0).to(args.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=-1).item()

            # Check if correct
            is_correct = (pred_idx == correct_idx)
            if is_correct:
                correct_total += 1

            # Track by query type
            for query_type, value in item["query_type"].items():
                if value == 1:
                    query_type_stats[query_type]["total"] += 1
                    if is_correct:
                        query_type_stats[query_type]["correct"] += 1

            # Store detailed prediction
            all_predictions.append({
                "question": question,
                "choices": option_texts,
                "correct_idx": correct_idx,
                "predicted_idx": pred_idx,
                "is_correct": is_correct,
                "query_types": [k for k, v in item["query_type"].items() if v == 1],
                "correctness_explanation": item.get("correctness_explanation", {}),
            })

    # Calculate overall accuracy
    overall_accuracy = correct_total / len(dataset) * 100

    # Calculate per-query-type accuracy
    query_type_accuracy = {}
    for qtype, stats in query_type_stats.items():
        if stats["total"] > 0:
            query_type_accuracy[qtype] = (stats["correct"] / stats["total"]) * 100
        else:
            query_type_accuracy[qtype] = 0.0

    metrics = {
        "overall_accuracy": overall_accuracy,
        "correct": correct_total,
        "total": len(dataset),
        "query_type_accuracy": query_type_accuracy,
        "query_type_counts": {k: v["total"] for k, v in query_type_stats.items()},
    }

    return metrics, all_predictions


def print_results(metrics: Dict, predictions: List[Dict], args: argparse.Namespace):
    """Print evaluation results to console."""

    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}RECIPE-MPR EVALUATION RESULTS{Colors.END}")
    print("=" * 80)

    # Overall accuracy
    accuracy = metrics["overall_accuracy"]
    goal = 70.0

    print(f"\n{Colors.BOLD}Overall Performance:{Colors.END}")
    print(f"  Accuracy: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")

    if accuracy >= goal:
        print(f"  {Colors.GREEN}✓ Goal achieved! (target: {goal}%){Colors.END}")
    else:
        gap = goal - accuracy
        print(f"  {Colors.RED}✗ Below goal by {gap:.2f}% (target: {goal}%){Colors.END}")

    # Per-query-type breakdown
    print(f"\n{Colors.BOLD}Accuracy by Query Type:{Colors.END}")
    print(f"  {'Type':<15} {'Accuracy':<12} {'Count':<10}")
    print(f"  {'-' * 40}")

    query_types_sorted = sorted(metrics["query_type_accuracy"].items(),
                                 key=lambda x: x[1], reverse=True)

    for qtype, acc in query_types_sorted:
        count = metrics["query_type_counts"][qtype]
        color = Colors.GREEN if acc >= goal else Colors.YELLOW if acc >= goal - 10 else Colors.RED
        print(f"  {qtype:<15} {color}{acc:>6.2f}%{Colors.END}      {count:<10}")

    # Example predictions
    print(f"\n{Colors.BOLD}Example Predictions:{Colors.END}")

    # Get correct and incorrect examples
    correct_preds = [p for p in predictions if p["is_correct"]]
    incorrect_preds = [p for p in predictions if not p["is_correct"]]

    random.seed(args.seed)

    # Show correct examples
    num_show = min(args.num_examples, len(correct_preds))
    if num_show > 0:
        print(f"\n{Colors.GREEN}Correct Predictions (showing {num_show}):{Colors.END}")
        for i, pred in enumerate(random.sample(correct_preds, num_show), 1):
            print(f"\n  {i}. Question: {pred['question'][:100]}...")
            print(f"     Answer: {pred['choices'][pred['correct_idx']][:80]}")
            print(f"     Query types: {', '.join(pred['query_types'])}")

    # Show incorrect examples
    num_show = min(args.num_examples, len(incorrect_preds))
    if num_show > 0:
        print(f"\n{Colors.RED}Incorrect Predictions (showing {num_show}):{Colors.END}")
        for i, pred in enumerate(random.sample(incorrect_preds, num_show), 1):
            print(f"\n  {i}. Question: {pred['question'][:100]}...")
            print(f"     {Colors.RED}Predicted: {pred['choices'][pred['predicted_idx']][:80]}{Colors.END}")
            print(f"     {Colors.GREEN}Correct:   {pred['choices'][pred['correct_idx']][:80]}{Colors.END}")
            print(f"     Query types: {', '.join(pred['query_types'])}")

    print("\n" + "=" * 80 + "\n")


def save_results(metrics: Dict, predictions: List[Dict], output_path: Path):
    """Save detailed results to JSON file."""
    results = {
        "metrics": metrics,
        "predictions": predictions,
    }

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Check model path exists
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at: {args.model_path}")

    logger.info(f"Loading model from: {args.model_path}")
    logger.info(f"Using device: {args.device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_path)

    # Load dataset
    dataset = load_dataset(args.dataset_path)

    # Evaluate
    metrics, predictions = evaluate_model(model, tokenizer, dataset, args)

    # Print results
    print_results(metrics, predictions, args)

    # Save results if requested
    if args.save_results:
        save_results(metrics, predictions, Path(args.save_results))

    # Return success/failure based on goal
    goal_met = metrics["overall_accuracy"] >= 70.0
    return 0 if goal_met else 1


if __name__ == "__main__":
    exit(main())
