#!/usr/bin/env python
"""
Evaluate the BASE Qwen model (without fine-tuning) on Recipe-MPR.

This provides a baseline to compare against the fine-tuned model.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ANSI colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BASE Qwen (no fine-tuning) on Recipe-MPR"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../data/500QA.json",
        help="Path to Recipe-MPR dataset",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to show",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="base_model_eval_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    args.dataset_path = Path(args.dataset_path)

    return args


def load_dataset(dataset_path: Path) -> List[Dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def format_prompt(question: str, choices: List[str]) -> str:
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


def extract_answer(text: str) -> str:
    text = text.strip()
    match = re.match(r'^([A-E])', text)
    if match:
        return match.group(1)
    match = re.search(r'([A-E])', text[:10])
    if match:
        return match.group(1)
    return 'X'


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    args: argparse.Namespace,
) -> Tuple[Dict, List[Dict]]:
    model.eval()

    all_predictions = []
    correct_total = 0
    query_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    logger.info(f"Evaluating BASE model on {len(dataset)} examples...")

    choice_letters = ['A', 'B', 'C', 'D', 'E']

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating"):
            question = item["query"].strip()
            options_items = list(item["options"].items())
            option_texts = [text for _, text in options_items]

            answer_id = item["answer"]
            correct_idx = next(
                i for i, (option_id, _) in enumerate(options_items)
                if option_id == answer_id
            )
            correct_letter = choice_letters[correct_idx]

            prompt = format_prompt(question, option_texts)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            predicted_letter = extract_answer(generated_text)

            try:
                pred_idx = choice_letters.index(predicted_letter)
            except ValueError:
                pred_idx = -1

            is_correct = (pred_idx == correct_idx)
            if is_correct:
                correct_total += 1

            for query_type, value in item["query_type"].items():
                if value == 1:
                    query_type_stats[query_type]["total"] += 1
                    if is_correct:
                        query_type_stats[query_type]["correct"] += 1

            all_predictions.append({
                "question": question,
                "choices": option_texts,
                "correct_idx": correct_idx,
                "correct_letter": correct_letter,
                "predicted_idx": pred_idx,
                "predicted_letter": predicted_letter,
                "generated_text": generated_text,
                "is_correct": is_correct,
                "query_types": [k for k, v in item["query_type"].items() if v == 1],
                "correctness_explanation": item.get("correctness_explanation", {}),
            })

    overall_accuracy = correct_total / len(dataset) * 100

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
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}BASE QWEN MODEL EVALUATION (NO FINE-TUNING){Colors.END}")
    print("=" * 80)

    accuracy = metrics["overall_accuracy"]
    goal = 75.0

    print(f"\n{Colors.BOLD}Overall Performance:{Colors.END}")
    print(f"  Accuracy: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")

    if accuracy >= goal:
        print(f"  {Colors.GREEN}✓ Exceeds goal (target: {goal}%){Colors.END}")
    else:
        gap = goal - accuracy
        print(f"  {Colors.YELLOW}Below goal by {gap:.2f}% (target: {goal}%){Colors.END}")

    print(f"\n{Colors.BOLD}Accuracy by Query Type:{Colors.END}")
    print(f"  {'Type':<15} {'Accuracy':<12} {'Count':<10}")
    print(f"  {'-' * 40}")

    query_types_sorted = sorted(
        metrics["query_type_accuracy"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for qtype, acc in query_types_sorted:
        count = metrics["query_type_counts"][qtype]
        color = Colors.GREEN if acc >= goal else Colors.YELLOW if acc >= goal - 10 else Colors.RED
        print(f"  {qtype:<15} {color}{acc:>6.2f}%{Colors.END}      {count:<10}")

    print(f"\n{Colors.BOLD}Example Predictions:{Colors.END}")

    correct_preds = [p for p in predictions if p["is_correct"]]
    incorrect_preds = [p for p in predictions if not p["is_correct"]]

    random.seed(args.seed)

    num_show = min(args.num_examples, len(correct_preds))
    if num_show > 0:
        print(f"\n{Colors.GREEN}Correct Predictions (showing {num_show}):{Colors.END}")
        for i, pred in enumerate(random.sample(correct_preds, num_show), 1):
            print(f"\n  {i}. Q: {pred['question'][:100]}...")
            print(f"     Model answered: {pred['predicted_letter']}")
            print(f"     Answer: {pred['choices'][pred['correct_idx']][:80]}")
            print(f"     Query types: {', '.join(pred['query_types'])}")

    num_show = min(args.num_examples, len(incorrect_preds))
    if num_show > 0:
        print(f"\n{Colors.RED}Incorrect Predictions (showing {num_show}):{Colors.END}")
        for i, pred in enumerate(random.sample(incorrect_preds, num_show), 1):
            print(f"\n  {i}. Q: {pred['question'][:100]}...")
            print(f"     {Colors.RED}Model: {pred['predicted_letter']} - {pred['choices'][pred['predicted_idx']][:70] if pred['predicted_idx'] >= 0 else 'INVALID'}{Colors.END}")
            print(f"     {Colors.GREEN}Correct: {pred['correct_letter']} - {pred['choices'][pred['correct_idx']][:70]}{Colors.END}")
            print(f"     Generated: '{pred['generated_text']}'")
            print(f"     Query types: {', '.join(pred['query_types'])}")

    print("\n" + "=" * 80 + "\n")


def save_results(metrics: Dict, predictions: List[Dict], output_path: Path):
    results = {
        "metrics": metrics,
        "predictions": predictions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main():
    args = parse_args()

    logger.info(f"Loading BASE model: {args.base_model}")
    logger.info("NOTE: This is the un-fine-tuned model for baseline comparison")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Model loaded on device: {model.device}")

    dataset = load_dataset(args.dataset_path)

    metrics, predictions = evaluate_model(model, tokenizer, dataset, args)

    print_results(metrics, predictions, args)

    save_results(metrics, predictions, Path(args.save_results))

    return 0


if __name__ == "__main__":
    exit(main())
