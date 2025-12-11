#!/usr/bin/env python
"""
Evaluate baseline (non-fine-tuned) BERT-family models on Recipe-MPR dataset.
This tests the model with a randomly initialized classification head (no training).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on Recipe-MPR")
    parser.add_argument("--model-name", type=str, default="roberta-base",
                       help="HuggingFace model name (default: roberta-base)")
    parser.add_argument("--dataset-path", type=str, default="../data/500QA.json",
                       help="Path to dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\nEvaluating baseline: {args.model_name}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # Load tokenizer and model with multiple choice head
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model configured for multiple choice (randomly initialized classifier)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True  # classifier head will be randomly initialized
    )
    model.to(args.device)
    model.eval()

    # Load dataset
    dataset_path = Path(args.dataset_path)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} examples")

    # Set random seed
    rng = np.random.RandomState(args.seed)

    correct_total = 0
    query_type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating"):
            question = item["query"].strip()
            options_items = list(item["options"].items())
            option_texts = [text for _, text in options_items]

            # Find correct answer index
            answer_id = item["answer"]
            correct_idx = next(
                i for i, (option_id, _) in enumerate(options_items)
                if option_id == answer_id
            )

            # Shuffle choices (same as training)
            shuffled_indices = rng.permutation(len(option_texts))
            shuffled_option_texts = [option_texts[i] for i in shuffled_indices]
            shuffled_correct_idx = int(np.where(shuffled_indices == correct_idx)[0][0])

            # Tokenize
            first_sentences = [question] * len(shuffled_option_texts)
            second_sentences = shuffled_option_texts

            inputs = tokenizer(
                first_sentences,
                second_sentences,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.unsqueeze(0).to(args.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=-1).item()

            is_correct = (pred_idx == shuffled_correct_idx)
            if is_correct:
                correct_total += 1

            # Track by query type
            for query_type, value in item["query_type"].items():
                if value == 1:
                    query_type_stats[query_type]["total"] += 1
                    if is_correct:
                        query_type_stats[query_type]["correct"] += 1

    # Results
    overall_accuracy = correct_total / len(dataset) * 100

    print("\n" + "=" * 50)
    print(f"BASELINE RESULTS: {args.model_name}")
    print("=" * 50)
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({correct_total}/{len(dataset)})")
    print(f"\nRandom chance: 20.00% (1/5 choices)")

    print(f"\nAccuracy by Query Type:")
    print(f"  {'Type':<15} {'Accuracy':<12} {'Count':<10}")
    print(f"  {'-' * 40}")

    for qtype, stats in sorted(query_type_stats.items(),
                                key=lambda x: x[1]["correct"]/max(x[1]["total"],1),
                                reverse=True):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {qtype:<15} {acc:>6.2f}%      {stats['total']:<10}")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
