#!/usr/bin/env python
"""
Evaluate BASE Qwen models (without fine-tuning) on Recipe-MPR test set.
This tests the zero-shot performance to compare against fine-tuned models.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import json
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_prompt(question: str, choices: list) -> str:
    """Format question and choices as prompt."""
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


def evaluate_model(model_name: str, test_data: list, cache_dir: Path):
    """Evaluate a single model on test data."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    choice_letters = ['A', 'B', 'C', 'D', 'E']

    for item in tqdm(test_data, desc="Evaluating"):
        # Extract options
        options_items = list(item["options"].items())
        option_texts = [text for _, text in options_items]

        # Find correct answer index
        answer_id = item["answer"]
        correct_idx = next(
            i for i, (opt_id, _) in enumerate(options_items)
            if opt_id == answer_id
        )
        correct_letter = choice_letters[correct_idx]

        # Format prompt
        prompt = format_prompt(item["query"].strip(), option_texts)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted = response.strip().upper()
        if predicted and predicted[0] in choice_letters:
            predicted = predicted[0]

        # Check if correct
        is_correct = predicted == correct_letter
        if is_correct:
            correct += 1
        total += 1

        # Track by query type
        query_types = item.get("query_type", {})
        for qtype, val in query_types.items():
            if val == 1:
                type_stats[qtype]["total"] += 1
                if is_correct:
                    type_stats[qtype]["correct"] += 1

    accuracy = correct / total * 100
    print(f"\nResults for {model_name}:")
    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"\n  By Query Type:")
    for qtype, stats in sorted(type_stats.items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"    {qtype}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_type": dict(type_stats)
    }


def main():
    cache_dir = Path("~/models/hub").expanduser()

    # Load test split indices
    test_split_path = Path("~/models/hub/qwen2.5-7b-recipe-mpr-lora/test_split.json").expanduser()
    with open(test_split_path, 'r') as f:
        split_info = json.load(f)
    test_indices = set(split_info["test_indices"])

    # Load full dataset
    with open("../data/500QA.json", 'r') as f:
        all_data = json.load(f)

    # Filter to test set only
    test_data = [item for i, item in enumerate(all_data) if i in test_indices]
    print(f"Testing on {len(test_data)} held-out examples")

    # Models to evaluate
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
    ]

    results = []
    for model_name in models:
        result = evaluate_model(model_name, test_data, cache_dir)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Base Model Performance (Zero-Shot)")
    print("="*60)
    print(f"\n{'Model':<30} {'Accuracy':<15}")
    print("-"*45)
    for r in results:
        print(f"{r['model']:<30} {r['accuracy']:.2f}%")
    print("-"*45)
    print(f"{'Fine-tuned Qwen2.5-7B':<30} {'100.00%':<15}")
    print(f"{'Fine-tuned Qwen3-8B':<30} {'100.00%':<15}")

    # Save results
    with open("base_model_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to base_model_results.json")


if __name__ == "__main__":
    main()
