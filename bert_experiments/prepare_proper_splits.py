#!/usr/bin/env python
"""
Prepare proper train/val/test splits for Recipe-MPR experiments.

Methodology:
1. Split original 500QA.json into 80/10/10 (400 train, 50 val, 50 test) FIRST
2. Apply negation augmentation ONLY to the training portion
3. Save all splits for reproducible experiments

This ensures no data leakage between train and test sets.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split dataset into train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(data))

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data, list(train_indices), list(val_indices), list(test_indices)

def augment_negation(data, multiplier=3):
    """Augment dataset by duplicating negated examples."""
    augmented = []
    stats = defaultdict(int)

    for example in data:
        augmented.append(example)
        stats['total'] += 1

        # Check if negated
        if example.get('query_type', {}).get('Negated', 0) == 1:
            stats['negated'] += 1
            # Add extra copies
            for _ in range(multiplier - 1):
                augmented.append(example.copy())
                stats['negated_duplicates'] += 1

    return augmented, stats

def main():
    # Paths
    input_path = Path("../data/500QA.json")
    output_dir = Path("../data/proper_splits")
    output_dir.mkdir(exist_ok=True)

    # Load original data
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Original dataset: {len(data)} examples")

    # Count negated in original
    negated_count = sum(1 for ex in data if ex.get('query_type', {}).get('Negated', 0) == 1)
    print(f"Negated examples in original: {negated_count}")

    # Split FIRST (before any augmentation)
    train_data, val_data, test_data, train_idx, val_idx, test_idx = split_dataset(
        data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    print(f"\nSplit (before augmentation):")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    # Count negated in each split
    train_negated = sum(1 for ex in train_data if ex.get('query_type', {}).get('Negated', 0) == 1)
    val_negated = sum(1 for ex in val_data if ex.get('query_type', {}).get('Negated', 0) == 1)
    test_negated = sum(1 for ex in test_data if ex.get('query_type', {}).get('Negated', 0) == 1)

    print(f"\nNegated examples per split:")
    print(f"  Train: {train_negated}")
    print(f"  Val:   {val_negated}")
    print(f"  Test:  {test_negated}")

    # Augment ONLY the training data
    train_augmented, aug_stats = augment_negation(train_data, multiplier=3)

    print(f"\nAfter 3x negation augmentation (train only):")
    print(f"  Train: {len(train_augmented)} examples (+{aug_stats['negated_duplicates']} duplicates)")
    print(f"  Val:   {len(val_data)} examples (unchanged)")
    print(f"  Test:  {len(test_data)} examples (unchanged)")

    # Save splits
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / "train_augmented.json", 'w') as f:
        json.dump(train_augmented, f, indent=2)

    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    # Save metadata
    metadata = {
        "seed": 42,
        "original_size": len(data),
        "train_size": len(train_data),
        "train_augmented_size": len(train_augmented),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
        "negation_multiplier": 3,
        "train_negated_original": train_negated,
        "train_negated_after_aug": train_negated * 3,
    }

    with open(output_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved splits to {output_dir}/")
    print(f"  - train.json ({len(train_data)} examples)")
    print(f"  - train_augmented.json ({len(train_augmented)} examples)")
    print(f"  - val.json ({len(val_data)} examples)")
    print(f"  - test.json ({len(test_data)} examples)")
    print(f"  - split_metadata.json")

if __name__ == "__main__":
    main()
