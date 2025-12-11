#!/bin/bash
# Evaluate fine-tuned Qwen2.5-7B-Instruct on held-out test set
# Uses test_split.json from training for proper held-out evaluation

set -e

echo "=========================================="
echo "Evaluating Qwen2.5-7B on HELD-OUT Test Set"
echo "=========================================="
echo ""

MODEL_PATH="~/models/hub/qwen2.5-7b-recipe-mpr-lora"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH="../data/500QA.json"
RESULTS_PATH="$MODEL_PATH/results.json"

echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $DATASET_PATH"
echo "  Using: --use-test-split (50 held-out examples)"
echo ""
echo "Starting evaluation..."
echo ""

python evaluate_qwen_recipe_mpr.py \
    --model-path "$MODEL_PATH" \
    --base-model "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --use-test-split \
    --save-results "$RESULTS_PATH" \
    --num-examples 5 \
    --seed 42

echo ""
echo "Evaluation complete!"
echo "Results saved to: $RESULTS_PATH"
