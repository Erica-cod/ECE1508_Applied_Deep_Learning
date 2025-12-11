#!/bin/bash
# Evaluate fine-tuned Qwen3-8B-Instruct model
# Uses the test split from training (80/10/10)

set -e

echo "=========================================="
echo "Evaluating Qwen3-8B"
echo "=========================================="
echo ""

MODEL_PATH="~/models/hub/qwen3-8b-recipe-mpr-lora"
BASE_MODEL="Qwen/Qwen3-8B"
DATASET_PATH="../data/500QA.json"
RESULTS_PATH="$MODEL_PATH/results.json"

echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $DATASET_PATH"
echo "  Using test split from training (10% of data)"
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
echo ""
