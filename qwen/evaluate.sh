#!/bin/bash
# Evaluate a fine-tuned Qwen model on Recipe-MPR dataset

set -e

# Default configuration
MODEL_PATH="${1:-~/models/hub/qwen2.5-7b-recipe-mpr-lora}"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH="../data/500QA.json"
NUM_EXAMPLES=5

echo "=========================================="
echo "Qwen Recipe-MPR Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $DATASET_PATH"
echo "  Examples to show: $NUM_EXAMPLES"
echo ""
echo "Starting evaluation..."
echo ""

python evaluate_qwen_recipe_mpr.py \
    --model-path "$MODEL_PATH" \
    --base-model "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --num-examples $NUM_EXAMPLES \
    --save-results "${MODEL_PATH}/eval_results.json"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${MODEL_PATH}/eval_results.json"
echo "=========================================="
