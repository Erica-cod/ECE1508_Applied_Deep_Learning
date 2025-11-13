#!/bin/bash
# Evaluate the aggressive fine-tuned Qwen model

set -e

MODEL_PATH="~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH="../data/500QA.json"

echo "=========================================="
echo "Evaluating AGGRESSIVE Qwen Model"
echo "=========================================="
echo ""

python evaluate_qwen_recipe_mpr.py \
    --model-path "$MODEL_PATH" \
    --base-model "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --num-examples 5 \
    --save-results "${MODEL_PATH}/eval_results.json"

echo ""
echo "Done!"
