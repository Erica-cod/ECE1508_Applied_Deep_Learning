#!/bin/bash
# Evaluate all available Qwen models

set -e

echo "=========================================="
echo "Evaluating All Qwen Models"
echo "=========================================="
echo ""

BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH="../data/500QA.json"

# List of model paths to check
MODELS=(
    "~/models/hub/qwen2.5-7b-recipe-mpr-lora:Standard"
    "~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive:Aggressive"
)

FOUND_MODELS=0
RESULTS=()

for MODEL_INFO in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_INFO%%:*}"
    MODEL_NAME="${MODEL_INFO##*:}"

    # Expand path
    EXPANDED_PATH=$(eval echo "$MODEL_PATH")

    if [ -d "$EXPANDED_PATH" ]; then
        echo "Found: $MODEL_NAME model at $MODEL_PATH"
        FOUND_MODELS=$((FOUND_MODELS + 1))

        echo "Evaluating $MODEL_NAME..."
        echo "────────────────────────────────────────"

        # Run evaluation and capture output
        OUTPUT=$(python evaluate_qwen_recipe_mpr.py \
            --model-path "$MODEL_PATH" \
            --base-model "$BASE_MODEL" \
            --dataset-path "$DATASET_PATH" \
            --num-examples 3 \
            --save-results "${MODEL_PATH}/eval_results.json" 2>&1)

        echo "$OUTPUT"

        # Extract accuracy
        ACCURACY=$(echo "$OUTPUT" | grep "Accuracy:" | head -1 | awk '{print $2}')
        RESULTS+=("$MODEL_NAME:$ACCURACY")

        echo ""
        echo ""
    fi
done

if [ $FOUND_MODELS -eq 0 ]; then
    echo "❌ No trained models found!"
    echo ""
    echo "Please train a model first:"
    echo "  ./train_qwen.sh          - Standard training"
    echo "  ./train_qwen_aggressive.sh - Aggressive training"
    exit 1
fi

echo "=========================================="
echo "Summary: All Models"
echo "=========================================="
echo ""
printf "%-20s %s\n" "Model" "Accuracy"
echo "────────────────────────────────────────"
for RESULT in "${RESULTS[@]}"; do
    MODEL="${RESULT%%:*}"
    ACC="${RESULT##*:}"
    printf "%-20s %s\n" "$MODEL" "$ACC"
done
echo ""
echo "Goal: 75.00%"
echo ""
echo "Detailed results saved to respective model directories"
