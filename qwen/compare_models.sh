#!/bin/bash
# Compare standard vs aggressive Qwen models

set -e

echo "=========================================="
echo "Comparing Qwen Models"
echo "=========================================="
echo ""

# Check if models exist
STANDARD_PATH=~/models/hub/qwen2.5-7b-recipe-mpr-lora
AGGRESSIVE_PATH=~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive

if [ ! -d "$STANDARD_PATH" ]; then
    echo "❌ Standard model not found at: $STANDARD_PATH"
    echo "Run ./train_qwen.sh first"
    exit 1
fi

if [ ! -d "$AGGRESSIVE_PATH" ]; then
    echo "❌ Aggressive model not found at: $AGGRESSIVE_PATH"
    echo "Run ./train_qwen_aggressive.sh first"
    exit 1
fi

echo "Evaluating STANDARD model..."
echo "────────────────────────────────────────"
./evaluate.sh "$STANDARD_PATH" > /tmp/standard_eval.txt 2>&1
cat /tmp/standard_eval.txt

echo ""
echo ""
echo "Evaluating AGGRESSIVE model..."
echo "────────────────────────────────────────"
./evaluate_aggressive.sh > /tmp/aggressive_eval.txt 2>&1
cat /tmp/aggressive_eval.txt

echo ""
echo "=========================================="
echo "Comparison Summary"
echo "=========================================="
echo ""

# Extract accuracies from results
STANDARD_ACC=$(grep "Accuracy:" /tmp/standard_eval.txt | head -1 | awk '{print $2}')
AGGRESSIVE_ACC=$(grep "Accuracy:" /tmp/aggressive_eval.txt | head -1 | awk '{print $2}')

echo "Standard model:   $STANDARD_ACC"
echo "Aggressive model: $AGGRESSIVE_ACC"
echo ""

# Compare to goal
echo "Goal: 75.00%"
echo ""

if [ ! -z "$STANDARD_ACC" ]; then
    echo "Standard model: $(grep -q "Goal achieved" /tmp/standard_eval.txt && echo "✅ Goal achieved" || echo "❌ Below goal")"
fi

if [ ! -z "$AGGRESSIVE_ACC" ]; then
    echo "Aggressive model: $(grep -q "Goal achieved" /tmp/aggressive_eval.txt && echo "✅ Goal achieved" || echo "❌ Below goal")"
fi

echo ""
echo "Detailed results saved to:"
echo "  - ${STANDARD_PATH}/eval_results.json"
echo "  - ${AGGRESSIVE_PATH}/eval_results.json"
