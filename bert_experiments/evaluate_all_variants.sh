#!/bin/bash
# Evaluate all trained BERT variants and compare results

# Note: Don't use set -e here because evaluation script returns 1 if accuracy < 70%
# We want to continue evaluating all models even if one doesn't meet the goal

echo "=========================================="
echo "Evaluating All BERT Variants"
echo "=========================================="
echo ""

# Model paths
MODELS=(
    "distilbert-finetuned-recipe-mpr:DistilBERT"
    "bert-base-recipe-mpr:BERT-base"
    "roberta-base-recipe-mpr:RoBERTa-base"
    "deberta-v3-base-recipe-mpr:DeBERTa-v3"
    "bert-large-recipe-mpr:BERT-large"
)

# Results storage
declare -A RESULTS
FOUND_MODELS=()

for model_info in "${MODELS[@]}"; do
    MODEL_DIR="${model_info%%:*}"
    MODEL_NAME="${model_info##*:}"
    MODEL_PATH=~/models/hub/"$MODEL_DIR"

    # Check if model exists
    if [ -d "$MODEL_PATH" ]; then
        echo "Found: $MODEL_NAME at $MODEL_PATH"
        FOUND_MODELS+=("$MODEL_NAME:$MODEL_PATH")
    fi
done

if [ ${#FOUND_MODELS[@]} -eq 0 ]; then
    echo "❌ No trained models found!"
    echo ""
    echo "Please train models first:"
    echo "  ./train_bert_variant.sh bert-base-uncased"
    echo "  ./train_bert_variant.sh roberta-base"
    echo "  etc."
    exit 1
fi

echo ""
echo "Evaluating ${#FOUND_MODELS[@]} model(s)..."
echo ""

# Evaluate each model
for model_info in "${FOUND_MODELS[@]}"; do
    MODEL_NAME="${model_info%%:*}"
    MODEL_PATH="${model_info##*:}"

    echo "=========================================="
    echo "Evaluating: $MODEL_NAME"
    echo "=========================================="
    echo ""

    # Run evaluation and capture output
    OUTPUT=$(python ../distilbert/evaluate_distilbert_recipe_mpr.py \
        --model-path "$MODEL_PATH" \
        --dataset-path "../data/500QA.json" \
        --num-examples 3 \
        2>&1)

    echo "$OUTPUT"

    # Extract accuracy
    ACCURACY=$(echo "$OUTPUT" | grep "Accuracy:" | head -1 | awk '{print $2}')

    if [ ! -z "$ACCURACY" ]; then
        RESULTS["$MODEL_NAME"]="$ACCURACY"
    else
        RESULTS["$MODEL_NAME"]="ERROR"
    fi

    echo ""
    echo ""
done

echo "=========================================="
echo "COMPARISON SUMMARY"
echo "=========================================="
echo ""
printf "%-20s %10s %15s\n" "Model" "Accuracy" "vs Goal (75%)"
echo "──────────────────────────────────────────────────"

# Sort by accuracy (descending)
for model in "${!RESULTS[@]}"; do
    echo "$model:${RESULTS[$model]}"
done | sort -t: -k2 -rn | while IFS=: read -r model accuracy; do
    if [ "$accuracy" != "ERROR" ]; then
        # Remove % sign for calculation
        acc_num=$(echo "$accuracy" | sed 's/%//')

        if (( $(echo "$acc_num >= 75" | bc -l) )); then
            status="✅ Pass"
        else
            diff=$(echo "75 - $acc_num" | bc -l)
            status="❌ -${diff}%"
        fi

        printf "%-20s %10s %15s\n" "$model" "$accuracy" "$status"
    else
        printf "%-20s %10s %15s\n" "$model" "ERROR" "❌ Failed"
    fi
done

echo ""
echo "Goal: 75% accuracy"
echo ""
echo "Detailed results saved in respective model directories"
echo ""
