#!/bin/bash
# Train all BERT variants sequentially

set -e

echo "=========================================="
echo "Training All BERT Variants"
echo "=========================================="
echo ""
echo "This will train 4 different BERT variants:"
echo "  1. bert-base-uncased (110M params)"
echo "  2. roberta-base (125M params)"
echo "  3. microsoft/deberta-v3-base (184M params)"
echo "  4. bert-large-uncased (340M params)"
echo ""
echo "Estimated total time: ~60-90 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

MODELS=(
    "bert-base-uncased"
    "roberta-base"
    "microsoft/deberta-v3-base"
    "bert-large-uncased"
)

FAILED_MODELS=()
SUCCESSFUL_MODELS=()

for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training: $model"
    echo "=========================================="
    echo ""

    if ./train_bert_variant.sh "$model"; then
        SUCCESSFUL_MODELS+=("$model")
        echo "✓ $model trained successfully"
    else
        FAILED_MODELS+=("$model")
        echo "✗ $model training failed"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "Training Summary"
echo "=========================================="
echo ""
echo "Successful (${#SUCCESSFUL_MODELS[@]}):"
for model in "${SUCCESSFUL_MODELS[@]}"; do
    echo "  ✓ $model"
done
echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed (${#FAILED_MODELS[@]}):"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model"
    done
    echo ""
fi

echo "To evaluate all models:"
echo "  ./evaluate_all_variants.sh"
echo ""
