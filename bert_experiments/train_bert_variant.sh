#!/bin/bash
# Train different BERT variants on Recipe-MPR
# Usage: ./train_bert_variant.sh [model_name]

set -e

# Get model name from argument or use default
MODEL_NAME="${1:-bert-base-uncased}"

# Configuration based on model
case "$MODEL_NAME" in
    "bert-base-uncased")
        OUTPUT_DIR="~/models/hub/bert-base-recipe-mpr"
        EPOCHS=10
        LR=3e-5
        BATCH_SIZE=16
        ;;
    "bert-large-uncased")
        OUTPUT_DIR="~/models/hub/bert-large-recipe-mpr"
        EPOCHS=8
        LR=2e-5
        BATCH_SIZE=8
        ;;
    "roberta-base")
        OUTPUT_DIR="~/models/hub/roberta-base-recipe-mpr"
        EPOCHS=10
        LR=3e-5
        BATCH_SIZE=16
        ;;
    "roberta-large")
        OUTPUT_DIR="~/models/hub/roberta-large-recipe-mpr"
        EPOCHS=8
        LR=2e-5
        BATCH_SIZE=8
        ;;
    "microsoft/deberta-v3-base")
        OUTPUT_DIR="~/models/hub/deberta-v3-base-recipe-mpr"
        EPOCHS=10
        LR=2e-5
        BATCH_SIZE=16
        ;;
    "albert-base-v2")
        OUTPUT_DIR="~/models/hub/albert-base-recipe-mpr"
        EPOCHS=10
        LR=3e-5
        BATCH_SIZE=16
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Supported models:"
        echo "  - bert-base-uncased"
        echo "  - bert-large-uncased"
        echo "  - roberta-base"
        echo "  - roberta-large"
        echo "  - microsoft/deberta-v3-base"
        echo "  - albert-base-v2"
        exit 1
        ;;
esac

echo "=========================================="
echo "Training $MODEL_NAME on Recipe-MPR"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "Starting training..."
echo ""

python ../distilbert/finetune_distilbert_recipe_mpr.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "../data/500QA.json" \
    --num-train-epochs $EPOCHS \
    --learning-rate $LR \
    --per-device-train-batch-size $BATCH_SIZE \
    --per-device-eval-batch-size $BATCH_SIZE \
    --gradient-accumulation-steps 2 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --max-length 256 \
    --logging-steps 25 \
    --eval-steps 100 \
    --save-steps 100 \
    --fp16 \
    --seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python ../scripts/evaluate_distilbert_recipe_mpr.py --model-path $OUTPUT_DIR"
echo ""
