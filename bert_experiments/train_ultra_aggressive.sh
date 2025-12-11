#!/bin/bash
# Ultra-aggressive training for BERT-large with extended epochs and negation focus
# Targeting 65%+ accuracy on Recipe-MPR
# Usage: ./train_ultra_aggressive.sh [model_name]

set -e

# Get model name from argument or use default
MODEL_NAME="${1:-bert-large-uncased}"

# Ultra-aggressive configuration
case "$MODEL_NAME" in
    "bert-large-uncased")
        OUTPUT_DIR="~/models/hub/bert-large-ultra-recipe-mpr"
        EPOCHS=20
        LR=1.5e-5
        BATCH_SIZE=4
        GRAD_ACCUM=8
        WARMUP_RATIO=0.15
        ;;
    "distilbert-base-uncased")
        OUTPUT_DIR="~/models/hub/distilbert-ultra-recipe-mpr"
        EPOCHS=25
        LR=2e-5
        BATCH_SIZE=8
        GRAD_ACCUM=4
        WARMUP_RATIO=0.15
        ;;
    "roberta-base")
        OUTPUT_DIR="~/models/hub/roberta-base-ultra-recipe-mpr"
        EPOCHS=25
        LR=2e-5
        BATCH_SIZE=8
        GRAD_ACCUM=4
        WARMUP_RATIO=0.15
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Supported models:"
        echo "  - bert-large-uncased"
        echo "  - distilbert-base-uncased"
        echo "  - roberta-base"
        exit 1
        ;;
esac

echo "=========================================="
echo "ULTRA-AGGRESSIVE Training: $MODEL_NAME"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS (2.5x standard)"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: ${GRAD_ACCUM}× (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Warmup ratio: $WARMUP_RATIO"
echo "  Max length: 384"
echo "  Label smoothing: 0.1"
echo ""
echo "Ultra-Aggressive Features:"
echo "  • Extended training: 20 epochs (vs 8 standard)"
echo "  • Higher warmup: 15% (vs 10%)"
echo "  • Longer context: 384 tokens"
echo "  • Early stopping disabled for full training"
echo ""
echo "Starting ultra-aggressive training..."
echo ""

python ../distilbert/finetune_distilbert_recipe_mpr.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "../data/500QA_negation_augmented.json" \
    --num-train-epochs $EPOCHS \
    --learning-rate $LR \
    --per-device-train-batch-size $BATCH_SIZE \
    --per-device-eval-batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --warmup-ratio $WARMUP_RATIO \
    --weight-decay 0.01 \
    --max-length 384 \
    --label-smoothing 0.1 \
    --logging-steps 25 \
    --eval-steps 50 \
    --save-steps 50 \
    --fp16 \
    --seed 42

echo ""
echo "=========================================="
echo "Ultra-aggressive training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python ../distilbert/evaluate_distilbert_recipe_mpr.py --model-path $OUTPUT_DIR --dataset-path '../data/500QA.json' --use-test-split"
echo ""
