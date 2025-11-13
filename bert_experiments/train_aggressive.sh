#!/bin/bash
# Aggressive training for BERT variants that failed to reach 75% goal
# Usage: ./train_aggressive.sh [model_name]

# Get model name from argument or use default
MODEL_NAME="${1:-bert-base-uncased}"

# Aggressive configuration based on model
case "$MODEL_NAME" in
    "distilbert-base-uncased")
        OUTPUT_DIR="~/models/hub/distilbert-aggressive-recipe-mpr"
        EPOCHS=15
        LR=2e-5
        BATCH_SIZE=16
        GRAD_ACCUM=2
        WARMUP=300
        ;;
    "bert-base-uncased")
        OUTPUT_DIR="~/models/hub/bert-base-aggressive-recipe-mpr"
        EPOCHS=20
        LR=1.5e-5
        BATCH_SIZE=8
        GRAD_ACCUM=4
        WARMUP=400
        ;;
    "roberta-base")
        OUTPUT_DIR="~/models/hub/roberta-base-aggressive-recipe-mpr"
        EPOCHS=25
        LR=1e-5
        BATCH_SIZE=8
        GRAD_ACCUM=4
        WARMUP=500
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Supported models:"
        echo "  - distilbert-base-uncased"
        echo "  - bert-base-uncased"
        echo "  - roberta-base"
        exit 1
        ;;
esac

echo "=========================================="
echo "AGGRESSIVE Training: $MODEL_NAME"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: ${GRAD_ACCUM}× (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Warmup steps: $WARMUP"
echo ""
echo "Changes from standard training:"
case "$MODEL_NAME" in
    "distilbert-base-uncased")
        echo "  • Epochs: 3 → 15 (+400%)"
        echo "  • LR: 3e-5 → 2e-5 (lower, more stable)"
        echo "  • Warmup: 200 → 300 (+50%)"
        ;;
    "bert-base-uncased")
        echo "  • Epochs: 10 → 20 (+100%)"
        echo "  • LR: 3e-5 → 1.5e-5 (-50%, more careful)"
        echo "  • Batch: 16 → 8 (smaller steps)"
        echo "  • Grad accum: 2× → 4× (effective batch 32)"
        echo "  • Warmup: 200 → 400 (+100%)"
        ;;
    "roberta-base")
        echo "  • Epochs: 10 → 25 (+150%)"
        echo "  • LR: 3e-5 → 1e-5 (-67%, much lower)"
        echo "  • Batch: 16 → 8 (smaller steps)"
        echo "  • Grad accum: 2× → 4× (effective batch 32)"
        echo "  • Warmup: 200 → 500 (+150%)"
        ;;
esac
echo ""
echo "Starting aggressive training..."
echo ""

python ../distilbert/finetune_distilbert_recipe_mpr.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "../data/500QA.json" \
    --num-train-epochs $EPOCHS \
    --learning-rate $LR \
    --per-device-train-batch-size $BATCH_SIZE \
    --per-device-eval-batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --warmup-steps $WARMUP \
    --weight-decay 0.01 \
    --max-length 256 \
    --logging-steps 50 \
    --eval-steps 100 \
    --save-steps 100 \
    --fp16 \
    --seed 42

echo ""
echo "=========================================="
echo "Aggressive training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python ../distilbert/evaluate_distilbert_recipe_mpr.py --model-path $OUTPUT_DIR --dataset-path '../data/500QA.json'"
echo ""
