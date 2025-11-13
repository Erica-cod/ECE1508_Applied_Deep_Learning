#!/bin/bash
# Aggressive Qwen2.5-7B training for maximum accuracy
# Uses larger LoRA rank and more epochs

set -e

echo "=========================================="
echo "Qwen2.5-7B AGGRESSIVE Training"
echo "Target: 80%+ accuracy"
echo "=========================================="
echo ""

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive"
DATASET_PATH="../data/500QA.json"

# Aggressive LoRA parameters
LORA_R=32              # Higher rank = more parameters
LORA_ALPHA=64          # 2x rank typically
LORA_DROPOUT=0.05

# Aggressive training
NUM_EPOCHS=10          # More epochs
LEARNING_RATE=1.5e-4   # Slightly lower for stability
BATCH_SIZE=4
GRAD_ACCUM=4
WARMUP_STEPS=150       # Longer warmup
WEIGHT_DECAY=0.01

MAX_LENGTH=512
LOGGING_STEPS=10
EVAL_STEPS=50          # Evaluate more frequently
SAVE_STEPS=50

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  LoRA: r=$LORA_R (AGGRESSIVE), alpha=$LORA_ALPHA"
echo "  Epochs: $NUM_EPOCHS (more training)"
echo "  Learning rate: $LEARNING_RATE"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""
echo "Estimated VRAM usage: ~20-24 GB (higher due to larger LoRA)"
echo ""
echo "Starting aggressive training..."
echo ""

python ../scripts/finetune_qwen_recipe_mpr.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "$DATASET_PATH" \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --num-train-epochs $NUM_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --per-device-train-batch-size $BATCH_SIZE \
    --per-device-eval-batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --max-length $MAX_LENGTH \
    --logging-steps $LOGGING_STEPS \
    --eval-steps $EVAL_STEPS \
    --save-steps $SAVE_STEPS \
    --bf16 \
    --seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate the aggressive model, run:"
echo "  ./evaluate_aggressive.sh"
echo ""
echo "Or to compare with standard model:"
echo "  ./compare_models.sh"
echo ""
