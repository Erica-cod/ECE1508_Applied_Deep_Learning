#!/bin/bash
# Fine-tune Qwen2.5-7B on Recipe-MPR with LoRA
# Optimized for 30GB VRAM

set -e

echo "=========================================="
echo "Qwen2.5-7B Recipe-MPR Training with LoRA"
echo "Target: 75%+ accuracy"
echo "=========================================="
echo ""

# Configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="~/models/hub/qwen2.5-7b-recipe-mpr-lora"
DATASET_PATH="../data/500QA.json"

# LoRA parameters
LORA_R=16              # LoRA rank (higher = more params)
LORA_ALPHA=32          # LoRA alpha (scaling factor)
LORA_DROPOUT=0.05      # Dropout for LoRA layers

# Training hyperparameters
NUM_EPOCHS=5           # Number of training epochs
LEARNING_RATE=2e-4     # Learning rate (higher than BERT due to LoRA)
BATCH_SIZE=4           # Per-device batch size
GRAD_ACCUM=4           # Gradient accumulation (effective batch = 4*4 = 16)
WARMUP_STEPS=100       # Warmup steps
WEIGHT_DECAY=0.01      # Weight decay

# Other settings
MAX_LENGTH=512         # Maximum sequence length
LOGGING_STEPS=10       # Log every N steps
EVAL_STEPS=100         # Evaluate every N steps
SAVE_STEPS=100         # Save every N steps

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Max length: $MAX_LENGTH"
echo ""
echo "Estimated VRAM usage: ~18-22 GB"
echo ""
echo "Starting training..."
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
echo "To evaluate the model, run:"
echo "  ./evaluate.sh"
echo ""
echo "Or to compare multiple models:"
echo "  ./evaluate_all.sh"
echo ""
