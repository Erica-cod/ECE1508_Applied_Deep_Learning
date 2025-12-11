#!/bin/bash
# Train Qwen2.5-7B-Instruct with LoRA on Recipe-MPR
# 80/10/10 split with proper held-out evaluation

set -e

echo "=========================================="
echo "Training Qwen2.5-7B-Instruct with LoRA"
echo "=========================================="
echo ""

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="~/models/hub/qwen2.5-7b-recipe-mpr-lora"
DATASET_PATH="../data/500QA.json"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Dataset: $DATASET_PATH"
echo "  Split: 80/10/10 (400 train / 50 val / 50 test)"
echo "  LoRA: r=16, alpha=32, dropout=0.05"
echo "  Epochs: 5"
echo "  Learning rate: 2e-4"
echo ""
echo "Starting training..."
echo ""

python ../scripts/finetune_qwen3_recipe_mpr.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "$DATASET_PATH" \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --num-train-epochs 5 \
    --learning-rate 2e-4 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --bf16 \
    --seed 42

echo ""
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
