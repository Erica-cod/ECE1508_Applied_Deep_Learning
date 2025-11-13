#!/bin/bash
# Aggressive training configuration if improved.sh doesn't reach 75%
# Uses more epochs and lower learning rate for maximum convergence

set -e

echo "=========================================="
echo "DistilBERT Training - AGGRESSIVE MODE"
echo "Target: 75%+ accuracy"
echo "=========================================="
echo ""

# Training configuration
MODEL_NAME="distilbert-base-uncased"
OUTPUT_DIR="~/models/hub/distilbert-recipe-mpr-aggressive"
DATASET_PATH="../data/500QA.json"

# Aggressive hyperparameters
NUM_EPOCHS=15           # Even more training
LEARNING_RATE=2e-5      # Lower LR for fine-grained optimization
BATCH_SIZE=16
WARMUP_STEPS=300        # Longer warmup
WEIGHT_DECAY=0.01
MAX_LENGTH=256
GRAD_ACCUM=2

# Evaluation
EVAL_STEPS=50
LOGGING_STEPS=25

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $NUM_EPOCHS (aggressive)"
echo "  Learning rate: $LEARNING_RATE (lower for stability)"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Warmup steps: $WARMUP_STEPS"
echo ""
echo "Starting aggressive training..."
echo ""

python finetune_distilbert_recipe_mpr.py \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-path "$DATASET_PATH" \
    --num-train-epochs $NUM_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --per-device-train-batch-size $BATCH_SIZE \
    --per-device-eval-batch-size $BATCH_SIZE \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --max-length $MAX_LENGTH \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --eval-steps $EVAL_STEPS \
    --logging-steps $LOGGING_STEPS \
    --fp16 \
    --seed 42

echo ""
echo "Training complete! Evaluating..."
echo ""

python evaluate_distilbert_recipe_mpr.py \
    --model-path "$OUTPUT_DIR" \
    --save-results "${OUTPUT_DIR}/eval_results.json"

echo ""
echo "Done!"
