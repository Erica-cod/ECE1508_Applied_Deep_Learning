#!/bin/bash
# Improved training configuration to reach 75% accuracy
# Based on current 69.4% baseline with 3 epochs

set -e

echo "=========================================="
echo "DistilBERT Recipe-MPR Training - Improved"
echo "Target: 75% accuracy (current: 69.4%)"
echo "=========================================="
echo ""

# Training configuration
MODEL_NAME="distilbert-base-uncased"
OUTPUT_DIR="~/models/hub/distilbert-recipe-mpr-improved"
DATASET_PATH="../data/500QA.json"

# Improved hyperparameters
NUM_EPOCHS=10           # Increased from 3 to 10
LEARNING_RATE=3e-5      # Reduced from 5e-5 for better convergence
BATCH_SIZE=16           # Increased from 8 for more stable gradients
WARMUP_STEPS=200        # Increased from 100 for smoother warmup
WEIGHT_DECAY=0.01
MAX_LENGTH=256

# Gradient accumulation if memory is limited
GRAD_ACCUM=2            # Effective batch size = 16 * 2 = 32

# Evaluation strategy
EVAL_STEPS=50           # Evaluate frequently to track progress
LOGGING_STEPS=25        # Log more frequently

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)) with grad accum)"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting training..."
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
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Now evaluating on full dataset..."
echo ""

# Evaluate the trained model
python evaluate_distilbert_recipe_mpr.py \
    --model-path "$OUTPUT_DIR" \
    --save-results "${OUTPUT_DIR}/eval_results.json"

echo ""
echo "Done! Check results above."
echo ""
