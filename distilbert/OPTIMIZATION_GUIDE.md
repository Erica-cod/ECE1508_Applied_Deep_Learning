# DistilBERT Recipe-MPR Optimization Guide

## Current Status
- **Baseline Accuracy**: 69.4% (347/500)
- **Goal**: 75.0%
- **Gap**: +5.6% improvement needed

## Performance by Query Type (Current)
| Query Type   | Accuracy | Count | Status |
|--------------|----------|-------|--------|
| Specific     | 74.83%   | 151   | ✓ Good |
| Analogical   | 73.33%   | 30    | ✓ Good |
| Negated      | 69.72%   | 109   | Close  |
| Commonsense  | 68.66%   | 268   | Needs work |
| Temporal     | 62.50%   | 32    | **Weakest** |

## Baseline Configuration (3 epochs → 69.4%)
```bash
Epochs: 3
Learning Rate: 5e-5
Batch Size: 8
Warmup Steps: 100
Mixed Precision: No
```

## Optimization Strategy

### Approach 1: Train Longer (RECOMMENDED FIRST)
**Script**: `train_improved.sh`

**Key Changes**:
- ✅ **10 epochs** (↑ from 3) - More training time to converge
- ✅ **3e-5 learning rate** (↓ from 5e-5) - Finer-grained optimization
- ✅ **Batch size 16** (↑ from 8) - More stable gradients
- ✅ **Gradient accumulation 2x** - Effective batch size = 32
- ✅ **200 warmup steps** (↑ from 100) - Smoother learning rate ramp-up
- ✅ **FP16 training** - Faster training, less memory

**Expected Impact**: +3-5% accuracy improvement → **~72-74%**

**Usage**:
```bash
cd distilbert
./train_improved.sh
```

### Approach 2: Aggressive Training
**Script**: `train_aggressive.sh`

Use this if `train_improved.sh` doesn't reach 75%.

**Key Changes**:
- ✅ **15 epochs** - Maximum convergence
- ✅ **2e-5 learning rate** - Very fine-grained updates
- ✅ **300 warmup steps** - Very gradual warmup
- ✅ All other improvements from Approach 1

**Expected Impact**: +5-7% accuracy improvement → **~74-76%**

**Usage**:
```bash
cd distilbert
./train_aggressive.sh
```

## Why These Changes Work

### 1. More Epochs (10-15 vs 3)
- **Problem**: 3 epochs may not be enough for the model to fully learn the patterns
- **Solution**: Train longer to allow the model to converge properly
- **Risk**: Low (early stopping can prevent overfitting)

### 2. Lower Learning Rate (2e-5 to 3e-5 vs 5e-5)
- **Problem**: Higher LR may cause the model to "jump around" and miss the optimal solution
- **Solution**: Smaller steps allow fine-tuned convergence to a better minimum
- **Trade-off**: Requires more epochs, but we're increasing those anyway

### 3. Larger Effective Batch Size (32 vs 8)
- **Problem**: Small batches have noisy gradients
- **Solution**: Larger batches provide more stable gradient estimates
- **Implementation**: batch_size=16 + gradient_accumulation=2

### 4. More Warmup Steps (200-300 vs 100)
- **Problem**: Starting with high LR can destabilize early training
- **Solution**: Gradual warmup prevents early instability
- **Benefit**: Better convergence path from the start

### 5. Mixed Precision (FP16)
- **Problem**: Training can be slow
- **Solution**: FP16 speeds up training with minimal accuracy impact
- **Benefit**: Train longer in less time

## Alternative Approaches (If 75% Not Reached)

### Option A: Switch to Larger Model
```bash
# Use bert-base instead of distilbert
python finetune_distilbert_recipe_mpr.py \
    --model-name bert-base-uncased \
    --num-train-epochs 10 \
    --learning-rate 2e-5 \
    ...
```
**Expected**: +2-3% over DistilBERT

### Option B: Switch to RoBERTa
```bash
# RoBERTa often outperforms BERT
python finetune_distilbert_recipe_mpr.py \
    --model-name roberta-base \
    --num-train-epochs 10 \
    --learning-rate 2e-5 \
    ...
```
**Expected**: +3-5% over DistilBERT

### Option C: Ensemble Methods
Train 3-5 models with different seeds and average predictions:
```bash
for seed in 42 123 456; do
    python finetune_distilbert_recipe_mpr.py --seed $seed ...
done
```
**Expected**: +1-2% over single model

## Monitoring Training

Watch for these signs during training:

✅ **Good Signs**:
- Training loss steadily decreasing
- Eval accuracy improving over epochs
- No large spikes in gradient norm

⚠️ **Warning Signs**:
- Training loss stops decreasing early
- Eval accuracy plateaus before reaching 75%
- Large gradient norm spikes (>10)

## Quick Start

1. **Try improved training** (recommended first):
   ```bash
   cd distilbert
   ./train_improved.sh
   ```

2. **If accuracy < 75%, try aggressive**:
   ```bash
   ./train_aggressive.sh
   ```

3. **Still not 75%? Consider upgrading model**:
   ```bash
   # Edit train_improved.sh and change:
   MODEL_NAME="bert-base-uncased"  # or "roberta-base"
   ```

## Expected Timeline

| Approach | Training Time | Expected Accuracy |
|----------|--------------|-------------------|
| Improved (10 epochs) | ~5-8 minutes | 72-74% |
| Aggressive (15 epochs) | ~8-12 minutes | 74-76% |
| BERT-base (10 epochs) | ~10-15 minutes | 75-78% |

## Results Tracking

After each run, results are automatically saved:
- Model: `~/models/hub/distilbert-recipe-mpr-{config}/`
- Evaluation: `~/models/hub/distilbert-recipe-mpr-{config}/eval_results.json`

Compare results:
```bash
python evaluate_distilbert_recipe_mpr.py --model-path ~/models/hub/distilbert-recipe-mpr-improved
python evaluate_distilbert_recipe_mpr.py --model-path ~/models/hub/distilbert-recipe-mpr-aggressive
```
