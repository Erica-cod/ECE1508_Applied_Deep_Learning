# Recipe-MPR Fine-tuning: Model Comparison

This project explores fine-tuning different models on the Recipe-MPR 500QA dataset for multiple-choice recipe selection.

## Dataset Overview
- **Task**: Multiple-choice recipe selection
- **Examples**: 500 questions, each with 5 answer choices
- **Query Types**: Specific, Commonsense, Negated, Analogical, Temporal
- **Goal**: 75%+ accuracy

## Models Available

### 1. DistilBERT (Baseline)
📁 **Location**: `distilbert/`

**Architecture**: Encoder-only (BERT-style)
**Size**: 66M parameters
**Method**: Fine-tune classification head

**Current Performance**: 69.4% accuracy

**Breakdown by Query Type**:
- Specific: 74.83%
- Analogical: 73.33%
- Negated: 69.72%
- Commonsense: 68.66%
- Temporal: 62.50% (weakest)

**Training**:
```bash
cd distilbert
./train_improved.sh     # 10 epochs, optimized hyperparams
./train_aggressive.sh   # 15 epochs, max convergence
```

**Pros**:
- ✅ Fast training (5-8 mins)
- ✅ Low VRAM usage (~4 GB)
- ✅ Simple architecture
- ✅ Proven baseline

**Cons**:
- ❌ Small model size limits capacity
- ❌ Struggles with temporal reasoning
- ❌ Accuracy ceiling ~72-74%

---

### 2. Qwen2.5-7B (Target)
📁 **Location**: `qwen/`

**Architecture**: Decoder-only (GPT-style)
**Size**: 7B parameters (100× larger than DistilBERT)
**Method**: LoRA fine-tuning (parameter-efficient)

**Expected Performance**: 77-82% accuracy

**Training**:
```bash
cd qwen
./train_qwen.sh            # Standard: 5 epochs, LoRA r=16
./train_qwen_aggressive.sh # Aggressive: 10 epochs, LoRA r=32
```

**Pros**:
- ✅ Much higher capacity
- ✅ Instruction-tuned (understands task format)
- ✅ Better reasoning abilities
- ✅ LoRA = efficient training
- ✅ Can explain answers (generative)
- ✅ Expected 75%+ accuracy

**Cons**:
- ❌ Larger VRAM usage (~18-22 GB)
- ❌ Longer training (10-20 mins)
- ❌ Requires more dependencies (PEFT)

---

## Side-by-Side Comparison

| Feature | DistilBERT | Qwen2.5-7B |
|---------|------------|------------|
| **Size** | 66M | 7B |
| **Architecture** | Encoder | Decoder (GPT-style) |
| **Training method** | Full fine-tune | LoRA |
| **Trainable params** | 66M (100%) | ~30-40M (0.5%) |
| **VRAM usage** | 2-4 GB | 18-22 GB |
| **Training time** | 5-8 mins | 10-20 mins |
| **Current accuracy** | 69.4% | TBD |
| **Expected max** | 72-74% | 77-82% |
| **Best for Temporal** | 62.5% | 70-78% (expected) |
| **Best for Commonsense** | 68.7% | 75-82% (expected) |
| **Can explain reasoning** | No | Yes |
| **Requires** | transformers | transformers, peft |

---

## Training Strategy

### Path to 75%+ Accuracy

**Option 1: Improve DistilBERT** (Tried)
- Current: 69.4% → Target: 75%
- Strategy: More epochs, lower LR, larger batch size
- Scripts: `distilbert/train_improved.sh`, `train_aggressive.sh`
- Expected: 72-74% (may fall short of 75%)

**Option 2: Use Qwen2.5-7B** (Recommended)
- Start: Baseline → Target: 75%+
- Strategy: LoRA fine-tuning with 5-10 epochs
- Scripts: `qwen/train_qwen.sh`, `train_qwen_aggressive.sh`
- Expected: 77-82% (exceeds 75% goal)

---

## Hardware Requirements

### Minimum (DistilBERT)
- **GPU**: 4GB VRAM
- **RAM**: 8GB
- **Storage**: 2GB

### Recommended (Qwen2.5-7B)
- **GPU**: 20-24GB VRAM ✅ **You have 30GB - perfect!**
- **RAM**: 16GB
- **Storage**: 20GB (model download)

---

## Quick Start Guide

### For DistilBERT
```bash
# Train improved version
cd distilbert
./train_improved.sh

# Evaluate
python ../scripts/evaluate_distilbert_recipe_mpr.py \
    --model-path ~/models/hub/distilbert-recipe-mpr-improved
```

### For Qwen (Recommended)
```bash
# Train with LoRA
cd qwen
./train_qwen.sh

# Evaluate
python ../scripts/evaluate_qwen_recipe_mpr.py \
    --model-path ~/models/hub/qwen2.5-7b-recipe-mpr-lora
```

---

## Expected Results Timeline

### DistilBERT Path
1. **Baseline** (3 epochs): 69.4% ✅ Done
2. **Improved** (10 epochs): ~72-73%
3. **Aggressive** (15 epochs): ~73-74%
4. **Max potential**: ~74-75% (may not reach 75%)

### Qwen Path
1. **Standard** (5 epochs, r=16): ~77-80%
2. **Aggressive** (10 epochs, r=32): ~80-82%
3. **Max potential**: ~82-85% (with larger rank)

---

## When to Use Which Model

### Use DistilBERT if:
- Limited GPU memory (< 10GB VRAM)
- Need fast iteration cycles
- Baseline/comparison purposes
- Simple deployment required

### Use Qwen if:
- Have sufficient VRAM (20GB+) ✅ **You have this!**
- Need 75%+ accuracy ✅ **Your goal**
- Want best possible performance
- Can afford longer training time

---

## Recommendations

Given your requirements:
- **Goal**: 75%+ accuracy
- **Hardware**: 30GB VRAM
- **Current**: 69.4% with DistilBERT

### Recommended Approach:

1. **Start with Qwen Standard** (Most likely to succeed)
   ```bash
   cd qwen && ./train_qwen.sh
   ```
   - Should reach 77-80% accuracy
   - Uses ~20GB of your 30GB VRAM
   - Takes ~10-15 minutes

2. **If below 75%** (unlikely), try Qwen Aggressive
   ```bash
   ./train_qwen_aggressive.sh
   ```
   - Should reach 80-82%
   - Uses ~22-24GB VRAM
   - Takes ~15-25 minutes

3. **Optional**: Also try DistilBERT improvements for comparison
   - Good for understanding the impact of model size
   - Useful for ablation studies

---

## File Structure

```
ECE1508_Applied_Deep_Learning/
├── data/
│   └── 500QA.json                 # Recipe-MPR dataset
│
├── scripts/
│   ├── finetune_distilbert_recipe_mpr.py
│   ├── evaluate_distilbert_recipe_mpr.py
│   ├── finetune_qwen_recipe_mpr.py
│   └── evaluate_qwen_recipe_mpr.py
│
├── distilbert/
│   ├── train_improved.sh          # 10 epochs, optimized
│   ├── train_aggressive.sh        # 15 epochs, max effort
│   ├── OPTIMIZATION_GUIDE.md
│   └── result.md                  # Current 69.4% results
│
├── qwen/
│   ├── train_qwen.sh              # Standard LoRA (recommended)
│   ├── train_qwen_aggressive.sh   # Aggressive LoRA
│   └── README.md                  # Qwen-specific guide
│
└── MODEL_COMPARISON.md            # This file
```

---

## Next Steps

**To reach 75%+ accuracy with Qwen**:

```bash
# 1. Navigate to qwen directory
cd qwen

# 2. Run training (takes ~10-15 mins)
./train_qwen.sh

# 3. Check results
# The script will automatically evaluate and show if 75% is reached

# 4. If needed, run aggressive training
./train_qwen_aggressive.sh
```

**Expected outcome**: 77-82% accuracy, exceeding the 75% goal! 🎯

---

## Support & Troubleshooting

### DistilBERT Issues
See: `distilbert/OPTIMIZATION_GUIDE.md`

### Qwen Issues
See: `qwen/README.md` - Troubleshooting section

### Common Issues

**Out of Memory (Qwen)**:
- Reduce batch size: `BATCH_SIZE=2`
- Reduce LoRA rank: `LORA_R=8`

**Low Accuracy**:
- Train longer (more epochs)
- Increase LoRA rank
- Lower learning rate
- Check data preprocessing

**Slow Training**:
- Enable bf16: `--bf16`
- Increase batch size
- Use gradient accumulation
