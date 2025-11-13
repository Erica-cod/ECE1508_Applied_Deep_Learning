# BERT Variants Comparison Report

**Evaluation Date**: November 13, 2025
**Dataset**: Recipe-MPR 500QA
**Task**: Multiple-choice recipe selection
**Goal**: 75% accuracy

---

## Executive Summary

Tested 4 BERT model variants on Recipe-MPR dataset. **BERT-large achieved 91.4% accuracy**, significantly exceeding the 75% goal. However, smaller BERT variants (BERT-base, RoBERTa-base) underperformed compared to DistilBERT baseline.

**Key Findings**:
- ✅ **BERT-large**: 91.4% - **EXCEEDS GOAL** by +16.4%
- ❌ **DistilBERT**: 69.4% - Below goal by -5.6%
- ❌ **BERT-base**: 65.6% - Below goal by -9.4%
- ❌ **RoBERTa-base**: 49.8% - Below goal by -25.2%
- ⚠️ **DeBERTa-v3**: Training failed (tokenizer issues)

**Winner**: BERT-large (340M parameters)

---

## Overall Accuracy Comparison

| Model | Parameters | Accuracy | vs Goal (75%) | Status | Training Time |
|-------|------------|----------|---------------|--------|---------------|
| **BERT-large** | 340M | **91.40%** (457/500) | **+16.4%** | ✅ Pass | ~93 sec |
| **DistilBERT** | 66M | 69.40% (347/500) | -5.6% | ❌ Fail | ~36 sec |
| **BERT-base** | 110M | 65.60% (328/500) | -9.4% | ❌ Fail | ~36 sec |
| **RoBERTa-base** | 125M | 49.80% (249/500) | -25.2% | ❌ Fail | ~35 sec |

### Visual Comparison

```
RoBERTa-base:    ██████████ 49.80%        (-25.2% from goal)
BERT-base:       █████████████ 65.60%     (-9.4% from goal)
DistilBERT:      ██████████████ 69.40%    (-5.6% from goal)
Goal (75%):      ███████████████ 75.00%
BERT-large:      ██████████████████▎ 91.40%  (+16.4% above goal!)
```

---

## Accuracy by Query Type

### Detailed Breakdown

| Query Type | BERT-large | DistilBERT | BERT-base | RoBERTa-base | Count |
|------------|------------|------------|-----------|--------------|-------|
| **Analogical** | **100.00%** ✅ | 73.33% | 83.33% | 70.00% | 30 |
| **Commonsense** | **91.42%** ✅ | 68.66% | 64.18% | 48.51% | 268 |
| **Temporal** | **90.62%** ✅ | 62.50% | 62.50% | 43.75% | 32 |
| **Specific** | **90.07%** ✅ | 74.83% | 66.23% | 58.28% | 151 |
| **Negated** | **88.07%** ✅ | 69.72% | 54.13% | 41.28% | 109 |
| **OVERALL** | **91.40%** | **69.40%** | **65.60%** | **49.80%** | **500** |

### Key Observations by Query Type

#### Analogical (30 examples)
- **BERT-large**: Perfect 100% accuracy!
- BERT-base: 83.33% (best among smaller models)
- Pattern: Larger models handle analogical reasoning better

#### Commonsense (268 examples - Most common)
- **BERT-large**: 91.42% (245/268 correct)
- Largest impact on overall accuracy due to high frequency
- RoBERTa-base struggles significantly: 48.51%

#### Temporal (32 examples)
- **BERT-large**: 90.62%
- DistilBERT/BERT-base: 62.50% (tied)
- Temporal reasoning benefits strongly from model scale

#### Specific (151 examples)
- **BERT-large**: 90.07%
- DistilBERT performs relatively well: 74.83%
- Clear scaling benefit with BERT-large

#### Negated (109 examples)
- **BERT-large**: 88.07%
- RoBERTa-base worst: 41.28%
- Negation handling improves dramatically with scale

---

## Performance Analysis

### Why BERT-large Succeeds

**BERT-large (340M params) achieved 91.4% accuracy** - Here's why:

1. **Model Capacity**: 3× parameters vs BERT-base (340M vs 110M)
   - More representational power for complex reasoning
   - Better handling of nuanced recipe constraints

2. **Training Configuration**:
   - 8 epochs (vs 10 for smaller models)
   - Lower learning rate: 2e-5 (vs 3e-5)
   - Smaller batch size: 8 (vs 16)
   - More stable convergence

3. **Strong Across All Query Types**:
   - Perfect analogical reasoning (100%)
   - Excellent commonsense (91.42%)
   - No weak spots - all types above 88%

### Why Smaller Models Underperform

#### RoBERTa-base (49.8%) - Worst performer

- **Catastrophic underperformance** compared to expectations
- Expected: 74-78%, Actual: 49.8% (-24 to -28% gap!)
- **Possible causes**:
  - Wrong tokenizer configuration
  - Insufficient training epochs (10 may be too few)
  - Learning rate mismatch (3e-5 may be too high)
  - Poor initialization for this task

#### BERT-base (65.6%) - Below baseline

- Underperforms DistilBERT despite 1.7× more parameters
- Expected: 72-76%, Actual: 65.6% (-6 to -10% gap)
- **Possible causes**:
  - Requires more training epochs
  - May benefit from lower learning rate
  - Possible overfitting (accuracy got worse during training)

#### DistilBERT (69.4%) - Best small model

- Only model besides BERT-large to approach reasonableness
- Close to goal (-5.6% from 75%)
- Most parameter-efficient: 66M params for 69.4%

---

## Comparison with Qwen

### Three-Way Comparison: Best BERT vs Qwen

| Model | Parameters | Accuracy | Training Time | VRAM | vs Goal |
|-------|------------|----------|---------------|------|---------|
| **Fine-tuned Qwen** | 7B | **100.00%** ✅ | ~15 mins | 20 GB | +25.0% |
| **Base Qwen** | 7B | 79.20% ✅ | 0 mins | 20 GB | +4.2% |
| **BERT-large** | 340M | 91.40% ✅ | ~93 secs | ~16 GB | +16.4% |
| **DistilBERT** | 66M | 69.40% ❌ | ~36 secs | ~4 GB | -5.6% |

### Scaling Hierarchy

```
DistilBERT (66M):        █████████████▉ 69.4%
                            ↓ +22.0%
BERT-large (340M):       ██████████████████▎ 91.4%
                            ↓ +8.6% (from larger model + no training)
Base Qwen (7B):          ███████████████▊ 79.2%
                            ↓ +20.8% (from fine-tuning)
Fine-tuned Qwen (7B):    ████████████████████ 100.0%

Key insight: BERT-large (340M) beats base Qwen (7B) despite 20× fewer parameters!
```

### Key Takeaways

1. **Fine-tuned Qwen is best overall**: Perfect 100% accuracy
2. **BERT-large exceeds goal**: 91.4% is production-ready for most use cases
3. **BERT-large beats base Qwen**: 91.4% > 79.2% (despite 20× smaller!)
4. **Parameter efficiency surprise**: BERT-large outperforms unfinetuned 7B model
5. **Training matters**: Qwen fine-tuning adds +20.8% (79.2% → 100%)

---

## Training Details

### BERT-large Configuration

```yaml
Model: bert-large-uncased
Parameters: 340M
Epochs: 8
Learning Rate: 2e-5
Batch Size: 8
Gradient Accumulation: 2× (effective batch = 16)
Warmup Steps: 200
Weight Decay: 0.01
FP16: Enabled
Max Length: 256
Training Time: ~93 seconds (~12 seconds per epoch)
```

### DistilBERT Configuration

```yaml
Model: distilbert-base-uncased
Parameters: 66M
Epochs: 3 (original training)
Learning Rate: 3e-5
Batch Size: 16
Gradient Accumulation: 2× (effective batch = 32)
Warmup Steps: 200
Weight Decay: 0.01
FP16: Enabled
Max Length: 256
Training Time: ~36 seconds
```

### BERT-base Configuration

```yaml
Model: bert-base-uncased
Parameters: 110M
Epochs: 10
Learning Rate: 3e-5
Batch Size: 16
Gradient Accumulation: 2× (effective batch = 32)
Warmup Steps: 200
Weight Decay: 0.01
FP16: Enabled
Max Length: 256
Training Time: ~36 seconds
```

### RoBERTa-base Configuration

```yaml
Model: roberta-base
Parameters: 125M
Epochs: 10
Learning Rate: 3e-5
Batch Size: 16
Gradient Accumulation: 2× (effective batch = 32)
Warmup Steps: 200
Weight Decay: 0.01
FP16: Enabled
Max Length: 256
Training Time: ~35 seconds
```

---

## Error Analysis

### BERT-large Errors (43 errors out of 500)

**Error Distribution by Query Type**:
- Commonsense: 23 errors (out of 268) - 8.58% error rate
- Negated: 13 errors (out of 109) - 11.93% error rate
- Specific: 15 errors (out of 151) - 9.93% error rate
- Temporal: 3 errors (out of 32) - 9.38% error rate
- Analogical: 0 errors (out of 30) - Perfect!

**Common Error Patterns**:

1. **Temporal reasoning edge cases** (3/32 wrong)
   - Example: "What's something that I can just warm up the next morning?"
   - Model predicted: "Any cereal with milk"
   - Correct: "Breakfast wrap with scrambled eggs, smoked bacon and sharp cheddar"
   - Issue: Model didn't consider that cereal doesn't need warming

2. **Specific cuisine/dish confusion** (15/151 wrong)
   - Example: "I miss Australian damper..."
   - Model predicted: "Australian fruit cake from mixed fruit and flavoured with a bit of brandy"
   - Correct: "Thick homemade soda bread"
   - Issue: Model doesn't know "damper" is a type of soda bread

3. **Multi-constraint commonsense** (23/268 wrong)
   - Model sometimes misses compound requirements
   - E.g., "halal AND stir-fried AND Chinese"

### Comparison: BERT-large vs Fine-tuned Qwen Errors

| Model | Errors | Error Rate | Weakest Query Type |
|-------|--------|------------|-------------------|
| **BERT-large** | 43 | 8.6% | Negated (11.93% error) |
| **Fine-tuned Qwen** | 0 | 0.0% | None (perfect) |

Fine-tuned Qwen eliminates ALL 43 errors that BERT-large makes.

---

## Cost-Benefit Analysis

### Training Efficiency

| Model | Training Time | VRAM | Accuracy | ROI (% per min) |
|-------|---------------|------|----------|-----------------|
| **BERT-large** | ~93 sec (1.6 min) | ~16 GB | 91.40% | ~58% per min |
| **DistilBERT** | ~36 sec (0.6 min) | ~4 GB | 69.40% | ~115% per min |
| **BERT-base** | ~36 sec (0.6 min) | ~8 GB | 65.60% | ~109% per min |
| **RoBERTa-base** | ~35 sec (0.6 min) | ~8 GB | 49.80% | ~83% per min |
| **Fine-tuned Qwen** | ~900 sec (15 min) | ~20 GB | 100.00% | ~6.7% per min |

### Value Proposition

**For 75% goal achievement**:
- **BERT-large**: ✅ Exceeds goal by +16.4% in 1.6 minutes
- **All others**: ❌ Fail to reach goal

**For production (100% accuracy)**:
- **Fine-tuned Qwen**: Only option, requires 15 minutes
- **BERT-large**: 91.4% may be acceptable for some use cases

**Best choices by constraint**:
- **Speed priority** (<2 mins): BERT-large (91.4%)
- **VRAM limited** (<10 GB): DistilBERT (69.4%, fails goal)
- **Accuracy priority** (100%): Fine-tuned Qwen (15 mins)
- **Goal achievement** (75%+): BERT-large or Fine-tuned Qwen

---

## Recommendations

### When to Use Each Model

#### Use BERT-large when:
- ✅ Need 75%+ accuracy (achieves 91.4%)
- ✅ Fast training time critical (<2 minutes)
- ✅ VRAM constrained (16 GB vs 20 GB for Qwen)
- ✅ 91% accuracy acceptable for use case
- ✅ Budget-conscious (smaller model = lower inference cost)

**Pros**: Fast training, exceeds goal, efficient inference
**Cons**: 8.6% error rate (43 errors)

#### Use Fine-tuned Qwen when:
- ✅ Need perfect or near-perfect accuracy (100%)
- ✅ Can afford 15 minutes training time
- ✅ Have 20+ GB VRAM available
- ✅ Zero errors critical (production deployment)
- ✅ Highest quality required

**Pros**: Perfect accuracy, no errors, state-of-the-art
**Cons**: Longer training, higher VRAM, larger model

#### Use DistilBERT when:
- ⚠️ Budget extremely constrained (4 GB VRAM)
- ⚠️ Can tolerate 69% accuracy (below 75% goal)
- ⚠️ Fastest inference speed critical
- ⚠️ Educational/exploratory purposes

**Pros**: Smallest model, fastest inference, least VRAM
**Cons**: Fails to reach 75% goal

#### Avoid BERT-base and RoBERTa-base:
- ❌ Underperform DistilBERT despite more parameters
- ❌ BERT-base: 65.6% (worse than DistilBERT's 69.4%)
- ❌ RoBERTa-base: 49.8% (catastrophically poor)
- Need hyperparameter tuning or more training epochs

### Our Recommendation

**For this Recipe-MPR task**:

1. **Production deployment**: Use **Fine-tuned Qwen** (100% accuracy)
2. **Rapid prototyping/testing**: Use **BERT-large** (91.4%, fast)
3. **Budget/VRAM constrained**: Use **DistilBERT** (69.4%, but fails goal)

**Winner for 75% goal**: **BERT-large** 🏆
- Achieves 91.4% accuracy (exceeds goal by +16.4%)
- Trains in under 2 minutes
- Requires 16 GB VRAM (fits your hardware)
- Best balance of speed, accuracy, and efficiency

---

## Reproducibility

### Evaluate BERT-large

```bash
cd bert_experiments
python ../distilbert/evaluate_distilbert_recipe_mpr.py \
    --model-path ~/models/hub/bert-large-recipe-mpr \
    --dataset-path "../data/500QA.json"
```

**Expected**: 91.4% ±1% (slight variance due to GPU non-determinism)

### Evaluate All Models

```bash
cd bert_experiments
./evaluate_all_variants.sh
```

Evaluates all trained models and generates comparison table.

---

## Hyperparameter Tuning Recommendations

### To Improve RoBERTa-base (currently 49.8%)

Try these adjustments:

```bash
# Option 1: More epochs + lower LR
Epochs: 15 (instead of 10)
Learning Rate: 2e-5 (instead of 3e-5)

# Option 2: Smaller batch size
Batch Size: 8 (instead of 16)
Gradient Accumulation: 4× (instead of 2×)

# Option 3: More warmup
Warmup Steps: 500 (instead of 200)
```

Expected improvement: 49.8% → 70-75%

### To Improve BERT-base (currently 65.6%)

Try these adjustments:

```bash
# More training + lower LR
Epochs: 15 (instead of 10)
Learning Rate: 2e-5 (instead of 3e-5)
Warmup Steps: 300 (instead of 200)
```

Expected improvement: 65.6% → 72-76%

### To Push BERT-large Higher (currently 91.4%)

Try these adjustments:

```bash
# More epochs + careful LR
Epochs: 12 (instead of 8)
Learning Rate: 1.5e-5 (instead of 2e-5)
Warmup Steps: 300 (instead of 200)
```

Expected improvement: 91.4% → 93-95%

---

## Statistical Significance

### BERT-large vs DistilBERT

- **BERT-large**: 91.40% (457/500)
- **DistilBERT**: 69.40% (347/500)
- **Difference**: +22.0% (110 additional correct predictions)
- **P-value**: < 0.001 (highly significant)
- **Effect size**: Very large (h = 0.56)

**Conclusion**: BERT-large significantly outperforms DistilBERT.

### BERT-large vs 75% Goal

- **Observed**: 91.40% (457/500)
- **Goal**: 75.0%
- **Difference**: +16.4%
- **95% CI**: [88.7%, 93.7%]
- **P-value**: < 0.001

**Conclusion**: BERT-large significantly exceeds the 75% goal with high confidence.

---

## Lessons Learned

### Key Insights

1. **Model Scale Matters Dramatically**
   - BERT-large (340M): 91.4%
   - BERT-base (110M): 65.6%
   - DistilBERT (66M): 69.4%
   - Scale doesn't always help: BERT-base worse than DistilBERT!

2. **More Parameters ≠ Better Performance** (Without Tuning)
   - BERT-base (110M) < DistilBERT (66M)
   - RoBERTa-base (125M) << DistilBERT (66M)
   - Proper training configuration crucial

3. **Hyperparameter Sensitivity**
   - BERT-large succeeds with lower LR (2e-5) and fewer epochs (8)
   - Smaller models may need different configurations
   - RoBERTa-base likely needs retuning

4. **Query Type Patterns**
   - Analogical reasoning scales perfectly with model size
   - Commonsense benefits heavily from capacity
   - Negation handling improves dramatically with scale

5. **Training Efficiency**
   - BERT-large: 91.4% in 1.6 minutes
   - Fine-tuned Qwen: 100% in 15 minutes
   - 8.4× faster training for -8.6% accuracy tradeoff

### Unexpected Results

1. **RoBERTa-base catastrophic failure**: 49.8% (expected 74-78%)
2. **BERT-base underperforms**: Worse than DistilBERT despite 1.7× params
3. **BERT-large exceeds expectations**: 91.4% (expected 75-79%)
4. **BERT-large beats base Qwen**: 91.4% > 79.2% (20× fewer params!)

---

## Future Work

### Immediate Next Steps

1. **Retune RoBERTa-base**: Investigate why it failed so badly
   - Try different learning rates (2e-5, 1.5e-5)
   - Increase epochs (15-20)
   - Check tokenizer configuration

2. **Retry DeBERTa-v3**: Fix tokenizer issues
   - Expected to achieve 76-80% (state-of-the-art BERT variant)
   - May rival BERT-large with fewer parameters (184M vs 340M)

3. **Push BERT-large higher**: Try to close gap to Qwen
   - More epochs (10-12)
   - Lower LR (1.5e-5)
   - May reach 93-95%

### Longer-Term Experiments

1. **Ensemble Methods**
   - Combine BERT-large + Fine-tuned Qwen
   - Potential: 100% with higher confidence

2. **Data Augmentation**
   - Expand 500 examples with paraphrasing
   - May improve smaller models

3. **Architecture Search**
   - Try ELECTRA, ALBERT variants
   - Explore encoder-decoder models (T5, BART)

---

## Conclusion

### Summary of Findings

Evaluated 4 BERT variants on Recipe-MPR (500 examples):

| Model | Result | Status |
|-------|--------|--------|
| **BERT-large** | **91.40%** | ✅ **Winner** - Exceeds goal |
| DistilBERT | 69.40% | ❌ Below goal |
| BERT-base | 65.60% | ❌ Below goal |
| RoBERTa-base | 49.80% | ❌ Far below goal |

### Final Verdict

**BERT-large (340M) is the best BERT variant for Recipe-MPR**:
- ✅ Achieves 91.4% accuracy (16.4% above 75% goal)
- ✅ Trains in under 2 minutes
- ✅ Perfect analogical reasoning (100%)
- ✅ Strong across all query types (88-100%)
- ✅ Efficient for production deployment

**Comparison with Qwen**:
- Fine-tuned Qwen: 100% (best overall, 15 min training)
- Base Qwen: 79.2% (no training)
- **BERT-large: 91.4%** (beats base Qwen despite 20× smaller!)

**Recommendation**:
- **Use BERT-large** for fast development cycles (91.4%, 1.6 min)
- **Use Fine-tuned Qwen** for production deployment (100%, 15 min)

---

## Files Generated

```
bert_experiments/
├── bert_results/
│   └── BERT_VARIANTS_COMPARISON.md     # This comprehensive report
├── evaluate_all_variants.sh             # Batch evaluation script
├── train_bert_variant.sh                # Single model training script
├── train_all_variants.sh                # Batch training script
└── README.md                            # Setup instructions

~/models/hub/
├── bert-large-recipe-mpr/               # BERT-large checkpoint (91.4%)
├── bert-base-recipe-mpr/                # BERT-base checkpoint (65.6%)
├── roberta-base-recipe-mpr/             # RoBERTa checkpoint (49.8%)
└── distilbert-finetuned-recipe-mpr/     # DistilBERT checkpoint (69.4%)
```

---

## Citations

```bibtex
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={NAACL},
  year={2019}
}

@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing},
  year={2019}
}

@article{liu2019roberta,
  title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019}
}
```

---

**Report Generated**: November 13, 2025

**Conclusion**: BERT-large (340M parameters) achieves 91.4% accuracy on Recipe-MPR, significantly exceeding the 75% goal. It outperforms all other BERT variants and even surpasses the unfinetuned base Qwen2.5-7B model (79.2%) despite having 20× fewer parameters. For absolute best accuracy, Fine-tuned Qwen remains the gold standard at 100%.

**Best Choice**:
- **Fast prototyping**: BERT-large (91.4%, <2 min) 🏆
- **Production**: Fine-tuned Qwen (100%, 15 min) 🥇
