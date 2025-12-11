# Qwen3-8B LoRA Fine-tuning Results

**Date**: December 11, 2025
**Task**: Recipe-MPR Multiple-Choice QA

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-8B |
| Parameters | 8.23B total, 43.6M trainable (0.53%) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Epochs | 5 |
| Learning rate | 2e-4 |
| Batch size | 1 (effective 16 with grad accum) |
| Max length | 512 |
| Data split | 80/10/10 (400 train / 50 val / 50 test) |
| Dataset | 500QA.json (original, no augmentation) |

---

## Training Progress

| Epoch | Loss | Grad Norm | Learning Rate |
|-------|------|-----------|---------------|
| 0.4 | 2.9123 | 1.739 | 1.8e-5 |
| 0.8 | 2.7039 | 1.229 | 3.8e-5 |
| 1.2 | 2.0936 | 0.710 | 5.8e-5 |
| 1.6 | 1.8534 | 0.470 | 7.8e-5 |
| 2.0 | 1.7535 | 0.450 | 9.8e-5 |
| 2.4 | 1.6961 | 0.524 | 1.18e-4 |
| 2.8 | 1.6058 | 0.586 | 1.38e-4 |
| 3.2 | 1.5350 | 0.714 | 1.58e-4 |
| 3.6 | 1.4003 | 1.079 | 1.78e-4 |
| 4.0 | 1.3544 | 1.059 | 1.98e-4 |
| 4.4 | 1.0630 | 1.378 | 1.28e-4 |
| 4.8 | 0.9985 | 1.412 | 4.8e-5 |

---

## Final Metrics

### Training Metrics
- **Final train loss**: 1.7176
- **Training time**: 10:19 (619 seconds)
- **Samples/sec**: 3.23
- **Steps/sec**: 0.202

### Validation Metrics
- **Eval loss**: 1.55

---

## Held-Out Test Results (Proper Evaluation)

**Evaluated on 50 held-out test examples (10% of data)**

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **100.00% (50/50)** |
| Target | 75% |
| Gap from Target | +25% |

### Accuracy by Query Type

| Query Type | Accuracy | Count |
|------------|----------|-------|
| Commonsense | **100.00%** | 26 |
| Temporal | **100.00%** | 4 |
| Specific | **100.00%** | 11 |
| Negated | **100.00%** | 9 |
| Analogical | **100.00%** | 3 |

---

## Comparison with BERT Models

| Model | Parameters | Held-Out Test Accuracy | Training Time |
|-------|-----------|----------------------|---------------|
| **Qwen3-8B LoRA** | **8.23B (43.6M trainable)** | **100.00%** | ~10 min |
| BERT-large (5-fold CV) | 340M | 46.6% ± 6.3% | ~7.8 min |
| RoBERTa-base (5-fold CV) | 125M | 40.0% ± 6.5% | ~3.2 min |
| DistilBERT (5-fold CV) | 66M | 31.4% ± 3.3% | ~1.7 min |
| Random baseline | - | 20% | - |

---

## Key Findings

1. **Qwen3-8B dramatically outperforms BERT variants** on held-out evaluation (100% vs 46.6%)

2. **LoRA is highly effective** - Only 0.53% of parameters are trainable, yet achieves perfect accuracy

3. **Proper train/test separation maintained** - 80/10/10 split with test set held out during training

4. **All query types handled perfectly** - Including negated queries (which BERT struggled with initially)

---

## Model Artifacts

- **Model path**: `~/models/hub/qwen3-8b-recipe-mpr-lora/`
- **Test split info**: `test_split.json`
- **Results**: `results.json`

---

## Reproducibility

```bash
# Training
cd /home/kevinlin/utece/ECE1508_Applied_Deep_Learning/qwen
./train_qwen3.sh

# Evaluation (held-out test set)
./evaluate_qwen3.sh

# Evaluation (full dataset for comparison)
./evaluate_qwen3_full.sh
```
