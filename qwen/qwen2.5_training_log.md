# Qwen2.5-7B-Instruct LoRA Fine-tuning Results

**Date**: December 11, 2025
**Task**: Recipe-MPR Multiple-Choice QA

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Parameters | 7.66B total, 40.4M trainable (0.53%) |
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
| 0.4 | 2.9500 | 2.464 | 1.8e-5 |
| 0.8 | 2.4995 | 1.085 | 3.8e-5 |
| 1.2 | 1.9424 | 0.833 | 5.8e-5 |
| 1.6 | 1.7868 | 0.560 | 7.8e-5 |
| 2.0 | 1.7096 | 0.645 | 9.8e-5 |
| 2.4 | 1.6546 | 0.670 | 1.18e-4 |
| 2.8 | 1.5691 | 0.841 | 1.38e-4 |
| 3.2 | 1.4891 | 0.925 | 1.58e-4 |
| 3.6 | 1.3524 | 1.524 | 1.78e-4 |
| 4.0 | 1.3142 | 1.272 | 1.98e-4 |
| 4.4 | 1.0186 | 1.562 | 1.28e-4 |
| 4.8 | 0.9407 | 1.825 | 4.8e-5 |

---

## Final Metrics

### Training Metrics
- **Final train loss**: 1.6561
- **Training time**: 7:43 (463 seconds)
- **Samples/sec**: 4.32
- **Steps/sec**: 0.27

### Validation Metrics
- **Eval loss**: 1.5426

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

## Comparison with Other Models

| Model | Parameters | Held-Out Test Accuracy | Training Time |
|-------|-----------|----------------------|---------------|
| **Qwen2.5-7B LoRA** | **7.66B (40.4M trainable)** | **100.00%** | ~7.7 min |
| Qwen3-8B LoRA | 8.23B (43.6M trainable) | 100.00% | ~10 min |
| BERT-large (5-fold CV) | 340M | 46.6% ± 6.3% | ~7.8 min |
| RoBERTa-base (5-fold CV) | 125M | 40.0% ± 6.5% | ~3.2 min |
| DistilBERT (5-fold CV) | 66M | 31.4% ± 3.3% | ~1.7 min |
| Random baseline | - | 20% | - |

---

## Key Findings

1. **Qwen2.5-7B matches Qwen3-8B performance** - Both achieve 100% on held-out test

2. **Slightly faster training** - 7.7 min vs 10 min for Qwen3-8B

3. **Smaller model, same accuracy** - 7.66B vs 8.23B parameters

4. **LoRA is highly effective** - Only 0.53% of parameters trainable

---

## Model Artifacts

- **Model path**: `~/models/hub/qwen2.5-7b-recipe-mpr-lora/`
- **Test split info**: `test_split.json`
- **Results**: `results.json`

---

## Reproducibility

```bash
# Training
cd /home/kevinlin/utece/ECE1508_Applied_Deep_Learning/qwen
./train.sh

# Evaluation (held-out test set)
./evaluate.sh
```
