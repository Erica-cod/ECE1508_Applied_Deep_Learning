# BERT Variants Experiments

Compare different BERT model variants on the Recipe-MPR dataset.

## Available Models

| Model | Parameters | Description | Expected Accuracy |
|-------|------------|-------------|-------------------|
| **DistilBERT** | 66M | Distilled BERT (baseline) | 69.4% ✅ Tested |
| **BERT-base** | 110M | Original BERT | 72-76% |
| **RoBERTa-base** | 125M | Robustly optimized BERT | 74-78% |
| **DeBERTa-v3** | 184M | State-of-the-art BERT variant | 76-80% |
| **BERT-large** | 340M | Larger BERT | 75-79% |

## Quick Start

### Train a Single Model

```bash
cd bert_experiments

# BERT-base (recommended first)
./train_bert_variant.sh bert-base-uncased

# RoBERTa-base (usually best)
./train_bert_variant.sh roberta-base

# DeBERTa-v3 (state-of-the-art)
./train_bert_variant.sh microsoft/deberta-v3-base

# BERT-large (most powerful)
./train_bert_variant.sh bert-large-uncased
```

### Train All Models (Sequential)

```bash
./train_all_variants.sh
```

**Warning**: This will take 60-90 minutes total!

### Evaluate All Models

```bash
./evaluate_all_variants.sh
```

Shows comparison table with all trained models.

## Individual Training Commands

### BERT-base (110M params)
```bash
./train_bert_variant.sh bert-base-uncased
```
- **Time**: ~12-15 mins
- **VRAM**: ~8-10 GB
- **Expected**: 72-76%

### RoBERTa-base (125M params)
```bash
./train_bert_variant.sh roberta-base
```
- **Time**: ~12-15 mins
- **VRAM**: ~8-10 GB
- **Expected**: 74-78%
- **Note**: Often outperforms BERT

### DeBERTa-v3-base (184M params)
```bash
./train_bert_variant.sh microsoft/deberta-v3-base
```
- **Time**: ~15-18 mins
- **VRAM**: ~10-12 GB
- **Expected**: 76-80%
- **Note**: State-of-the-art performance

### BERT-large (340M params)
```bash
./train_bert_variant.sh bert-large-uncased
```
- **Time**: ~20-25 mins
- **VRAM**: ~14-16 GB
- **Expected**: 75-79%
- **Note**: Most parameters, not always best

## Configuration

Each model uses optimized hyperparameters:

| Model | Epochs | LR | Batch Size | Notes |
|-------|--------|----|-----------| ------|
| BERT-base | 10 | 3e-5 | 16 | Balanced |
| RoBERTa-base | 10 | 3e-5 | 16 | Same as BERT |
| DeBERTa-v3 | 10 | 2e-5 | 16 | Lower LR |
| BERT-large | 8 | 2e-5 | 8 | Smaller batch |

All models use:
- Gradient accumulation: 2× (effective batch = 2× listed)
- Warmup steps: 200
- Weight decay: 0.01
- FP16: Enabled (faster training)
- Seed: 42 (reproducibility)

## Expected Results

Based on model capabilities and task complexity:

| Model | Accuracy | vs DistilBERT | vs Goal (75%) |
|-------|----------|---------------|---------------|
| DistilBERT | 69.4% | Baseline | -5.6% ❌ |
| BERT-base | 72-76% | +2-7% | -3% to +1% |
| RoBERTa-base | 74-78% | +5-9% | -1% to +3% ✅ |
| DeBERTa-v3 | 76-80% | +7-11% | +1% to +5% ✅ |
| BERT-large | 75-79% | +6-10% | 0% to +4% ✅ |

**Goal**: 75% accuracy

**Most likely to succeed**: RoBERTa-base or DeBERTa-v3

## Why Try Different Models?

### BERT-base
- Original BERT architecture
- Good baseline
- Well-studied and reliable
- 1.7× DistilBERT size

### RoBERTa-base
- Improved training procedure
- Often outperforms BERT
- Better handling of commonsense reasoning
- Same size as BERT-base

### DeBERTa-v3-base
- State-of-the-art architecture
- Disentangled attention
- Enhanced mask decoder
- Best performance/size ratio

### BERT-large
- Most parameters (340M)
- Highest capacity
- May overfit on 500 examples
- Requires more VRAM

## Hardware Requirements

| Model | Min VRAM | Recommended VRAM |
|-------|----------|------------------|
| BERT-base | 6 GB | 8 GB |
| RoBERTa-base | 6 GB | 8 GB |
| DeBERTa-v3 | 8 GB | 10 GB |
| BERT-large | 12 GB | 16 GB |

**Your hardware**: 30 GB VRAM ✅ Can run all models!

## Workflow

### Recommended Approach

1. **Start with RoBERTa-base** (most likely to hit 75%)
   ```bash
   ./train_bert_variant.sh roberta-base
   python ../scripts/evaluate_distilbert_recipe_mpr.py \
       --model-path ~/models/hub/roberta-base-recipe-mpr
   ```

2. **If RoBERTa doesn't hit 75%, try DeBERTa-v3**
   ```bash
   ./train_bert_variant.sh microsoft/deberta-v3-base
   python ../scripts/evaluate_distilbert_recipe_mpr.py \
       --model-path ~/models/hub/deberta-v3-base-recipe-mpr
   ```

3. **Compare all trained models**
   ```bash
   ./evaluate_all_variants.sh
   ```

### Batch Training Approach

Train all models overnight:
```bash
./train_all_variants.sh
```

Then evaluate and compare:
```bash
./evaluate_all_variants.sh
```

## Comparison with Qwen

| Aspect | BERT Variants | Qwen2.5-7B |
|--------|---------------|------------|
| **Size** | 66M - 340M | 7B |
| **Training Time** | 12-25 mins | 10-15 mins |
| **VRAM** | 6-16 GB | 20 GB |
| **Expected Accuracy** | 72-80% | 100% ✅ |
| **Architecture** | Encoder-only | Decoder-only |
| **Best Use** | Comparison study | Production |

**Qwen is superior** but BERT variants are useful for:
- Understanding model scaling
- Ablation studies
- Resource-constrained deployment
- Academic comparison

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
# Edit train_bert_variant.sh
# Change BATCH_SIZE from 16 to 8 or 4
```

Or use gradient accumulation:
```bash
# Already set to 2× in the script
# Can increase to 4× if needed
```

### Low Accuracy

Try:
1. More epochs (12-15 instead of 10)
2. Lower learning rate (2e-5 instead of 3e-5)
3. Different model (try RoBERTa or DeBERTa)

### Training Too Slow

Enable FP16 (already enabled):
```bash
--fp16
```

Or reduce logging frequency:
```bash
--logging-steps 100  # instead of 25
```

## File Structure

```
bert_experiments/
├── README.md                   # This file
├── train_bert_variant.sh       # Train single model
├── train_all_variants.sh       # Train all models
└── evaluate_all_variants.sh    # Evaluate and compare

~/models/hub/
├── bert-base-recipe-mpr/       # BERT-base checkpoint
├── roberta-base-recipe-mpr/    # RoBERTa checkpoint
├── deberta-v3-base-recipe-mpr/ # DeBERTa checkpoint
└── bert-large-recipe-mpr/      # BERT-large checkpoint
```

## Expected Timeline

| Task | Time |
|------|------|
| BERT-base training | 12-15 mins |
| RoBERTa-base training | 12-15 mins |
| DeBERTa-v3 training | 15-18 mins |
| BERT-large training | 20-25 mins |
| Each evaluation | 2-3 mins |
| **Total (sequential)** | **~70 mins** |

## Tips

1. **Start with RoBERTa**: Usually best performance
2. **Monitor training loss**: Should decrease steadily
3. **Check eval accuracy**: Should improve each epoch
4. **Compare results**: Use `evaluate_all_variants.sh`
5. **Consider Qwen**: If you need >80% accuracy

## References

- BERT: Devlin et al., 2019
- RoBERTa: Liu et al., 2019
- DeBERTa: He et al., 2021
- DistilBERT: Sanh et al., 2019

## Support

For issues:
1. Check VRAM usage: `nvidia-smi`
2. Review logs in model directories
3. Verify dataset path: `../data/500QA.json`
4. Check model compatibility with transformers version
