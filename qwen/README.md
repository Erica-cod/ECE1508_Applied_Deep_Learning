# Qwen2.5-7B Fine-tuning for Recipe-MPR

Training Qwen2.5-7B-Instruct on Recipe-MPR using LoRA (Parameter-Efficient Fine-Tuning).

## Why Qwen?

**Advantages over DistilBERT**:
- 🚀 **Much larger** (7B vs 66M params) - Better understanding of complex reasoning
- 🧠 **Instruction-tuned** - Already trained to follow task instructions
- 📈 **Higher ceiling** - Potential for 80-85%+ accuracy vs DistilBERT's ~72-74%
- 💬 **Generative** - Can explain reasoning, not just pick answers

**With LoRA**:
- ⚡ **Efficient** - Only trains ~1-2% of parameters
- 💾 **Low memory** - Fits in 30GB VRAM easily
- ⏱️ **Fast** - Training takes 10-20 minutes

## Current Baseline
- **DistilBERT**: 69.4% accuracy (3 epochs)
- **Target with Qwen**: 75-80%+ accuracy

## Quick Start

### 1. Install Dependencies
```bash
pip install transformers peft accelerate bitsandbytes
```

### 2. Run Training
```bash
cd qwen
./train_qwen.sh
```

This will:
- Download Qwen2.5-7B-Instruct (~15GB)
- Train with LoRA for 5 epochs (~10-15 mins)
- Save model to `~/models/hub/qwen2.5-7b-recipe-mpr-lora/`

### 3. Run Evaluation
```bash
./evaluate.sh
```

This will:
- Evaluate on full 500QA dataset
- Show accuracy with 75% goal check
- Display per-query-type breakdown
- Show example predictions
- Save results to JSON

## Training Scripts

### `train_qwen.sh` - Standard Training
**Recommended first try**

Configuration:
- **LoRA rank**: 16 (good balance)
- **Epochs**: 5
- **Learning rate**: 2e-4
- **Effective batch size**: 16
- **VRAM usage**: ~18-22 GB

**Expected**: 75-80% accuracy

### `train_qwen_aggressive.sh` - Maximum Performance
**Use if standard doesn't reach 75%**

Configuration:
- **LoRA rank**: 32 (more parameters)
- **Epochs**: 10
- **Learning rate**: 1.5e-4
- **VRAM usage**: ~20-24 GB

**Expected**: 80-85% accuracy

## LoRA Hyperparameters

### What is LoRA?
LoRA (Low-Rank Adaptation) adds small trainable matrices to the model while keeping the base model frozen. This dramatically reduces:
- Trainable parameters (~99% fewer)
- Memory usage (~70% less VRAM)
- Training time (~50% faster)

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | Rank of LoRA matrices (higher = more capacity) |
| `lora_alpha` | 32 | Scaling factor (typically 2×r) |
| `lora_dropout` | 0.05 | Dropout rate for regularization |

**Rule of thumb**:
- Small dataset (500 examples): r=8-16
- Medium dataset (5K examples): r=16-32
- Large dataset (50K+ examples): r=32-64

## Training Details

### Data Format
Qwen receives prompts like:
```
Given the following recipe question and options, select the best answer.

Question: I want to make a warm dish containing oysters

Options:
A) Simple creamy oyster soup
B) Seasoned salted crackers shaped like oysters
C) Creamy clam chowder made with whole milk and baby clams
D) Tomato mussel soup containing dry white wine
E) Warm vegetable soup containing tomatoes, peas, corn, carrots, and potatoes

Answer: A
```

### Model Architecture
- **Base**: Qwen2.5-7B-Instruct (7 billion parameters)
- **LoRA targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params**: ~30-40 million (0.5% of total)

### Hardware Requirements
| Configuration | VRAM | Training Time |
|---------------|------|---------------|
| Standard (r=16) | 18-22 GB | 10-15 mins |
| Aggressive (r=32) | 20-24 GB | 15-25 mins |

Your 30GB VRAM is perfect for both!

## Evaluation Scripts

After training, use the dedicated evaluation scripts:

### Quick Evaluation
```bash
./evaluate.sh                 # Evaluate standard model
./evaluate_aggressive.sh      # Evaluate aggressive model
```

### Compare Models
```bash
./compare_models.sh          # Compare standard vs aggressive
./evaluate_all.sh            # Evaluate all available models
```

### Advanced Evaluation
```bash
python evaluate_qwen_recipe_mpr.py \
    --model-path ~/models/hub/qwen2.5-7b-recipe-mpr-lora \
    --num-examples 10 \
    --save-results detailed_results.json
```

**Output includes**:
- ✅ Overall accuracy (with 75% goal check)
- 📊 Per-query-type breakdown (Specific, Commonsense, Negated, Analogical, Temporal)
- 📝 Example correct/incorrect predictions
- 💾 Detailed JSON results file

**For complete evaluation documentation**, see: `EVALUATION_GUIDE.md`

## Comparison: Qwen vs DistilBERT

| Aspect | DistilBERT | Qwen2.5-7B |
|--------|------------|------------|
| Parameters | 66M | 7B (100× larger) |
| Architecture | Encoder-only | Decoder-only (GPT-style) |
| Training method | Multiple-choice head | LoRA causal LM |
| VRAM usage | ~2-4 GB | ~18-22 GB (with LoRA) |
| Training time | ~5 mins | ~10-15 mins |
| Baseline accuracy | 69.4% | TBD |
| Expected max | ~72-74% | ~80-85% |
| Can explain reasoning | ❌ | ✅ |

## Tips for Best Results

### 1. Start with Standard Config
```bash
./train_qwen.sh
```
Check if it reaches 75%. If yes, you're done!

### 2. If Below 75%, Try Aggressive
```bash
./train_qwen_aggressive.sh
```
Larger LoRA rank + more epochs usually helps.

### 3. Experiment with Learning Rate
Lower LR for more stable convergence:
```bash
# Edit train_qwen.sh
LEARNING_RATE=1e-4  # Instead of 2e-4
```

### 4. Increase LoRA Rank
For maximum capacity:
```bash
# Edit train_qwen.sh
LORA_R=64
LORA_ALPHA=128
```
Note: Uses ~24-28 GB VRAM

### 5. Monitor Training
Watch for:
- ✅ Loss decreasing smoothly
- ✅ Eval loss tracking train loss
- ⚠️ Eval loss increasing = overfitting (stop early)

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
BATCH_SIZE=2          # Instead of 4
GRAD_ACCUM=8          # Instead of 4 (keep effective batch same)
```

Or use smaller LoRA:
```bash
LORA_R=8              # Instead of 16
LORA_ALPHA=16
```

### Model Not Learning
- Increase learning rate: `3e-4` instead of `2e-4`
- Increase LoRA rank: `32` instead of `16`
- Train longer: `10` epochs instead of `5`

### Accuracy Plateaus Early
- Decrease learning rate: `1e-4` instead of `2e-4`
- Increase warmup: `200` steps instead of `100`
- Add more regularization: `lora_dropout=0.1`

## File Structure

```
qwen/
├── train_qwen.sh              # Standard training script
├── train_qwen_aggressive.sh   # Aggressive training script
└── README.md                  # This file

scripts/
├── finetune_qwen_recipe_mpr.py    # Training code
└── evaluate_qwen_recipe_mpr.py    # Evaluation code

Output (after training):
~/models/hub/qwen2.5-7b-recipe-mpr-lora/
├── adapter_config.json        # LoRA configuration
├── adapter_model.safetensors  # LoRA weights (~100MB)
├── tokenizer files           # Tokenizer
├── eval_results.json         # Detailed results
└── training logs
```

## Expected Performance

Based on model size and task complexity:

| Query Type | DistilBERT | Qwen (Expected) |
|------------|------------|-----------------|
| Specific | 74.8% | **80-85%** |
| Analogical | 73.3% | **78-83%** |
| Negated | 69.7% | **75-80%** |
| Commonsense | 68.7% | **75-82%** |
| Temporal | 62.5% | **70-78%** |
| **Overall** | **69.4%** | **77-82%** |

Qwen's instruction-following and reasoning abilities should particularly help with:
- **Temporal reasoning** (currently weakest at 62.5%)
- **Commonsense** (largest category, needs improvement)
- **Negated questions** (requires careful reading)

## Next Steps

1. **Run standard training**:
   ```bash
   cd qwen && ./train_qwen.sh
   ```

2. **Check results** - Look for 75%+ accuracy

3. **If below 75%**, run aggressive:
   ```bash
   ./train_qwen_aggressive.sh
   ```

4. **Compare with DistilBERT**:
   - Which query types improved most?
   - Is the gap consistent across types?
   - Check example predictions for insights

5. **Optional**: Try Qwen2.5-14B for even better performance (uses ~26-30GB VRAM)

Good luck! 🚀
