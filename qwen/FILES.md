# Qwen Directory - File Overview

Complete reference for all files in the `qwen/` directory.

## Directory Structure

```
qwen/
├── README.md                      # Main documentation
├── EVALUATION_GUIDE.md            # Evaluation documentation
├── FILES.md                       # This file
│
├── train_qwen.sh                  # Standard training script
├── train_qwen_aggressive.sh       # Aggressive training script
│
├── evaluate.sh                    # Evaluate standard model
├── evaluate_aggressive.sh         # Evaluate aggressive model
├── compare_models.sh              # Compare both models
├── evaluate_all.sh                # Evaluate all available models
└── evaluate_qwen_recipe_mpr.py    # Python evaluation script
```

## File Descriptions

### Documentation

#### `README.md`
**Purpose**: Main guide for Qwen fine-tuning
**Contains**:
- Quick start instructions
- Training script descriptions
- LoRA hyperparameter explanations
- Model architecture details
- Comparison with DistilBERT
- Troubleshooting tips

**Read this first!**

---

#### `EVALUATION_GUIDE.md`
**Purpose**: Comprehensive evaluation documentation
**Contains**:
- All evaluation script usage
- Output interpretation
- Workflow examples
- JSON results format
- Advanced analysis tips
- Troubleshooting

**Read this for evaluation details!**

---

#### `FILES.md` (This file)
**Purpose**: Quick reference for all files
**Contains**:
- File structure
- Description of each file
- Quick usage guide

---

### Training Scripts

#### `train_qwen.sh` ⭐
**Purpose**: Standard training configuration
**What it does**:
- Downloads Qwen2.5-7B-Instruct if needed
- Trains with LoRA (r=16, 5 epochs)
- Saves to `~/models/hub/qwen2.5-7b-recipe-mpr-lora/`

**Usage**:
```bash
./train_qwen.sh
```

**Expected result**: 77-80% accuracy

**Training time**: ~10-15 minutes

**VRAM usage**: ~18-22 GB

**Configuration**:
- LoRA rank: 16
- LoRA alpha: 32
- Epochs: 5
- Learning rate: 2e-4
- Batch size: 4 (effective 16 with grad accum)
- Max length: 512

---

#### `train_qwen_aggressive.sh`
**Purpose**: Maximum performance training
**What it does**:
- Trains with larger LoRA (r=32, 10 epochs)
- Saves to `~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive/`

**Usage**:
```bash
./train_qwen_aggressive.sh
```

**Expected result**: 80-82% accuracy

**Training time**: ~15-25 minutes

**VRAM usage**: ~20-24 GB

**Configuration**:
- LoRA rank: 32 (2× standard)
- LoRA alpha: 64
- Epochs: 10 (2× standard)
- Learning rate: 1.5e-4 (lower for stability)
- Everything else same as standard

**When to use**: If standard model doesn't reach 75%

---

### Evaluation Scripts

#### `evaluate.sh` ⭐
**Purpose**: Evaluate standard model
**What it does**:
- Loads model from `~/models/hub/qwen2.5-7b-recipe-mpr-lora/`
- Evaluates on all 500 examples
- Shows results with 75% goal check
- Saves JSON results

**Usage**:
```bash
./evaluate.sh
```

**Or with custom model**:
```bash
./evaluate.sh /path/to/model
```

**Output**:
- Overall accuracy
- Per-query-type breakdown
- 5 correct examples
- 5 incorrect examples
- Results saved to `{model_path}/eval_results.json`

**Time**: ~2-3 minutes

---

#### `evaluate_aggressive.sh`
**Purpose**: Evaluate aggressive model
**What it does**:
- Same as `evaluate.sh` but for aggressive model
- Loads from `~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive/`

**Usage**:
```bash
./evaluate_aggressive.sh
```

**Output**: Same as `evaluate.sh`

---

#### `compare_models.sh`
**Purpose**: Compare standard vs aggressive models
**What it does**:
- Runs `evaluate.sh` on standard model
- Runs `evaluate_aggressive.sh` on aggressive model
- Shows side-by-side comparison

**Usage**:
```bash
./compare_models.sh
```

**Requires**: Both models must be trained

**Output**:
```
Standard model:   78.20%
Aggressive model: 81.40%

Goal: 75.00%

Standard model: ✅ Goal achieved
Aggressive model: ✅ Goal achieved
```

**Time**: ~4-6 minutes (both evaluations)

---

#### `evaluate_all.sh`
**Purpose**: Evaluate all available models
**What it does**:
- Automatically finds all trained Qwen models
- Evaluates each one
- Shows summary table

**Usage**:
```bash
./evaluate_all.sh
```

**Finds models at**:
- `~/models/hub/qwen2.5-7b-recipe-mpr-lora`
- `~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive`

**Output**:
```
Model                 Accuracy
────────────────────────────────────────
Standard              78.20%
Aggressive            81.40%

Goal: 75.00%
```

**Time**: Depends on number of models found

---

#### `evaluate_qwen_recipe_mpr.py`
**Purpose**: Python evaluation script (used by shell scripts)
**What it does**:
- Loads Qwen model with LoRA weights
- Runs inference on dataset
- Calculates accuracy metrics
- Saves detailed results

**Direct usage**:
```bash
python evaluate_qwen_recipe_mpr.py \
    --model-path ~/models/hub/qwen2.5-7b-recipe-mpr-lora \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --dataset-path ../data/500QA.json \
    --num-examples 10 \
    --save-results results.json
```

**Parameters**:
- `--model-path` (required): Path to LoRA checkpoint
- `--base-model`: Base Qwen model name
- `--dataset-path`: Path to 500QA.json
- `--max-length`: Max sequence length (default: 512)
- `--max-new-tokens`: Tokens to generate (default: 2)
- `--num-examples`: Examples to show (default: 5)
- `--save-results`: Path to save JSON
- `--seed`: Random seed (default: 42)
- `--batch-size`: Inference batch size (default: 1)

**Note**: Usually called via shell scripts, not directly

---

## Quick Usage Guide

### First Time Setup

1. **Install dependencies**:
   ```bash
   pip install transformers peft accelerate bitsandbytes
   ```

2. **Train standard model**:
   ```bash
   cd qwen
   ./train_qwen.sh
   ```

3. **Evaluate**:
   ```bash
   ./evaluate.sh
   ```

---

### Standard Workflow

```bash
# Train
./train_qwen.sh

# Evaluate
./evaluate.sh

# If below 75%, train aggressive
./train_qwen_aggressive.sh

# Compare
./compare_models.sh
```

---

### Evaluation Only

If models are already trained:

```bash
# Evaluate one model
./evaluate.sh

# Evaluate all models
./evaluate_all.sh

# Compare two models
./compare_models.sh
```

---

## File Dependencies

### Training Scripts Require:
- `../scripts/finetune_qwen_recipe_mpr.py`
- `../data/500QA.json`
- Internet (first run, to download Qwen model)

### Evaluation Scripts Require:
- `evaluate_qwen_recipe_mpr.py` (in same directory)
- `../data/500QA.json`
- Trained model (in `~/models/hub/`)
- Base Qwen model (downloaded during training)

---

## Output Locations

### Models
```
~/models/hub/
├── qwen2.5-7b-recipe-mpr-lora/          # Standard model
│   ├── adapter_config.json               # LoRA config
│   ├── adapter_model.safetensors         # LoRA weights (~100MB)
│   ├── tokenizer files                   # Tokenizer
│   └── eval_results.json                 # Evaluation results (after eval)
│
└── qwen2.5-7b-recipe-mpr-lora-aggressive/  # Aggressive model
    └── (same structure as above)
```

### Cache
```
~/models/hub/
└── models--Qwen--Qwen2.5-7B-Instruct/   # Base model cache (~15GB)
    └── snapshots/
        └── <hash>/
            └── (model files)
```

---

## Common Tasks

### Check if model exists
```bash
ls -lh ~/models/hub/ | grep qwen
```

### View training logs
```bash
# Last training run (if saved)
cat ~/models/hub/qwen2.5-7b-recipe-mpr-lora/trainer_state.json
```

### View evaluation results
```bash
cat ~/models/hub/qwen2.5-7b-recipe-mpr-lora/eval_results.json | jq '.metrics'
```

### Delete a model
```bash
rm -rf ~/models/hub/qwen2.5-7b-recipe-mpr-lora
```

### Re-evaluate a model
```bash
./evaluate.sh ~/models/hub/qwen2.5-7b-recipe-mpr-lora
```

---

## File Size Reference

| File/Directory | Size | Description |
|----------------|------|-------------|
| Base Qwen model | ~15 GB | Downloaded once, cached |
| LoRA checkpoint | ~100 MB | Small! (vs 15GB full model) |
| Tokenizer | ~1 MB | Saved with checkpoint |
| eval_results.json | ~500 KB | Detailed predictions |
| Training logs | ~100 KB | Trainer state |

**Total for one checkpoint**: ~100 MB (LoRA weights only)
**Total with base model**: ~15 GB (base model + LoRA)

---

## Script Execution Order

### Typical workflow:
1. `train_qwen.sh` → Trains model
2. `evaluate.sh` → Evaluates trained model
3. (Optional) `train_qwen_aggressive.sh` → Train better model
4. (Optional) `compare_models.sh` → Compare both

### Evaluation only:
1. `evaluate.sh` or `evaluate_aggressive.sh` or `evaluate_all.sh`

### No dependencies between evaluation scripts - run in any order!

---

## Related Files (Outside qwen/)

### In `scripts/`:
- `finetune_qwen_recipe_mpr.py` - Training implementation (called by train_*.sh)
- `evaluate_qwen_recipe_mpr.py` - Evaluation implementation (copied to qwen/)

### In `data/`:
- `500QA.json` - Recipe-MPR dataset (500 examples)

### In project root:
- `MODEL_COMPARISON.md` - Compare DistilBERT vs Qwen

---

## Need Help?

- **Training questions**: See `README.md`
- **Evaluation questions**: See `EVALUATION_GUIDE.md`
- **File locations**: This file!
- **Issues**: Check both README files for troubleshooting sections
