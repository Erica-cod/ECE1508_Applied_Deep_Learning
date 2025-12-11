# Recipe-MPR Llama3 Fine-tuning Experiment

This project compares fine-tuned Llama models with GPT-3 Embedding Baseline on the Recipe-MPR dataset for recipe recommendation tasks.

## Prerequisites

Before running this project, you need to:

### 1. Download Recipe-MPR Dataset
```bash
git clone https://github.com/Interactive-NLP/Recipe-MPR.git
# Or download from: https://github.com/Interactive-NLP/Recipe-MPR
```

The dataset file should be at: `Recipe-MPR/data/500QA.json`

### 2. Download Llama Models

You need to download the following models from Hugging Face:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ~/models/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ~/models/Llama-3.2-3B-Instruct

# Option 2: Using git lfs
cd ~/models
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
git clone https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the complete list of dependencies.

### 4. (Optional) GPT-3 Embeddings

If you want to run the GPT-3 baseline:
- Get an OpenAI API key from https://platform.openai.com/api-keys
- Pre-computed embeddings are already provided in `testing_results/embeddings_with_aspects.json`


##  Project Structure

```
llamaFineTune/
├── data/                          # Dataset
│   ├── train.jsonl               # Training set (300 samples)
│   ├── valid.jsonl               # Validation set (100 samples)  
│   └── test.jsonl                # Test set (100 samples)
│
├── Recipe-MPR/                    # Original dataset and reference code
│   └── data/500QA.json           # Original 500 recipe QA pairs
│
├── outputs/                       # Training outputs
│   └── llama3-mpr-sft/
│       └── final/                # Final fine-tuned model
│
│
├── testing_results/               # All testing data and predictions
│   ├── embeddings_with_aspects.json  # GPT-3 pre-computed embeddings (101MB)
│   ├── mpr_preds_1b_zeroshot.jsonl   # 1B model zero-shot predictions
│   ├── mpr_preds_1b_ft_80_10_10.jsonl  # 1B model fine-tuned predictions
│   ├── mpr_preds_3b_zeroshot.jsonl   # 3B model zero-shot predictions
│   ├── mpr_preds_3b_ft_80_10_10.jsonl  # 3B model fine-tuned predictions
│   ├── emb_preds_80_10_10.jsonl      # GPT-3 embedding predictions
│   └── emb_preds.jsonl               # GPT-3 embedding predictions (legacy)
│
├── baselines/                     # GPT-3 baseline scripts
│   └── generate_GPT3_embeddings.py  # Generate embeddings via API
│
├── training_on_80_10_10.log       # Training logs (loss, metrics, time)
├── prep_mpr.py                    # Data preparation script
├── train_sft.py                   # Model fine-tuning script
├── eval_mpr.py                    # Evaluate fine-tuned model
├── eval_embedding_baseline.py     # Evaluate embedding baseline
├── compare_runs.py                # Compare two model results
├── analyze_by_query_type.py       # Analyze by query type
└── requirements.txt               # Python dependencies
```

## Usage

### 1. Prepare Data

```bash
python prep_mpr.py \
    --infile Recipe-MPR/data/500QA.json \
    --outdir data \
    --seed 42 \
    --split "80,10,10"  # train/valid/test split ratio
``` 
This generates:
- `data/train.jsonl`: Training set (400 samples for 80/10/10 split)
- `data/valid.jsonl`: Validation set (50 samples)
- `data/test.jsonl`: Test set (50 samples)

**Important**: Option orders are randomly shuffled to avoid position bias.

### 2. (Optional) Generate GPT-3 Embeddings

If you need to run the GPT-3 embedding baseline, first generate embeddings:

```bash
cd baselines
# Set your OpenAI API key
export OPENAI_API_KEY=your-key-here
python generate_GPT3_embeddings.py
```

This will:
- Pre-compute embeddings for all 500 queries
- Pre-compute embeddings for all 2,500 options (500 × 5)
- Cache results in `embeddings_with_aspects.json` (~101 MB)
- Support resumption if interrupted

**Note**: The pre-computed `embeddings_with_aspects.json` is already provided in `testing_results/` folder.

### 3. Train Model

```bash
python train_sft.py \
    --model_dir ~/models/Llama-3.2-3B-Instruct \
    --output_dir outputs/llama3-mpr-sft \
    --epochs 5
```

Requirements:
- Llama-3.2-3B-Instruct base model (in `~/models/Llama-3.2-3B-Instruct/`)
- 8GB+ GPU memory
- ~15 minutes training time (400 samples, 5 epochs)

**Training Details** (from `training_on_80_10_10.log`):
- **3B Model**: 
  - Trainable params: 2.3M (0.07% of 3.2B)
  - Training time: ~15 minutes
  - Final eval loss: 1.629
- **1B Model**:
  - Trainable params: 852K (0.07% of 1.2B)
  - Training time: ~6 minutes
  - Final eval loss: 1.717

### 4. Evaluate Models

**Evaluate fine-tuned model**:
```bash
python eval_mpr.py \
    --data data/test.jsonl \
    --model_dir ~/models/Llama-3.2-3B-Instruct \
    --adapter_dir outputs/llama3-mpr-sft/final \
    --save_pred mpr_preds.jsonl
```

**Evaluate Embedding Baseline**:
```bash
python eval_embedding_baseline.py \
    --data data/test.jsonl \
    --raw_json Recipe-MPR/data/500QA.json \
    --emb testing_results/embeddings_with_aspects.json \
    --save_pred emb_preds.jsonl
```

### 5. Compare Results

```bash
python compare_runs.py \
    --mpr_preds mpr_preds.jsonl \
    --emb_preds emb_preds.jsonl \
    --test_data data/test.jsonl
```

### 6. Analyze by Query Type

```bash
python analyze_by_query_type.py
```

This generates a detailed breakdown by query type (Specific, Analogical, Negated, Commonsense, Temporal) and provides LaTeX table code ready for paper insertion.

## Quick Start with Pre-computed Results

If you want to quickly see the results without training:

```bash
# Copy pre-computed results from testing_results/
cp testing_results/mpr_preds_*.jsonl .
cp testing_results/emb_preds_80_10_10.jsonl .

# Run analysis
python analyze_by_query_type.py
```

All testing data and predictions are available in `testing_results/` folder.

##  View Results

Complete experiment report in `compare-result/FINAL_REPORT.md`

All prediction files and testing data are organized in `testing_results/` folder:
- Model predictions (1B/3B, zero-shot/fine-tuned)
- GPT-3 embedding predictions
- Pre-computed embeddings cache

Training logs are available in `training_on_80_10_10.log`, which contains:
- Complete training process for both 1B and 3B models
- Loss curves and evaluation metrics at each epoch
- Training time: 3B (~15 min), 1B (~6 min)
- Final eval loss: 3B (1.629), 1B (1.717)

##  Key Findings

1. **Data Bias Issue**: Discovered and fixed critical bias where all answers were at position A
2. **No Data Leakage**: Train/validation/test sets are completely disjoint (verified with fixed seed=42)
3. **Fine-tuning Effectiveness**: 3B model achieves 86% accuracy with only 400 training samples
4. **Model Scale Matters**: 3B consistently outperforms 1B (86% vs 74%)
5. **Generative > Retrieval**: Even zero-shot Llama (80%) outperforms GPT-3 Embedding (56%)

##  Dependencies

All required packages are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `transformers==4.57.1` - Hugging Face Transformers
- `peft==0.17.1` - Parameter-Efficient Fine-Tuning (LoRA)
- `datasets==4.4.1` - Hugging Face Datasets
- `bitsandbytes==0.48.1` - 4-bit quantization
- `torch==2.5.1` - PyTorch with CUDA support
- `accelerate==1.11.0` - Distributed training utilities
- `openai==0.28.1` - OpenAI API (optional, for generating embeddings)

##  Data Split Methodology

The dataset is split using a deterministic random shuffle:

1. **Fixed Seed**: `random.seed(42)` ensures reproducible splits
2. **Shuffle**: All 500 indices are randomly shuffled
3. **Sequential Split**: Shuffled indices are split sequentially (no overlap possible)
   - Train: indices[0:400]
   - Valid: indices[400:450]
   - Test: indices[450:500]

**Verification**: Train/valid/test sets are mathematically guaranteed to be disjoint.

