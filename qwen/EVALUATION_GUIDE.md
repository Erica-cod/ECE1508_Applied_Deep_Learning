# Qwen Model Evaluation Guide

Comprehensive guide for evaluating fine-tuned Qwen models on Recipe-MPR.

## Available Evaluation Scripts

### 1. `evaluate.sh` - Evaluate Standard Model
Evaluate the standard Qwen model (trained with `train_qwen.sh`).

```bash
./evaluate.sh
```

**Output**:
- Overall accuracy with 75% goal check
- Per-query-type breakdown (Specific, Commonsense, Negated, Analogical, Temporal)
- 5 correct prediction examples
- 5 incorrect prediction examples
- Results saved to JSON

**Default model path**: `~/models/hub/qwen2.5-7b-recipe-mpr-lora`

**Custom model path**:
```bash
./evaluate.sh ~/path/to/your/model
```

---

### 2. `evaluate_aggressive.sh` - Evaluate Aggressive Model
Evaluate the aggressive Qwen model (trained with `train_qwen_aggressive.sh`).

```bash
./evaluate_aggressive.sh
```

**Output**: Same as `evaluate.sh` but for the aggressive model.

**Default model path**: `~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive`

---

### 3. `compare_models.sh` - Compare Standard vs Aggressive
Run both evaluations and show a side-by-side comparison.

```bash
./compare_models.sh
```

**Output**:
- Standard model results
- Aggressive model results
- Summary table comparing accuracies
- Goal achievement status for each

**Requires**: Both models must be trained first.

---

### 4. `evaluate_all.sh` - Evaluate All Available Models
Automatically finds and evaluates all trained Qwen models.

```bash
./evaluate_all.sh
```

**Output**:
- Evaluates each found model
- Summary table with all accuracies
- Comparison to 75% goal

**Finds models at**:
- `~/models/hub/qwen2.5-7b-recipe-mpr-lora`
- `~/models/hub/qwen2.5-7b-recipe-mpr-lora-aggressive`

---

## Direct Python Script Usage

For more control, use the Python script directly:

```bash
python evaluate_qwen_recipe_mpr.py \
    --model-path ~/models/hub/qwen2.5-7b-recipe-mpr-lora \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --dataset-path ../data/500QA.json \
    --num-examples 10 \
    --save-results results.json
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | Required | Path to fine-tuned LoRA model |
| `--base-model` | `Qwen/Qwen2.5-7B-Instruct` | Base model name |
| `--dataset-path` | `data/500QA.json` | Path to dataset |
| `--max-length` | 512 | Max sequence length |
| `--max-new-tokens` | 2 | Tokens to generate (for answer) |
| `--num-examples` | 5 | Number of examples to show |
| `--save-results` | None | Path to save JSON results |
| `--seed` | 42 | Random seed |
| `--batch-size` | 1 | Inference batch size |

---

## Understanding the Output

### Overall Performance Section

```
Overall Performance:
  Accuracy: 78.20% (391/500)
  ✓ Goal achieved! (target: 75.0%)
```

- **Accuracy**: Percentage of correct predictions
- **Count**: (correct/total) examples
- **Goal check**: Green ✓ if ≥75%, yellow if close, red ✗ if far

### Query Type Breakdown

```
Accuracy by Query Type:
  Type            Accuracy     Count
  ----------------------------------------
  Specific         82.78%      151
  Analogical       80.00%      30
  Negated          77.06%      109
  Commonsense      76.12%      268
  Temporal         71.88%      32
```

Shows performance on each reasoning type:
- **Specific**: Direct recipe requests
- **Commonsense**: Requires common sense reasoning
- **Negated**: Contains negation ("not", "without")
- **Analogical**: Requires analogical reasoning
- **Temporal**: Time/sequence-based reasoning

### Example Predictions

**Correct examples** show where the model succeeded:
```
Correct Predictions (showing 5):
  1. Q: I want to make a warm dish containing oysters...
     Model answered: A
     Answer: Simple creamy oyster soup
     Query types: Specific, Commonsense
```

**Incorrect examples** help identify failure patterns:
```
Incorrect Predictions (showing 5):
  1. Q: Today's really hot and I'm craving for a peach...
     Model: B - A jar of peaches covered in peach wine
     Correct: C - Gelato made from peach, heavy cream
     Generated: 'B'
     Query types: Specific, Temporal
```

---

## Saved Results JSON

When using `--save-results`, a detailed JSON file is saved:

```json
{
  "metrics": {
    "overall_accuracy": 78.20,
    "correct": 391,
    "total": 500,
    "query_type_accuracy": {
      "Specific": 82.78,
      "Commonsense": 76.12,
      ...
    },
    "query_type_counts": {
      "Specific": 151,
      ...
    }
  },
  "predictions": [
    {
      "question": "...",
      "choices": [...],
      "correct_idx": 0,
      "correct_letter": "A",
      "predicted_idx": 0,
      "predicted_letter": "A",
      "generated_text": "A",
      "is_correct": true,
      "query_types": ["Specific"],
      ...
    },
    ...
  ]
}
```

Use this for:
- Detailed error analysis
- Custom visualizations
- Comparison across runs
- Paper/report figures

---

## Workflow Examples

### Basic Workflow

1. **Train model**:
   ```bash
   ./train_qwen.sh
   ```

2. **Evaluate**:
   ```bash
   ./evaluate.sh
   ```

3. **Check results** - Did you hit 75%?
   - ✅ Yes → You're done!
   - ❌ No → Try aggressive training

---

### Optimization Workflow

1. **Train standard**:
   ```bash
   ./train_qwen.sh
   ```

2. **Evaluate standard**:
   ```bash
   ./evaluate.sh
   ```
   Result: 77.2%

3. **Train aggressive**:
   ```bash
   ./train_qwen_aggressive.sh
   ```

4. **Compare both**:
   ```bash
   ./compare_models.sh
   ```

5. **Choose best** based on accuracy and query type performance

---

### Detailed Analysis Workflow

1. **Train model**:
   ```bash
   ./train_qwen.sh
   ```

2. **Run detailed evaluation** with more examples:
   ```bash
   python evaluate_qwen_recipe_mpr.py \
       --model-path ~/models/hub/qwen2.5-7b-recipe-mpr-lora \
       --num-examples 20 \
       --save-results detailed_results.json
   ```

3. **Analyze results**:
   ```bash
   # View JSON results
   cat detailed_results.json | jq '.metrics'

   # Find all incorrect temporal predictions
   cat detailed_results.json | jq '.predictions[] | select(.is_correct == false and (.query_types | contains(["Temporal"])))'
   ```

4. **Identify patterns**:
   - Which query types have lowest accuracy?
   - What kinds of errors is the model making?
   - Are there specific recipe categories causing issues?

---

## Comparing to DistilBERT Baseline

To compare Qwen performance with DistilBERT:

1. **Evaluate DistilBERT** (if not done):
   ```bash
   cd ../distilbert
   python ../scripts/evaluate_distilbert_recipe_mpr.py \
       --model-path ~/models/hub/distilbert-finetuned-recipe-mpr \
       --save-results distilbert_results.json
   ```

2. **Evaluate Qwen**:
   ```bash
   cd ../qwen
   ./evaluate.sh
   ```

3. **Compare results**:
   - DistilBERT: 69.4% overall
   - Qwen: ??? % overall
   - Improvement: +X.X%

4. **Compare by query type**:
   - Which types improved most?
   - Where is Qwen strongest?
   - Did Temporal performance improve? (was 62.5% in DistilBERT)

---

## Troubleshooting

### "Model not found" Error

**Problem**: Script can't find the model.

**Solutions**:
```bash
# Check if model exists
ls ~/models/hub/

# Verify model path in script
cat evaluate.sh | grep MODEL_PATH

# Manually specify path
python evaluate_qwen_recipe_mpr.py --model-path /full/path/to/model
```

---

### Out of Memory During Evaluation

**Problem**: GPU runs out of memory.

**Solutions**:
```bash
# Reduce batch size
python evaluate_qwen_recipe_mpr.py \
    --model-path ~/models/hub/qwen2.5-7b-recipe-mpr-lora \
    --batch-size 1

# Use CPU (slower but works)
CUDA_VISIBLE_DEVICES="" python evaluate_qwen_recipe_mpr.py ...
```

---

### Model Generates Invalid Answers

**Problem**: Model outputs something other than A/B/C/D/E.

**Observation in output**:
```
Predicted: X - INVALID
```

**Causes**:
- Model not properly fine-tuned
- Insufficient training epochs
- Wrong prompt format

**Solutions**:
- Train for more epochs
- Check training loss converged
- Verify dataset format

---

### Evaluation is Slow

**Problem**: Evaluation takes too long.

**Current**: ~2-3 minutes for 500 examples

**To speed up**:
```bash
# Use larger batch size (if memory allows)
python evaluate_qwen_recipe_mpr.py \
    --batch-size 4 \
    ...

# Reduce max tokens (already at 2, optimal)
--max-new-tokens 2

# Evaluate on subset for quick checks
python evaluate_qwen_recipe_mpr.py ... --num-examples 100
```

---

## Tips for Best Results

### 1. Run Multiple Seeds

For more robust results, train and evaluate with different seeds:

```bash
for seed in 42 123 456; do
    python ../scripts/finetune_qwen_recipe_mpr.py \
        --seed $seed \
        --output-dir ~/models/hub/qwen-seed-$seed \
        ...

    python evaluate_qwen_recipe_mpr.py \
        --model-path ~/models/hub/qwen-seed-$seed \
        --seed $seed \
        --save-results results_seed_${seed}.json
done
```

Then average the accuracies.

### 2. Focus on Weak Query Types

If Temporal is still weak after training:
- Check Temporal examples specifically
- Consider data augmentation
- Try different LoRA hyperparameters

### 3. Use Saved Results for Analysis

The JSON results file is great for:
```python
import json

# Load results
with open('eval_results.json') as f:
    results = json.load(f)

# Analyze
temporal_preds = [
    p for p in results['predictions']
    if 'Temporal' in p['query_types']
]

accuracy = sum(p['is_correct'] for p in temporal_preds) / len(temporal_preds)
print(f"Temporal accuracy: {accuracy:.2%}")
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Evaluate standard model | `./evaluate.sh` |
| Evaluate aggressive model | `./evaluate_aggressive.sh` |
| Compare both models | `./compare_models.sh` |
| Evaluate all models | `./evaluate_all.sh` |
| Custom evaluation | `python evaluate_qwen_recipe_mpr.py --model-path PATH` |
| Save detailed results | Add `--save-results results.json` |
| Show more examples | Add `--num-examples 10` |

---

## Expected Results

Based on Qwen2.5-7B capabilities and LoRA fine-tuning:

| Configuration | Expected Accuracy |
|--------------|-------------------|
| Standard (r=16, 5 epochs) | 77-80% |
| Aggressive (r=32, 10 epochs) | 80-82% |

**By query type** (expected):
- Specific: 82-85%
- Analogical: 78-83%
- Negated: 75-80%
- Commonsense: 76-82%
- Temporal: 70-78% (biggest improvement over DistilBERT's 62.5%)

All should comfortably exceed the 75% goal! 🎯
