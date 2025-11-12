# Recipe-MPR Llama3 微调实验

本项目对比了微调的 Llama3-MPR-SFT 模型与 GPT-3 Embedding Baseline 在 Recipe-MPR 数据集上的性能。

## 📊 实验结果

| 模型 | 准确率 |
|------|--------|
| **Llama3-MPR-SFT** | **84.00%** |
| GPT-3 Embedding | 54.55% |

**提升**: +29.45 个百分点

## 📁 项目结构

```
llamaFineTune/
├── data/                          # 数据集
│   ├── train.jsonl               # 训练集 (300 samples)
│   ├── valid.jsonl               # 验证集 (100 samples)
│   └── test.jsonl                # 测试集 (100 samples)
│
├── Recipe-MPR/                    # 原始数据集和参考代码
│   └── data/500QA.json           # 原始 500 个食谱问答
│
├── outputs/                       # 训练输出
│   └── llama3-mpr-sft/
│       └── final/                # 最终微调模型
│
├── compare-result/                # 实验结果
│   ├── 最终实验报告.md            # 完整实验报告 ⭐
│   ├── detailed_errors.json      # 详细错误案例
│   ├── overall_stats.csv         # 总体统计
│   └── stats_by_type.csv         # 按查询类型统计
│
├── prep_mpr.py                    # 数据准备脚本
├── train_sft.py                   # 模型微调脚本
├── eval_mpr.py                    # 评估微调模型
├── eval_embedding_baseline.py     # 评估 embedding baseline
├── compare_runs.py                # 对比两个模型结果
│
├── embeddings_with_aspects.json   # GPT-3 预计算的 embeddings
├── mpr_preds.jsonl               # Llama3-MPR-SFT 预测结果
└── emb_preds.jsonl               # GPT-3 Embedding 预测结果
```

## 🚀 使用方法

### 1. 准备数据

```bash
python prep_mpr.py \
    --infile Recipe-MPR/data/500QA.json \
    --outdir data \
    --seed 42
```

### 2. 训练模型

```bash
python train_sft.py
```

需要：
- Llama-3.2-3B-Instruct 基础模型（放在 `~/models/Llama-3.2-3B-Instruct/`）
- 8GB+ GPU 显存
- 约 12-15 分钟训练时间

### 3. 评估模型

**评估微调模型**：
```bash
python eval_mpr.py \
    --data data/test.jsonl \
    --model_dir ~/models/Llama-3.2-3B-Instruct \
    --adapter_dir outputs/llama3-mpr-sft/final \
    --save_pred mpr_preds.jsonl
```

**评估 Embedding Baseline**：
```bash
python eval_embedding_baseline.py \
    --data data/test.jsonl \
    --raw_json Recipe-MPR/data/500QA.json \
    --emb embeddings_with_aspects.json \
    --save_pred emb_preds.jsonl
```

### 4. 对比结果

```bash
python compare_runs.py \
    --raw500 Recipe-MPR/data/500QA.json \
    --mpr_preds mpr_preds.jsonl \
    --emb_preds emb_preds.jsonl
```

## 📖 查看结果

完整的实验报告在 `compare-result/最终实验报告.md`

## 🔑 关键发现

1. **数据偏差问题**：发现并修复了原始数据中所有答案都在位置 A 的问题
2. **微调效果显著**：3B 模型经过微调后超越了通用的 GPT-3 Embedding
3. **任务特化重要**：针对任务的微调比模型规模更关键

## 📚 依赖环境

```
transformers
datasets
peft
bitsandbytes
torch
numpy
```

## 📄 许可

本项目基于 Recipe-MPR 数据集进行实验。

