# -*- coding: utf-8 -*-
"""
批量训练多个 Llama 模型并对比性能
"""
import os
import subprocess
import json
import time

# 定义要训练的模型
MODELS = {
    "llama-3.2-1b": {
        "path": "~/models/Llama-3.2-1B-Instruct",
        "output": "outputs/llama-3.2-1b-mpr-sft"
    },
    "llama-3.2-3b": {
        "path": "~/models/Llama-3.2-3B-Instruct",
        "output": "outputs/llama-3.2-3b-mpr-sft"
    },
    # 如果你想训练更大的模型，取消下面的注释
    # "llama-3-4b": {
    #     "path": "~/models/Meta-Llama-3-4B-Instruct",
    #     "output": "outputs/llama-3-4b-mpr-sft"
    # },
    # "llama-3-8b": {
    #     "path": "~/models/Meta-Llama-8B-Instruct",
    #     "output": "outputs/llama-3-8b-mpr-sft"
    # },
}

def train_model(model_name, model_config):
    """训练单个模型"""
    print("=" * 70)
    print(f"开始训练: {model_name}")
    print("=" * 70)
    
    model_dir = os.path.expanduser(model_config["path"])
    output_dir = model_config["output"]
    
    # 创建临时训练脚本
    train_script = f"""
import os, json, inspect
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments,
                          DataCollatorForLanguageModeling, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_DIR = "{model_dir}"
OUTPUT_DIR = "{output_dir}"

def load_jsonl(p):
    rows=[]
    with open(p,'r',encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def ensure_answer_suffix(txt:str)->str:
    t = txt.rstrip()
    if not t.endswith("Answer:"):
        t = t + "\\nAnswer:"
    return t

def join_prompt_and_label(rec):
    return {{"text": ensure_answer_suffix(rec["text"]) + " " + rec["label"].strip()}}

def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy", "epoch")
    else:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "epoch")
    return TrainingArguments(**kwargs)

# 1) 读数据
train_rows = [join_prompt_and_label(r) for r in load_jsonl("data/train.jsonl")]
valid_rows = [join_prompt_and_label(r) for r in load_jsonl("data/valid.jsonl")]
ds_train = Dataset.from_list(train_rows)
ds_valid = Dataset.from_list(valid_rows)

# 2) Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 3) 量化配置
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
)

# 4) 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=bnb,
    device_map="auto",
    offload_folder="offload",
    max_memory={{0: "8GiB", "cpu": "32GiB"}},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True
)
model.config.use_cache = False

# 5) LoRA 适配
peft_cfg = LoraConfig(
    r=4, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# 6) 数据 tokenize
def tok_fn(batch):
    out = tok(batch["text"], truncation=True, max_length=128)
    out["labels"] = out["input_ids"].copy()
    return out

ds_train = ds_train.map(tok_fn, remove_columns=["text"], batched=True)
ds_valid = ds_valid.map(tok_fn, remove_columns=["text"], batched=True)

# 7) Data collator
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# 8) 训练参数
args = make_training_args(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=5,
    logging_steps=10,
    eval_strategy="no",
    save_strategy="epoch",
    bf16=False, fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    report_to=[]
)

# 9) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    data_collator=collator,
    tokenizer=tok
)

trainer.train()
trainer.save_model(OUTPUT_DIR + "/final")
print(f"\\n模型已保存到: {{OUTPUT_DIR}}/final")
"""
    
    # 保存并执行
    temp_script = f"temp_train_{model_name}.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(train_script)
    
    start_time = time.time()
    try:
        result = subprocess.run(["python", temp_script], check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        elapsed = time.time() - start_time
        print(f"\n训练完成！耗时: {elapsed/60:.1f} 分钟")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败！")
        print(e.stderr)
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_script):
            os.remove(temp_script)

def evaluate_model(model_name, model_config):
    """评估单个模型"""
    print(f"\n评估 {model_name}...")
    
    model_dir = os.path.expanduser(model_config["path"])
    adapter_dir = model_config["output"] + "/final"
    pred_file = f"mpr_preds_{model_name}.jsonl"
    
    cmd = [
        "python", "eval_mpr.py",
        "--data", "data/test.jsonl",
        "--model_dir", model_dir,
        "--adapter_dir", adapter_dir,
        "--save_pred", pred_file,
        "--max_new_tokens", "2"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # 读取准确率
        with open(pred_file, 'r') as f:
            preds = [json.loads(line) for line in f if line.strip()]
        correct = sum(1 for p in preds if p.get('pred_letter') == p.get('gold_letter'))
        acc = correct / len(preds) * 100
        
        return {
            "model": model_name,
            "accuracy": acc,
            "correct": correct,
            "total": len(preds),
            "pred_file": pred_file
        }
    except Exception as e:
        print(f"评估失败: {e}")
        return None

def main():
    print("=" * 70)
    print("多模型训练与对比实验")
    print("=" * 70)
    print()
    
    print("将训练以下模型:")
    for name, config in MODELS.items():
        print(f"  - {name}: {config['path']}")
    print()
    
    input("按 Enter 开始训练...")
    
    results = []
    
    # 训练所有模型
    for model_name, model_config in MODELS.items():
        success = train_model(model_name, model_config)
        if not success:
            print(f"⚠️  {model_name} 训练失败，跳过评估")
            continue
        
        # 评估模型
        result = evaluate_model(model_name, model_config)
        if result:
            results.append(result)
    
    # 生成对比报告
    print("\n" + "=" * 70)
    print("多模型对比结果")
    print("=" * 70)
    print()
    
    if results:
        print("| 模型 | 准确率 | 正确数/总数 |")
        print("|------|--------|-------------|")
        for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            print(f"| {r['model']:<20} | {r['accuracy']:.2f}% | {r['correct']}/{r['total']} |")
        print()
        
        # 保存结果
        with open("compare-result/multi_model_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("详细结果已保存到: compare-result/multi_model_results.json")
    else:
        print("没有成功的评估结果")

if __name__ == "__main__":
    main()

