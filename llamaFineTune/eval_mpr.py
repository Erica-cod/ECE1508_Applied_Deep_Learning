# -*- coding: utf-8 -*-
import os, json, re, argparse, random, torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel

    _has_peft = True
except Exception:
    _has_peft = False

A2E = "ABCDE"


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def first_A2E(s: str) -> Optional[str]:
    m = re.search(r"[A-E]", s.upper())
    return m.group(0) if m else None


def ensure_answer_suffix(txt: str) -> str:
    t = txt.rstrip()
    return t if t.endswith("Answer:") else (t + "\nAnswer:")


def format_fewshot(records: List[Dict], k: int) -> str:
    """records: [{'text': '...Answer:', 'label': 'C'}, ...]"""
    if k <= 0: return ""
    blocks = []
    for r in records[:k]:
        blocks.append(ensure_answer_suffix(r["text"]) + " " + r["label"].strip())
    return "\n\n".join(blocks) + "\n\n"


def build_prompt(example: Dict, fewshot_block: str) -> str:
    # 我们的样本结构是 {"text": "...Answer:", "label": "C"} （来自 prep_mpr.py）
    return fewshot_block + ensure_answer_suffix(example["text"])


def load_model_and_tokenizer(model_dir: str, adapter_dir: Optional[str], offload_folder: Optional[str],
                             use_flash_attn: bool):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    )
    kwargs = dict(
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if offload_folder:
        kwargs["offload_folder"] = offload_folder
        kwargs["max_memory"] = {0: "7.2GiB", "cpu": "32GiB"}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["attn_implementation"] = "sdpa"

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, **kwargs)
    model.eval()

    if adapter_dir:
        if not _has_peft:
            raise RuntimeError("需要 peft 才能加载 LoRA 适配器，请先 pip install peft")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model.eval()

    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl", help="测试集路径（prep_mpr.py 生成的）")
    ap.add_argument("--model_dir", required=True, help="本地基础模型目录，如 ~/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter_dir", default=None, help="LoRA 目录，如 outputs/llama3-mpr-sft/final；留空=零样本/无微调")
    ap.add_argument("--fewshot", type=int, default=0, help="few-shot 示例数量（从 train.jsonl 采样）")
    ap.add_argument("--fewshot_source", default="data/train.jsonl", help="few-shot 样本来源文件")
    ap.add_argument("--limit", type=int, default=0, help="仅评测前 N 条（0=全部）")
    ap.add_argument("--max_new_tokens", type=int, default=2)
    ap.add_argument("--offload_folder", default=None, help="显存紧张时指定，如 offload")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_errors", type=int, default=5, help="打印前 K 条错题（0=不打印）")
    ap.add_argument("--use_flash_attn", action="store_true", help="已安装 flash-attn2 时可开启")
    ap.add_argument("--save_pred", default=None, help="保存每题预测到 jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    # Few-shot block
    fewshot_block = ""
    if args.fewshot > 0:
        fs = load_jsonl(args.fewshot_source)
        random.shuffle(fs)
        fewshot_block = format_fewshot(fs, args.fewshot)

    # Load model/tokenizer
    model_dir = os.path.expanduser(args.model_dir)
    adapter_dir = os.path.expanduser(args.adapter_dir) if args.adapter_dir else None
    tok, model = load_model_and_tokenizer(model_dir, adapter_dir, args.offload_folder, args.use_flash_attn)

    # Load test set
    data = load_jsonl(args.data)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    correct = 0
    wrong_cases = []
    total = len(data)

    # 循环开始前：
    out_f = open(args.save_pred, "w", encoding="utf-8") if args.save_pred else None

    for i, ex in enumerate(data, 1):
        prompt = build_prompt(ex, fewshot_block)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = first_A2E(gen or "")
        gold = ex["label"].strip().upper()

        import re
        m = re.search(r"Question:\n(.*?)(?:\n\nOptions:|\Z)", ex["text"], flags=re.S)
        query_text = m.group(1).strip() if m else None

        if out_f:
            out_f.write(json.dumps({
                "query": query_text,
                "pred_letter": pred,  # A/B/C/D/E or None
                "gold_letter": gold  # A/B/C/D/E
            }, ensure_ascii=False) + "\n")

        if pred == gold:
            correct += 1
        else:
            if len(wrong_cases) < args.show_errors:
                # 尽量还原原题，用简单提取：把 few-shot 部分去掉，只打印本题主干
                wrong_cases.append({
                    "pred": pred,
                    "gold": gold,
                    "gen": gen.strip(),
                    "text_preview": ex["text"].split("\nOptions:\n")[0][-200:] + " ...",
                })

        if i % 50 == 0:
            print(f"[{i}/{total}] running acc: {correct / i:.4f}")

    # 循环结束后：
    if out_f: out_f.close()

    acc = correct / total if total else 0.0
    print("=" * 60)
    print(f"Adapter: {adapter_dir or 'None (zero-shot)'}")
    print(f"Data: {args.data}  |  Total: {total}")
    print(f"Accuracy: {acc:.4f}")

    if wrong_cases:
        print("\n-- Wrong cases (first {}):".format(len(wrong_cases)))
        for j, w in enumerate(wrong_cases, 1):
            print(f"[{j}] pred={w['pred']} gold={w['gold']}  gen='{w['gen']}'")
            print(f"    {w['text_preview']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
