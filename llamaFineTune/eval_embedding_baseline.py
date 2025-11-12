# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np
from typing import Dict, List


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cos(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=np.float32);
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a);
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return -1.0
    return float(a @ b) / float(na * nb + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl",
                    help="prep_mpr.py 生成的 test 文件（含 text/label）或原始 500QA.json")
    ap.add_argument("--raw_json", default="Recipe-MPR/data/500QA.json", help="原始 500QA.json（供读取选项ID→文本的映射）")
    ap.add_argument("--emb", default="embeddings_with_aspects.json", help="上一阶段生成的向量缓存 JSON")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_pred", default="emb_preds.jsonl")
    args = ap.parse_args()

    # 开头打开文件：
    out_f = open(args.save_pred, "w", encoding="utf-8") if args.save_pred else None

    # 读 embeddings（{文本: 向量}）
    E: Dict[str, List[float]] = load_json(args.emb)

    # 读原始数据（拿到 query、options、answer）
    raw = load_json(args.raw_json)

    # 如需只测一部分
    data = raw[:args.limit] if args.limit and args.limit > 0 else raw

    correct = 0;
    total = 0
    miss = 0

    for ex in data:
        q = ex.get("query", "")
        opts: Dict[str, str] = ex.get("options", {}) or {}
        ans_id = ex.get("answer")

        # 拿向量
        qv = E.get(q)
        if qv is None:
            miss += 1
            continue

        # 选项逐个比较相似度
        best_s, best_id = -1.0, None
        for oid, otext in opts.items():
            ov = E.get(otext)
            if ov is None:
                continue
            s = cos(qv, ov)
            if s > best_s:
                best_s, best_id = s, oid

        if best_id is None:
            # 该题所有选项都缺向量，跳过
            miss += 1
            continue

        total += 1
        if best_id == ans_id:
            correct += 1

            # 循环得出 best_id 后，写一行：
        if out_f:
            out_f.write(json.dumps({
                "query": q,
                "pred_id": best_id,
                "gold_id": ans_id
            }, ensure_ascii=False) + "\n")

    if out_f:
        out_f.close()

    acc = correct / total if total else 0.0
    print("=" * 60)
    print(f"Embedding baseline")
    print(f"Data(total usable): {total} | Missing pairs: {miss}")
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
