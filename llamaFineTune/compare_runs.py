# -*- coding: utf-8 -*-
import json, argparse, sys
from collections import OrderedDict

def load_json(path):
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def load_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for l in f:
            if l.strip(): yield json.loads(l)

LETTER = "ABCDE"

def build_query2_ordered_ids(raw500_path):
    """从 500QA.json 构造：query -> [option_ids in insertion order]"""
    raw = load_json(raw500_path)
    q2ids = {}
    for ex in raw:
        # Python 3.7+ 保持 JSON dict insertion order
        ids_in_order = list(OrderedDict(ex["options"]).keys())
        q2ids[ex["query"]] = {
            "ids_in_order": ids_in_order,
            "gold_id": ex["answer"]
        }
    return q2ids

def letter_to_id(letter, ids_in_order):
    if not letter or letter not in LETTER: return None
    idx = LETTER.index(letter)
    if idx >= len(ids_in_order): return None
    return ids_in_order[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw500", default="Recipe-MPR/data/500QA.json",
                    help="原始 500QA.json，用于恢复 选项顺序 与 gold id")
    ap.add_argument("--mpr_preds", default="mpr_preds.jsonl",
                    help="eval_mpr.py 保存的预测（含 query, pred_letter, gold_letter）")
    ap.add_argument("--emb_preds", default="emb_preds.jsonl",
                    help="eval_embedding_baseline.py 保存的预测（含 query, pred_id, gold_id）")
    args = ap.parse_args()

    q2 = build_query2_ordered_ids(args.raw500)

    # 1) 读 MPR（生成式）预测，转成 id
    mpr_right = mpr_tot = 0
    for r in load_jsonl(args.mpr_preds):
        q = r.get("query")
        pl = (r.get("pred_letter") or "").strip().upper()
        meta = q2.get(q)
        if not q or not meta:
            continue
        pred_id = letter_to_id(pl, meta["ids_in_order"])
        gold_id = meta["gold_id"]
        if pred_id is None:
            continue
        mpr_tot += 1
        if pred_id == gold_id:
            mpr_right += 1

    # 2) 读 Embedding 预测
    emb_right = emb_tot = 0
    for r in load_jsonl(args.emb_preds):
        q = r.get("query")
        pred_id = r.get("pred_id")
        gold_id = r.get("gold_id")
        if q not in q2:
            continue
        if pred_id is None:
            continue
        emb_tot += 1
        if pred_id == gold_id:
            emb_right += 1

    print("="*60)
    print("Unified comparison (both mapped to option_id exact-match)")
    if mpr_tot:
        print(f"MPR (fine-tuned)   : {mpr_right}/{mpr_tot}  acc={mpr_right/mpr_tot:.4f}")
    else:
        print("MPR (fine-tuned)   : no comparable samples")
    if emb_tot:
        print(f"Embedding baseline : {emb_right}/{emb_tot}  acc={emb_right/emb_tot:.4f}")
    else:
        print("Embedding baseline : no comparable samples")
    print("="*60)

if __name__ == "__main__":
    main()
