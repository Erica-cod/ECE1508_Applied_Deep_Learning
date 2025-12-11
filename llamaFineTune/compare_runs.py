# -*- coding: utf-8 -*-
"""
Simplified comparison script: directly compares predictions without complex ID mapping
Note: Since option order is shuffled, MPR predictions compare letters, Embedding predictions compare IDs
"""
import json, argparse

def load_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for l in f:
            if l.strip(): yield json.loads(l)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpr_preds", default="mpr_preds.jsonl",
                    help="Predictions saved by eval_mpr.py (contains query, pred_letter, gold_letter)")
    ap.add_argument("--emb_preds", default="emb_preds.jsonl",
                    help="Predictions saved by eval_embedding_baseline.py (contains query, pred_id, gold_id)")
    ap.add_argument("--test_data", default="data/test.jsonl",
                    help="Test data for filtering embedding predictions")
    args = ap.parse_args()

    # 1) Read MPR (generative) predictions, directly compare letters
    mpr_right = mpr_tot = 0
    for r in load_jsonl(args.mpr_preds):
        pred = (r.get("pred_letter") or "").strip().upper()
        gold = (r.get("gold_letter") or "").strip().upper()
        if not pred or not gold:
            continue
        mpr_tot += 1
        if pred == gold:
            mpr_right += 1

    # 2) Read test set queries
    test_queries = set()
    for line in open(args.test_data, 'r', encoding='utf-8'):
        data = json.loads(line)
        lines = data['text'].split('\n')
        for i, l in enumerate(lines):
            if l.startswith('Question:') and i+1 < len(lines):
                test_queries.add(lines[i+1].strip())
                break

    # 3) Read Embedding predictions, filter test set samples
    emb_right = emb_tot = 0
    for r in load_jsonl(args.emb_preds):
        query = r.get("query")
        # Only count test set samples
        if query not in test_queries:
            continue
        pred_id = r.get("pred_id")
        gold_id = r.get("gold_id")
        if pred_id is None or gold_id is None:
            continue
        emb_tot += 1
        if pred_id == gold_id:
            emb_right += 1

    print("="*70)
    print("Model Comparison Results (direct prediction comparison)")
    print("="*70)
    if mpr_tot:
        print(f"Llama3-MPR-SFT     : {mpr_right}/{mpr_tot}  accuracy={mpr_right/mpr_tot:.4f} ({mpr_right/mpr_tot*100:.2f}%)")
    else:
        print("Llama3-MPR-SFT     : No valid predictions")
    if emb_tot:
        print(f"GPT-3 Embedding    : {emb_right}/{emb_tot}  accuracy={emb_right/emb_tot:.4f} ({emb_right/emb_tot*100:.2f}%)")
    else:
        print("GPT-3 Embedding    : No valid predictions")
    
    if mpr_tot and emb_tot:
        improvement = (mpr_right/mpr_tot - emb_right/emb_tot) * 100
        print(f"\nImprovement: {improvement:+.2f} percentage points")
    print("="*70)

if __name__ == "__main__":
    main()
