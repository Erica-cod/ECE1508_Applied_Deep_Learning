# -*- coding: utf-8 -*-
"""
Analyze model accuracy by query type
"""
import json
from collections import defaultdict

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_query_from_text(text):
    """Extract query from formatted text"""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('Question:') and i+1 < len(lines):
            return lines[i+1].strip()
    return None

def main():
    # Load raw data to get query types
    raw_data = load_json('/home/zhengyangli/1508project/Recipe-MPR/data/500QA.json')
    query_to_type = {}
    for ex in raw_data:
        # Strip leading/trailing spaces to avoid matching issues
        clean_query = ex['query'].strip()
        query_to_type[clean_query] = ex['query_type']
    
    # Load test set
    test_data = load_jsonl('data/test.jsonl')
    test_queries = {}
    for item in test_data:
        query = extract_query_from_text(item['text'])
        if query:
            test_queries[query] = item['label']
    
    print(f"Test set samples: {len(test_queries)}")
    print()
    
    # Analyze prediction files from different models (50-sample test set)
    models = {
        'Llama3-1B Zero-shot': ('mpr', 'mpr_preds_1b_zeroshot.jsonl'),
        'Llama3-1B Fine-tuned': ('mpr', 'mpr_preds_1b_ft_80_10_10.jsonl'),
        'Llama3-3B Zero-shot': ('mpr', 'mpr_preds_3b_zeroshot.jsonl'),
        'Llama3-3B Fine-tuned': ('mpr', 'mpr_preds_3b_ft_80_10_10.jsonl'),
        'GPT-3 Embedding': ('emb', 'emb_preds_80_10_10.jsonl'),
    }
    
    results = {}
    overall_stats = {}  # 存储每个模型的整体准确率（不重复计数）
    
    for model_name, (pred_type, pred_file) in models.items():
        try:
            preds = load_jsonl(pred_file)
            
            # 按查询类型统计
            type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            
            # 整体统计（每个查询只计数一次）
            overall_correct = 0
            overall_total = 0
            
            for pred in preds:
                query = pred.get('query')
                if not query:
                    continue
                
                # 清理查询文本（去除前导空格）
                query_clean = query.strip()
                
                # 只统计测试集中的查询
                if query_clean not in test_queries:
                    continue
                
                # 根据类型判断是否正确
                if pred_type == 'mpr':
                    pred_letter = pred.get('pred_letter', '').strip().upper()
                    gold_letter = pred.get('gold_letter', '').strip().upper()
                    if not pred_letter or not gold_letter:
                        continue
                    is_correct = (pred_letter == gold_letter)
                else:  # emb
                    pred_id = pred.get('pred_id')
                    gold_id = pred.get('gold_id')
                    if pred_id is None or gold_id is None:
                        continue
                    is_correct = (pred_id == gold_id)
                
                # 整体统计（所有查询，包括无类型标签的）
                overall_total += 1
                if is_correct:
                    overall_correct += 1
                
                # 如果查询有类型标签，才统计到各个类型中
                if query_clean in query_to_type:
                    query_types = query_to_type[query_clean]
                    # 统计每个类型（一个查询可能属于多个类型）
                    for qtype, value in query_types.items():
                        if value == 1:  # 该查询属于这个类型
                            type_stats[qtype]['total'] += 1
                            if is_correct:
                                type_stats[qtype]['correct'] += 1
            
            results[model_name] = type_stats
            overall_stats[model_name] = {'correct': overall_correct, 'total': overall_total}
            
        except FileNotFoundError:
            print(f"Warning: {pred_file} does not exist, skipping {model_name}")
            continue
    
    # Print results
    print("=" * 80)
    print("Accuracy by Query Type")
    print("=" * 80)
    print()
    
    # Query types
    query_types = ['Specific', 'Analogical', 'Negated', 'Commonsense', 'Temporal']
    
    # Print header
    print(f"{'Query Type':<15}", end='')
    for model_name in results.keys():
        print(f"{model_name:<25}", end='')
    print()
    print("-" * 80)
    
    # Print results for each type
    for qtype in query_types:
        print(f"{qtype:<15}", end='')
        for model_name in results.keys():
            stats = results[model_name].get(qtype, {'correct': 0, 'total': 0})
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"{acc:5.2f}% ({stats['correct']}/{stats['total']})  ", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()
    
    # Print overall (true overall accuracy, no duplicate counting)
    print("-" * 80)
    print(f"{'Overall':<15}", end='')
    for model_name in results.keys():
        stats = overall_stats.get(model_name, {'correct': 0, 'total': 0})
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"{acc:5.2f}% ({stats['correct']}/{stats['total']})  ", end='')
        else:
            print(f"{'N/A':<20}", end='')
    print()
    print("=" * 80)
    print(f"\nNote: Queries may belong to multiple types, so type-wise totals > test set size.")
    print(f"The 'Overall' row shows true overall accuracy (each query counted once).")
    
    # Generate LaTeX table
    print("\n\nGenerated LaTeX table code (all models):")
    print("=" * 80)
    
    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Accuracy by query type across different models.}")
    print("\\label{tab:accuracy_by_type}")
    print("\\small")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Query Type} & \\textbf{1B-Zero (\\%)} & \\textbf{1B-FT (\\%)} & \\textbf{3B-Zero (\\%)} & \\textbf{3B-FT (\\%)} & \\textbf{GPT-3 Emb (\\%)} \\\\")
    print("\\midrule")
    
    # 获取所有模型的结果
    model1b_zero = results.get('Llama3-1B Zero-shot', {})
    model1b_ft = results.get('Llama3-1B Fine-tuned', {})
    model3b_zero = results.get('Llama3-3B Zero-shot', {})
    model3b_ft = results.get('Llama3-3B Fine-tuned', {})
    gpt3_emb = results.get('GPT-3 Embedding', {})
    
    for qtype in query_types:
        stats_list = [
            model1b_zero.get(qtype, {'correct': 0, 'total': 0}),
            model1b_ft.get(qtype, {'correct': 0, 'total': 0}),
            model3b_zero.get(qtype, {'correct': 0, 'total': 0}),
            model3b_ft.get(qtype, {'correct': 0, 'total': 0}),
            gpt3_emb.get(qtype, {'correct': 0, 'total': 0}),
        ]
        
        print(f"{qtype}", end='')
        for stats in stats_list:
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f" & {acc:.2f}", end='')
            else:
                print(f" & --", end='')
        print(" \\\\")
    
    print("\\midrule")
    
    # Overall（使用真实的整体准确率，不重复计数）
    overall_list = []
    for model_name in ['Llama3-1B Zero-shot', 'Llama3-1B Fine-tuned', 'Llama3-3B Zero-shot', 'Llama3-3B Fine-tuned', 'GPT-3 Embedding']:
        stats = overall_stats.get(model_name, {'correct': 0, 'total': 0})
        if stats['total'] > 0:
            overall_list.append(stats['correct'] / stats['total'] * 100)
        else:
            overall_list.append(None)
    
    print(f"Overall", end='')
    for acc in overall_list:
        if acc is not None:
            print(f" & {acc:.2f}", end='')
        else:
            print(f" & --", end='')
    print(" \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")
    print("\\vspace{0.2cm}")
    print("\\footnotesize{Note: Queries may belong to multiple types. Overall accuracy counts each query only once.}")
    print("=" * 80)

if __name__ == "__main__":
    main()

