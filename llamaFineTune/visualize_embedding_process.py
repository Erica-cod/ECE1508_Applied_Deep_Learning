# -*- coding: utf-8 -*-
"""
Visualize the GPT-3 Embedding similarity matching process
"""
import json
import numpy as np

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(a @ b) / float(na * nb + 1e-8)

def print_ascii_diagram():
    """Print ASCII diagram of the embedding process"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║         GPT-3 Embedding-based Recipe Recommendation Process        ║
╚════════════════════════════════════════════════════════════════════╝

Step 1: Input Query and Options
────────────────────────────────────────────────────────────────────
  Query: "I want a warm dish containing oysters"
  
  Options:
    A: Simple creamy oyster soup
    B: Seasoned salted crackers shaped like oysters
    C: Creamy clam chowder made with whole milk
    D: Tomato mussel soup containing dry white wine
    E: Warm vegetable soup with tomatoes and corn


Step 2: Get Pre-computed Embeddings (1536-dimensional vectors)
────────────────────────────────────────────────────────────────────
  Query Embedding:
    q_vec = [0.023, -0.045, 0.078, ..., 0.012]  (1536 dims)
  
  Option Embeddings:
    opt_A = [0.034, -0.023, 0.089, ..., 0.021]  (1536 dims)
    opt_B = [-0.012, 0.067, -0.045, ..., 0.033] (1536 dims)
    opt_C = [0.028, -0.031, 0.072, ..., 0.015]  (1536 dims)
    opt_D = [0.019, -0.028, 0.065, ..., 0.018]  (1536 dims)
    opt_E = [0.015, -0.035, 0.058, ..., 0.011]  (1536 dims)


Step 3: Calculate Cosine Similarity
────────────────────────────────────────────────────────────────────
  Formula: cos_sim(q, opt) = (q · opt) / (||q|| × ||opt||)
  
  Similarities:
    sim(q, opt_A) = 0.87  ← Highest! ✓
    sim(q, opt_B) = 0.62
    sim(q, opt_C) = 0.73
    sim(q, opt_D) = 0.69
    sim(q, opt_E) = 0.58


Step 4: Select Option with Highest Similarity
────────────────────────────────────────────────────────────────────
  Prediction: Option A (similarity = 0.87)
  Ground Truth: Option A
  Result: ✓ Correct


Visualization of Similarity Scores:
────────────────────────────────────────────────────────────────────
  A: ████████████████████████████████████  0.87 ← Selected
  B: ████████████████████                  0.62
  C: ████████████████████████████          0.73
  D: ██████████████████████████            0.69
  E: ██████████████████                    0.58

╔════════════════════════════════════════════════════════════════════╗
║  Final Answer: Option A - "Simple creamy oyster soup"             ║
╚════════════════════════════════════════════════════════════════════╝
""")

def demonstrate_with_real_example():
    """Demonstrate with a real example from the dataset"""
    print("\n" + "="*70)
    print("Real Example Demonstration")
    print("="*70 + "\n")
    
    # Load embeddings
    try:
        embeddings = load_json("embeddings_with_aspects.json")
        raw_data = load_json("Recipe-MPR/data/500QA.json")
    except:
        print("Note: Run this from the project root directory")
        return
    
    # Get first example
    example = raw_data[0]
    query = example["query"]
    options = example["options"]
    answer_id = example["answer"]
    
    print(f"Query: \"{query}\"\n")
    print("Options:")
    for i, (opt_id, opt_text) in enumerate(options.items(), 1):
        marker = "✓" if opt_id == answer_id else " "
        print(f"  {chr(64+i)}: {opt_text} {marker}")
    
    print("\n" + "-"*70)
    print("Computing Similarities...")
    print("-"*70 + "\n")
    
    # Get query embedding
    q_emb = embeddings.get(query)
    if q_emb is None:
        print("Query embedding not found")
        return
    
    # Calculate similarities
    similarities = []
    for opt_id, opt_text in options.items():
        opt_emb = embeddings.get(opt_text)
        if opt_emb is None:
            sim = -1.0
        else:
            sim = cosine_similarity(q_emb, opt_emb)
        similarities.append((opt_id, opt_text, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Display results
    print("Similarity Scores (sorted):")
    max_sim = max(s[2] for s in similarities)
    for i, (opt_id, opt_text, sim) in enumerate(similarities, 1):
        marker = "✓" if opt_id == answer_id else " "
        selected = "← SELECTED" if i == 1 else ""
        bar_length = int((sim / max_sim * 40)) if sim > 0 else 0
        bar = "█" * bar_length
        print(f"  {chr(64+i)}: {sim:.4f} {bar} {selected} {marker}")
        print(f"      {opt_text[:60]}...")
    
    # Show prediction result
    pred_id = similarities[0][0]
    is_correct = (pred_id == answer_id)
    
    print("\n" + "="*70)
    print(f"Prediction: {pred_id}")
    print(f"Ground Truth: {answer_id}")
    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print("="*70)

def explain_algorithm():
    """Explain the algorithm in detail"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                    Algorithm Explanation                           ║
╚════════════════════════════════════════════════════════════════════╝

Input:
  - Query: User's recipe request (text)
  - Options: 5 candidate recipes (text)
  - Embeddings: Pre-computed vectors for all texts

Process:
  1. Retrieve query embedding: q_vec = embeddings[query]
  2. For each option i in {A, B, C, D, E}:
       a. Retrieve option embedding: opt_i_vec = embeddings[option_i]
       b. Compute cosine similarity: sim_i = cos_sim(q_vec, opt_i_vec)
  3. Select option with maximum similarity: pred = argmax(sim_1...sim_5)

Output:
  - Prediction: The option ID with highest similarity
  
Key Properties:
  ✓ Fast: O(n) similarity calculations where n=5 options
  ✓ No training required: Uses pre-trained embeddings
  ✓ Interpretable: Similarity scores show matching degree
  ✗ Limited reasoning: Cannot handle complex logic or negation
  ✗ Shallow matching: Based on lexical/semantic similarity only

""")

def main():
    print_ascii_diagram()
    print("\n\n")
    explain_algorithm()
    print("\n\n")
    demonstrate_with_real_example()

if __name__ == "__main__":
    main()

