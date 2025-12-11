import json, random, os, argparse
from collections import OrderedDict

def letter(i): return "ABCDE"[i]

def build_record(ex, shuffle_options=True):
    # options: {option_id: option_text}
    # answer: correct option_id
    opts = list(OrderedDict(ex["options"]).items())  # [(id, text), ...]
    # Randomly shuffle option order to avoid answer always at position A
    if shuffle_options:
        random.shuffle(opts)
    lines = []
    for i, (_, txt) in enumerate(opts):
        lines.append(f"{letter(i)}: {txt}")
    prompt = (
        "You are a helpful cooking assistant.\n"
        "Choose the single best answer (A-E). Reply with just the letter.\n\n"
        f"Question:\n{ex['query'].strip()}\n\nOptions:\n" +
        "\n".join(lines) + "\n\nAnswer:"
    )
    # Map correct ID to letter label
    ans_id = ex["answer"]
    correct_index = [i for i,(oid,_) in enumerate(opts) if oid == ans_id]
    if not correct_index:
        raise ValueError("answer id not found in options")
    label = letter(correct_index[0])
    return {"text": prompt, "label": label}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="/Recipe-MPR/data/500QA.json")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="60,20,20")  # percentages
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle and split by ratio
    idx = list(range(len(data)))
    random.shuffle(idx)
    p = list(map(int, args.split.split(",")))
    assert sum(p)==100
    n = len(idx)
    n_train = n * p[0] // 100
    n_valid = n * p[1] // 100
    splits = {
        "train": idx[:n_train],
        "valid": idx[n_train:n_train+n_valid],
        "test":  idx[n_train+n_valid:]
    }

    for name, indices in splits.items():
        outp = os.path.join(args.outdir, f"{name}.jsonl")
        with open(outp, "w", encoding="utf-8") as w:
            for i in indices:
                rec = build_record(data[i])
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(name, len(indices), outp)

if __name__ == "__main__":
    main()
