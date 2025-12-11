# -*- coding: utf-8 -*-
"""
Generate GPT-3 embeddings for all texts in Recipe-MPR dataset
Pre-computes and caches embeddings to avoid redundant API calls
"""
import os, json, time, random
from tqdm import tqdm

import openai
from openai.embeddings_utils import get_embedding
from openai.error import RateLimitError, APIError, Timeout

# -------------------- Configuration --------------------
# Use environment variable for security: export OPENAI_API_KEY=sk-xxx
# You should set your own API key here :) I deleted my key b
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-api-key-here"

INPUT_JSON = "../data/500QA.json"
OUT_JSON    = "embeddings_with_aspects.json"
EMBED_ENGINE = "text-embedding-ada-002"   # For openai==0.28.1, use 'engine' parameter

SLEEP_EVERY = 10     # Sleep after processing N texts to avoid rate limits
SLEEP_SECS  = 2

MAX_RETRIES = 6
BACKOFF_BASE = 2.0
# -------------------------------------------------------

def safe_get_embedding(text: str, engine: str):
    """
    Get embedding with retry logic (compatible with openai==0.28.1)
    """
    # Return None for empty texts
    if not text or not text.strip():
        return None
    for attempt in range(MAX_RETRIES):
        try:
            return get_embedding(text, engine=engine)
        except (RateLimitError, Timeout):
            time.sleep(BACKOFF_BASE ** attempt)  # Exponential backoff
        except APIError as e:
            # Retry 5xx errors; 4xx are parameter/quota issues
            if getattr(e, "http_status", 500) >= 500:
                time.sleep(BACKOFF_BASE ** attempt)
            else:
                raise
    # Final failure: raise error to notify
    raise RuntimeError(f"Embedding failed after retries for text[:60]={text[:60]!r}")

def main():
    # 1) Read data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    N = len(all_data)
    print("Total examples:", N)

    # 2) Load (or initialize) cache, supports resumption
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
        print(f"Loaded existing cache: {OUT_JSON} (keys={len(embeddings)})")
    else:
        embeddings = {}

    # 3) Main loop
    counter = 0
    for i in tqdm(range(N)):
        data = all_data[i]

        # a) query
        q = data.get("query", "")
        if q not in embeddings:
            embeddings[q] = safe_get_embedding(q, engine=EMBED_ENGINE)
            counter += 1

        # b) options: dict values
        opts = data.get("options", {}) or {}
        for opt in opts.values():
            if opt not in embeddings:
                embeddings[opt] = safe_get_embedding(opt, engine=EMBED_ENGINE)
                counter += 1

        # c) correctness_explanation: dict keys
        aspects = data.get("correctness_explanation", {}) or {}
        for aspect in aspects.keys():
            if aspect not in embeddings:
                embeddings[aspect] = safe_get_embedding(aspect, engine=EMBED_ENGINE)
                counter += 1

        # Simple rate limiting: sleep after processing a batch
        if counter and counter % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECS)

        # Save periodically to avoid data loss on interruption
        if (i + 1) % 20 == 0:
            with open(OUT_JSON, "w", encoding="utf-8") as w:
                json.dump(embeddings, w, ensure_ascii=False)

    # 4) Final save
    with open(OUT_JSON, "w", encoding="utf-8") as w:
        json.dump(embeddings, w, ensure_ascii=False)
    print("Saved:", OUT_JSON, "keys:", len(embeddings))

if __name__ == "__main__":
    main()
