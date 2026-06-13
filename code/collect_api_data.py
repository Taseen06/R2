import os
import time
import json
import argparse
import pandas as pd
import requests
from tqdm import tqdm

DEFAULT_INPUT = "data/prompts/prompts_main.csv"   # EDIT path
OUTPUT_DIR    = "data/raw_outputs"        # EDIT path
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {"code": "M1", "name": "gemma3:27b-cloud"},      # Google  (non-reasoning)
    {"code": "M2", "name": "gemma4:31b-cloud"},      # Google  (non-reasoning)
    {"code": "M3", "name": "gpt-oss:120b-cloud"},    # OpenAI  (reasoning)
    {"code": "M4", "name": "qwen3-next:80b-cloud"},  # Alibaba (reasoning)
    {"code": "M5", "name": "nemotron-3-super:cloud"},# NVIDIA  (reasoning)
]

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = "Answer the following question accurately and concisely."  # LOCKED
TEMPERATURE   = 0.0
GEN_CAP        = 8192
LONG_THRESHOLD = 2048    # not a cap; only flags "long" (verbose-but-complete) outputs
TOP_LOGPROBS  = 20
THINK         = False

REQUEST_DELAY       = 0.8
BATCH_SAVE_INTERVAL = 10
RETRY_LIMIT         = 3
TIMEOUT             = 300

def summarize_logprobs(logprob_entries):
    vals = [float(e["logprob"]) for e in (logprob_entries or [])
            if isinstance(e, dict) and e.get("logprob") is not None]
    if not vals:
        return None, None, None, 0
    return round(sum(vals)/len(vals), 6), round(min(vals), 6), round(vals[0], 6), len(vals)

def call_ollama(model_name, prompt, temperature=TEMPERATURE):
    api_num_predict = GEN_CAP

    payload = {
        "model":  model_name,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "options": {"temperature": temperature, "num_predict": api_num_predict, "top_p": 1.0},
    }

    if not THINK:
        if any(keyword in model_name for keyword in ["qwen", "gpt-oss", "nemotron"]):
            payload["reasoning_effort"] = "minimal"
        else:
            payload["think"] = False
    else:
        if any(keyword in model_name for keyword in ["qwen", "gpt-oss", "nemotron"]):
            payload["reasoning_effort"] = "high"
        else:
            payload["think"] = True

    last_err = ""
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                mean_lp, min_lp, first_lp, n_tok = summarize_logprobs(data.get("logprobs"))
                thinking = data.get("thinking", "") or ""
                return {
                    "text": data.get("response", "") or "",
                    "thinking": thinking,
                    "logprobs_raw": data.get("logprobs"),
                    "logprob_mean": mean_lp, "logprob_min": min_lp, "logprob_first": first_lp,
                    "token_count": data.get("eval_count", n_tok),
                    "retry_count": attempt, "error": "",
                }
            last_err = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last_err = str(e)[:160]
        time.sleep(1.5 * (attempt + 1))
    return {"text": "", "thinking": "", "logprobs_raw": None, "logprob_mean": None,
            "logprob_min": None, "logprob_first": None, "token_count": 0,
            "retry_count": RETRY_LIMIT, "error": last_err}

def run_one_model(model_cfg, df, tag, limit=None):
    code, name = model_cfg["code"], model_cfg["name"]
    out_csv  = os.path.join(OUTPUT_DIR, f"r2_raw_{tag}_{code}.csv")
    lp_jsonl = os.path.join(OUTPUT_DIR, f"r2_logprobs_{tag}_{code}.jsonl")

    rows, processed = [], set()
    if os.path.exists(out_csv):
        try:
            old = pd.read_csv(out_csv, dtype={"question_id": str})
            rows = old.to_dict("records")
            processed = set(old["question_id"].astype(str) + "|" + old["prompt_version"].astype(str))
            print(f"[{tag}/{code}] Resuming - {len(processed)} already collected.")
        except Exception:
            pass

    key = df["question_id"].astype(str) + "|" + df["prompt_version"].astype(str)
    todo = df[~key.isin(processed)]
    if limit:
        todo = todo.head(limit)
    print(f"--- [{tag}/{code}] {name}: {len(todo)} prompts to collect ---")

    warned_lp = False
    lp_fh = open(lp_jsonl, "a", encoding="utf-8")
    try:
        for i, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo))):
            res = call_ollama(name, row["prompt_text"])

            if not warned_lp and res["error"] == "":
                if res["logprob_mean"] is None:
                    print(f"  [note] {code}: no logprobs returned (cloud model).")
                warned_lp = True

            rows.append({
                "question_id": row["question_id"], "condition": row["condition"],
                "prompt_version": row["prompt_version"], "source": row.get("source", ""),
                "domain": row.get("domain", ""), "model_code": code, "model_name": name,
                "prompt_text": row["prompt_text"], "expected_answer": row.get("expected_answer", ""),
                "raw_output": res["text"], "token_count": res["token_count"],
                "truncated": int(res["token_count"] >= GEN_CAP),
                "long_output": int(LONG_THRESHOLD <= res["token_count"] < GEN_CAP),
                "thinking_tokens": len(res["thinking"].split()) if res["thinking"] else 0,
                "logprob_mean": res["logprob_mean"], "logprob_min": res["logprob_min"],
                "logprob_first": res["logprob_first"], "retry_count": res["retry_count"],
                "error": res["error"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

            if res["thinking"] or res["logprobs_raw"] is not None:
                lp_fh.write(json.dumps({"question_id": row["question_id"],
                                        "prompt_version": row["prompt_version"], "model_code": code,
                                        "thinking": res["thinking"],
                                        "logprobs": res["logprobs_raw"]}) + "\n")
            if (i + 1) % BATCH_SAVE_INTERVAL == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False); lp_fh.flush()
            time.sleep(REQUEST_DELAY)
    except KeyboardInterrupt:
        print(f"\n[{tag}/{code}] Interrupted - saving progress...")
    finally:
        pd.DataFrame(rows).to_csv(out_csv, index=False); lp_fh.close()
    print(f"[{tag}/{code}] Done. {len(rows)} rows -> {out_csv}")

def run(input_csv, tag, model_filter=None, limit=None):
    if not os.path.exists(input_csv):
        print(f"Input CSV not found: {input_csv}"); return
    df = pd.read_csv(input_csv, dtype={"question_id": str})
    required = {"question_id", "condition", "prompt_version", "prompt_text"}
    missing = required - set(df.columns)
    if missing:
        print(f"Input CSV missing columns: {missing}"); return
    models = MODELS if not model_filter else [m for m in MODELS if m["code"] in model_filter]
    for m in models:
        run_one_model(m, df, tag, limit=limit)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="*", default=None, help="e.g. M3 M4 M5")
    ap.add_argument("--limit", type=int, default=None, help="cap prompts (pilot)")
    args = ap.parse_args()
    run(args.input, args.tag, model_filter=args.models, limit=args.limit)
