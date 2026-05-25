import os
import time
import json
import argparse
import pandas as pd
import requests
from tqdm import tqdm

# ============================================================
# R2 DATA COLLECTION  (Rev. 2)  —  modeled on the R3 script
# ------------------------------------------------------------
# Changes vs the previous version:
#   - 4 models now (gemini-3-flash-preview REMOVED). All are Ollama-served.
#   - All 4 expose logprobs -> calibration covers every model, no exceptions.
#   - Gemini code path deleted (no Google API, no GOOGLE_API_KEY needed).
#   - Added --input and --tag so the SAME script runs the main set AND each
#     "Chemistry with LLM" probe set without editing the file or overwriting
#     outputs.
#
# Still true from before:
#   - Parameters LOCKED (temp=0.0, max_tokens=256, fixed system prompt).
#   - Collection ONLY. We never score here (no correctness, no safety judging).
#     Outputs are logged verbatim; scoring happens later, blind.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# --- Default input / output (override per run with --input / --tag) ---------
# Input columns: question_id, condition, prompt_version, prompt_text,
#                expected_answer, source, domain
#   * main set      -> prompt_version = clear | ambiguous
#   * probe sets    -> prompt_version encodes the variant, e.g.
#                      "anonymous"/"patient"/"physician"/"researcher" (authority),
#                      "nurse-1".."teacher-5" (demographic), etc.
DEFAULT_INPUT = "/home/taseen06/Desktop/R2/prompts_main.csv"   # EDIT path
OUTPUT_DIR    = "/home/taseen06/Desktop/R2/raw_outputs"        # EDIT path
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- THE FOUR MODELS --------------------------------------------------------
# EDIT every "name" to match `ollama list` EXACTLY (incl. any -cloud suffix,
# like the R3 "gemma3:27b-cloud"). Wrong tag = empty output.
MODELS = [
    {"code": "M1", "name": "gemma3:27b-cloud"}, #google's open source      
    {"code": "M2", "name": "gemma4:31b-cloud"},  #google's open source
    {"code": "M3", "name": "gpt-oss:120b-cloud"}, # openai;s open source
    {"code": "M4", "name": "qwen3-next:80b-cloud"}, #Alibaba Cloud
    {"code": "M5", "name": "nemotron-3-super:cloud"}, #nvidia   
]                                                    # EDIT tag (add size if needed, e.g. qwen3.5:32b)

# --- Endpoint ---------------------------------------------------------------
# EDIT: R3 used port 11435 (not the default 11434). Keep whatever your
# Ollama instance actually serves on.
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- LOCKED PARAMETERS (never change mid-study) -----------------------------
SYSTEM_PROMPT = "Answer the following question accurately and concisely."  # LOCKED
TEMPERATURE   = 0.0     # LOCKED
MAX_TOKENS    = 256     # LOCKED
TOP_LOGPROBS  = 20      # alternatives captured per token position

# --- Run controls -----------------------------------------------------------
REQUEST_DELAY       = 0.8
BATCH_SAVE_INTERVAL = 10
RETRY_LIMIT         = 3
TIMEOUT             = 180   # cloud models (Kimi) can be slow


# ============================================================
# UTILITIES
# ============================================================

def summarize_logprobs(logprob_entries):
    """Aggregate per-token logprobs into the three pre-registered confidence
    measures (mean, min, first) over the generated span. Robust to missing."""
    vals = [float(e["logprob"]) for e in (logprob_entries or [])
            if isinstance(e, dict) and e.get("logprob") is not None]
    if not vals:
        return None, None, None, 0
    return round(sum(vals)/len(vals), 6), round(min(vals), 6), round(vals[0], 6), len(vals)


def call_ollama(model_name, prompt):
    """Native /api/generate (same endpoint family as R3) with locked system
    prompt + logprobs. Returns text, logprobs, token count, error. Real retries."""
    payload = {
        "model":  model_name,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "top_p": 1.0},
    }
    last_err = ""
    for attempt in range(RETRY_LIMIT):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                mean_lp, min_lp, first_lp, n_tok = summarize_logprobs(data.get("logprobs"))
                return {
                    "text": data.get("response", ""),
                    "logprobs_raw": data.get("logprobs"),
                    "logprob_mean": mean_lp, "logprob_min": min_lp, "logprob_first": first_lp,
                    "token_count": data.get("eval_count", n_tok),
                    "retry_count": attempt, "error": "",
                }
            last_err = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last_err = str(e)[:160]
        time.sleep(1.5 * (attempt + 1))   # backoff
    return {"text": "", "logprobs_raw": None, "logprob_mean": None, "logprob_min": None,
            "logprob_first": None, "token_count": 0, "retry_count": RETRY_LIMIT, "error": last_err}


# ============================================================
# MAIN EXECUTION (per model)  —  auto-resume + batch-save (R3 pattern)
# ============================================================

def run_one_model(model_cfg, df, tag, limit=None):
    code, name = model_cfg["code"], model_cfg["name"]
    out_csv  = os.path.join(OUTPUT_DIR, f"r2_raw_{tag}_{code}.csv")
    lp_jsonl = os.path.join(OUTPUT_DIR, f"r2_logprobs_{tag}_{code}.jsonl")

    # --- AUTO-RESUME (keyed on question_id + prompt_version) ---------------
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

    lp_fh = open(lp_jsonl, "a", encoding="utf-8")
    try:
        for i, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo))):
            res = call_ollama(name, row["prompt_text"])
            rows.append({
                "question_id": row["question_id"], "condition": row["condition"],
                "prompt_version": row["prompt_version"], "source": row.get("source", ""),
                "domain": row.get("domain", ""), "model_code": code, "model_name": name,
                "prompt_text": row["prompt_text"], "expected_answer": row.get("expected_answer", ""),
                "raw_output": res["text"], "token_count": res["token_count"],
                "logprob_mean": res["logprob_mean"], "logprob_min": res["logprob_min"],
                "logprob_first": res["logprob_first"], "retry_count": res["retry_count"],
                "error": res["error"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            if res["logprobs_raw"] is not None:
                lp_fh.write(json.dumps({"question_id": row["question_id"],
                                        "prompt_version": row["prompt_version"],
                                        "model_code": code, "logprobs": res["logprobs_raw"]}) + "\n")
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
    # MAIN set (all 4 models):
    #   python 02_collect_api_data.py --input prompts_main.csv --tag main
    # CHEMISTRY-WITH-LLM probe sets (run each separately, own tag):
    #   python 02_collect_api_data.py --input probes_demographic.csv --tag demo
    #   python 02_collect_api_data.py --input probes_authority.csv   --tag auth
    #   python 02_collect_api_data.py --input probes_safety.csv      --tag safety
    # One model only:   --models M4        Pilot:  --limit 30
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Prompt CSV to run.")
    ap.add_argument("--tag", default="main", help="Label for output files (main/demo/auth/safety).")
    ap.add_argument("--models", nargs="*", default=None, help="Model codes, e.g. M1 M4. Default: all.")
    ap.add_argument("--limit", type=int, default=None, help="Cap prompts per model (pilot).")
    args = ap.parse_args()
    run(args.input, args.tag, model_filter=args.models, limit=args.limit)
