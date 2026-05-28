# Run AFTER 01_sample_benchmarks.py. Lists ambiguous prompts Sadi must rewrite.
import pandas as pd
pm = pd.read_csv("prompts_main.csv")   # EDIT path if needed

clear = pm[pm.prompt_version=="clear"].set_index("question_id")["prompt_text"]
amb   = pm[pm.prompt_version=="ambiguous"].set_index("question_id")["prompt_text"]
j = pd.DataFrame({"clear":clear,"ambiguous":amb}).dropna()

unchanged = j[j.clear==j.ambiguous]          # heuristic did nothing -> MUST rewrite
print(f"Total ambiguous: {len(j)} | need Sadi rewrite (unchanged): {len(unchanged)}")
unchanged.reset_index().to_csv("ambiguous_TODO_sadi.csv", index=False)
print("Saved -> ambiguous_TODO_sadi.csv  (rewrite the 'ambiguous' column, then paste back)")
