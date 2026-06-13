import pandas as pd
from sklearn.metrics import cohen_kappa_score

SADI = "data/scoring/scoring_sadi_filled.csv"
TASIN = "data/scoring/scoring_tasin_filled.csv"
OVERLAP = "data/scoring/kappa_overlap_uids.csv"
COLUMNS = ["correctness", "confidence_tone", "opinion_balance", "bias_signal", "safety_statement"]


def main():
    sadi = pd.read_csv(SADI)
    tasin = pd.read_csv(TASIN)
    overlap = set(pd.read_csv(OVERLAP).row_uid)

    s = sadi[sadi.row_uid.isin(overlap)].set_index("row_uid").sort_index()
    t = tasin[tasin.row_uid.isin(overlap)].set_index("row_uid").sort_index()

    rows = []
    for col in COLUMNS:
        mask = (s[col].fillna("").astype(str).str.strip() != "") & \
               (t[col].fillna("").astype(str).str.strip() != "")
        a = s.loc[mask, col].astype(str)
        b = t.loc[mask, col].astype(str)
        k = cohen_kappa_score(a, b)
        agree = (a.values == b.values).mean()
        rows.append({"column": col, "n": len(a), "cohen_kappa": round(k, 3),
                     "agreement": round(agree, 3),
                     "status": "pass" if k >= 0.80 else "below"})

    out = pd.DataFrame(rows)
    out.to_csv("results/cohens_kappa_report.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
