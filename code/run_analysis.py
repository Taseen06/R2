import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

SADI = "data/scoring/scoring_sadi_filled.csv"
TASIN = "data/scoring/scoring_tasin_filled.csv"
KEY = "data/scoring/blind_key.csv"
OVERLAP = "data/scoring/kappa_overlap_uids.csv"
FRAMINGS = ["anonymous", "patient", "physician", "researcher"]


def load():
    s = pd.read_csv(SADI)
    t = pd.read_csv(TASIN)
    key = pd.read_csv(KEY)
    overlap = set(pd.read_csv(OVERLAP).row_uid)
    d = pd.concat([s, t[~t.row_uid.isin(overlap)]], ignore_index=True)
    d = d.merge(key, on="row_uid", how="left")
    d["framing"] = d["prompt_version"]
    return d


def pct_ci(c, n):
    lo, hi = proportion_confint(c, n, method="wilson")
    return 100 * c / n, 100 * lo, 100 * hi


def odds_ratio(table):
    (a, b), (c, d) = table
    return (a * d) / (b * c)


def main():
    d = load()
    pvals = {}

    fac = d[(d.tag == "main") & d.correctness.isin(["correct", "incorrect"])].copy()
    cmap = {"C1": ("clear", "general"), "C2": ("clear", "niche"),
            "C4": ("ambiguous", "general"), "C5": ("ambiguous", "niche")}
    fac = fac[fac.condition.isin(cmap)]
    fac["err"] = (fac.correctness == "incorrect").astype(int)
    fac["clarity"] = fac.condition.map(lambda c: cmap[c][0])
    fac["domain"] = fac.condition.map(lambda c: cmap[c][1])

    cl, am = fac[fac.clarity == "clear"].err, fac[fac.clarity == "ambiguous"].err
    tab = [[am.sum(), len(am) - am.sum()], [cl.sum(), len(cl) - cl.sum()]]
    chi, p, _, _ = stats.chi2_contingency(tab)
    pvals["H1_clarity"] = p
    print("H1  clear %.1f%% vs ambiguous %.1f%%  chi2=%.1f p=%.1e OR=%.2f"
          % (100 * cl.mean(), 100 * am.mean(), chi, p, odds_ratio(tab)))

    ge, ni = fac[fac.domain == "general"].err, fac[fac.domain == "niche"].err
    tab3 = [[ni.sum(), len(ni) - ni.sum()], [ge.sum(), len(ge) - ge.sum()]]
    chi3, p3, _, _ = stats.chi2_contingency(tab3)
    pvals["H3_domain"] = p3
    print("H3  general %.1f%% vs niche %.1f%%  chi2=%.1f p=%.1e OR=%.2f"
          % (100 * ge.mean(), 100 * ni.mean(), chi3, p3, odds_ratio(tab3)))

    g = d[(d.tag == "main") & d.correctness.isin(["correct", "incorrect"])].copy()
    g = g[g.confidence_tone.astype(str).isin(["1", "1.0", "2", "2.0", "3", "3.0"])]
    g["hedged"] = (g.confidence_tone.astype(float) == 1).astype(int)
    inc, cor = g[g.correctness == "incorrect"], g[g.correctness == "correct"]
    nonhedge = pct_ci((inc.hedged == 0).sum(), len(inc))
    tab2 = [[(inc.hedged == 0).sum(), inc.hedged.sum()],
            [(cor.hedged == 0).sum(), cor.hedged.sum()]]
    chi2, p2, _, _ = stats.chi2_contingency(tab2)
    pvals["H2_confidence"] = p2
    print("H2  non-hedged|incorrect %.1f%% [%.1f,%.1f]  hedge inc %.1f%% vs cor %.1f%% chi2=%.1f p=%.1e"
          % (*nonhedge, 100 * inc.hedged.mean(), 100 * cor.hedged.mean(), chi2, p2))

    dem = d[d.tag == "demo"]
    ctab = pd.crosstab(dem.framing, dem.bias_signal)
    chi4, p4, dof, _ = stats.chi2_contingency(ctab)
    n = ctab.values.sum()
    V = np.sqrt(chi4 / (n * (min(ctab.shape) - 1)))
    pvals["H4_pronoun"] = p4
    print("H4 pronoun  chi2=%.1f dof=%d p=%.1e CramerV=%.2f" % (chi4, dof, p4, V))

    au = d[d.tag == "auth"].copy()
    au["w"] = au.raw_output.fillna("").str.split().apply(len)
    H, pH = stats.kruskal(*[au[au.framing == f].w for f in FRAMINGS])
    pvals["H4_authority"] = pH
    means = {f: round(au[au.framing == f].w.mean()) for f in FRAMINGS}
    print("H4 authority  means %s  Kruskal H=%.1f p=%.1e" % (means, H, pH))

    sa = d[d.tag == "safety"].copy()
    sa["present"] = (sa.safety_statement == "present").astype(int)
    stab = pd.crosstab(sa.framing, sa.present)
    chiS, pS, dofS, _ = stats.chi2_contingency(stab)
    pvals["H4_safety"] = pS
    by = {f: round(100 * sa[sa.framing == f].present.mean(), 1) for f in FRAMINGS}
    print("H4 safety  present%% %s  chi2=%.1f p=%.3f" % (by, chiS, pS))

    keys = list(pvals)
    rej, padj, _, _ = multipletests([pvals[k] for k in keys], method="fdr_bh")
    print("\nBenjamini-Hochberg FDR:")
    for k, pa, r in zip(keys, padj, rej):
        print("  %-14s adj_p=%.1e  %s" % (k, pa, "significant" if r else "n.s."))


if __name__ == "__main__":
    main()
