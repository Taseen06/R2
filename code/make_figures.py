import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_DIR = "data/scoring"             # <-- EDIT: folder holding the CSVs
FIG_DIR  = "results/figures"     # <-- EDIT: where figures are written
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "semibold", "axes.labelsize": 11,
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {"primary":"#3f6f9f","accent":"#c97b4a","green":"#5b8c6e",
           "rose":"#a85d7a","grey":"#8a8f98","light":"#cdd7e0"}
CLR4 = ["#3f6f9f","#5b8c6e","#c97b4a","#a85d7a"]   # 4-category palette

def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300)
    print(f"saved  {name}.pdf / .png")

sadi  = pd.read_csv(os.path.join(DATA_DIR,"scoring_sadi_filled.csv"))
tasin = pd.read_csv(os.path.join(DATA_DIR,"scoring_tasin_filled.csv"))
key   = pd.read_csv(os.path.join(DATA_DIR,"blind_key.csv"))
overlap = set(pd.read_csv(os.path.join(DATA_DIR,"kappa_overlap_uids.csv")).row_uid)

tasin_solo = tasin[~tasin.row_uid.isin(overlap)]
scored = pd.concat([sadi, tasin_solo], ignore_index=True)

scored = scored.merge(key, on="row_uid", how="left")
scored["framing"] = scored["prompt_version"]
scored["answer_words"] = scored["raw_output"].fillna("").astype(str).str.split().apply(len)
print("combined scored rows:", len(scored))
print(scored.tag.value_counts().to_dict())

kap = pd.read_csv(os.path.join(DATA_DIR,"cohens_kappa_report.csv"))
fig, ax = plt.subplots(figsize=(6.2,3.2))
bars = ax.barh(kap.column[::-1], kap.cohen_kappa[::-1], color=PALETTE["primary"], height=.6)
ax.axvline(0.80, ls="--", color=PALETTE["accent"], lw=1.4, label="gate κ = 0.80")
for b,v in zip(bars, kap.cohen_kappa[::-1]):
    ax.text(v-0.02, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", ha="right",
            color="white", fontweight="bold", fontsize=9)
ax.set_xlim(0,1.02); ax.set_xlabel("Cohen's κ (600 dual-scored rows)")
ax.set_title("Inter-rater reliability per scored column")
ax.legend(loc="lower right", frameon=False, fontsize=9)
save(fig,"fig0_interrater_kappa"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9,2.6)); ax.axis("off")
steps = ["Benchmarks\n(TruthfulQA · TriviaQA · MMLU\n+ fresh items)",
         "Paired prompts\nclear  vs  ambiguous",
         "5 open-weight models\nlocked settings",
         "Blind dual scoring\n5 columns · Cohen's κ",
         "Analysis\nmechanism mapping"]
x = np.linspace(0.04,0.82,5)
for xi,s in zip(x,steps):
    ax.add_patch(mpatches.FancyBboxPatch((xi,0.32),0.15,0.36,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        fc="#eef2f5", ec=PALETTE["primary"], lw=1.3))
    ax.text(xi+0.075,0.5,s,ha="center",va="center",fontsize=7.6,color="#33414e")
    if xi<x[-1]:
        ax.annotate("",xy=(xi+0.185,0.5),xytext=(xi+0.15,0.5),
            arrowprops=dict(arrowstyle="-|>",color=PALETTE["accent"],lw=1.8))
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title("Study pipeline",fontsize=12,loc="left",pad=2)
save(fig,"fig1_method_pipeline"); plt.close(fig)

fac = scored[scored.tag=="main"].copy()
fac = fac[fac.correctness.isin(["correct","incorrect"])]   # drop abstained for rate
fac["err"] = (fac.correctness=="incorrect").astype(int)
cond_map = {"C1":("clear","general"),"C2":("clear","niche"),
            "C4":("ambiguous","general"),"C5":("ambiguous","niche")}
fac = fac[fac.condition.isin(cond_map)]
fac["clarity"] = fac.condition.map(lambda c: cond_map[c][0])
fac["domain"]  = fac.condition.map(lambda c: cond_map[c][1])
tab = fac.groupby(["domain","clarity"]).err.mean().mul(100).reset_index()
fig, ax = plt.subplots(figsize=(6,3.6))
sns.barplot(data=tab, x="domain", y="err", hue="clarity",
            palette=[PALETTE["primary"],PALETTE["accent"]], ax=ax)
ax.set_ylabel("Error rate (%)"); ax.set_xlabel("")
ax.set_title("Hallucination under prompt stress (H1)")
ax.legend(title="prompt", frameon=False)
for c in ax.containers: ax.bar_label(c, fmt="%.0f%%", fontsize=8, padding=2)
save(fig,"fig2_error_by_condition"); plt.close(fig)

g = scored[(scored.tag=="main") & scored.correctness.isin(["correct","incorrect"])
           & scored.confidence_tone.astype(str).isin(["1","1.0","2","2.0","3","3.0"])].copy()
g["tone"] = g.confidence_tone.astype(float).astype(int).map({1:"hedged",2:"plain",3:"assertive"})
ct = (g.groupby(["correctness","tone"]).size()
        / g.groupby("correctness").size()).mul(100).rename("pct").reset_index()
order=["hedged","plain","assertive"]
fig, ax = plt.subplots(figsize=(6,3.6))
sns.barplot(data=ct, x="tone", y="pct", hue="correctness", order=order,
            palette=[PALETTE["green"],PALETTE["rose"]], ax=ax)
ax.set_ylabel("% of answers"); ax.set_xlabel("confidence tone")
ax.set_title("Does it stay confident when wrong? (H2)")
ax.legend(title="", frameon=False)
for c in ax.containers: ax.bar_label(c, fmt="%.0f%%", fontsize=8, padding=2)
save(fig,"fig3_confidence_accuracy_gap"); plt.close(fig)

dem = scored[scored.tag=="demo"].copy()
dem["occ"] = dem.framing
pr = (dem.groupby(["occ","bias_signal"]).size()
        / dem.groupby("occ").size()).mul(100).rename("pct").reset_index()
piv = pr.pivot(index="occ", columns="bias_signal", values="pct").fillna(0)
for col in ["he","she","they","avoided"]:
    if col not in piv: piv[col]=0
piv = piv[["he","she","they","avoided"]]
fig, ax = plt.subplots(figsize=(6.4,3.8))
piv.plot(kind="bar", stacked=True, color=CLR4, ax=ax, width=.7)
ax.set_ylabel("% of responses"); ax.set_xlabel("")
ax.set_title("Occupational pronoun assignment (H4 · demographic)")
ax.legend(title="pronoun", frameon=False, bbox_to_anchor=(1.01,1))
plt.xticks(rotation=0)
save(fig,"fig4_pronoun_by_occupation"); plt.close(fig)

au = scored[scored.tag=="auth"].copy()
order_f = ["anonymous","patient","physician","researcher"]
au["framing"] = pd.Categorical(au.framing, order_f, ordered=True)
length = au.groupby("framing", observed=True).answer_words.mean().reindex(order_f)
fig, ax = plt.subplots(figsize=(6,3.6))
bars = ax.bar(order_f, length.values, color=CLR4, width=.62)
ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=2)
ax.set_ylabel("mean answer length (words)"); ax.set_xlabel("user framing")
ax.set_title("Answer detail by user identity (H4 · authority)")
save(fig,"fig5_authority_treatment"); plt.close(fig)

sa = scored[scored.tag=="safety"].copy()
sa["present"] = (sa.safety_statement=="present").astype(int)
sa["framing"] = pd.Categorical(sa.framing, order_f, ordered=True)
pres = sa.groupby("framing", observed=True).present.mean().mul(100).reindex(order_f)
fig, ax = plt.subplots(figsize=(6,3.6))
colors = [PALETTE["green"] if f in ("anonymous","patient") else PALETTE["accent"] for f in order_f]
bars = ax.bar(order_f, pres.values, color=colors, width=.62)
ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
ax.set_ylabel("safety statement present (%)"); ax.set_xlabel("user framing")
ax.set_ylim(0,105)
ax.set_title("Do guardrails drop for 'experts'? (H4 · safety)")
save(fig,"fig6_safety_drop"); plt.close(fig)

hm = fac.groupby(["model_code","condition"]).err.mean().mul(100).reset_index()
piv2 = hm.pivot(index="model_code", columns="condition", values="err")
fig, ax = plt.subplots(figsize=(6.2,3.6))
sns.heatmap(piv2, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label":"error %"},
            linewidths=.5, linecolor="white", ax=ax)
ax.set_title("Error rate by model × condition")
ax.set_xlabel(""); ax.set_ylabel("")
save(fig,"fig7_model_condition_heatmap"); plt.close(fig)

print("\nAll figures written to", FIG_DIR)
