# -*- coding: utf-8 -*-
"""Makale (Ingilizce) icin Ingilizce etiketli sentetik-etki grafigi uretir."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\ahmet\Desktop\YTU_Computer_Engineering_Project_Template__1_\Article_Template\fig_synth_effect_en.png"

r = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
REG = ["R0", "R1", "R2", "R3"]
X = [0, 1000, 2000, 3000]
CATS = {"abarti": "Hyperbole", "mecaz": "Metaphor", "deyim": "Idiom"}
COLORS = {"abarti": "#d62728", "mecaz": "#1f77b4", "deyim": "#ff7f0e"}

def get(cat, reg, split):
    for x in r:
        if x["category"] == cat and x["regime"] == reg:
            return x[split]["f1"]
    raise KeyError((cat, reg))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, split, title in [(axes[0], "test", "Test F1 (100 hand-labeled sentences)"),
                         (axes[1], "train", "Training F1")]:
    for cat, disp in CATS.items():
        ys = [get(cat, reg, split) for reg in REG]
        ax.plot(X, ys, marker="o", label=disp, color=COLORS[cat])
    macro = [sum(get(c, reg, split) for c in CATS) / 3 for reg in REG]
    ax.plot(X, macro, marker="s", linestyle="--", color="black", label="Macro avg.")
    ax.set_xlabel("Synthetic sentences added per category")
    ax.set_ylabel("F1 score")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(X)
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=160)
print("kaydedildi:", OUT)
