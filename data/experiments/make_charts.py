# -*- coding: utf-8 -*-
"""Deney sonuclarindan rapor icin grafik uretir."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = r"C:\Users\ahmet\Desktop\YTU_Computer_Engineering_Project_Template__1_"

r = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
REG = ["R0", "R1", "R2", "R3"]
X = [0, 1000, 2000, 3000]
CATS = {"abarti": "ABARTI", "mecaz": "MECAZ", "deyim": "DEYİM"}
COLORS = {"abarti": "#d62728", "mecaz": "#1f77b4", "deyim": "#ff7f0e"}

def get(cat, reg, split):
    for x in r:
        if x["category"] == cat and x["regime"] == reg:
            return x[split]["f1"]
    raise KeyError((cat, reg))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, split, title in [(axes[0], "test", "Test Başarısı (100 el ile etiketli cümle)"),
                         (axes[1], "train", "Eğitim Başarısı")]:
    macro = []
    for cat, disp in CATS.items():
        ys = [get(cat, reg, split) for reg in REG]
        ax.plot(X, ys, marker="o", label=disp, color=COLORS[cat])
    for i, reg in enumerate(REG):
        macro.append(sum(get(c, reg, split) for c in CATS) / 3)
    ax.plot(X, macro, marker="s", linestyle="--", color="black", label="Makro Ort.")
    ax.set_xlabel("Eklenen sentetik cümle sayısı (kategori başına)")
    ax.set_ylabel("F1 skoru")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(X)
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(TEMPLATE, "sekil_5_1_sentetik_etki.png")
plt.savefig(out, dpi=160)
print("kaydedildi:", out)

# makro tablosu (rapora yazilacak sayilar)
print("\nMakro ortalamalar:")
for reg in REG:
    tr = sum(get(c, reg, "train") for c in CATS) / 3
    te = sum(get(c, reg, "test") for c in CATS) / 3
    print(f"{reg}: train={tr:.4f} test={te:.4f}")
