# -*- coding: utf-8 -*-
"""
Deney verisi hazirlama scripti.

1. Manuel veriyi git'teki orijinal commit'ten ayiklar (390/302/1187 - duzeltmeler sonrasi 389/302/1173)
2. Sentetik havuzu mevcut 3000'lik dosyalardan cikarir, eksigi uretip kategori basina tam 3000'e tamamlar
3. Egitim rejim dosyalarini olusturur:
   R0: sadece manuel
   R1: manuel + 1000 sentetik
   R2: manuel + 2000 sentetik
   R3: manuel + 3000 sentetik
4. Test dosyalarini dataset_test.json'dan kategoriye gore ayirir (testte 0 sentetik)
"""
import json
import os
import re
import subprocess
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
ROOT = os.path.dirname(DATA)

sys.path.insert(0, DATA)  # _to3000 jeneratorlerini kullanmak icin

CATS = {
    "abarti": "ABARTI",
    "mecaz": "MECAZ",
    "deyim": "DEYIM",
}


def key(x):
    return " ".join(x["tokens"])


def check_bio(x, lab=None):
    assert len(x["tokens"]) == len(x["labels"]), x
    prev = "O"
    for l in x["labels"]:
        if lab:
            assert l == "O" or l.endswith(lab), x
        assert not (l.startswith("I-") and prev not in ("B-" + l[2:], "I-" + l[2:])), x
        prev = l


def git_show_json(path):
    out = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, cwd=ROOT).stdout
    return json.loads(out.decode("utf-8"))


def main():
    random.seed(20250608)

    # test seti anahtarlari (hicbir egitim dosyasiyla cakismamali)
    test_all = json.load(open(os.path.join(DATA, "dataset_test.json"), encoding="utf-8"))
    test_keys = {key(x) for x in test_all}

    summary = {}

    for cat, LAB in CATS.items():
        cur = json.load(open(os.path.join(DATA, f"dataset_{cat}.json"), encoding="utf-8"))
        head = git_show_json(f"data/dataset_{cat}.json")
        head_keys = {key(x) for x in head}

        manual = [x for x in cur if key(x) in head_keys]
        synth = [x for x in cur if key(x) not in head_keys]

        # eksik sentetigi uret (kategori basina tam 3000)
        need = 3000 - len(synth)
        if need > 0:
            existing = {key(x) for x in cur} | test_keys
            import importlib
            gen_mod = importlib.import_module("_to3000")
            gen_fn = {"abarti": gen_mod.gen_abarti, "mecaz": gen_mod.gen_mecaz, "deyim": gen_mod.gen_deyim}[cat]
            produced = 0
            attempt = 0
            while produced < need and attempt < 30:
                attempt += 1
                random.seed(31337 + attempt * 101)
                cands = gen_fn()
                random.shuffle(cands)
                for s in cands:
                    if produced >= need:
                        break
                    s = re.sub(r"\s+", " ", s).strip()
                    ex = gen_mod.build(s, LAB)
                    try:
                        gen_mod.validate(ex, s)
                    except AssertionError:
                        continue
                    k = key(ex)
                    if k in existing:
                        continue
                    existing.add(k)
                    synth.append(ex)
                    produced += 1
            assert produced == need, f"{cat}: sentetik uretim yetersiz ({produced}/{need})"

        # dogrulama
        for x in manual + synth:
            check_bio(x, LAB)
        assert len(synth) >= 3000, f"{cat}: sentetik {len(synth)} < 3000"
        synth = synth[:3000]

        # sentetigi sabit seed ile karistir -> R1 ⊂ R2 ⊂ R3 artan kumeler
        random.seed(42)
        random.shuffle(synth)

        json.dump(manual, open(os.path.join(HERE, f"manual_{cat}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(synth, open(os.path.join(HERE, f"synthetic_{cat}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        for rname, n in [("R0", 0), ("R1", 1000), ("R2", 2000), ("R3", 3000)]:
            train = manual + synth[:n]
            # egitim/test sizintisi kontrolu
            assert not ({key(x) for x in train} & test_keys), f"{cat} {rname}: test sizintisi!"
            json.dump(train, open(os.path.join(HERE, f"train_{cat}_{rname}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # kategoriye ozel test dosyasi
        test_cat = [x for x in test_all if any(l.endswith(LAB) for l in x["labels"] if l != "O")]
        for x in test_cat:
            check_bio(x, LAB)
        json.dump(test_cat, open(os.path.join(HERE, f"test_{cat}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        summary[cat] = {
            "manual": len(manual),
            "synthetic_pool": len(synth),
            "test": len(test_cat),
            "R0": len(manual), "R1": len(manual) + 1000,
            "R2": len(manual) + 2000, "R3": len(manual) + 3000,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
