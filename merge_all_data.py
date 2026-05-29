# merge_all_data.py
# Tum egitim dosyalarini (txt + mevcut JSON) tek bir temiz dataset.json'a birlestirir.
# .txt dosyalari satir-satir JSON objesi iceriyor (bazi satirlar geersiz [, ]) -> tolere ediyoruz.
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_PATH = os.path.join(DATA_DIR, "dataset.json")

VALID_LABELS = {"O", "B-DEYIM", "I-DEYIM", "B-MECAZ", "I-MECAZ", "B-ABARTI", "I-ABARTI"}

# Hem text hem JSON dosyalarindan okuyacagiz
SOURCES = [
    os.path.join(BASE, "abartı.txt"),
    os.path.join(BASE, "deyim_A_harfi.txt"),
    os.path.join(BASE, "gerçek.txt"),
    os.path.join(BASE, "mecaz_gemini.txt"),
    os.path.join(DATA_DIR, "dataset_general.json"),
    os.path.join(DATA_DIR, "dataset_deyim_250.json"),
    os.path.join(DATA_DIR, "dataset_overfit.json"),
]

def parse_records_from_text(text):
    """Bir metin parcasindan {tokens, labels} kayitlarini cikarir.
    Once tum metni JSON olarak parse etmeyi dener; basarisizsa satir bazinda parse eder.
    Sondaki kapanmamis virguller, fazla `]` karakterleri tolere edilir."""
    records = []

    # 1) Saglam JSON dizisi olarak dene
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict) and "tokens" in r and "labels" in r]
    except Exception:
        pass

    # 2) Satir bazinda regex ile {...} bloklarini yakala
    # cok satirli olmadigini varsayiyoruz (txt dosyalari her kayit bir satir)
    obj_pattern = re.compile(r"\{[^{}]*\}")
    for match in obj_pattern.finditer(text):
        chunk = match.group(0)
        # Sondaki virgul vs olabilir, json.loads bununla ilgilenmez (yok zaten match icinde)
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        if isinstance(obj, dict) and "tokens" in obj and "labels" in obj:
            records.append(obj)
    return records


def validate(rec):
    toks = rec.get("tokens")
    labs = rec.get("labels")
    if not isinstance(toks, list) or not isinstance(labs, list):
        return False
    if len(toks) == 0 or len(toks) != len(labs):
        return False
    if not all(isinstance(t, str) and t for t in toks):
        return False
    if not all(isinstance(l, str) and l in VALID_LABELS for l in labs):
        return False
    return True


def main():
    all_records = []
    seen = set()  # dedup anahtari: tokens'in tuple hali
    stats = {}

    for src in SOURCES:
        if not os.path.exists(src):
            print(f"[SKIP] yok: {src}")
            continue
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
        recs = parse_records_from_text(text)
        kept = 0
        for r in recs:
            if not validate(r):
                continue
            key = tuple(r["tokens"])
            if key in seen:
                continue
            seen.add(key)
            all_records.append({"tokens": r["tokens"], "labels": r["labels"]})
            kept += 1
        stats[os.path.basename(src)] = (len(recs), kept)
        print(f"[OK]  {os.path.basename(src):30s} parsed={len(recs):4d}  added={kept:4d}")

    # Etiket dagilimi
    label_counts = {}
    for r in all_records:
        for l in r["labels"]:
            label_counts[l] = label_counts.get(l, 0) + 1

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print("\n=== OZET ===")
    print(f"Toplam benzersiz cumle: {len(all_records)}")
    print(f"Yazilan dosya: {OUT_PATH}")
    print("Etiket dagilimi:")
    for k in sorted(label_counts):
        print(f"  {k:10s} {label_counts[k]}")


if __name__ == "__main__":
    main()
