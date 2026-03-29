import json
import os
from config import BASE_DIR

# Overfitting testi: tek bir cümleyi 100 kere tekrarlıyoruz.
# Eğer model doğru çalışıyorsa, bu cümleyi ezberleyip %100 doğru tahmin etmeli.
sentence = {
    "tokens": ["Ayağını", "denk", "al", "yoksa", "görürsün", "gününü", "."],
    "labels": ["B-DEYIM", "I-DEYIM", "I-DEYIM", "O", "B-DEYIM", "I-DEYIM", "O"]
}

dataset = [sentence.copy() for _ in range(200)]

if __name__ == "__main__":
    out_path = os.path.join(BASE_DIR, "data", "dataset_overfit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"{len(dataset)} adet ayni cumle ile overfitting dataseti '{out_path}' konumuna kaydedildi!")
