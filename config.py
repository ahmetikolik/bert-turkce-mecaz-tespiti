# config.py
# Proje sabitleri, etiket haritaları ve hiperparametreler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kullanılacak ön-eğitimli temel model
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# ───────────────────────────────────────────────
# Her kategori için ayrı 3-etiketli BIO şeması
# ───────────────────────────────────────────────
CATEGORIES = {
    "DEYIM": {
        "label_list": ["O", "B-DEYIM", "I-DEYIM"],
        "dataset": os.path.join(BASE_DIR, "data", "dataset_deyim.json"),
        "model_dir": os.path.join(BASE_DIR, "best_model_deyim"),
        "display_name": "Deyim",
        "color": "#ffb703",       # Turuncu/Sarımsı
    },
    "MECAZ": {
        "label_list": ["O", "B-MECAZ", "I-MECAZ"],
        "dataset": os.path.join(BASE_DIR, "data", "dataset_mecaz.json"),
        "model_dir": os.path.join(BASE_DIR, "best_model_mecaz"),
        "display_name": "Mecaz",
        "color": "#8ecae6",       # Açık Mavi
    },
    "ABARTI": {
        "label_list": ["O", "B-ABARTI", "I-ABARTI"],
        "dataset": os.path.join(BASE_DIR, "data", "dataset_abarti.json"),
        "model_dir": os.path.join(BASE_DIR, "best_model_abarti"),
        "display_name": "Abartı",
        "color": "#ffb5a7",       # Açık Kırmızı/Pembe
    },
}

# Her kategori için id2label / label2id sözlükleri
for cat_key, cat_cfg in CATEGORIES.items():
    cat_cfg["id2label"] = {i: lbl for i, lbl in enumerate(cat_cfg["label_list"])}
    cat_cfg["label2id"] = {lbl: i for i, lbl in enumerate(cat_cfg["label_list"])}

# Hiperparametreler
HYPERPARAMS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3.0,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
}

# Genel çıktı dizini (checkpoint'ler için)
OUTPUT_DIR = os.path.join(BASE_DIR, "model_output")
