# config.py
# Proje sabitleri, etiket haritaları ve hiperparametreler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kullanılacak ön-eğitimli temel model
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# 7 sınıflı BIO etiket şeması
# B-DEYIM, I-DEYIM: Deyim başlangıcı ve devamı
# B-MECAZ, I-MECAZ: Mecaz başlangıcı ve devamı
# B-ABARTI, I-ABARTI: Abartı başlangıcı ve devamı
# O: Diğer (Söz sanatı olmayan kelimeler)
LABEL_LIST = [
    "O",
    "B-DEYIM",
    "I-DEYIM",
    "B-MECAZ",
    "I-MECAZ",
    "B-ABARTI",
    "I-ABARTI"
]

# ID'den etikete ve etiketten ID'ye dönüşüm sözlükleri
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}

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

# Veri yolları
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_output")
LOGGING_DIR = os.path.join(BASE_DIR, "logs")
BEST_MODEL_DIR = os.path.join(BASE_DIR, "best_model")
