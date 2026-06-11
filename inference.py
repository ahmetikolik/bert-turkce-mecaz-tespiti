# inference.py
import torch
import os
from transformers import AutoModelForTokenClassification, pipeline
from config import CATEGORIES
from tokenizer_utils import tokenizer


def load_all_pipelines():
    """
    Tüm kategori modellerini yükler ve pipeline sözlüğü döndürür.
    Yüklenen modeller: { "DEYIM": pipeline, "MECAZ": pipeline, "ABARTI": pipeline }
    Yüklenemeyen modeller atlanır.
    """
    pipelines = {}
    for category, cat_cfg in CATEGORIES.items():
        model_dir = cat_cfg["model_dir"]
        if not os.path.exists(model_dir):
            print(f"[{category}] Model bulunamadı: {model_dir} — atlanıyor.")
            continue
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            nlp = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
            pipelines[category] = nlp
            print(f"[{category}] Model yüklendi.")
        except Exception as e:
            print(f"[{category}] Model yüklenemedi: {e}")
    return pipelines


def predict_entities(text, pipelines=None):
    """
    Verilen metin üzerinde tüm kategorilerdeki söz sanatlarını tespit eder.
    3 ayrı model çalıştırılır ve sonuçlar birleştirilir.
    """
    if pipelines is None:
        pipelines = load_all_pipelines()

    if not pipelines:
        print("Hiçbir model yüklü değil. Lütfen önce modelleri eğitin.")
        return []

    print(f"\nİncelenen Metin: '{text}'")

    all_predictions = []
    for category, nlp in pipelines.items():
        preds = nlp(text)
        for pred in preds:
            if pred['entity_group'] != 'O':
                pred['category'] = category
                all_predictions.append(pred)

    if not all_predictions:
        print("-> Metinde herhangi bir söz sanatı tespit edilemedi.")
        return []

    print("-> Tespit Edilen Söz Sanatları:")
    for pred in sorted(all_predictions, key=lambda x: x['start']):
        entity = pred['entity_group']
        word = pred['word']
        score = pred['score']
        cat = pred['category']
        print(f"  - [{cat}] (Eminlik: %{score*100:.1f}) => {word}")

    return all_predictions


def run_interactive_inference():
    """
    Kullanıcıdan sürekli girdi alan interaktif çıkarım döngüsü.
    """
    print("\n--- İnteraktif NLP Test Aracı ---")
    print("Çıkmak için 'q' veya 'quit' yazın.")

    pipelines = load_all_pipelines()

    while True:
        text = input("\nTest edilecek cümleyi girin: ")
        if text.lower() in ['q', 'quit', 'exit']:
            break
        if not text.strip():
            continue
        predict_entities(text, pipelines)


if __name__ == "__main__":
    test_sentences = [
        "Ona o kadar kızdım ki gözümden ateş fışkırıyordu.",
        "Bu projeyi bitirmek için taşın altına elimizi koymalıyız.",
        "Çocuk sevinçten havalara uçtu."
    ]

    pipelines = load_all_pipelines()
    for sentence in test_sentences:
        predict_entities(sentence, pipelines)
