# inference.py
import torch
from transformers import AutoModelForTokenClassification, pipeline
from config import MODEL_NAME, LABEL_LIST, BEST_MODEL_DIR
from tokenizer_utils import tokenizer

def predict_entities(text, model_path=BEST_MODEL_DIR):
    """
    Verilen metin üzerindeki deyim, mecaz ve abartı ifadelerini tahmin eder.
    Hugging Face pipeline'ını kullanır.
    """
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Model yüklenemedi. Lütfen önce 'main.py --mode train' ile modeli eğitin.")
        print(f"Hata detayı: {str(e)}")
        return

    # Pipeline oluştur
    # aggregation_strategy="simple" ile alt-kelimeler (subwords) otomatik olarak
    # birleştirilerek anlamlı kelimelere/ifadelere dönüştürülür.
    nlp_pipeline = pipeline(
        "token-classification", 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy="simple"
    )

    print(f"\nİncelenen Metin: '{text}'")
    predictions = nlp_pipeline(text)
    
    if not predictions:
        print("-> Metinde herhangi bir söz sanatı tespit edilemedi (Tümü 'O').")
        return

    print("-> Tespit Edilen Söz Sanatları:")
    for pred in predictions:
        # Puan (confidence score), etiket (entity_group) ve metin parçası
        entity_group = pred['entity_group']
        word = pred['word']
        score = pred['score']
        
        # Sadece bizim aradığımız sınıfları göster (LABEL_LIST'teki ana sınıflar DEYIM, MECAZ, ABARTI)
        if entity_group != "O":
            print(f"  - [{entity_group}] (Eminlik: %{score*100:.1f}) => {word}")

def run_interactive_inference(model_path=BEST_MODEL_DIR):
    """
    Kullanıcıdan sürekli girdi alan interaktif çıkarım döngüsü.
    """
    print("\n--- İnteraktif NLP Test Aracı ---")
    print("Çıkmak için 'q' veya 'quit' yazın.")
    
    while True:
        text = input("\nTest edilecek cümleyi girin: ")
        if text.lower() in ['q', 'quit', 'exit']:
            break
            
        if not text.strip():
            continue
            
        predict_entities(text, model_path)

if __name__ == "__main__":
    # Test amaçlı
    test_sentences = [
        "Ona o kadar kızdım ki gözümden ateş fışkırıyordu.", # Abartı/Mecaz
        "Bu projeyi bitirmek için taşın altına elimizi koymalıyız.", # Deyim
        "Çocuk sevinçten havalara uçtu." # Abartı/Deyim
    ]
    
    for sentence in test_sentences:
        predict_entities(sentence)
