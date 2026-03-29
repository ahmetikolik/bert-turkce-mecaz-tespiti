# tokenizer_utils.py
from transformers import AutoTokenizer
from config import MODEL_NAME, LABEL2ID

# Sabit tokenizer yüklenmesi
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    """
    Bu fonksiyon, gelen metinleri (kelime listesi halindeki) tokene çevirir 
    ve ardından etiketleri yeni oluşturulan alt-kelime (subword) tokenleriyle hizalar.
    
    WordPiece tokenizer (BERTurk'un kullandığı) bir kelimeyi birden fazla parçaya bölebilir.
    Örnek: "Gözyaşları" -> "Göz", "##yaş", "##ları"
    
    Orijinal etiketi yalnızca KELİMENİN İLK PARÇASINA atıyoruz.
    Sonraki parçalara ise -100 etiketi atıyoruz ki kayıp (loss) hesaplamasında dikkate alınmasın.
    [CLS], [SEP] ve [PAD] tokenlerine de -100 atanır.
    """
    # is_split_into_words=True, metnin zaten kelimelere ayrılmış olduğunu belirtir
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding=False # Padding işlemi Trainer içindeki DataCollator tarafından yapılacak
    )

    labels = []
    
    # Her bir örnekteki cümle için (batch işlemi)
    for i, label in enumerate(examples["labels"]):
        # Tokenizer'ın ürettiği kelime indeksleri haritasını alıyoruz
        # Örnek: [None, 0, 1, 1, 2, None] -> [CLS], Word0, Word1(P1), Word1(P2), Word2, [SEP]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        # Etiketleri ID'lere çevir
        label_ids = [LABEL2ID[l] for l in label]
        
        new_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], [PAD] gibi özel tokenler için -100 atarız
                new_labels.append(-100)
            elif word_idx != previous_word_idx:
                # Bir kelimenin İLK parçası (ilk subword token) ise orijinal etiketi atarız
                new_labels.append(label_ids[word_idx])
            else:
                # Aynı kelimenin SONRAKİ parçaları ise -100 atarız
                new_labels.append(-100)
                
            previous_word_idx = word_idx

        labels.append(new_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == "__main__":
    # Test script - fonksiyonun çalışıp çalışmadığını kontrol eder
    test_example = {
        "tokens": [["Ayağını", "denk", "al", "yoksa", "başına", "iş", "açarsın"]],
        "labels": [["B-DEYIM", "I-DEYIM", "I-DEYIM", "O", "B-DEYIM", "I-DEYIM", "I-DEYIM"]]
    }
    
    print("Orijinal Cümle:", test_example["tokens"][0])
    print("Orijinal Etiketler:", test_example["labels"][0])
    
    result = tokenize_and_align_labels(test_example)
    
    tokens = tokenizer.convert_ids_to_tokens(result["input_ids"][0])
    aligned_labels = result["labels"][0]
    
    print("\nTokenize Edilmiş ve Hizalanmış Sonuç:")
    for token, label in zip(tokens, aligned_labels):
        print(f"{token:15} -> {label}")
