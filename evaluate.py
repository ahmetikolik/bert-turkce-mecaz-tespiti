# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, Trainer
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score

from config import LABEL_LIST, BEST_MODEL_DIR, BASE_DIR
from tokenizer_utils import tokenizer
from train import compute_metrics, DataCollatorForTokenClassification
from data_loader import load_and_split_dataset
import os

def evaluate_model(model_path=BEST_MODEL_DIR, data_path="data/dataset.json"):
    """
    Test veri seti üzerinde modeli değerlendirir ve raporlama yapar.
    """
    print(f"{model_path} modeli test seti ile değerlendiriliyor...")
    
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Model yüklenemedi. Önce modeli eğitmeniz gerekiyor. Hata: {str(e)}")
        return
        
    dataset = load_and_split_dataset(file_path=data_path)
    if 'test' not in dataset or len(dataset['test']) == 0:
        print("Uyarı: Test veri seti bulunamadı. Değerlendirme yapılamıyor.")
        return
        
    from tokenizer_utils import tokenize_and_align_labels
    test_tokenized = dataset["test"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["test"].column_names
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Ham tahminleri al
    predictions, labels, _ = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions, axis=2)
    
    # -100 olanları temizle ve flatten et
    true_predictions_flat = []
    true_labels_flat = []
    
    true_predictions_seq = []
    true_labels_seq = []
    
    for prediction, label in zip(predictions, labels):
        pred_seq = []
        lbl_seq = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_predictions_flat.append(LABEL_LIST[p])
                true_labels_flat.append(LABEL_LIST[l])
                
                pred_seq.append(LABEL_LIST[p])
                lbl_seq.append(LABEL_LIST[l])
                
        true_predictions_seq.append(pred_seq)
        true_labels_seq.append(lbl_seq)
    
    print("\n" + "="*50)
    print("NER METRİKLERİ (SEQEVAL - Kurum Seviyesi Değerlendirme)")
    print("="*50)
    print(classification_report(true_labels_seq, true_predictions_seq))
    
    print("\n" + "="*50)
    print("COHEN'S KAPPA (Token Seviyesi Uyum)")
    print("="*50)
    kappa = cohen_kappa_score(true_labels_flat, true_predictions_flat)
    print(f"Cohen's Kappa Skoru: {kappa:.4f}")
    print("* Not: NER görevlerinde 'O' etiketinin yoğunluğu sebebiyle bu skor tek başına yanıltıcı olabilir.")
    
    print("\n" + "="*50)
    print("CONFUSION MATRIX (Karmaşıklık Matrisi)")
    print("="*50)
    # Etiketleri sabit bir sırada tutmak için:
    labels_order = list(LABEL_LIST)
    cm = confusion_matrix(true_labels_flat, true_predictions_flat, labels=labels_order)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Token-Level Confusion Matrix")
    plt.tight_layout()
    
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix {cm_path} olarak kaydedildi.")
    
if __name__ == "__main__":
    evaluate_model()
