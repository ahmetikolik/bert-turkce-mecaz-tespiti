# train.py
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from config import MODEL_NAME, LABEL_LIST, ID2LABEL, LABEL2ID, HYPERPARAMS, OUTPUT_DIR, LOGGING_DIR, BEST_MODEL_DIR
from tokenizer_utils import tokenizer, tokenize_and_align_labels
from data_loader import load_and_split_dataset

def compute_metrics(p):
    """
    Model tahminleri ile gerçek etiketleri karşılaştırıp seqeval metriklerini hesaplar.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Padding olan tokenleri (-100) dikkate almıyoruz
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

def train_model(data_path="data/dataset.json", epochs=None):
    """
    Dataset'i yükler, hazırlar, modeli oluşturur ve Trainer API ile eğitimi başlatır.
    """
    print("Veri seti yükleniyor ve bölünüyor...")
    dataset = load_and_split_dataset(file_path=data_path)
    
    print("Veri seti tokenize ediliyor ve etiketler hizalanıyor...")
    # Dataset'i özellik çıkarma amaçlı map'liyoruz
    tokenized_datasets = dataset.map(
        # Burada tokenizer_utils.py içinde düzeltmemiz gereken ufak bir detay var. 
        # Trainer.map beklediği fonksiyon formatına uygun küçük bir wrapper.
        lambda examples: tokenize_and_align_labels(examples),
        batched=True,
        remove_columns=dataset["train"].column_names # Orijinal metin kolonlarını siliyoruz ki model sadece tensor algılasın
    )
    
    print("Model yükleniyor...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Padding için Data Collator (dinamik olarak her batch'teki en uzun diziye göre padding yapar)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_epochs = epochs if epochs is not None else HYPERPARAMS["num_train_epochs"]

    print(f"Eğitim başlıyor... ({training_epochs} Epoch)")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=HYPERPARAMS["learning_rate"],
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=training_epochs,
        weight_decay=HYPERPARAMS["weight_decay"],
        eval_strategy=HYPERPARAMS["eval_strategy"],
        save_strategy=HYPERPARAMS["save_strategy"],
        load_best_model_at_end=HYPERPARAMS["load_best_model_at_end"],
        metric_for_best_model=HYPERPARAMS["metric_for_best_model"],
        logging_dir=LOGGING_DIR,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Eğitimi başlat
    trainer.train()
    
    print(f"Eğitim tamamlandı! En iyi model kaydediliyor: {BEST_MODEL_DIR}")
    trainer.save_model(BEST_MODEL_DIR)
    
    return trainer, tokenized_datasets

if __name__ == "__main__":
    train_model()
