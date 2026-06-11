# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import os

from config import CATEGORIES, BASE_DIR
from tokenizer_utils import tokenizer, tokenize_and_align_labels
from train import _make_compute_metrics
from data_loader import load_and_split_dataset


def evaluate_model(category, data_path=None):
    """
    Belirtilen kategori modeli için test seti değerlendirmesi yapar.
    """
    cat_cfg = CATEGORIES[category]
    label_list = cat_cfg["label_list"]
    label2id = cat_cfg["label2id"]
    model_dir = cat_cfg["model_dir"]
    dataset_path = data_path or cat_cfg["dataset"]

    print(f"[{category}] {model_dir} modeli değerlendiriliyor...")

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    except Exception as e:
        print(f"[{category}] Model yüklenemedi. Önce modeli eğitin. Hata: {str(e)}")
        return

    dataset = load_and_split_dataset(file_path=dataset_path)
    if 'test' not in dataset or len(dataset['test']) == 0:
        print(f"[{category}] Test veri seti bulunamadı.")
        return

    test_tokenized = dataset["test"].map(
        lambda examples: tokenize_and_align_labels(examples, label2id),
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        eval_dataset=test_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(label_list)
    )

    predictions, labels, _ = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions, axis=2)

    true_predictions_flat = []
    true_labels_flat = []
    true_predictions_seq = []
    true_labels_seq = []

    for prediction, label in zip(predictions, labels):
        pred_seq = []
        lbl_seq = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_predictions_flat.append(label_list[p])
                true_labels_flat.append(label_list[l])
                pred_seq.append(label_list[p])
                lbl_seq.append(label_list[l])
        true_predictions_seq.append(pred_seq)
        true_labels_seq.append(lbl_seq)

    print(f"\n{'='*50}")
    print(f"[{category}] NER METRİKLERİ (SEQEVAL)")
    print(f"{'='*50}")
    print(classification_report(true_labels_seq, true_predictions_seq))

    print(f"\n{'='*50}")
    print(f"[{category}] COHEN'S KAPPA")
    print(f"{'='*50}")
    kappa = cohen_kappa_score(true_labels_flat, true_predictions_flat)
    print(f"Cohen's Kappa Skoru: {kappa:.4f}")

    print(f"\n{'='*50}")
    print(f"[{category}] CONFUSION MATRIX")
    print(f"{'='*50}")
    labels_order = list(label_list)
    cm = confusion_matrix(true_labels_flat, true_predictions_flat, labels=labels_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"{category} - Token-Level Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(BASE_DIR, f"confusion_matrix_{category.lower()}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix {cm_path} olarak kaydedildi.")


def get_f1_score(category, data_path=None):
    """
    Belirtilen kategori modeli için F1, Precision, Recall ve Kappa skorlarını
    hesaplayıp sözlük olarak döndürür. GUI'den çağrılmak üzere tasarlandı.
    Hata durumunda None döner.
    """
    from seqeval.metrics import precision_score, recall_score, f1_score

    cat_cfg = CATEGORIES[category]
    label_list = cat_cfg["label_list"]
    label2id = cat_cfg["label2id"]
    model_dir = cat_cfg["model_dir"]
    dataset_path = data_path or cat_cfg["dataset"]

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    except Exception:
        return None

    try:
        dataset = load_and_split_dataset(file_path=dataset_path)
    except Exception:
        return None

    if 'test' not in dataset or len(dataset['test']) == 0:
        return None

    test_tokenized = dataset["test"].map(
        lambda examples: tokenize_and_align_labels(examples, label2id),
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        eval_dataset=test_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(label_list)
    )

    predictions, labels, _ = trainer.predict(test_tokenized)
    predictions = np.argmax(predictions, axis=2)

    true_predictions_seq = []
    true_labels_seq = []
    true_predictions_flat = []
    true_labels_flat = []

    for prediction, label in zip(predictions, labels):
        pred_seq = []
        lbl_seq = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_predictions_flat.append(label_list[p])
                true_labels_flat.append(label_list[l])
                pred_seq.append(label_list[p])
                lbl_seq.append(label_list[l])
        true_predictions_seq.append(pred_seq)
        true_labels_seq.append(lbl_seq)

    kappa = cohen_kappa_score(true_labels_flat, true_predictions_flat)

    return {
        "precision": precision_score(true_labels_seq, true_predictions_seq),
        "recall": recall_score(true_labels_seq, true_predictions_seq),
        "f1": f1_score(true_labels_seq, true_predictions_seq),
        "kappa": kappa,
    }


def evaluate_all():
    """Tüm kategori modellerini değerlendirir."""
    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f" {category} MODELİ DEĞERLENDİRİLİYOR")
        print(f"{'='*60}")
        evaluate_model(category)


if __name__ == "__main__":
    evaluate_all()
