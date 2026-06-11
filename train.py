# train.py
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from config import MODEL_NAME, CATEGORIES, HYPERPARAMS, OUTPUT_DIR
from tokenizer_utils import tokenizer, tokenize_and_align_labels
from data_loader import load_and_split_dataset


def _make_compute_metrics(label_list):
    """Verilen etiket listesine göre metrik hesaplama fonksiyonu üretir."""
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }
    return compute_metrics


def train_model(category, epochs=None, data_path=None):
    """
    Belirtilen kategori (DEYIM / MECAZ / ABARTI) için modeli eğitir.
    """
    cat_cfg = CATEGORIES[category]
    label_list = cat_cfg["label_list"]
    label2id = cat_cfg["label2id"]
    id2label = cat_cfg["id2label"]
    model_dir = cat_cfg["model_dir"]
    dataset_path = data_path or cat_cfg["dataset"]

    print(f"[{category}] Veri seti yükleniyor: {dataset_path}")
    dataset = load_and_split_dataset(file_path=dataset_path)

    print(f"[{category}] Tokenize ediliyor ve etiketler hizalanıyor...")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, label2id),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    print(f"[{category}] Model yükleniyor ({MODEL_NAME})...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_epochs = epochs if epochs is not None else HYPERPARAMS["num_train_epochs"]

    output_dir = f"{OUTPUT_DIR}_{category.lower()}"

    print(f"[{category}] Eğitim başlıyor... ({training_epochs} Epoch)")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=HYPERPARAMS["learning_rate"],
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=training_epochs,
        weight_decay=HYPERPARAMS["weight_decay"],
        eval_strategy=HYPERPARAMS["eval_strategy"],
        save_strategy=HYPERPARAMS["save_strategy"],
        load_best_model_at_end=HYPERPARAMS["load_best_model_at_end"],
        metric_for_best_model=HYPERPARAMS["metric_for_best_model"],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(label_list)
    )

    trainer.train()

    print(f"[{category}] Eğitim tamamlandı! Model kaydediliyor: {model_dir}")
    trainer.save_model(model_dir)

    return trainer, tokenized_datasets


def train_all(epochs=None):
    """Tüm kategoriler için ayrı ayrı model eğitir."""
    results = {}
    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f" {category} MODELİ EĞİTİLİYOR")
        print(f"{'='*60}")
        trainer, datasets = train_model(category, epochs=epochs)
        results[category] = (trainer, datasets)
    return results


if __name__ == "__main__":
    train_all()
