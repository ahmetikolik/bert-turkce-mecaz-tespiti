# -*- coding: utf-8 -*-
"""
Sentetik veri etkisi deneyleri.

4 rejim x 3 kategori = 12 egitim:
  R0: sadece manuel
  R1: manuel + 1000 sentetik
  R2: manuel + 2000 sentetik
  R3: manuel + 3000 sentetik

Her kosuda:
  - Egitim verisi %90 train / %10 validation olarak bolunur (best model secimi icin)
  - Egitim sonrasi TUM egitim verisi uzerinde F1 (train basarisi)
  - El ile etiketlenmis harici test seti uzerinde F1 (test basarisi - 0 sentetik)

Sonuclar results.json'a kosudan kosuya yazilir.
"""
import json
import os
import sys
import time
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score

from config import MODEL_NAME, CATEGORIES, HYPERPARAMS
from tokenizer_utils import tokenizer, tokenize_and_align_labels

RESULTS_PATH = os.path.join(HERE, "results.json")
REGIMES = ["R0", "R1", "R2", "R3"]
CATS = {"abarti": "ABARTI", "mecaz": "MECAZ", "deyim": "DEYIM"}


def seq_metrics(trainer, tokenized, label_list):
    predictions, labels, _ = trainer.predict(tokenized)
    predictions = np.argmax(predictions, axis=2)
    preds_seq, labels_seq = [], []
    for prediction, label in zip(predictions, labels):
        p_seq, l_seq = [], []
        for p, l in zip(prediction, label):
            if l != -100:
                p_seq.append(label_list[p])
                l_seq.append(label_list[l])
        preds_seq.append(p_seq)
        labels_seq.append(l_seq)
    return {
        "precision": round(precision_score(labels_seq, preds_seq), 4),
        "recall": round(recall_score(labels_seq, preds_seq), 4),
        "f1": round(f1_score(labels_seq, preds_seq), 4),
    }


def run_one(cat, regime, save_final=False):
    LAB = CATS[cat]
    cat_cfg = CATEGORIES[LAB]
    label_list = cat_cfg["label_list"]
    label2id = cat_cfg["label2id"]
    id2label = cat_cfg["id2label"]

    train_data = json.load(open(os.path.join(HERE, f"train_{cat}_{regime}.json"), encoding="utf-8"))
    test_data = json.load(open(os.path.join(HERE, f"test_{cat}.json"), encoding="utf-8"))

    full_train = Dataset.from_list(train_data)
    split = full_train.train_test_split(test_size=0.1, seed=42)
    ds_train, ds_val = split["train"], split["test"]
    ds_test = Dataset.from_list(test_data)

    tok = lambda ds: ds.map(
        lambda ex: tokenize_and_align_labels(ex, label2id),
        batched=True, remove_columns=ds.column_names,
    )
    tk_train, tk_val, tk_full, tk_test = tok(ds_train), tok(ds_val), tok(full_train), tok(ds_test)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id,
    )

    out_dir = os.path.join(HERE, f"tmp_out_{cat}_{regime}")
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=HYPERPARAMS["learning_rate"],
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=HYPERPARAMS["num_train_epochs"],
        weight_decay=HYPERPARAMS["weight_decay"],
        eval_strategy=HYPERPARAMS["eval_strategy"],
        save_strategy=HYPERPARAMS["save_strategy"],
        load_best_model_at_end=HYPERPARAMS["load_best_model_at_end"],
        metric_for_best_model=HYPERPARAMS["metric_for_best_model"],
        save_total_limit=1,
        report_to="none",
        push_to_hub=False,
        logging_steps=50,
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        tp = [[label_list[q] for (q, l) in zip(pr, lb) if l != -100] for pr, lb in zip(predictions, labels)]
        tl = [[label_list[l] for (q, l) in zip(pr, lb) if l != -100] for pr, lb in zip(predictions, labels)]
        return {"f1": f1_score(tl, tp)}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tk_train, eval_dataset=tk_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    dur = time.time() - t0

    train_scores = seq_metrics(trainer, tk_full, label_list)
    test_scores = seq_metrics(trainer, tk_test, label_list)

    if save_final:
        trainer.save_model(cat_cfg["model_dir"])
        print(f"[{cat}/{regime}] final model kaydedildi: {cat_cfg['model_dir']}")

    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "category": cat, "regime": regime,
        "train_size": len(train_data), "test_size": len(test_data),
        "train": train_scores, "test": test_scores,
        "duration_sec": round(dur, 1),
    }


def main():
    results = []
    if os.path.exists(RESULTS_PATH):
        results = json.load(open(RESULTS_PATH, encoding="utf-8"))
    done = {(r["category"], r["regime"]) for r in results}

    for regime in REGIMES:
        for cat in CATS:
            if (cat, regime) in done:
                print(f"[SKIP] {cat}/{regime} zaten var")
                continue
            print(f"\n{'='*60}\n  KOSU: {cat.upper()} / {regime}\n{'='*60}", flush=True)
            res = run_one(cat, regime, save_final=(regime == "R3"))
            results.append(res)
            json.dump(results, open(RESULTS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"[OK] {cat}/{regime}: train_f1={res['train']['f1']} test_f1={res['test']['f1']} ({res['duration_sec']}s)", flush=True)

    print("\nTUM DENEYLER TAMAMLANDI")


if __name__ == "__main__":
    main()
