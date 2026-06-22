# -*- coding: utf-8 -*-
"""
Tek model (7 sinifli) vs Uc model (ensemble) karsilastirmasi.
Her iki yaklasim da AYNI 100 cumlelik el-etiketli test seti uzerinde,
ayni metrik (seqeval entity-level F1) ile degerlendirilir.

Uc-model tarafi: her token icin 3 modelin tahminleri arasindan en yuksek
guvenli non-O etiketi secilir (max_score ensemble, token seviyesinde).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification, TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from config import MODEL_NAME, CATEGORIES, HYPERPARAMS
from tokenizer_utils import tokenizer, tokenize_and_align_labels

LABEL7 = ["O", "B-DEYIM", "I-DEYIM", "B-MECAZ", "I-MECAZ", "B-ABARTI", "I-ABARTI"]
L2I7 = {l: i for i, l in enumerate(LABEL7)}
I2L7 = {i: l for i, l in enumerate(LABEL7)}
CATS = {"abarti": "ABARTI", "mecaz": "MECAZ", "deyim": "DEYIM"}


def load(fn):
    return json.load(open(os.path.join(HERE, fn), encoding="utf-8"))


def tok7(ds):
    return ds.map(lambda ex: tokenize_and_align_labels(ex, L2I7),
                  batched=True, remove_columns=ds.column_names)


def train_generic(train_list, label_list, label2id, id2label, tag):
    full = Dataset.from_list(train_list)
    split = full.train_test_split(test_size=0.1, seed=42)
    tok = lambda ds: ds.map(lambda ex: tokenize_and_align_labels(ex, label2id),
                            batched=True, remove_columns=ds.column_names)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    args = TrainingArguments(
        output_dir=os.path.join(HERE, f"tmp_cmp_{tag}"),
        learning_rate=HYPERPARAMS["learning_rate"],
        per_device_train_batch_size=HYPERPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMS["per_device_eval_batch_size"],
        num_train_epochs=HYPERPARAMS["num_train_epochs"],
        weight_decay=HYPERPARAMS["weight_decay"],
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        save_total_limit=1, report_to="none", logging_steps=100,
    )

    def cm(p):
        pred, lab = p
        pred = np.argmax(pred, axis=2)
        tp = [[label_list[a] for (a, l) in zip(pr, lb) if l != -100] for pr, lb in zip(pred, lab)]
        tl = [[label_list[l] for (a, l) in zip(pr, lb) if l != -100] for pr, lb in zip(pred, lab)]
        return {"f1": f1_score(tl, tp)}

    trainer = Trainer(model=model, args=args,
                      train_dataset=tok(split["train"]), eval_dataset=tok(split["test"]),
                      processing_class=tokenizer,
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      compute_metrics=cm)
    trainer.train()
    return trainer.model


def predict_logits(model, examples, label2id):
    """Her cumle icin kelime-hizali (word-level) etiket olasiliklarini dondurur.
    Donen: liste; her eleman np.array [n_kelime, n_etiket]."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for ex in examples:
        enc = tokenizer(ex["tokens"], truncation=True, is_split_into_words=True, return_tensors="pt")
        word_ids = enc.word_ids(0)
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [n_subtoken, n_label]
        # her kelimenin ilk alt-tokeninin olasiligini al
        word_probs = []
        prev = None
        for i, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                continue
            word_probs.append(probs[i])
            prev = wid
        out.append(np.array(word_probs))
    return out


def eval_seq(true_seqs, pred_seqs):
    return {
        "precision": round(precision_score(true_seqs, pred_seqs), 4),
        "recall": round(recall_score(true_seqs, pred_seqs), 4),
        "f1": round(f1_score(true_seqs, pred_seqs), 4),
        "accuracy": round(accuracy_score(true_seqs, pred_seqs), 4),
    }


def main():
    # --- birlesik manuel egitim + birlesik 100 test ---
    manual = load("manual_abarti.json") + load("manual_mecaz.json") + load("manual_deyim.json")
    test = load("test_abarti.json") + load("test_mecaz.json") + load("test_deyim.json")
    gold = [ex["labels"] for ex in test]
    print(f"birlesik manuel egitim: {len(manual)} cumle | birlesik test: {len(test)} cumle", flush=True)

    results = {}

    # === TEK MODEL (7 sinifli) ===
    print("\n=== TEK MODEL (7 sinifli) egitiliyor ===", flush=True)
    single = train_generic(manual, LABEL7, L2I7, I2L7, "single")
    sp = predict_logits(single, test, L2I7)
    single_pred = [[I2L7[int(np.argmax(row))] for row in wp] for wp in sp]
    results["single"] = eval_seq(gold, single_pred)
    print("TEK MODEL test:", results["single"], flush=True)

    # === UC MODEL (ensemble) ===
    cat_models = {}
    for cat, LAB in CATS.items():
        print(f"\n=== {LAB} modeli (R0) egitiliyor ===", flush=True)
        cfg = CATEGORIES[LAB]
        cat_models[cat] = (train_generic(load(f"manual_{cat}.json"),
                                         cfg["label_list"], cfg["label2id"], cfg["id2label"], cat),
                           cfg["label_list"])

    # her modelin 100 test uzerindeki kelime-olasiliklari
    per_model = {}
    for cat, (model, llist) in cat_models.items():
        per_model[cat] = (predict_logits(model, test, CATEGORIES[CATS[cat]]["label2id"]), llist)

    # token seviyesinde max-skor birlestirme
    ens_pred = []
    for si in range(len(test)):
        n = len(test[si]["tokens"])
        seq = []
        for ti in range(n):
            best_label, best_score = "O", 0.0
            for cat, (wp_list, llist) in per_model.items():
                row = wp_list[si][ti]
                j = int(np.argmax(row))
                lab = llist[j]
                if lab != "O" and row[j] > best_score:
                    best_label, best_score = lab, row[j]
            seq.append(best_label)
        # BIO tutarliligi: basta I- gelirse B-'ye cevir
        for ti in range(n):
            if seq[ti].startswith("I-"):
                t = seq[ti][2:]
                if ti == 0 or seq[ti-1] not in ("B-"+t, "I-"+t):
                    seq[ti] = "B-"+t
        ens_pred.append(seq)
    results["ensemble"] = eval_seq(gold, ens_pred)
    print("UC MODEL (ensemble) test:", results["ensemble"], flush=True)

    json.dump(results, open(os.path.join(HERE, "compare_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nKAYDEDILDI: compare_results.json", flush=True)
    # gecici klasorleri temizle
    import shutil
    for tag in ["single", "abarti", "mecaz", "deyim"]:
        shutil.rmtree(os.path.join(HERE, f"tmp_cmp_{tag}"), ignore_errors=True)


if __name__ == "__main__":
    main()
