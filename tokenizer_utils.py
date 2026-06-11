# tokenizer_utils.py
from transformers import AutoTokenizer
from config import MODEL_NAME

# Sabit tokenizer yüklenmesi (tüm modeller aynı tokenizer'ı kullanır)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(examples, label2id):
    """
    Gelen metinleri (kelime listesi halindeki) tokene çevirir
    ve etiketleri yeni oluşturulan alt-kelime (subword) tokenleriyle hizalar.

    label2id: Kullanılacak kategoriye özel etiket-ID sözlüğü.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [label2id[l] for l in label]

        new_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                new_labels.append(-100)
            elif word_idx != previous_word_idx:
                new_labels.append(label_ids[word_idx])
            else:
                new_labels.append(-100)
            previous_word_idx = word_idx

        labels.append(new_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    from config import CATEGORIES
    cat_cfg = CATEGORIES["DEYIM"]

    test_example = {
        "tokens": [["Ayağını", "denk", "al", "yoksa", "başına", "iş", "açarsın"]],
        "labels": [["B-DEYIM", "I-DEYIM", "I-DEYIM", "O", "B-DEYIM", "I-DEYIM", "I-DEYIM"]]
    }

    print("Orijinal Cümle:", test_example["tokens"][0])
    print("Orijinal Etiketler:", test_example["labels"][0])

    result = tokenize_and_align_labels(test_example, cat_cfg["label2id"])

    tokens = tokenizer.convert_ids_to_tokens(result["input_ids"][0])
    aligned_labels = result["labels"][0]

    print("\nTokenize Edilmiş ve Hizalanmış Sonuç:")
    for token, label in zip(tokens, aligned_labels):
        print(f"{token:15} -> {label}")
