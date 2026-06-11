# data_loader.py
import json
import os
from datasets import Dataset, DatasetDict


def load_and_split_dataset(file_path, test_size=0.2, val_size=0.5, seed=42):
    """
    JSON dosyasından veriyi okur, train/validation/test split yapar
    ve Hugging Face Dataset formuna dönüştürür.
    Varsayılan split oranları: %80 Train, %10 Val, %10 Test
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Veri seti bulunamadı: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    hf_dataset = Dataset.from_list(data)

    if len(hf_dataset) < 10:
        print("Uyarı: Veri seti boyutu çok küçük (< 10). Split yapılmıyor, tüm veri train olarak alındı.")
        return DatasetDict({
            'train': hf_dataset,
            'validation': hf_dataset,
            'test': hf_dataset
        })
    else:
        train_testval = hf_dataset.train_test_split(test_size=test_size, seed=seed)

        if len(train_testval['test']) >= 2:
            test_val = train_testval['test'].train_test_split(test_size=val_size, seed=seed)
            dataset = DatasetDict({
                'train': train_testval['train'],
                'validation': test_val['train'],
                'test': test_val['test']
            })
        else:
            dataset = DatasetDict({
                'train': train_testval['train'],
                'validation': train_testval['test'],
                'test': train_testval['test']
            })

        return dataset


if __name__ == "__main__":
    from config import CATEGORIES
    for cat_key, cat_cfg in CATEGORIES.items():
        print(f"\n--- {cat_key} ---")
        try:
            ds = load_and_split_dataset(cat_cfg["dataset"])
            print(ds)
            if 'train' in ds and len(ds['train']) > 0:
                print("Örnek:", ds['train'][0])
        except FileNotFoundError as e:
            print(e)
