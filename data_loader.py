# data_loader.py
import json
import os
from datasets import Dataset, DatasetDict
from config import DEFAULT_DATA_PATH

def create_sample_dataset(file_path):
    """
    Kullanıcı gerçek veri setini sağlayana kadar test amaçlı kullanılabilecek
    küçük bir örnek veri seti oluşturur.
    """
    sample_data = [
        {
            "tokens": ["Ayağını", "denk", "al", "yoksa", "başın", "belaya", "girer"],
            "labels": ["B-DEYIM", "I-DEYIM", "I-DEYIM", "O", "B-DEYIM", "I-DEYIM", "I-DEYIM"]
        },
        {
            "tokens": ["Gözden", "düştü", "ama", "yılmadı", "çalışmaya", "devam", "etti"],
            "labels": ["B-DEYIM", "I-DEYIM", "O", "O", "O", "O", "O"]
        },
        {
            "tokens": ["O", "kadar", "açım", "ki", "bir", "öküzü", "yiyebilirim"],
            "labels": ["O", "O", "O", "O", "B-ABARTI", "I-ABARTI", "I-ABARTI"]
        },
        {
            "tokens": ["Kalbi", "kırılmış", "bir", "şekilde", "odadan", "ayrıldı"],
            "labels": ["B-MECAZ", "I-MECAZ", "O", "O", "O", "O"]
        },
        {
            "tokens": ["Bu", "işte", "parmağı", "olan", "herkes", "hesap", "verecek"],
            "labels": ["O", "O", "B-DEYIM", "I-DEYIM", "O", "O", "O"]
        },
        {
            "tokens": ["Dünyalar", "kadar", "işim", "var", "bugün"],
            "labels": ["B-ABARTI", "I-ABARTI", "O", "O", "O"]
        },
        {
            "tokens": ["Karanlık", "düşünceler", "zihnini", "kemiriyordu"],
            "labels": ["B-MECAZ", "I-MECAZ", "O", "B-MECAZ"]
        },
        {
            "tokens": ["Göz", "dağı", "vermek", "için", "böyle", "konuştu"],
            "labels": ["B-DEYIM", "I-DEYIM", "I-DEYIM", "O", "O", "O"]
        },
        {
            "tokens": ["Gözyaşları", "sel", "oldu", "aktı"],
            "labels": ["O", "B-ABARTI", "I-ABARTI", "O"]
        },
        {
            "tokens": ["Taş", "kalpli", "adam", "hiç", "acıma", "göstermedi"],
            "labels": ["B-MECAZ", "I-MECAZ", "O", "O", "O", "O"]
        }
    ]
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Örnek veri seti oluşturuldu: {file_path}")

def load_and_split_dataset(file_path=DEFAULT_DATA_PATH, test_size=0.2, val_size=0.5, seed=42):
    """
    JSON dosyasından veriyi okur, train/validation/test split yapar 
    ve Hugging Face Dataset formuna dönüştürür.
    Varsayılan split oranları: %80 Train, %10 Val, %10 Test
    """
    if not os.path.exists(file_path):
        print(f"Uyarı: {file_path} bulunamadı.")
        print("Test için örnek bir veri seti oluşturuluyor...")
        create_sample_dataset(file_path)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Hugging Face formatına çevirme
    hf_dataset = Dataset.from_list(data)
    
    # Train ve Test statlarına ayırma
    # Eğer veri seti çok küçükse (örnekteki gibi), train_test_split çalışmayabilir
    # Bu durumda tüm veriyi eğitimde kullanacak şekilde bir yapı kuruyoruz
    if len(hf_dataset) < 10:
        print("Uyarı: Veri seti boyutu çok küçük (< 10). Split yapılmıyor, tüm veri train olarak alındı.")
        return DatasetDict({
            'train': hf_dataset,
            'validation': hf_dataset,
            'test': hf_dataset
        })
    else:
        # Önce train ve test_val olarak ikiye ayır (örnek: 80% train, 20% test_val)
        train_testval = hf_dataset.train_test_split(test_size=test_size, seed=seed)
        
        # Sonra test_val kısmını validation ve test olarak ikiye ayır (örnek: %50-%50 -> %10-%10)
        # Sadece yeterli veri varsa ayır
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
    # Test script - bağımsız çalıştırıldığında veri seti istatistiklerini basar
    ds = load_and_split_dataset()
    print("Veri seti yapısı:")
    print(ds)
    if 'train' in ds and len(ds['train']) > 0:
        print("\nÖrnek veri (Train setinden):")
        print(ds['train'][0])
