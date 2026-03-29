# main.py
import argparse
import os
import sys

from train import train_model
from evaluate import evaluate_model
from inference import predict_entities, run_interactive_inference
from config import DEFAULT_DATA_PATH, BEST_MODEL_DIR

def main():
    parser = argparse.ArgumentParser(description="Türkçe Söz Sanatları Tespiti (BERTurk NER)")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="interactive",
        choices=["train", "evaluate", "inference", "interactive", "all"],
        help="Çalıştırılacak mod: train, evaluate, inference, interactive, veya all (Varsayılan: interactive)"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        default=DEFAULT_DATA_PATH, 
        help="Veri seti JSON dosyasının yolu"
    )
    
    parser.add_argument(
        "--epochs", 
        type=float, 
        default=3.0, 
        help="Eğitim için epoch sayısı"
    )
    
    parser.add_argument(
        "--text", 
        type=str, 
        default="Ayağını denk al yoksa başın belaya girer.", 
        help="Inference modunda test edilecek örnek metin"
    )

    args = parser.parse_args()

    print("="*60)
    print(" TÜRKÇE SÖZ SANATLARI TESPİTİ - BERTURK NER PİPELINE")
    print("="*60)

    if args.mode in ["train", "all"]:
        print(f"\n---> [MOD: EĞİTİM (TRAIN)] <---")
        train_model(data_path=args.data, epochs=args.epochs)
        
    if args.mode in ["evaluate", "all"]:
        print(f"\n---> [MOD: DEĞERLENDİRME (EVALUATE)] <---")
        if not os.path.exists(BEST_MODEL_DIR):
            print(f"HATA: Değerlendirme için '{BEST_MODEL_DIR}' bulunamadı.")
            print("Lütfen önce modeli eğitin: python main.py --mode train")
            if args.mode != "all":
                sys.exit(1)
        else:
            evaluate_model(model_path=BEST_MODEL_DIR, data_path=args.data)
            
    if args.mode in ["inference", "all"]:
        print(f"\n---> [MOD: ÇIKARIM (INFERENCE)] <---")
        if not os.path.exists(BEST_MODEL_DIR):
            print(f"HATA: Çıkarım için '{BEST_MODEL_DIR}' bulunamadı.")
            print("Lütfen önce modeli eğitin: python main.py --mode train")
        else:
            predict_entities(args.text, model_path=BEST_MODEL_DIR)
            
    if args.mode == "interactive":
        if not os.path.exists(BEST_MODEL_DIR):
            print(f"HATA: Çıkarım için '{BEST_MODEL_DIR}' bulunamadı.")
            print("Lütfen önce modeli eğitin: python main.py --mode train")
        else:
            run_interactive_inference(model_path=BEST_MODEL_DIR)

if __name__ == "__main__":
    main()
