# main.py
import argparse
import os
import sys

from config import CATEGORIES
from train import train_model, train_all
from evaluate import evaluate_model, evaluate_all
from inference import predict_entities, run_interactive_inference, load_all_pipelines


def main():
    parser = argparse.ArgumentParser(description="Türkçe Söz Sanatları Tespiti (BERTurk NER) — 3 Ayrı Model")

    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["train", "evaluate", "inference", "interactive", "all"],
        help="Çalıştırılacak mod (Varsayılan: interactive)"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["DEYIM", "MECAZ", "ABARTI"],
        help="Eğitim/değerlendirme için kategori. Belirtilmezse tüm kategoriler işlenir."
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

    print("=" * 60)
    print(" TÜRKÇE SÖZ SANATLARI TESPİTİ — 3 AYRI MODEL")
    print("=" * 60)

    if args.mode in ["train", "all"]:
        print(f"\n---> [MOD: EĞİTİM (TRAIN)] <---")
        if args.category:
            train_model(args.category, epochs=args.epochs)
        else:
            train_all(epochs=args.epochs)

    if args.mode in ["evaluate", "all"]:
        print(f"\n---> [MOD: DEĞERLENDİRME (EVALUATE)] <---")
        if args.category:
            evaluate_model(args.category)
        else:
            evaluate_all()

    if args.mode in ["inference", "all"]:
        print(f"\n---> [MOD: ÇIKARIM (INFERENCE)] <---")
        pipelines = load_all_pipelines()
        if pipelines:
            predict_entities(args.text, pipelines)
        else:
            print("HATA: Hiçbir model bulunamadı. Lütfen önce modelleri eğitin.")

    if args.mode == "interactive":
        run_interactive_inference()


if __name__ == "__main__":
    main()
