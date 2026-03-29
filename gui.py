# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import threading
from transformers import AutoModelForTokenClassification, pipeline
from config import BEST_MODEL_DIR, LABEL_LIST, BASE_DIR
from tokenizer_utils import tokenizer
from train import train_model

# Kategori renkleri tanımlaması
COLORS = {
    "DEYIM": "#ffb703",   # Turuncu/Sarımsı
    "MECAZ": "#8ecae6",   # Açık Mavi
    "ABARTI": "#ffb5a7",  # Açık Kırmızı/Pembe
}

class NLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Türkçe Söz Sanatları Tespiti (NER)")
        self.root.geometry("800x600")
        self.root.configure(padx=20, pady=20)
        
        self.nlp_pipeline_word = None
        self.nlp_pipeline_token = None
        self.setup_ui()
        
        # Modeli arka planda yükle ki arayüz donmasın
        self.load_model_thread()

    def setup_ui(self):
        # Başlık
        title_lbl = tk.Label(self.root, text="Türkçe Söz Sanatları Tespiti", font=("Arial", 18, "bold"))
        title_lbl.pack(pady=(0, 10))
        
        desc_lbl = tk.Label(self.root, text="BERTurk modeli kullanarak Deyim, Mecaz ve Abartı (Mübalağa) tespiti yapar.", font=("Arial", 10))
        desc_lbl.pack(pady=(0, 20))

        # Eğitim Alanı Çerçevesi
        train_frame = tk.LabelFrame(self.root, text="Yapay Zeka Eğitim Ayarları", font=("Arial", 10, "bold"), padx=10, pady=10)
        train_frame.pack(fill=tk.X, expand=False, pady=5)
        
        train_inner = tk.Frame(train_frame)
        train_inner.pack(fill=tk.X)
        
        tk.Label(train_inner, text="Eğitim Verisi Seçin:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(train_inner, textvariable=self.dataset_var, state="readonly", width=30)
        self.dataset_combo['values'] = ("data/dataset_overfit.json", "data/dataset_general.json")
        self.dataset_combo.current(0)
        self.dataset_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(train_inner, text="Epoch:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 5))
        self.epoch_var = tk.StringVar(value="3.0")
        self.epoch_spin = ttk.Spinbox(train_inner, from_=1.0, to=10.0, increment=1.0, textvariable=self.epoch_var, width=5)
        self.epoch_spin.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = tk.Button(train_inner, text="Eğit (Train)", font=("Arial", 10, "bold"), bg="#2196F3", fg="white", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=10)

        # Girdi Alanı Çerçevesi
        input_frame = tk.LabelFrame(self.root, text="Metin Analizi", font=("Arial", 10, "bold"), padx=10, pady=10)
        input_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=5, font=("Arial", 12))
        self.text_input.pack(fill=tk.BOTH, expand=True)
        self.text_input.insert(tk.END, "Ayağını denk al yoksa başın belaya girer.")

        # Durum Çubuğu
        self.status_var = tk.StringVar()
        self.status_var.set("⏳ Model yükleniyor... Lütfen bekleyin.")
        status_lbl = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 10, "italic"), fg="blue")
        status_lbl.pack(pady=5)

        # Analiz Butonu
        self.analyze_btn = tk.Button(self.root, text="Analiz Et", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                                  command=self.analyze_text, state=tk.DISABLED, width=20, height=2)
        self.analyze_btn.pack(pady=10)

        # Sonuç Gösterim Alanı (Rich Text)
        result_frame = tk.LabelFrame(self.root, text="Analiz Sonucu", font=("Arial", 10, "bold"), padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Etiket bilgileri
        info_frame = tk.Frame(result_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(info_frame, text="Renk Kodları: ", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(info_frame, text=" DEYİM ", bg=COLORS["DEYIM"], font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Label(info_frame, text=" MECAZ ", bg=COLORS["MECAZ"], font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Label(info_frame, text=" ABARTI ", bg=COLORS["ABARTI"], font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=10, font=("Arial", 14), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Renk taglerini ayarla
        for entity_type, color in COLORS.items():
            self.result_text.tag_config(entity_type, background=color, foreground="black", font=("Arial", 14, "bold"))
            
        self.result_text.tag_config("NORMAL", font=("Arial", 14))

    def load_model_thread(self):
        thread = threading.Thread(target=self._load_model)
        thread.daemon = True
        thread.start()

    def start_training(self):
        dataset_choice = self.dataset_var.get()
        if not dataset_choice: return
        
        try:
            epochs = float(self.epoch_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli bir Epoch sayısı girin.")
            return

        full_path = os.path.join(BASE_DIR, *dataset_choice.split("/"))
        if not os.path.exists(full_path):
            messagebox.showerror("Hata", f"Veri seti bulunamadı: {full_path}")
            return
            
        self.train_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.dataset_combo.config(state=tk.DISABLED)
        self.epoch_spin.config(state=tk.DISABLED)
        self.status_var.set("⏳ Model eğitiliyor... Bu işlem biraz zaman alabilir, lütfen bekleyin.")
        
        thread = threading.Thread(target=self._train_thread, args=(full_path, epochs))
        thread.daemon = True
        thread.start()

    def _train_thread(self, data_path, epochs):
        try:
            train_model(data_path=data_path, epochs=epochs)
            self.root.after(0, self._train_success)
        except Exception as e:
            self.root.after(0, self._show_error, "Eğitim Hatası", f"Eğitim sırasında bir hata oluştu:\n{str(e)}")
            self.root.after(0, self._train_fail)
            
    def _train_success(self):
        self.status_var.set("✅ Eğitim tamamlandı! Yeni model yükleniyor...")
        self.load_model_thread()
        self.train_btn.config(state=tk.NORMAL)
        self.dataset_combo.config(state=tk.NORMAL)
        self.epoch_spin.config(state=tk.NORMAL)
        messagebox.showinfo("Başarılı", "Eğitim başarıyla tamamlandı ve model kaydedildi!")

    def _train_fail(self):
        self.status_var.set("❌ Eğitim başarısız.")
        self.train_btn.config(state=tk.NORMAL)
        self.dataset_combo.config(state=tk.NORMAL)
        self.epoch_spin.config(state=tk.NORMAL)
        self.analyze_btn.config(state=tk.NORMAL if self.nlp_pipeline_word else tk.DISABLED)

    def _load_model(self):
        try:
            if not os.path.exists(BEST_MODEL_DIR):
                self.root.after(0, self._show_error, "Model Bulunamadı", 
                               f"Eğitilmiş model '{BEST_MODEL_DIR}' dizininde bulunamadı.\nLütfen önce 'python main.py --mode train' ile modeli eğitin.")
                self.status_var.set("❌ Model bulunamadı.")
                return
                
            model = AutoModelForTokenClassification.from_pretrained(BEST_MODEL_DIR)
            self.nlp_pipeline_word = pipeline(
                "token-classification", 
                model=model, 
                tokenizer=tokenizer, 
                aggregation_strategy="simple",
                ignore_labels=[] # NORMAL (O) etiketlerini de gormek icin bu ayar sart
            )
            
            self.nlp_pipeline_token = pipeline(
                "token-classification", 
                model=model, 
                tokenizer=tokenizer, 
                aggregation_strategy="none",
                ignore_labels=[] # Token modunda da NORMAL etiketler dahil olsun
            )
            
            # UI güncellemeleri ana thread'de olmalı
            self.root.after(0, self._model_loaded_success)
            
        except Exception as e:
            self.root.after(0, self._show_error, "Yükleme Hatası", f"Model yüklenirken bir hata oluştu:\n{str(e)}")
            self.status_var.set("❌ Model yükleme hatası.")

    def _model_loaded_success(self):
        self.status_var.set("✅ Model başarıyla yüklendi. Hazır!")
        self.analyze_btn.config(state=tk.NORMAL)

    def _show_error(self, title, message):
        messagebox.showerror(title, message)

    def analyze_text(self):
        if not self.nlp_pipeline_word or not self.nlp_pipeline_token:
            messagebox.showwarning("Uyarı", "Model henüz yüklenmedi veya bir hata oluştu.")
            return

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Bilgi", "Lütfen analiz edilecek bir metin girin.")
            return

        self.status_var.set("🔍 Analiz ediliyor...")
        self.root.update()
        
        try:
            predictions_word = self.nlp_pipeline_word(text)
            predictions_token = self.nlp_pipeline_token(text)
            self.display_results(text, predictions_word, predictions_token)
            self.status_var.set("✅ Analiz tamamlandı.")
        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"Metin analiz edilirken hata oluştu:\n{str(e)}")
            self.status_var.set("❌ Analiz hatası.")

    def display_results(self, text, predictions_word, predictions_token):
        # Text kutusunu temizle ve yazdırılabilir yap
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        if not predictions_word and not predictions_token:
             self.result_text.insert(tk.END, text, "NORMAL")
             self.result_text.insert(tk.END, "\n\n(Metinde söz sanatı bulunamadı.)")
             self.result_text.config(state=tk.DISABLED)
             return

        # 1. BÖLÜM: KELİME BAZLI GÖSTERİM
        self.result_text.insert(tk.END, "▶ BÖLÜM 1: KELİME BAZLI ANALİZ (Birleştirilmiş)\n", "NORMAL")
        self.result_text.insert(tk.END, "-"*60 + "\n", "NORMAL")
        
        predictions_word = sorted(predictions_word, key=lambda x: x['start'])
        current_idx = 0
        details_word = []

        for pred in predictions_word:
            start = pred['start']
            end = pred['end']
            entity = pred['entity_group']
            word = pred['word']
            score = pred['score']
            
            if current_idx < start:
                self.result_text.insert(tk.END, text[current_idx:start], "NORMAL")
            
            tag = "NORMAL" if entity == "O" else entity
            word_text = text[start:end]
            display_text = f" {word_text} " if tag == "NORMAL" else f" [{word_text} - {tag} %{score*100:.1f}] "
            
            self.result_text.insert(tk.END, display_text, tag)
            details_word.append(f"• Kelime: '{word_text}' | Tespit: {tag} | Eminlik: %{score*100:.2f}")
            
            current_idx = end

        if current_idx < len(text):
            self.result_text.insert(tk.END, text[current_idx:], "NORMAL")
            
        # 2. BÖLÜM: TOKEN BAZLI GÖSTERİM
        self.result_text.insert(tk.END, "\n\n\n▶ BÖLÜM 2: TOKEN BAZLI ANALİZ (Alt-Kelimeler - Modelin Gözünden)\n", "NORMAL")
        self.result_text.insert(tk.END, "-"*60 + "\n", "NORMAL")
        
        predictions_token = sorted(predictions_token, key=lambda x: x['start'])
        current_idx = 0
        details_token = []

        for pred in predictions_token:
            start = pred['start']
            end = pred['end']
            entity = pred['entity']
            word = pred['word']
            score = pred['score']
            
            if current_idx < start:
                self.result_text.insert(tk.END, text[current_idx:start], "NORMAL")
            
            tag = "NORMAL" if entity == "O" else entity.replace("B-", "").replace("I-", "")
            
            word_text = text[start:end]
            display_text = f" {word} " if tag == "NORMAL" else f" [{word} - {entity} %{score*100:.1f}] "
            
            self.result_text.insert(tk.END, display_text, tag)
            details_token.append(f"• Token: '{word}' | Etiket: {entity} | Eminlik: %{score*100:.2f}")
            
            current_idx = end
            
        if current_idx < len(text):
            self.result_text.insert(tk.END, text[current_idx:], "NORMAL")
            
        # DETAYLAR BÖLÜMÜ
        self.result_text.insert(tk.END, "\n\n" + "="*70 + "\nDETAYLI TABLOLAR:\n", "NORMAL")
        self.result_text.insert(tk.END, "\n--- 1. Kelime Bazlı Detay ---\n" + "\n".join(details_word), "NORMAL")
        self.result_text.insert(tk.END, "\n\n--- 2. Token Bazlı (Alt-Kelimeler) Detay ---\n" + "\n".join(details_token), "NORMAL")

        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = NLPApp(root)
    root.mainloop()
