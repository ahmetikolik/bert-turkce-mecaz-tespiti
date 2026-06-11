# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import threading
from transformers import AutoModelForTokenClassification, pipeline
from config import CATEGORIES, BASE_DIR
from tokenizer_utils import tokenizer
from train import train_model
from evaluate import get_f1_score


class NLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Türkçe Söz Sanatları Tespiti (3 Ayrı Model)")
        self.root.geometry("900x750")
        self.root.configure(padx=20, pady=15)

        # Her kategori için pipeline'lar
        self.pipelines_word = {}   # aggregation_strategy="simple"
        self.pipelines_token = {}  # aggregation_strategy="none"

        self.setup_ui()

        # Mevcut modelleri arka planda yükle
        self.load_all_models_thread()

    def setup_ui(self):
        # ─── BAŞLIK ─────────────────────────────────────────
        title_lbl = tk.Label(self.root, text="Türkçe Söz Sanatları Tespiti", font=("Arial", 18, "bold"))
        title_lbl.pack(pady=(0, 5))

        desc_lbl = tk.Label(self.root,
                            text="BERTurk — Her kategori (Deyim, Mecaz, Abartı) için ayrı uzman model.",
                            font=("Arial", 10))
        desc_lbl.pack(pady=(0, 15))

        # ─── EĞİTİM ALANI ──────────────────────────────────
        train_frame = tk.LabelFrame(self.root, text="Yapay Zeka Eğitim Ayarları",
                                    font=("Arial", 10, "bold"), padx=10, pady=10)
        train_frame.pack(fill=tk.X, expand=False, pady=5)

        # Kategori seçimi
        row1 = tk.Frame(train_frame)
        row1.pack(fill=tk.X, pady=2)

        tk.Label(row1, text="Kategori:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(0, 5))

        self.cat_var = tk.StringVar(value="DEYIM")
        self.cat_combo = ttk.Combobox(row1, textvariable=self.cat_var, state="readonly", width=12,
                                      values=list(CATEGORIES.keys()))
        self.cat_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(row1, text="Epoch:", font=("Arial", 10)).pack(side=tk.LEFT, padx=(15, 5))
        self.epoch_var = tk.StringVar(value="3.0")
        self.epoch_spin = ttk.Spinbox(row1, from_=1.0, to=10.0, increment=1.0,
                                      textvariable=self.epoch_var, width=5)
        self.epoch_spin.pack(side=tk.LEFT, padx=5)

        self.train_btn = tk.Button(row1, text="Seçili Kategoriyi Eğit", font=("Arial", 10, "bold"),
                                   bg="#2196F3", fg="white", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=10)

        self.train_all_btn = tk.Button(row1, text="Hepsini Eğit", font=("Arial", 10, "bold"),
                                       bg="#FF9800", fg="white", command=self.start_training_all)
        self.train_all_btn.pack(side=tk.LEFT, padx=5)

        # Model durumları
        status_frame = tk.Frame(train_frame)
        status_frame.pack(fill=tk.X, pady=(8, 0))

        self.model_status_vars = {}
        for cat_key, cat_cfg in CATEGORIES.items():
            frm = tk.Frame(status_frame)
            frm.pack(side=tk.LEFT, padx=10)
            var = tk.StringVar(value=f"⏳ {cat_cfg['display_name']}")
            lbl = tk.Label(frm, textvariable=var, font=("Arial", 9),
                           bg=cat_cfg["color"], padx=8, pady=2)
            lbl.pack()
            self.model_status_vars[cat_key] = var

        # ─── METİN ANALİZİ ─────────────────────────────────
        input_frame = tk.LabelFrame(self.root, text="Metin Analizi",
                                    font=("Arial", 10, "bold"), padx=10, pady=10)
        input_frame.pack(fill=tk.BOTH, expand=False, pady=10)

        self.text_input = scrolledtext.ScrolledText(input_frame, height=4, font=("Arial", 12))
        self.text_input.pack(fill=tk.BOTH, expand=True)
        self.text_input.insert(tk.END, "Ayağını denk al yoksa başın belaya girer.")

        # Durum çubuğu
        self.status_var = tk.StringVar()
        self.status_var.set("⏳ Modeller yükleniyor...")
        status_lbl = tk.Label(self.root, textvariable=self.status_var,
                              font=("Arial", 10, "italic"), fg="blue")
        status_lbl.pack(pady=3)

        # Analiz Butonu
        self.analyze_btn = tk.Button(self.root, text="Analiz Et", font=("Arial", 12, "bold"),
                                     bg="#4CAF50", fg="white", command=self.analyze_text,
                                     state=tk.DISABLED, width=20, height=2)
        self.analyze_btn.pack(pady=8)

        # ─── SONUÇ GÖSTERİM ALANI ──────────────────────────
        result_frame = tk.LabelFrame(self.root, text="Analiz Sonucu",
                                     font=("Arial", 10, "bold"), padx=10, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Renk kodları bilgisi
        info_frame = tk.Frame(result_frame)
        info_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(info_frame, text="Renk Kodları: ", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        for cat_key, cat_cfg in CATEGORIES.items():
            tk.Label(info_frame, text=f" {cat_cfg['display_name']} ",
                     bg=cat_cfg["color"], font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        self.result_text = scrolledtext.ScrolledText(result_frame, height=10,
                                                     font=("Arial", 14), state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Renk tag'lerini ayarla
        for cat_key, cat_cfg in CATEGORIES.items():
            self.result_text.tag_config(cat_key, background=cat_cfg["color"],
                                        foreground="black", font=("Arial", 14, "bold"))
        self.result_text.tag_config("NORMAL", font=("Arial", 14))

    # ─── MODEL YÜKLEME ─────────────────────────────────────

    def load_all_models_thread(self):
        thread = threading.Thread(target=self._load_all_models)
        thread.daemon = True
        thread.start()

    def _load_all_models(self):
        loaded_count = 0
        for cat_key, cat_cfg in CATEGORIES.items():
            model_dir = cat_cfg["model_dir"]
            if not os.path.exists(model_dir):
                self.root.after(0, self._update_model_status, cat_key, f"❌ {cat_cfg['display_name']}")
                continue
            try:
                model = AutoModelForTokenClassification.from_pretrained(model_dir)

                self.pipelines_word[cat_key] = pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    ignore_labels=[]
                )

                self.pipelines_token[cat_key] = pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="none",
                    ignore_labels=[]
                )

                loaded_count += 1

                # F1 skoru hesapla
                try:
                    metrics = get_f1_score(cat_key)
                    if metrics:
                        f1 = metrics['f1']
                        self.root.after(0, self._update_model_status, cat_key,
                                        f"✅ {cat_cfg['display_name']} (F1: {f1:.2f})")
                    else:
                        self.root.after(0, self._update_model_status, cat_key,
                                        f"✅ {cat_cfg['display_name']}")
                except Exception:
                    self.root.after(0, self._update_model_status, cat_key,
                                    f"✅ {cat_cfg['display_name']}")

            except Exception as e:
                self.root.after(0, self._update_model_status, cat_key, f"❌ {cat_cfg['display_name']}")
                print(f"[{cat_key}] Model yükleme hatası: {e}")

        if loaded_count > 0:
            self.root.after(0, self._models_loaded, loaded_count)
        else:
            self.root.after(0, self._no_models_found)

    def _update_model_status(self, cat_key, text):
        self.model_status_vars[cat_key].set(text)

    def _models_loaded(self, count):
        total = len(CATEGORIES)
        self.status_var.set(f"✅ {count}/{total} model yüklendi. Hazır!")
        self.analyze_btn.config(state=tk.NORMAL)

    def _no_models_found(self):
        self.status_var.set("❌ Hiçbir model bulunamadı. Lütfen önce eğitim yapın.")

    # ─── EĞİTİM ────────────────────────────────────────────

    def start_training(self):
        category = self.cat_var.get()
        try:
            epochs = float(self.epoch_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Geçerli bir Epoch sayısı girin.")
            return

        self._disable_training_ui()
        self.status_var.set(f"⏳ {category} modeli eğitiliyor...")
        self._update_model_status(category, f"🔄 {CATEGORIES[category]['display_name']}")

        thread = threading.Thread(target=self._train_thread, args=([category], epochs))
        thread.daemon = True
        thread.start()

    def start_training_all(self):
        try:
            epochs = float(self.epoch_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Geçerli bir Epoch sayısı girin.")
            return

        self._disable_training_ui()
        self.status_var.set("⏳ Tüm modeller sırayla eğitiliyor...")
        for cat_key in CATEGORIES:
            self._update_model_status(cat_key, f"🔄 {CATEGORIES[cat_key]['display_name']}")

        categories = list(CATEGORIES.keys())
        thread = threading.Thread(target=self._train_thread, args=(categories, epochs))
        thread.daemon = True
        thread.start()

    def _disable_training_ui(self):
        self.train_btn.config(state=tk.DISABLED)
        self.train_all_btn.config(state=tk.DISABLED)
        self.analyze_btn.config(state=tk.DISABLED)
        self.cat_combo.config(state=tk.DISABLED)
        self.epoch_spin.config(state=tk.DISABLED)

    def _enable_training_ui(self):
        self.train_btn.config(state=tk.NORMAL)
        self.train_all_btn.config(state=tk.NORMAL)
        self.cat_combo.config(state="readonly")
        self.epoch_spin.config(state=tk.NORMAL)

    def _train_thread(self, categories, epochs):
        try:
            f1_results = {}
            for category in categories:
                self.root.after(0, lambda c=category:
                    self.status_var.set(f"⏳ {c} modeli eğitiliyor..."))
                train_model(category, epochs=epochs)

                # Eğitim sonrası F1 skoru hesapla
                self.root.after(0, lambda c=category:
                    self.status_var.set(f"📊 {c} F1 skoru hesaplanıyor..."))
                try:
                    metrics = get_f1_score(category)
                    if metrics:
                        f1_results[category] = metrics
                        self.root.after(0, self._update_model_status, category,
                                        f"✅ {CATEGORIES[category]['display_name']} (F1: {metrics['f1']:.2f})")
                    else:
                        self.root.after(0, self._update_model_status, category,
                                        f"✅ {CATEGORIES[category]['display_name']}")
                except Exception:
                    self.root.after(0, self._update_model_status, category,
                                    f"✅ {CATEGORIES[category]['display_name']}")

            self.root.after(0, self._train_success, categories, f1_results)
        except Exception as e:
            self.root.after(0, self._show_error, "Eğitim Hatası", str(e))
            self.root.after(0, self._train_fail)

    def _train_success(self, categories, f1_results=None):
        self._enable_training_ui()
        self.status_var.set("✅ Eğitim tamamlandı! Modeller yeniden yükleniyor...")
        self.load_all_models_thread()

        # F1 skorlarını mesaj kutusunda göster
        lines = []
        for cat in categories:
            name = CATEGORIES[cat]["display_name"]
            if f1_results and cat in f1_results:
                m = f1_results[cat]
                lines.append(f"{name}:\n"
                             f"  Precision: {m['precision']:.4f}\n"
                             f"  Recall:    {m['recall']:.4f}\n"
                             f"  F1-Score:  {m['f1']:.4f}\n"
                             f"  Kappa:     {m['kappa']:.4f}")
            else:
                lines.append(f"{name}: Skor hesaplanamadı")

        messagebox.showinfo("Eğitim Tamamlandı", "\n\n".join(lines))

    def _train_fail(self):
        self._enable_training_ui()
        self.status_var.set("❌ Eğitim başarısız.")
        if self.pipelines_word:
            self.analyze_btn.config(state=tk.NORMAL)

    def _show_error(self, title, message):
        messagebox.showerror(title, message)

    # ─── ANALİZ ─────────────────────────────────────────────

    def analyze_text(self):
        if not self.pipelines_word:
            messagebox.showwarning("Uyarı", "Hiçbir model yüklü değil.")
            return

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Bilgi", "Lütfen analiz edilecek bir metin girin.")
            return

        self.status_var.set("🔍 3 model ile analiz ediliyor...")
        self.root.update()

        try:
            all_word_preds = []
            all_token_preds = []

            for cat_key in self.pipelines_word:
                # Kelime bazlı
                word_preds = self.pipelines_word[cat_key](text)
                for p in word_preds:
                    p['category'] = cat_key
                all_word_preds.extend(word_preds)

                # Token bazlı
                if cat_key in self.pipelines_token:
                    token_preds = self.pipelines_token[cat_key](text)
                    for p in token_preds:
                        p['category'] = cat_key
                    all_token_preds.extend(token_preds)

            self.display_results(text, all_word_preds, all_token_preds)

            active = [CATEGORIES[c]["display_name"] for c in self.pipelines_word]
            self.status_var.set(f"✅ Analiz tamamlandı. ({', '.join(active)} modelleri kullanıldı)")
        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"Hata: {str(e)}")
            self.status_var.set("❌ Analiz hatası.")

    def display_results(self, text, predictions_word, predictions_token):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        # Sadece söz sanatı olanları filtrele
        word_preds = [p for p in predictions_word if p['entity_group'] != 'O']
        token_preds_filtered = [p for p in predictions_token
                                if p.get('entity', '') != 'O'
                                and not p.get('entity', '').startswith('O')]

        if not word_preds and not token_preds_filtered:
            self.result_text.insert(tk.END, text, "NORMAL")
            self.result_text.insert(tk.END, "\n\n(Metinde söz sanatı bulunamadı.)")
            self.result_text.config(state=tk.DISABLED)
            return

        # ── BÖLÜM 1: KELİME BAZLI ──────────────
        self.result_text.insert(tk.END, "▶ BÖLÜM 1: KELİME BAZLI ANALİZ (3 Model Birleşik)\n", "NORMAL")
        self.result_text.insert(tk.END, "-" * 60 + "\n", "NORMAL")

        # Çakışmaları çöz: aynı span'a birden fazla model tahmin yapmışsa en yüksek skoru al
        word_preds_sorted = sorted(word_preds, key=lambda x: (-x['score'], x['start']))
        best_preds = []
        used_ranges = []

        for pred in word_preds_sorted:
            start, end = pred['start'], pred['end']
            overlaps = False
            for us, ue in used_ranges:
                if start < ue and end > us:
                    overlaps = True
                    break
            if not overlaps:
                best_preds.append(pred)
                used_ranges.append((start, end))

        best_preds = sorted(best_preds, key=lambda x: x['start'])

        current_idx = 0
        details_word = []

        for pred in best_preds:
            start = pred['start']
            end = pred['end']
            category = pred['category']
            score = pred['score']

            if current_idx < start:
                self.result_text.insert(tk.END, text[current_idx:start], "NORMAL")

            word_text = text[start:end]
            display_name = CATEGORIES[category]["display_name"]
            display_text = f" [{word_text} - {display_name} %{score*100:.1f}] "
            self.result_text.insert(tk.END, display_text, category)
            details_word.append(
                f"• Kelime: '{word_text}' | Kategori: {display_name} | Eminlik: %{score*100:.2f}")

            current_idx = end

        if current_idx < len(text):
            self.result_text.insert(tk.END, text[current_idx:], "NORMAL")

        # ── BÖLÜM 2: TOKEN BAZLI ───────────────
        self.result_text.insert(tk.END,
            "\n\n\n▶ BÖLÜM 2: TOKEN BAZLI ANALİZ (Alt-Kelimeler - Her Modelden)\n", "NORMAL")
        self.result_text.insert(tk.END, "-" * 60 + "\n", "NORMAL")

        # Token bazlı sonuçları sadece bulunan kategoriler için göster
        active_cats = set(p['category'] for p in token_preds_filtered)

        for cat_key in sorted(active_cats):
            cat_tokens = [p for p in all_token_preds_for_cat(predictions_token, cat_key)]
            if not cat_tokens:
                continue

            display_name = CATEGORIES[cat_key]["display_name"]
            self.result_text.insert(tk.END, f"\n  [{display_name} Modeli]:\n", "NORMAL")

            cat_tokens = sorted(cat_tokens, key=lambda x: x['start'])
            current_idx = 0
            details_token = []

            for pred in cat_tokens:
                start = pred['start']
                end = pred['end']
                entity = pred['entity']
                word = pred['word']
                score = pred['score']

                if current_idx < start:
                    self.result_text.insert(tk.END, text[current_idx:start], "NORMAL")

                tag = "NORMAL" if entity == "O" else cat_key
                if entity == "O":
                    display_text = f" {word} "
                else:
                    display_text = f" [{word} - {entity} %{score*100:.1f}] "

                self.result_text.insert(tk.END, display_text, tag)
                if entity != "O":
                    details_token.append(
                        f"• Token: '{word}' | Etiket: {entity} | Eminlik: %{score*100:.2f}")

                current_idx = end

            if current_idx < len(text):
                self.result_text.insert(tk.END, text[current_idx:], "NORMAL")

        # DETAYLAR
        self.result_text.insert(tk.END, "\n\n" + "=" * 70 + "\nDETAYLI TABLO:\n", "NORMAL")
        self.result_text.insert(tk.END, "\n".join(details_word) if details_word else "(Yok)", "NORMAL")

        self.result_text.config(state=tk.DISABLED)


def all_token_preds_for_cat(predictions_token, cat_key):
    """Belirli bir kategorinin token tahminlerini filtreler."""
    return [p for p in predictions_token if p.get('category') == cat_key]


if __name__ == "__main__":
    root = tk.Tk()
    app = NLPApp(root)
    root.lift()
    root.attributes('-topmost', True)
    root.after(500, lambda: root.attributes('-topmost', False))
    root.focus_force()
    root.mainloop()
