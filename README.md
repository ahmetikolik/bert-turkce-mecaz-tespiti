# 🎭 BERTurk Türkçe Söz Sanatları Tespiti (NER)

Bu proje, geleneksel Token Sınıflandırma (NER - Named Entity Recognition) problemini **Türkçe söz sanatlarının (Deyim, Mecaz, Abartı)** tespit edilmesine uyarlamaktadır. Önceden eğitilmiş `dbmdz/bert-base-turkish-cased` (**BERTurk**) dil modeli üzerine kurulan bu sistem, yapay zeka eğitimini ve metin analizini tek bir masaüstü arayüzünden (`gui.py`) yapmanıza olanak tanır.

## 🚀 Özellikler
1. **İki Modlu Test Desteği:** İster "Overfit" (aşırı öğrenme) verisiyle modelin kapasitesini test edin, ister geniş metinlerle genel performans ölçümü yapın.
2. **Kelime & Token Bazlı İnceleme (UI):** Sadece söz sanatlarını değil, "O" (Normal) dahil her harfin, ekin ve kelimenin yüzde kaç ihtimalle hangi etikete girdiğini görebilirsiniz.
3. **Tek Tıkla Eğitim:** Hiç koda girmeden sadece arayüzü kullanarak Epoch (döngü) ve Dataset belirleyip yeni modeller eğitebilirsiniz.

---

## 🛠️ Kurulum Bilgileri

Projeyi bilgisayarınıza indirdikten sonra, kendi ortamınızda tüm yapay zeka kütüphanelerini kurmanız gerekmektedir. 

```bash
# Projeyi indirin
git clone https://github.com/ahmetikolik/bert-turkce-mecaz-tespiti.git
cd bert-turkce-mecaz-tespiti

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

---

## 🏃‍♂️ Nasıl Çalıştırılır?

Yapay zeka model ağırlıkları (Ağırlıklar yüzlerce megabayt yer kapladığı için) GitHub reposunda yüklü değildir! Bu sebeple kodu çalıştırdıktan sonra öncelikle ufak bir eğitim yaparak modeli kendi bilgisayarınızda inşa etmeniz gerekir.

**1. Veri setlerini (JSON) oluşturun:**
```bash
python create_overfit_dataset.py
python create_100_dataset.py
```

**2. Arayüzü (GUI) başlatın:**
```bash
python gui.py
```

**3. Eğitimi Başlatın (Çok Önemli!)**
Arayüz açıldığında "Analiz Et" butonu kapalı olacaktır çünkü `best_model` henüz bilgisayarınızda yok.
- Üst bölümden `dataset_overfit.json` veya `dataset_general.json` seçin.
- Yan taraftan Epoch'u ayarlayın (Önerilen: `3.0`).
- **"Eğit (Train)"** butonuna basın ve barın dolmasını bekleyin. Bu işlem internet hızınıza bağlı olarak BERTurk'ü indirecek ve eğitecektir.

**4. Analiz ve Çıkarım (Inference)**
Eğitim tamamlandığında "Analiz Et" butonu parlayacaktır. Ekrandaki metin kutusuna Türkçe bir cümle girip sistemin Deyim, Mecaz veya Abartı tespit etme olasılıklarını kelime/hece düzeyinde kontrol edebilirsiniz.
