# Kısa Vadeli Elektrik Tüketim Tahmini

LightGBM tabanlı makine öğrenmesi modeli ile Türkiye elektrik şebekesi için saatlik tüketim tahmini ve interaktif Streamlit dashboard'u.

## Genel Bakış

Bu proje, EPİAŞ saatlik tüketim verisi ve Open-Meteo hava durumu verisini birleştirerek **D+1 (ertesi gün)** bazında saatlik elektrik talebi tahmini yapar.

**Model Performansı:**
| Metrik | LightGBM | Baseline (Naive) |
|--------|----------|-----------------|
| MAPE   | **1.198%** | 4.277% |

Walk-Forward Validation (6 Fold) ile doğrulanan bu performans, baseline modele kıyasla ~%72 hata azaltımını temsil eder.

## Proje Yapısı

```
demand-forecast/
├── data/
│   ├── raw/              # Ham indirilen veri (EPİAŞ CSV vb.)
│   └── processed/        # Temizlenmiş / feature engineering sonrası veri
├── notebooks/
│   ├── 01_eda.ipynb                  # Keşifsel veri analizi
│   ├── 02_feature_engineering.ipynb  # Özellik mühendisliği
│   ├── 03_modeling.ipynb             # Model eğitimi ve değerlendirme
│   └── 04_shap_analysis.ipynb        # SHAP özellik önem analizi
├── src/
│   ├── data_loader.py    # EPİAŞ ve Open-Meteo veri yükleme fonksiyonları
│   ├── features.py       # Lag, rolling, CDD/HDD ve takvim özellikleri
│   └── utils.py          # Metrik hesaplama ve yardımcı fonksiyonlar
├── models/               # Eğitilmiş .pkl model dosyaları
├── outputs/
│   ├── figures/          # Grafikler ve SHAP görselleri
│   └── reports/          # Metrik raporları
├── app/
│   └── dashboard.py      # Streamlit dashboard (3 sayfa)
├── requirements.txt
└── README.md
```

## Metodoloji

### Veri Kaynakları
- **Tüketim:** EPİAŞ — saatlik toplam tüketim (MWh)
- **Hava Durumu:** Open-Meteo API — sıcaklık, nem, rüzgar hızı
- **Fiyat:** EPİAŞ spot piyasa fiyatları (TRY, USD, EUR / MWh)

### Özellik Mühendisliği
- **Lag özellikleri:** 24 sa, 48 sa, 168 sa (1 hafta), 336 sa (2 hafta) gecikmeli tüketim
- **Rolling istatistikler:** 24 sa / 168 sa ortalama ve standart sapma, 72 sa maksimum
- **Takvim:** Saat, gün, ay ve haftanın günü için sinüs/kosinüs döngüsel kodlama
- **Tatil bayrağı:** Türkiye resmi tatilleri
- **CDD / HDD:** Soğutma/ısıtma derece-gün değerleri

### Model
- **Algoritma:** LightGBM (Gradient Boosting)
- **Doğrulama:** Walk-Forward Validation — 6 katlı zaman serisi bölümlü çapraz doğrulama
- **Hedef:** Veri sızıntısını önlemek amacıyla üretim verisi (`total_generation_MWh` ve tüm kaynak bazlı üretim kolonları) modelden çıkarılmıştır

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Kullanım

### Notebook'ları sırayla çalıştırma

```bash
jupyter notebook
```

Notebook'ları `01 → 02 → 03 → 04` sırasıyla çalıştırın. Her aşama bir sonrakinin girdi dosyasını üretir.

### Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard üç sayfa içerir:

| Sayfa | İçerik |
|-------|--------|
| **D+1 Tahmin** | Seçilen gün için 24 saatlik saatlik tahmin ve gerçek tüketim karşılaştırması |
| **Gerçek vs Tahmin** | Test seti üzerinde tarih aralığı seçilerek MAPE / MAE / RMSE metrikleri ve hata dağılımı |
| **Model Bilgisi** | SHAP özellik önemi grafiği, LightGBM gain skoru tablosu ve test performansı özeti |

## Kullanılan Teknolojiler

| Amaç | Kütüphane |
|------|-----------|
| Modelleme | LightGBM |
| Veri işleme | pandas, numpy |
| Özellik önemi | SHAP |
| Görselleştirme | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Hava durumu API | openmeteo-requests |
| Tatil takvimi | holidays |
