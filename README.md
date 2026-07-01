# Demand Forecast — Kısa Vadeli Elektrik Tüketim Tahmini

KEPSAŞ staj projesi. LightGBM tabanlı kısa vadeli elektrik tüketim tahmin modeli
ve Streamlit dashboard'u içerir.

## Proje Yapısı

```
demand-forecast/
├── data/
│   ├── raw/              # Ham indirilen veri (EPİAŞ CSV vb.)
│   └── processed/        # Temizlenmiş / feature engineering sonrası veri
├── notebooks/
│   ├── 01_eda.ipynb              # Keşifsel veri analizi
│   ├── 02_feature_engineering.ipynb  # Özellik mühendisliği
│   ├── 03_modeling.ipynb         # Model eğitimi ve değerlendirme
│   └── 04_shap_analysis.ipynb    # SHAP özellik önem analizi
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Veri çekme/yükleme fonksiyonları
│   ├── features.py       # Lag, rolling, CDD/HDD feature engineering
│   └── utils.py          # Metrik hesaplama ve yardımcı fonksiyonlar
├── models/               # Eğitilmiş .pkl model dosyaları
├── outputs/
│   ├── figures/          # Grafikler, SHAP görselleri
│   └── reports/          # Metrik raporları
├── app/
│   └── dashboard.py      # Streamlit dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Kullanım

### Notebook'ları çalıştırma

```bash
jupyter notebook
```

### Streamlit dashboard'u başlatma

```bash
streamlit run app/dashboard.py
```

## Veri Kaynakları

- **Tüketim verisi:** EPİAŞ (Enerji Piyasaları İşletme A.Ş.) — saatlik tüketim CSV
- **Hava durumu:** Open-Meteo API — sıcaklık, nem, rüzgar vb.

## Kullanılan Teknolojiler

| Amaç | Kütüphane |
|---|---|
| Modelleme | LightGBM |
| Veri işleme | pandas, numpy |
| Özellik önemi | SHAP |
| Görselleştirme | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Hava durumu API | openmeteo-requests |
| Tatil takvimi | holidays |
