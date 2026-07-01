"""
Elektrik Talebi Tahmin Sistemi — Streamlit Dashboard
"""

import os
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"
META_PATH  = BASE_DIR / "models" / "model_meta.pkl"
DATA_PATH  = BASE_DIR / "data"   / "processed" / "featured_data.csv"
SHAP_IMG   = BASE_DIR / "outputs" / "figures" / "shap_summary_bar.png"

# Özellik açıklamaları (SHAP notebook'tan)
FEAT_DESC = {
    "lag_168":             "Geçen haftanın aynı saatindeki tüketim (168 saat)",
    "lag_24":              "Dün aynı saatteki tüketim (24 saat önce)",
    "lag_48":              "İki gün önceki aynı saatin tüketimi",
    "lag_336":             "İki hafta önceki aynı saatin tüketimi",
    "rolling_mean_24":     "Son 24 saatin ortalama tüketimi",
    "rolling_std_24":      "Son 24 saatin tüketim standart sapması",
    "rolling_mean_168":    "Son 168 saatin (1 hafta) ortalama tüketimi",
    "rolling_max_72":      "Son 72 saatin maksimum tüketimi",
    "hour_sin":            "Saat sinüs kodlaması — döngüsel günlük örüntü",
    "hour_cos":            "Saat kosinüs kodlaması — döngüsel günlük örüntü",
    "month_sin":           "Ay sinüs kodlaması — mevsimsel döngü",
    "month_cos":           "Ay kosinüs kodlaması — mevsimsel döngü",
    "day_of_week_sin":     "Haftanın günü sinüs kodlaması",
    "day_of_week_cos":     "Haftanın günü kosinüs kodlaması",
    "hour":                "Saat (0–23)",
    "month":               "Ay (1–12)",
    "day_of_year":         "Yılın günü (1–365)",
    "day_of_week":         "Haftanın günü (0=Pzt, 6=Paz)",
    "year":                "Yıl — uzun dönemli trend",
    "is_weekend":          "Hafta sonu bayrağı (1=hafta sonu)",
    "is_holiday":          "Resmi tatil bayrağı",
    "total_generation_MWh":"Toplam elektrik üretimi (MWh)",
    "natural_gas":         "Doğal gaz kaynaklı üretim",
    "hydro_dam":           "Baraj hidroelektrik üretimi",
    "lignite":             "Linyit kömürü üretimi",
    "hydro_river":         "Nehir akıntısı hidroelektrik üretimi",
    "coal_imported":       "İthal kömür üretimi",
    "wind":                "Rüzgar enerjisi üretimi",
    "solar":               "Güneş enerjisi üretimi",
    "fuel_oil":            "Fuel-oil üretimi",
    "geothermal":          "Jeotermal enerji üretimi",
    "asphaltite_coal":     "Asfaltit kömür üretimi",
    "hard_coal":           "Taş kömürü üretimi",
    "biomass":             "Biyokütle üretimi",
    "naphtha":             "Nafta üretimi",
    "LNG":                 "Sıvılaştırılmış doğal gaz üretimi",
    "international":       "Uluslararası enerji transferi",
    "waste_heat":          "Atık ısı geri kazanımı",
    "TRY/MWh":             "TL cinsinden elektrik spot fiyatı",
    "USD/MWh":             "Dolar cinsinden elektrik spot fiyatı",
    "EUR/MWh":             "Euro cinsinden elektrik spot fiyatı",
}

# ---------------------------------------------------------------------------
# Sayfa yapılandırması
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Elektrik Talebi Tahmin Sistemi",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Elektrik Talebi Tahmin Sistemi")

# ---------------------------------------------------------------------------
# Kaynak yükleme — cache
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Meta verisi yükleniyor…")
def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Veri seti yükleniyor…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, mae, rmse


# Yükleme hataları burada yakalanır
try:
    model = load_model()
    meta  = load_meta()
    df    = load_data()
except FileNotFoundError as exc:
    st.error(
        f"**Dosya bulunamadı:** `{exc.filename}`\n\n"
        "Lütfen model eğitim notebook'larının çalıştırıldığından emin olun."
    )
    st.stop()
except Exception as exc:
    st.error(f"**Yükleme hatası:** {exc}")
    st.stop()

FEATURES = meta["features"]
TARGET   = meta["target"]

# Test seti ayrımı (notebook'larla aynı %80 split)
_split_idx = int(len(df) * 0.80)
test_df    = df.iloc[_split_idx:].copy().reset_index(drop=True)

DATA_MIN: date = df["datetime"].dt.date.min()
DATA_MAX: date = df["datetime"].dt.date.max()
TEST_MIN: date = test_df["datetime"].dt.date.min()
TEST_MAX: date = test_df["datetime"].dt.date.max()

# ---------------------------------------------------------------------------
# Sol kenar çubuğu — navigasyon
# ---------------------------------------------------------------------------
st.sidebar.header("Navigasyon")
page = st.sidebar.radio(
    "Sayfa",
    ["D+1 Tahmin", "Gerçek vs Tahmin", "Model Bilgisi"],
    label_visibility="collapsed",
)
st.sidebar.divider()

# ===========================================================================
# SAYFA 1 — D+1 TAHMİN
# ===========================================================================
if page == "D+1 Tahmin":
    st.header("D+1 Saatlik Tüketim Tahmini")

    # Sidebar girdileri
    st.sidebar.subheader("Tarih Seçimi")
    selected_date: date = st.sidebar.date_input(
        "Tahmin tarihi",
        value=DATA_MAX,
        min_value=DATA_MIN,
        max_value=DATA_MAX,
        help=f"Veri aralığı: {DATA_MIN} – {DATA_MAX}",
    )

    # Seçilen güne ait satırlar
    day_mask = df["datetime"].dt.date == selected_date
    day_df   = df[day_mask].copy().reset_index(drop=True)

    if day_df.empty:
        st.warning(
            f"**{selected_date}** tarihi için veri bulunamadı. "
            "Lütfen farklı bir tarih seçin."
        )
        st.stop()

    # Tahmin
    try:
        X_day  = day_df[FEATURES]
        preds  = model.predict(X_day)
    except Exception as exc:
        st.error(f"Tahmin sırasında hata: {exc}")
        st.stop()

    hours        = day_df["datetime"].dt.hour.values
    actual       = day_df[TARGET].values
    total_pred   = preds.sum()
    total_actual = actual.sum()

    # Metrik kartları
    c1, c2, c3 = st.columns(3)
    c1.metric("Tahmin Toplam (MWh)", f"{total_pred:,.0f}")
    c2.metric("Gerçek Toplam (MWh)", f"{total_actual:,.0f}")
    delta_pct = (total_pred - total_actual) / total_actual * 100
    c3.metric(
        "Sapma",
        f"{abs(delta_pct):.2f}%",
        delta=f"{delta_pct:+.2f}%",
        delta_color="inverse",
    )

    # Çizgi grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=actual,
        mode="lines+markers",
        name="Gerçek",
        line=dict(color="#2c3e50", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=preds,
        mode="lines+markers",
        name="Tahmin",
        line=dict(color="#27ae60", width=2, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
    ))
    fig.update_layout(
        title=f"{selected_date.strftime('%d %B %Y')} — Saatlik Elektrik Tüketimi",
        xaxis_title="Saat",
        yaxis_title="Tüketim (MWh)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        height=460,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detay tablosu
    with st.expander("Saatlik Tahmin Tablosu"):
        tbl = pd.DataFrame({
            "Saat":          [f"{h:02d}:00" for h in hours],
            "Gerçek (MWh)":  [f"{a:,.1f}"  for a in actual],
            "Tahmin (MWh)":  [f"{p:,.1f}"  for p in preds],
            "Fark (MWh)":    [f"{p-a:+,.1f}" for p, a in zip(preds, actual)],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ===========================================================================
# SAYFA 2 — GERÇEK vs TAHMİN
# ===========================================================================
elif page == "Gerçek vs Tahmin":
    st.header("Test Seti — Gerçek Tüketim vs Model Tahmini")

    st.sidebar.subheader("Tarih Aralığı")
    default_end = min(TEST_MIN + timedelta(days=29), TEST_MAX)
    start_date: date = st.sidebar.date_input(
        "Başlangıç",
        value=TEST_MIN,
        min_value=TEST_MIN,
        max_value=TEST_MAX,
        key="start",
    )
    end_date: date = st.sidebar.date_input(
        "Bitiş",
        value=default_end,
        min_value=TEST_MIN,
        max_value=TEST_MAX,
        key="end",
    )

    if start_date > end_date:
        st.sidebar.error("Başlangıç tarihi bitiş tarihinden ileri olamaz.")
        st.stop()

    fmask    = (test_df["datetime"].dt.date >= start_date) & \
               (test_df["datetime"].dt.date <= end_date)
    filtered = test_df[fmask].copy()

    if filtered.empty:
        st.warning("Seçilen tarih aralığında test verisi bulunamadı.")
        st.stop()

    # Tahmin
    try:
        y_true = filtered[TARGET].values
        y_pred = model.predict(filtered[FEATURES])
    except Exception as exc:
        st.error(f"Tahmin sırasında hata: {exc}")
        st.stop()

    mape, mae, rmse = compute_metrics(y_true, y_pred)

    # Metrik kartları
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAPE",  f"{mape:.3f}%")
    m2.metric("MAE",   f"{mae:,.1f} MWh")
    m3.metric("RMSE",  f"{rmse:,.1f} MWh")
    m4.metric("Veri Noktası", f"{len(filtered):,} saat")

    # Plotly grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["datetime"], y=y_true,
        name="Gerçek",
        line=dict(color="#2c3e50", width=1.3),
    ))
    fig.add_trace(go.Scatter(
        x=filtered["datetime"], y=y_pred,
        name="Tahmin",
        line=dict(color="#e74c3c", width=1.3, dash="dot"),
    ))
    fig.update_layout(
        title=f"Gerçek vs Tahmin  |  {start_date} – {end_date}",
        xaxis_title="Tarih",
        yaxis_title="Tüketim (MWh)",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        height=500,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hata dağılımı (histogram)
    errors = y_pred - y_true
    fig2 = go.Figure(go.Histogram(
        x=errors,
        nbinsx=60,
        marker_color="#3498db",
        opacity=0.8,
        name="Hata",
    ))
    fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Sıfır hata")
    fig2.update_layout(
        title="Tahmin Hatası Dağılımı (Tahmin − Gerçek)",
        xaxis_title="Hata (MWh)",
        yaxis_title="Frekans",
        height=300,
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# SAYFA 3 — MODEL BİLGİSİ
# ===========================================================================
elif page == "Model Bilgisi":
    st.header("Model Bilgisi")

    # Model meta bilgileri
    from datetime import datetime as dt_cls
    try:
        mtime      = os.path.getmtime(MODEL_PATH)
        train_date = dt_cls.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M")
    except Exception:
        train_date = "Bilinmiyor"

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Model Türü",       "LightGBM")
    i2.metric("Eğitim Tarihi",    train_date)
    i3.metric("Özellik Sayısı",   str(len(FEATURES)))
    i4.metric("En İyi İterasyon", str(meta.get("best_iteration", "—")))

    st.divider()

    # Test performansı
    st.subheader("Test Seti Performansı")
    tm = meta.get("test_metrics", {})
    nm = meta.get("naive_metrics", {})

    p1, p2, p3 = st.columns(3)
    p1.metric(
        "MAPE",
        f"{tm.get('MAPE', 0):.3f}%",
        delta=f"Baseline: {nm.get('MAPE', 0):.3f}%",
        delta_color="inverse",
    )
    p2.metric(
        "MAE",
        f"{tm.get('MAE', 0):,.1f} MWh",
        delta=f"Baseline: {nm.get('MAE', 0):,.1f} MWh",
        delta_color="inverse",
    )
    p3.metric(
        "RMSE",
        f"{tm.get('RMSE', 0):,.1f} MWh",
        delta=f"Baseline: {nm.get('RMSE', 0):,.1f} MWh",
        delta_color="inverse",
    )

    wfv_mape = meta.get("wfv_mean_mape")
    wfv_std  = meta.get("wfv_std_mape")
    if wfv_mape is not None:
        st.caption(
            f"Walk-Forward Validation (6 Fold) ortalama MAPE: "
            f"**{wfv_mape:.3f}% ± {wfv_std:.3f}%**"
        )

    st.divider()

    # Veri aralığı bilgisi
    st.subheader("Veri Bilgisi")
    d1, d2, d3 = st.columns(3)
    d1.metric("Veri Başlangıcı", str(DATA_MIN))
    d2.metric("Veri Bitişi",     str(DATA_MAX))
    d3.metric("Toplam Saat",     f"{len(df):,}")

    st.divider()

    # SHAP görseli
    st.subheader("SHAP Özellik Önemi")
    if SHAP_IMG.exists():
        st.image(str(SHAP_IMG), caption="SHAP — Ortalama |Değer| (İlk 20 Özellik)")
    else:
        st.warning(
            f"`{SHAP_IMG.relative_to(BASE_DIR)}` bulunamadı. "
            "Lütfen `notebooks/04_shap_analysis.ipynb` dosyasını çalıştırın."
        )

    st.divider()

    # En önemli 10 özellik tablosu (LightGBM gain importance)
    st.subheader("En Önemli 10 Özellik")
    try:
        importances = model.feature_importances_
        feat_df = (
            pd.DataFrame({"Özellik": FEATURES, "Gain": importances})
            .sort_values("Gain", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        feat_df.index  = feat_df.index + 1
        feat_df["Açıklama"] = feat_df["Özellik"].map(
            lambda f: FEAT_DESC.get(f, "—")
        )
        feat_df["Gain"] = feat_df["Gain"].map(lambda v: f"{v:,.0f}")
        st.dataframe(
            feat_df[["Özellik", "Gain", "Açıklama"]],
            use_container_width=True,
        )
        st.caption("Gain: LightGBM'in iç özellik önemi skoru (SHAP ile tutarlı ancak farklı ölçek).")
    except Exception as exc:
        st.error(f"Özellik önemi yüklenirken hata: {exc}")
