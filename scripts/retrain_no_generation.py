# -*- coding: utf-8 -*-
"""
Uretim feature'lari (total_generation_MWh + tum kaynaklar) cikarilmis model.
Sadece lag, rolling, takvim ve fiyat feature'lari kullanilir.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# ── Veri ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/featured_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

TARGET = 'consumption_MWh'

# Çıkarılacak üretim feature'ları (data leakage)
GENERATION_COLS = [
    'total_generation_MWh',
    'natural_gas', 'hydro_dam', 'lignite', 'hydro_river',
    'coal_imported', 'wind', 'solar', 'fuel_oil', 'geothermal',
    'asphaltite_coal', 'hard_coal', 'biomass', 'naphtha',
    'LNG', 'international', 'waste_heat',
]

EXCLUDE = ['datetime', TARGET] + GENERATION_COLS
FEATURES = [c for c in df.columns if c not in EXCLUDE]

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

print(f'Toplam veri       : {df.shape[0]:,} satır')
print(f'Feature sayısı    : {len(FEATURES)}  (üretim sütunları çıkarıldı)')
print(f'Feature listesi   : {FEATURES}')

# ── Train / Test Split (%80 / %20 kronolojik) ─────────────────────────────────
split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

X_train = train_df[FEATURES]; y_train = train_df[TARGET]
X_test  = test_df[FEATURES];  y_test  = test_df[TARGET]

print(f'\nTrain : {train_df["datetime"].min().date()} - {train_df["datetime"].max().date()}  ({len(train_df):,})')
print(f'Test  : {test_df["datetime"].min().date()} - {test_df["datetime"].max().date()}  ({len(test_df):,})')

# ── Metrik fonksiyonu ─────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label=''):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if label:
        print(f'\n[{label}]')
    print(f'  MAPE : {mape:.3f}%')
    print(f'  MAE  : {mae:,.1f} MWh')
    print(f'  RMSE : {rmse:,.1f} MWh')
    return {'MAPE': mape, 'MAE': mae, 'RMSE': rmse}

# ── Naive Baseline ────────────────────────────────────────────────────────────
naive_pred = test_df['lag_168'].values
print('\n' + '='*50)
print('NAİVE BASELINE (lag_168) — TEST SETİ')
print('='*50)
metrics_naive = compute_metrics(y_test.values, naive_pred, label='Naive Baseline')

# ── LightGBM ──────────────────────────────────────────────────────────────────
val_split = int(len(X_train) * 0.90)
X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

lgbm_params = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

model = lgb.LGBMRegressor(**lgbm_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]
)
print(f'\nEn iyi iterasyon : {model.best_iteration_}')

lgbm_pred = model.predict(X_test)
print('\n' + '='*50)
print('LightGBM (üretimsiz) — TEST SETİ')
print('='*50)
metrics_lgbm = compute_metrics(y_test.values, lgbm_pred, label='LightGBM (no-gen)')

# ── Walk-Forward Validation ───────────────────────────────────────────────────
N_FOLDS = 6
TEST_WINDOW_HOURS = 4 * 30 * 24
X_all = df[FEATURES]; y_all = df[TARGET]; dates_all = df['datetime']
total = len(df); min_train = 8760
available = total - min_train; fold_step = available // N_FOLDS

wfv_results = []
print(f'\nWalk-Forward Validation — {N_FOLDS} Fold')
print('='*70)

for fold in range(N_FOLDS):
    train_end  = min_train + fold * fold_step
    test_start = train_end
    test_end   = min(test_start + TEST_WINDOW_HOURS, total)
    if test_start >= total:
        break

    X_tr_f = X_all.iloc[:train_end]; y_tr_f = y_all.iloc[:train_end]
    X_te_f = X_all.iloc[test_start:test_end]; y_te_f = y_all.iloc[test_start:test_end]

    v_split = int(len(X_tr_f) * 0.90)
    m = lgb.LGBMRegressor(**lgbm_params)
    m.fit(
        X_tr_f.iloc[:v_split], y_tr_f.iloc[:v_split],
        eval_set=[(X_tr_f.iloc[v_split:], y_tr_f.iloc[v_split:])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    preds = m.predict(X_te_f)
    y_te_arr = y_te_f.values
    mask = y_te_arr != 0
    mape = np.mean(np.abs((y_te_arr[mask] - preds[mask]) / y_te_arr[mask])) * 100
    mae  = mean_absolute_error(y_te_arr, preds)
    rmse = np.sqrt(mean_squared_error(y_te_arr, preds))

    wfv_results.append({'Fold': fold+1, 'MAPE': mape, 'MAE': mae, 'RMSE': rmse})
    print(f'Fold {fold+1} | Egitim: - {dates_all.iloc[train_end-1].date()} '
          f'| Test: {dates_all.iloc[test_start].date()} - {dates_all.iloc[test_end-1].date()} '
          f'| MAPE: {mape:.3f}% | MAE: {mae:,.0f} | RMSE: {rmse:,.0f}')

wfv_df = pd.DataFrame(wfv_results)
print('='*70)
print(f'Ortalama MAPE : {wfv_df["MAPE"].mean():.3f}% ± {wfv_df["MAPE"].std():.3f}%')

# ── Karşılaştırma Özeti ───────────────────────────────────────────────────────
print('\n' + '='*60)
print('          KARŞILAŞTIRMA ÖZETİ')
print('='*60)
print(f"{'Model':<30} {'MAPE':>8} {'MAE':>10} {'RMSE':>10}")
print('-'*60)
print(f"{'Naive Baseline':<30} {metrics_naive['MAPE']:>7.3f}% {metrics_naive['MAE']:>9,.0f} {metrics_naive['RMSE']:>9,.0f}")
print(f"{'LightGBM (ESKİ — üretimli)':<30} {'1.198':>7}% {'454':>9} {'741':>9}  ← data leakage!")
print(f"{'LightGBM (YENİ — üretimsiz)':<30} {metrics_lgbm['MAPE']:>7.3f}% {metrics_lgbm['MAE']:>9,.0f} {metrics_lgbm['RMSE']:>9,.0f}")
print('-'*60)
print(f"{'WFV Ortalama (yeni)':<30} {wfv_df['MAPE'].mean():>7.3f}% ± {wfv_df['MAPE'].std():.3f}%")
print('='*60)

# Model kaydet
with open('models/lgbm_model_no_gen.pkl', 'wb') as f:
    pickle.dump(model, f)

meta = {
    'features': FEATURES,
    'target': TARGET,
    'best_iteration': model.best_iteration_,
    'test_metrics': metrics_lgbm,
    'naive_metrics': metrics_naive,
    'wfv_mean_mape': wfv_df['MAPE'].mean(),
    'wfv_std_mape': wfv_df['MAPE'].std(),
    'generation_cols_removed': GENERATION_COLS,
}
with open('models/model_meta_no_gen.pkl', 'wb') as f:
    pickle.dump(meta, f)

print('\nModel kaydedildi: models/lgbm_model_no_gen.pkl')
