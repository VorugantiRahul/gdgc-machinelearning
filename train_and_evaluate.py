import os
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import extract_features

print("--- 1. Loading Datasets ---")
data_dir = "c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print(f"Train raw shape: {train_df.shape}")
print(f"Test raw shape: {test_df.shape}")

label_map = {'Benign': 0, 'Availability': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_label_map = {v: k for k, v in label_map.items()}
class_names = ['Benign', 'Availability', 'Reconnaissance', 'Spoofing']

y_train = train_df['Target'].map(label_map).values
test_ids = test_df['id'].values

print("\n--- 2. Feature Extraction ---")
t0 = time.time()
X_train_full = extract_features(train_df)
X_test = extract_features(test_df)
print(f"Features extracted in {time.time() - t0:.2f}s")

feature_cols = [c for c in X_train_full.columns if c not in ['id', 'Target']]
print(f"Total features: {len(feature_cols)}")

X_train = X_train_full[feature_cols].copy()
X_test = X_test[feature_cols].copy()

# Function to augment an unresponsive DoS variant in the training fold
def augment_unresponsive_dos(X_tr, y_tr, n_samples=8000, random_state=42):
    dos_indices = np.where((y_tr == 1) & (X_tr['id.resp_p'] == 21.0) & (X_tr['sig_ftp_dos'] == 1.0))[0]
    if len(dos_indices) == 0:
        return X_tr, y_tr
    
    n_sample = min(n_samples, len(dos_indices))
    rng = np.random.RandomState(random_state)
    chosen_idx = rng.choice(dos_indices, size=n_sample, replace=False)
    
    aug_X = X_tr.iloc[chosen_idx].copy()
    aug_y = y_tr[chosen_idx].copy()
    
    # Zero out backward fields and duration
    zero_cols = [
        'bwd_pkts_tot', 'bwd_data_pkts_tot', 'bwd_header_size_tot',
        'bwd_header_size_min', 'bwd_header_size_max', 'flow_duration',
        'flow_RST_flag_count', 'flow_ACK_flag_count', 'bwd_pkts_per_sec',
        'flow_pkts_per_sec', 'payload_bytes_per_second', 'bwd_pkts_payload.tot',
        'bwd_pkts_payload.avg', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max',
        'bwd_pkts_payload.std', 'flow_iat.tot', 'flow_iat.avg', 'flow_iat.min',
        'flow_iat.max', 'active.tot', 'active.avg', 'active.min', 'active.max',
        'bwd_init_window_size', 'bwd_to_fwd_pkts', 'bwd_to_fwd_bytes',
        'has_bwd_payload', 'bwd_payload_per_pkt', 'bwd_header_to_payload',
        'robust_bwd_pkts_per_sec', 'log_duration'
    ]
    for c in zero_cols:
        if c in aug_X.columns:
            aug_X[c] = 0.0
            
    aug_X['is_unidirectional'] = 1.0
    aug_X['is_zero_duration'] = 1.0
    aug_X['tot_pkts'] = 1.0
    aug_X['tot_bytes'] = 120.0
    aug_X['tot_pkt_mean_size'] = 120.0
    aug_X['robust_flow_pkts_per_sec'] = np.log1p(1.0 / 1e-6)
    aug_X['robust_flow_bytes_per_sec'] = np.log1p(120.0 / 1e-6)
    aug_X['robust_fwd_pkts_per_sec'] = np.log1p(1.0 / 1e-6)
    
    X_aug = pd.concat([X_tr, aug_X], ignore_index=True)
    y_aug = np.concatenate([y_tr, aug_y])
    return X_aug, y_aug

print("\n--- 3. Running 5-Fold Stratified Ensemble Cross-Validation ---")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(train_df), 4))
oof_xgb = np.zeros((len(train_df), 4))
oof_cat = np.zeros((len(train_df), 4))
oof_ens = np.zeros((len(train_df), 4))

test_probs_lgb = np.zeros((len(test_df), 4))
test_probs_xgb = np.zeros((len(test_df), 4))
test_probs_cat = np.zeros((len(test_df), 4))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    f_start = time.time()
    print(f"\n=================== FOLD {fold + 1} / {N_SPLITS} ===================")
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # Augment training split
    X_tr_aug, y_tr_aug = augment_unresponsive_dos(X_tr, y_tr, n_samples=8000, random_state=42 + fold)
    print(f"Training set size: {len(X_tr_aug)} (augmented from {len(X_tr)}), Val size: {len(X_va)}")
    
    # --- Model 1: LightGBM ---
    print("Training LightGBM...")
    model_lgb = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=350,
        learning_rate=0.06,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_tr_aug, y_tr_aug)
    val_preds_lgb = model_lgb.predict_proba(X_va)
    oof_lgb[val_idx] = val_preds_lgb
    test_probs_lgb += model_lgb.predict_proba(X_test) / N_SPLITS
    f1_lgb = f1_score(y_va, np.argmax(val_preds_lgb, axis=1), average='macro')
    print(f"  Fold {fold + 1} LightGBM Macro F1: {f1_lgb:.5f}")
    
    # --- Model 2: XGBoost ---
    print("Training XGBoost...")
    model_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_estimators=350,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42 + fold,
        n_jobs=-1
    )
    model_xgb.fit(X_tr_aug, y_tr_aug)
    val_preds_xgb = model_xgb.predict_proba(X_va)
    oof_xgb[val_idx] = val_preds_xgb
    test_probs_xgb += model_xgb.predict_proba(X_test) / N_SPLITS
    f1_xgb = f1_score(y_va, np.argmax(val_preds_xgb, axis=1), average='macro')
    print(f"  Fold {fold + 1} XGBoost Macro F1: {f1_xgb:.5f}")
    
    # --- Model 3: CatBoost ---
    print("Training CatBoost...")
    model_cat = CatBoostClassifier(
        iterations=400,
        learning_rate=0.07,
        depth=6,
        random_seed=42 + fold,
        thread_count=-1,
        verbose=0
    )
    model_cat.fit(X_tr_aug, y_tr_aug)
    val_preds_cat = model_cat.predict_proba(X_va)
    oof_cat[val_idx] = val_preds_cat
    test_probs_cat += model_cat.predict_proba(X_test) / N_SPLITS
    f1_cat = f1_score(y_va, np.argmax(val_preds_cat, axis=1), average='macro')
    print(f"  Fold {fold + 1} CatBoost Macro F1: {f1_cat:.5f}")
    
    # Fold Ensemble
    val_preds_ens = (val_preds_lgb + val_preds_xgb + val_preds_cat) / 3.0
    oof_ens[val_idx] = val_preds_ens
    f1_ens = f1_score(y_va, np.argmax(val_preds_ens, axis=1), average='macro')
    print(f"  Fold {fold + 1} Ensemble Macro F1: {f1_ens:.5f} (Fold finished in {time.time() - f_start:.1f}s)")

# Overall OOF Performance
print("\n=================== OVERALL OUT-OF-FOLD EVALUATION ===================")
score_lgb = f1_score(y_train, np.argmax(oof_lgb, axis=1), average='macro')
score_xgb = f1_score(y_train, np.argmax(oof_xgb, axis=1), average='macro')
score_cat = f1_score(y_train, np.argmax(oof_cat, axis=1), average='macro')
score_ens = f1_score(y_train, np.argmax(oof_ens, axis=1), average='macro')

print(f"LightGBM OOF Macro F1 : {score_lgb:.5f}")
print(f"XGBoost  OOF Macro F1 : {score_xgb:.5f}")
print(f"CatBoost OOF Macro F1 : {score_cat:.5f}")
print(f"Ensemble OOF Macro F1 : {score_ens:.5f}")

print("\nEnsemble Classification Report (Pre-threshold tuning):")
print(classification_report(y_train, np.argmax(oof_ens, axis=1), target_names=class_names, digits=5))

# --- 4. Macro F1 Threshold Multiplier Optimization ---
print("\n--- 4. Macro F1 Threshold Optimization ---")
def loss_func(log_w):
    w = np.exp(log_w)
    adj_probs = oof_ens * w
    preds = np.argmax(adj_probs, axis=1)
    return -f1_score(y_train, preds, average='macro')

res = minimize(loss_func, np.zeros(4), method='Nelder-Mead', options={'maxiter': 500})
best_weights = np.exp(res.x)
best_weights = best_weights / np.sum(best_weights)

tuned_oof_preds = np.argmax(oof_ens * best_weights, axis=1)
score_tuned = f1_score(y_train, tuned_oof_preds, average='macro')
print(f"Optimized Ensemble Macro F1: {score_tuned:.5f} (Gain: +{score_tuned - score_ens:.5f})")
print("Optimal Class Multipliers:", {class_names[i]: round(best_weights[i], 4) for i in range(4)})

print("\nEnsemble Classification Report (Post-threshold tuning):")
print(classification_report(y_train, tuned_oof_preds, target_names=class_names, digits=5))

# Confusion Matrix
cm = confusion_matrix(y_train, tuned_oof_preds)
print("\nConfusion Matrix:")
print("Actual \\ Predicted ->", class_names)
for i, name in enumerate(class_names):
    print(f"{name:15s}: {cm[i]}")

# --- 5. Generate Final Test Predictions ---
print("\n--- 5. Generating Test Predictions ---")
test_ensemble_probs = (test_probs_lgb + test_probs_xgb + test_probs_cat) / 3.0
final_test_preds = np.argmax(test_ensemble_probs * best_weights, axis=1)
final_labels = [inv_label_map[p] for p in final_test_preds]

sub = pd.DataFrame({'id': test_ids, 'Target': final_labels})

sub_path = os.path.join(data_dir, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"Submission saved successfully to: {sub_path}")

print("\nTest Predictions Class Distribution:")
pred_counts = sub['Target'].value_counts()
for name in class_names:
    cnt = pred_counts.get(name, 0)
    print(f"  {name:15s}: {cnt:6d} ({cnt / len(sub) * 100:5.2f}%)")

# Verify Submission formatting matches sample_submission.csv
print("\n--- 6. Verifying Submission Integrity ---")
assert len(sub) == len(sample_sub), f"Length mismatch: {len(sub)} vs {len(sample_sub)}"
assert list(sub.columns) == ['id', 'Target'], f"Column mismatch: {list(sub.columns)}"
assert sub['id'].equals(sample_sub['id']), "ID mismatch with sample submission!"
assert set(sub['Target'].unique()).issubset(set(class_names)), "Invalid class labels found!"
print("Verification Passed: 100% compliant with Kaggle submission rules!")

# Save evaluation report metadata
summary_data = {
    "num_train": len(train_df),
    "num_test": len(test_df),
    "num_features": len(feature_cols),
    "models": ["LightGBM", "XGBoost", "CatBoost"],
    "oof_macro_f1": {
        "LightGBM": round(score_lgb, 5),
        "XGBoost": round(score_xgb, 5),
        "CatBoost": round(score_cat, 5),
        "Ensemble_Raw": round(score_ens, 5),
        "Ensemble_Tuned": round(score_tuned, 5)
    },
    "optimal_weights": {class_names[i]: round(best_weights[i], 4) for i in range(4)},
    "test_predictions_distribution": {name: int(pred_counts.get(name, 0)) for name in class_names}
}

with open(os.path.join(data_dir, "models_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=2)
print("Saved models_summary.json")

print("\nAll training, evaluation, ensembling, and submission generation completed successfully!")
