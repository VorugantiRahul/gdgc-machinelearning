import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

data_dir = "c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

LABEL_MAP = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def engineer_clean_features(df):
    d = df.copy()
    rp, op = d['id.resp_p'], d['id.orig_p']

    # 1. Port Semantics (IANA standard tiers)
    d['is_well_known_resp'] = (rp < 1024).astype(np.int8)
    d['is_registered_resp'] = ((rp >= 1024) & (rp < 49152)).astype(np.int8)
    d['is_dynamic_resp'] = (rp >= 49152).astype(np.int8)
    d['is_same_port'] = (op == rp).astype(np.int8)

    # Standard service port flags
    d['is_ftp'] = (rp == 21).astype(np.int8)
    d['is_ssh'] = (rp == 22).astype(np.int8)
    d['is_dns'] = (rp == 53).astype(np.int8)
    d['is_http'] = (rp == 80).astype(np.int8)
    d['is_https'] = (rp == 443).astype(np.int8)
    d['is_mqtt'] = (rp == 1883).astype(np.int8)
    d['is_mdns'] = (rp == 5353).astype(np.int8)

    # 2. Attack Tool Domain Signatures
    # Hping3 SYN flood
    d['sig_hping_syn_flood'] = (
        (d['fwd_init_window_size'] == 64) &
        (d['fwd_pkts_payload.tot'] == 120) &
        (d['fwd_pkts_tot'] == 1)
    ).astype(np.int8)
    
    # Slowloris HTTP DoS
    d['sig_slowloris_dos'] = (
        (rp == 80) & (d['bwd_pkts_payload.tot'] == 0) & (d['fwd_pkts_tot'] >= 4)
    ).astype(np.int8)

    # Unanswered UDP probe (Nmap UDP scan)
    d['sig_udp_scan'] = ((d['proto'] == 'udp') & (d['bwd_pkts_tot'] == 0)).astype(np.int8)

    # TCP RST scan on closed non-standard ports (Nmap TCP connect / SYN scan)
    d['sig_tcp_rst_scan'] = (
        (d['proto'] == 'tcp') & (d['bwd_pkts_tot'] == 1) & (d['flow_RST_flag_count'] == 1)
        & (~rp.isin([21, 53, 80, 443, 1883]))
    ).astype(np.int8)

    # mDNS spoofing signature
    d['sig_mdns_spoof'] = (
        (rp == 5353) & (d['proto'] == 'udp') & (d['fwd_pkts_payload.tot'] == 29)
    ).astype(np.int8)

    # 3. Traffic Directionality & Packet Mechanics
    d['pkts_tot'] = d['fwd_pkts_tot'] + d['bwd_pkts_tot']
    d['data_pkts_tot'] = d['fwd_data_pkts_tot'] + d['bwd_data_pkts_tot']
    d['pkts_ratio'] = (d['fwd_pkts_tot'] + 1.0) / (d['bwd_pkts_tot'] + 1.0)
    d['data_pkts_ratio'] = (d['fwd_data_pkts_tot'] + 1.0) / (d['bwd_data_pkts_tot'] + 1.0)
    d['bytes_ratio'] = (d['fwd_pkts_payload.tot'] + 1.0) / (d['bwd_pkts_payload.tot'] + 1.0)
    d['bytes_per_pkt'] = d['flow_pkts_payload.tot'] / (d['pkts_tot'] + 1e-5)
    d['fwd_bytes_per_pkt'] = d['fwd_pkts_payload.tot'] / (d['fwd_pkts_tot'] + 1e-5)
    d['bwd_bytes_per_pkt'] = d['bwd_pkts_payload.tot'] / (d['bwd_pkts_tot'] + 1e-5)
    d['total_header_size'] = d['fwd_header_size_tot'] + d['bwd_header_size_tot']
    d['header_to_payload_ratio'] = d['total_header_size'] / (d['flow_pkts_payload.tot'] + 1.0)
    d['bwd_pkts_is_zero'] = (d['bwd_pkts_tot'] == 0).astype(np.int8)
    d['bwd_payload_is_zero'] = (d['bwd_pkts_payload.tot'] == 0).astype(np.int8)
    d['is_zero_duration'] = (d['flow_duration'] == 0.0).astype(np.int8)

    # 4. TCP Flags Dynamics
    d['total_flags'] = (
        d['flow_FIN_flag_count'] + d['flow_SYN_flag_count'] +
        d['flow_RST_flag_count'] + d['fwd_PSH_flag_count'] +
        d['bwd_PSH_flag_count'] + d['flow_ACK_flag_count']
    )
    d['syn_only'] = ((d['flow_SYN_flag_count'] > 0) & (d['flow_ACK_flag_count'] == 0)).astype(np.int8)
    d['fin_only'] = ((d['flow_FIN_flag_count'] > 0) & (d['flow_SYN_flag_count'] == 0)).astype(np.int8)
    d['rst_syn_ratio'] = (d['flow_RST_flag_count'] + 1e-5) / (d['flow_SYN_flag_count'] + 1.0)

    # 5. Window Mechanics
    d['window_ratio'] = (d['fwd_init_window_size'] + 1.0) / (d['bwd_init_window_size'] + 1.0)
    d['fwd_window_delta'] = d['fwd_init_window_size'] - d['fwd_last_window_size']
    d['is_zero_bwd_window'] = (d['bwd_init_window_size'] == 0).astype(np.int8)

    # 6. Robust Timing Transforms (eps floor for duration=0)
    eps = 1e-6
    d['log_flow_duration'] = np.log1p(np.maximum(0, d['flow_duration']))
    d['robust_flow_pkts_per_sec'] = np.log1p(d['pkts_tot'] / np.maximum(d['flow_duration'], eps))
    d['robust_flow_bytes_per_sec'] = np.log1p(d['flow_pkts_payload.tot'] / np.maximum(d['flow_duration'], eps))
    d['log_flow_iat_min'] = np.log1p(np.maximum(0, d['flow_iat.min']))
    d['log_flow_iat_max'] = np.log1p(np.maximum(0, d['flow_iat.max']))
    d['log_flow_iat_avg'] = np.log1p(np.maximum(0, d['flow_iat.avg']))
    d['log_flow_iat_tot'] = np.log1p(np.maximum(0, d['flow_iat.tot']))

    # Categoricals
    for p in ['tcp', 'udp', 'icmp']:
        d[f'proto_{p}'] = (d['proto'] == p).astype(np.int8)
    for s in ['-', 'dns', 'mqtt', 'http', 'ssl', 'ntp', 'dhcp', 'irc', 'radius']:
        d[f'service_{s}'] = (d['service'] == s).astype(np.int8)

    # Drop transient client port to prevent memorization
    d = d.drop(columns=['id.orig_p', 'proto', 'service', 'fwd_URG_flag_count'], errors='ignore')
    return d

print("Engineering clean features...")
train_fe = engineer_clean_features(train)
test_fe = engineer_clean_features(test)

feature_cols = [c for c in train_fe.columns if c not in ('id', 'Target')]
print(f"Total features: {len(feature_cols)}")

X_train = train_fe[feature_cols].copy().astype(np.float32)
X_test = test_fe[feature_cols].copy().astype(np.float32)
y_train = train['Target'].map(LABEL_MAP).values

# Group hash to prevent duplicate leakage
grp = pd.util.hash_pandas_object(train.drop(columns=['id', 'Target']), index=False).values
print(f"Unique flow groups: {len(np.unique(grp)):,} / {len(train):,}")

# Training fold augmentation (simulating unresponsive DoS server)
AUG_ZERO = [
    'bwd_pkts_tot', 'flow_duration', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec',
    'flow_pkts_per_sec', 'down_up_ratio', 'bwd_header_size_tot', 'bwd_header_size_min',
    'bwd_header_size_max', 'flow_RST_flag_count', 'flow_ACK_flag_count',
    'flow_pkts_payload.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot',
    'flow_iat.avg', 'flow_iat.std', 'payload_bytes_per_second', 'bwd_subflow_pkts',
    'active.min', 'active.max', 'active.tot', 'active.avg',
    'bwd_data_pkts_tot', 'bwd_pkts_payload.tot'
]

def make_fold_aug(raw_df, n=4000, seed=42):
    pool = raw_df[(raw_df['id.resp_p'] == 21) & (raw_df['Target'] == 'Availability')]
    if len(pool) == 0:
        return raw_df.iloc[0:0].copy()
    aug = pool.sample(n=min(n, len(pool)), random_state=seed).copy()
    for c in AUG_ZERO:
        if c in aug.columns:
            aug[c] = 0.0
    aug['flow_pkts_payload.min'] = 120.0
    aug['flow_pkts_payload.avg'] = 120.0
    return aug

# 5-Fold Stratified Group Cross-Validation
N_FOLDS = 5
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(train), 4))
oof_cat = np.zeros((len(train), 4))
oof_ens = np.zeros((len(train), 4))

test_probs_lgb = np.zeros((len(test), 4))
test_probs_cat = np.zeros((len(test), 4))

lgb_params = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=63, max_depth=8,
    min_child_samples=20, subsample=0.85, colsample_bytree=0.8,
    reg_lambda=1.0, class_weight='balanced', n_jobs=-1, verbose=-1, random_state=42
)

print(f"\n--- Training Pure ML Models Across {N_FOLDS} Stratified Group Folds ---")
for fold, (ti, vi) in enumerate(sgkf.split(X_train, y_train, groups=grp)):
    t_start = time.time()
    raw_aug = make_fold_aug(train.iloc[ti], n=4000, seed=fold)
    fe_aug = engineer_clean_features(raw_aug)[feature_cols].astype(np.float32)
    
    X_tr = pd.concat([X_train.iloc[ti], fe_aug], ignore_index=True)
    y_tr = np.concatenate([y_train[ti], raw_aug['Target'].map(LABEL_MAP).values])
    X_va, y_va = X_train.iloc[vi], y_train[vi]

    # 1. LightGBM
    clf_lgb = lgb.LGBMClassifier(**lgb_params)
    clf_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    val_p_lgb = clf_lgb.predict_proba(X_va)
    oof_lgb[vi] = val_p_lgb
    test_probs_lgb += clf_lgb.predict_proba(X_test) / N_FOLDS
    
    # 2. CatBoost
    clf_cat = CatBoostClassifier(iterations=600, learning_rate=0.07, depth=6, thread_count=-1, verbose=0, random_seed=42 + fold)
    clf_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
    val_p_cat = clf_cat.predict_proba(X_va)
    oof_cat[vi] = val_p_cat
    test_probs_cat += clf_cat.predict_proba(X_test) / N_FOLDS

    # Blend
    val_p_ens = (val_p_lgb + val_p_cat) / 2.0
    oof_ens[vi] = val_p_ens
    fold_f1 = f1_score(y_va, np.argmax(val_p_ens, axis=1), average='macro')
    print(f"Fold {fold + 1} Ensemble Macro F1: {fold_f1:.5f} ({time.time() - t_start:.1f}s)")

raw_oof = f1_score(y_train, np.argmax(oof_ens, axis=1), average='macro')
print(f"\nOverall OOF Macro F1: {raw_oof:.5f}")

# Threshold multiplier tuning
def tune_weights(probs, y_true):
    w = np.ones(4)
    best_s = f1_score(y_true, np.argmax(probs * w, axis=1), average='macro')
    grid = np.linspace(0.6, 2.0, 15)
    for _ in range(2):
        for k in range(4):
            for g in grid:
                cand = w.copy()
                cand[k] = g
                s = f1_score(y_true, np.argmax(probs * cand, axis=1), average='macro')
                if s > best_s:
                    best_s = s
                    w = cand
    return w, best_s

w_opt, tuned_oof = tune_weights(oof_ens, y_train)
print(f"Tuned OOF Macro F1: {tuned_oof:.5f}")
print("Optimal Multipliers:", w_opt)

# Generate final test predictions
test_ens = (test_probs_lgb + test_probs_cat) / 2.0
test_preds = np.argmax(test_ens * w_opt, axis=1)

# Apply high-precision network domain logic to eliminate scanner / probe edge confusion
# 1. Hping3 SYN flood -> Availability
mask_hping = (test['fwd_init_window_size'] == 64) & (test['fwd_pkts_payload.tot'] == 120) & (test['fwd_pkts_tot'] == 1)
test_preds[mask_hping.values] = 0

# 2. Slowloris HTTP DoS -> Availability
mask_slow = (test['id.resp_p'] == 80) & (test['bwd_pkts_payload.tot'] == 0) & (test['fwd_pkts_tot'] >= 4)
test_preds[mask_slow.values] = 0

# 3. MQTT port 1883 -> Benign
mask_mqtt = (test['id.resp_p'] == 1883)
test_preds[mask_mqtt.values] = 1

# 4. Unanswered UDP scans -> Reconnaissance
mask_udp = (test['proto'] == 'udp') & (test['bwd_pkts_tot'] == 0) & (test['id.resp_p'] != 5353)
test_preds[mask_udp.values] = 2

# 5. TCP RST scan on closed non-service ports -> Reconnaissance
mask_rst = (test['proto'] == 'tcp') & (test['bwd_pkts_tot'] == 1) & (test['flow_RST_flag_count'] == 1) & (~test['id.resp_p'].isin([21, 53, 80, 443, 1883]))
test_preds[mask_rst.values] = 2

# 6. mDNS Spoofing (ARP/mDNS poison on port 5353)
mask_mdns = (test['id.resp_p'] == 5353) & (test['proto'] == 'udp') & (test['fwd_pkts_payload.tot'] == 29)
test_preds[mask_mdns.values] = 3

final_labels = [INV_LABEL_MAP[p] for p in test_preds]
sub = pd.DataFrame({'id': test['id'], 'Target': final_labels})

sub_path = os.path.join(data_dir, "submission_my_independent_model.csv")
sub.to_csv(sub_path, index=False)
print(f"\nSaved independent submission to: {sub_path}")
print("\nClass distribution:")
print(sub['Target'].value_counts())
