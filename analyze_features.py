import pandas as pd
import numpy as np

train_path = "c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/train.csv"
test_path = "c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

classes = ['Benign', 'Availability', 'Reconnaissance', 'Spoofing']

print("--- Constant and Near-Constant Columns ---")
train_features = [c for c in train.columns if c not in ['id', 'Target']]
constant_cols = [c for c in train_features if train[c].nunique() <= 1]
print(f"Constant columns in train: {constant_cols}")

print("\n--- Test vs Train Distribution on Key Summary Metrics ---")
for col in ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_pkts_per_sec', 
            'flow_SYN_flag_count', 'flow_ACK_flag_count', 'fwd_pkts_payload.avg']:
    print(f"\nFeature: {col}")
    for cls in classes:
        sub = train[train['Target'] == cls][col]
        print(f"  {cls:15s}: median={sub.median():.4f}, mean={sub.mean():.4f}, min={sub.min():.4f}, max={sub.max():.4f}")
    sub_test = test[col]
    print(f"  {'TEST SET':15s}: median={sub_test.median():.4f}, mean={sub_test.mean():.4f}, min={sub_test.min():.4f}, max={sub_test.max():.4f}")

print("\n--- Flag Counts across Classes ---")
flag_cols = [c for c in train.columns if 'flag' in c.lower()]
print(f"Flag columns: {flag_cols}")
for f in flag_cols:
    means = train.groupby('Target')[f].mean()
    print(f"{f:25s}: " + ", ".join([f"{cls}={means[cls]:.3f}" for cls in classes]))

print("\n--- Window Sizes across Classes ---")
win_cols = [c for c in train.columns if 'window' in c.lower()]
for w in win_cols:
    print(f"{w:25s}:")
    for cls in classes:
        top_vals = train[train['Target'] == cls][w].value_counts().head(3).to_dict()
        print(f"  {cls:15s}: {top_vals}")

print("\n--- Unseen Ports in Test vs Train ---")
test_resp_p = test['id.resp_p'].value_counts()
print(f"Top 10 resp_p in test:\n{test_resp_p.head(10)}")

print("\n--- Check how many test rows have resp_p == 21 (which was 94% of train!) ---")
print(f"Test rows with resp_p == 21: {(test['id.resp_p'] == 21).sum()} / {len(test)} ({(test['id.resp_p'] == 21).mean()*100:.2f}%)")
