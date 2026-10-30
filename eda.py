import pandas as pd
import numpy as np

train_path = "c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/train.csv"
test_path = "c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/test.csv"

print("Loading train and test...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\n--- Target Distribution in Train ---")
target_counts = train['Target'].value_counts()
target_pct = train['Target'].value_counts(normalize=True) * 100
for cls, count in target_counts.items():
    print(f"  {cls:15s}: {count:6d} ({target_pct[cls]:5.2f}%)")

print("\n--- Data Types & Missing Values ---")
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0]
print(f"Columns with missing values in train: {len(missing_train)}")
if len(missing_train) > 0:
    print(missing_train)

missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0]
print(f"Columns with missing values in test: {len(missing_test)}")
if len(missing_test) > 0:
    print(missing_test)

print("\n--- Categorical Columns Check ---")
non_num = train.select_dtypes(include=['object']).columns.tolist()
print(f"Object columns in train: {non_num}")
for col in non_num:
    if col != 'id':
        print(f"\nUnique values in '{col}':")
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique()) if col in test.columns else set()
        print(f"  Train unique ({len(train_vals)}): {list(train_vals)[:15]}")
        if col in test.columns:
            print(f"  Test unique ({len(test_vals)}): {list(test_vals)[:15]}")
            new_in_test = test_vals - train_vals
            print(f"  Unseen in test: {len(new_in_test)} values: {list(new_in_test)[:10]}")

print("\n--- Port Analysis ---")
for col in ['id.orig_p', 'id.resp_p']:
    print(f"\n{col}:")
    print(f"  Train unique: {train[col].nunique()}, Test unique: {test[col].nunique()}")
    unseen_ports = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
    print(f"  Unseen ports in test: {len(unseen_ports)} / {test[col].nunique()} ({len(unseen_ports)/test[col].nunique()*100:.1f}%)")
    top_train = train[col].value_counts().head(5)
    print(f"  Top 5 in train:\n{top_train}")

print("\n--- Target by Service / Protocol ---")
if 'service' in train.columns:
    print(pd.crosstab(train['service'], train['Target'], margins=True))
if 'proto' in train.columns:
    print(pd.crosstab(train['proto'], train['Target'], margins=True))

print("\nDone EDA inspection.")
