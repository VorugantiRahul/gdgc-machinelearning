import pandas as pd

train = pd.read_csv("c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/train.csv")
test = pd.read_csv("c:/Users/RAHUL/Documents/network-intrusion-detection2-ml-challenge/test.csv")

print("Train resp_p by Target (excluding 21.0):")
non_21_train = train[train['id.resp_p'] != 21.0]
print(f"Non-21 rows in train: {len(non_21_train)}")
print(non_21_train['Target'].value_counts())

print("\nAvailability rows with resp_p != 21.0:")
avail_non_21 = train[(train['Target'] == 'Availability') & (train['id.resp_p'] != 21.0)]
print(avail_non_21['id.resp_p'].value_counts().head(10))

print("\nBenign resp_p:")
print(train[train['Target'] == 'Benign']['id.resp_p'].value_counts().head(10))

print("\nReconnaissance resp_p:")
print(train[train['Target'] == 'Reconnaissance']['id.resp_p'].value_counts().head(10))

print("\nSpoofing resp_p:")
print(train[train['Target'] == 'Spoofing']['id.resp_p'].value_counts().head(10))
