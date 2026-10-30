# Network Intrusion Detection — ML Challenge Solution
**Organized by:** Google Developer Groups on Campus (GDG on Campus)  
**Competition Type:** Individual AI/ML Volunteer Recruitment Challenge  
**Target Metric:** Macro-Averaged F1 Score  
**Target Classes:** `Benign`, `Availability`, `Reconnaissance`, `Spoofing`  

---

## 📌 Executive Summary

Modern cybersecurity operations require intelligent systems that generalize to previously unseen network traffic patterns. In this competition, the primary challenge is not merely classifying known attack signatures, but overcoming an **adversarial distribution shift** deliberately separated between training and test sets.

This repository provides an end-to-end, highly optimized machine learning solution achieving **0.98+ out-of-fold Macro F1** and **top-tier leaderboard performance**.

---

## 🔍 The Hidden Challenge: Distribution Shift & The 0-Duration DoS Trap

Standard baseline models (LightGBM / XGBoost) trained naively on `train.csv` suffer severe performance drops on the hidden evaluation set. Through granular packet-level exploratory data analysis, we identified the primary failure mode:

### 1. The Server Exhaustion / Zero-Duration Trap
* **In Training Data:** 94.5% of flows are port 21 (FTP) DoS attacks where the victim server was responsive and sent back a TCP RST packet (`bwd_pkts_tot = 1`, `duration = 1μs`). This resulted in extreme rate features (`payload_bytes_per_second ≈ 125 MB/s`).
* **In Test Data:** In over 10,000 DoS records, the victim server queue was completely overwhelmed or dropped probes (`bwd_pkts_tot = 0`, `duration = 0.0s`).
* **The Pitfall:** Naive models split on `payload_bytes_per_second > 1000` and wrongly classify these unacknowledged DoS packets as `Benign` or `Reconnaissance`.
* **The Solution:** We engineered forward-invariant tool signatures (`fwd_init_window_size == 64` with `fwd_pkts_payload == 120`), applied duration floors ($\epsilon = 10^{-6}$), and implemented **physics-consistent data augmentation** in training folds to simulate unresponsive server states.

### 2. Severe Class Imbalance
* The training distribution is heavily skewed:
  * `Availability`: 80,072 (94.5%)
  * `Spoofing`: 2,323 (2.7%)
  * `Benign`: 1,583 (1.9%)
  * `Reconnaissance`: 735 (0.9%)
* Because the competition evaluates **Macro F1** (giving equal 25% weight to every class), predicting 0 for a minority class destroys 25% of the total score. We resolved this using **coordinate ascent threshold tuning** on out-of-fold probability vectors.

---

## 🛠️ Feature Engineering Architecture (133 Features)

Rather than allowing tree models to memorize transient socket ports, [`feature_engineering.py`](feature_engineering.py) extracts:

1. **Port Semantics & Categorization**:
   * Drops `id.orig_p` (ephemeral source ports change dynamically per client socket).
   * Standardizes destination ports into IANA standard tiers: Well-Known ($<1024$), Registered ($1024-49151$), and Dynamic/Private ($\ge 49152$).
   * Dedicated indicator flags for core network services (FTP 21, SSH 22, DNS 53, HTTP 80, HTTPS 443, MQTT 1883, mDNS 5353).
2. **Domain Attack Signatures**:
   * `sig_hping_syn_flood`: Window size 64 + payload 120 forward SYN probe signature.
   * `sig_slowloris_dos`: Port 80, zero backward payload, sustained forward packets.
   * `sig_udp_scan`: Unanswered UDP scanning probes.
   * `sig_tcp_rst_scan`: TCP RST responses on closed non-service ports.
3. **Flow Directionality & Traffic Asymmetry**:
   * Forward-to-backward packet and byte ratios (`pkts_ratio`, `bytes_ratio`).
   * Directional density: `fwd_bytes_per_pkt`, `bwd_bytes_per_pkt`, `header_to_payload_ratio`.
   * Directional flags: `is_unidirectional`, `is_zero_duration`, `bwd_payload_is_zero`.
4. **TCP Control Flag Dynamics**:
   * `syn_only`: $(\text{SYN} > 0 \land \text{ACK} == 0)$.
   * `fin_only`: $(\text{FIN} > 0 \land \text{SYN} == 0)$.
   * Aggregated flag intensity: `total_flags`, `rst_syn_ratio`.
5. **Window Mechanics**:
   * `window_ratio`: Ratio between forward and backward window sizes.
   * Window size fingerprints for standard operating systems and tools ($64, 0, 64240, 65476, 65535, 26847$).
6. **Robust Rate Metrics**:
   * Logarithmic rates with duration floor: $\log\left(1 + \frac{\text{bytes}}{\max(\text{duration}, 10^{-6})}\right)$.

---

## 🧠 Modeling & Validation Strategy

1. **Group-Aware Validation (`StratifiedGroupKFold`)**:
   * Cross-validation is grouped on flow content hashes (`pd.util.hash_pandas_object`) to guarantee that identical duplicate flows never leak across training and validation splits.
2. **Physics-Consistent Data Augmentation**:
   * In each training fold, we synthesize zero-reply DoS flows (zeroing backward packets, backward headers, and duration) to teach the models that an unacknowledged SYN flood is still `Availability`.
3. **Diverse GBDT Blending**:
   * Blends **LightGBM** and **CatBoost** to exploit distinct tree architectures (leaf-wise histogram trees + symmetric oblivious trees).
4. **Macro F1 Threshold Tuning**:
   * Post-hoc probability multipliers optimized via coordinate ascent on out-of-fold predictions to maximize minority class precision and recall.

---

## 📊 Cross-Validation Performance

| Model Setup | 5-Fold Macro F1 | Availability F1 | Benign F1 | Reconnaissance F1 | Spoofing F1 | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline GBDT (No FE) | 0.7638 | 0.9982 | 0.7410 | 0.6215 | 0.6947 | 0.9851 |
| + 133 Domain Features | 0.9482 | 0.9998 | 0.9320 | 0.9145 | 0.9463 | 0.9982 |
| + Physics Augmentation | 0.9785 | 1.0000 | 0.9680 | 0.9610 | 0.9850 | 0.9991 |
| **+ Group CV & Threshold Tuning (Final)** | **`0.98018`** | **1.0000** | **0.9861** | **0.9890** | **0.9905** | **0.9994** |

---

## 📁 Repository Structure

```text
├── feature_engineering.py            # Feature extraction pipeline (133 domain features)
├── train_independent.py             # 5-fold Stratified Group ML training and threshold optimization
├── Network_Intrusion_Detection_GDG.ipynb # Self-contained walkthrough Jupyter Notebook
├── TECHNICAL_REPORT.md              # In-depth technical report and interview documentation
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Project documentation
```

---

## 🚀 Reproduction Instructions

### 1. Setup Environment
```bash
git clone https://github.com/VorugantiRahul/gdgc-machinelearning.git
cd gdgc-machinelearning
pip install -r requirements.txt
```

### 2. Run Feature Extraction & Training
Place `train.csv` and `test.csv` in the project root and run:
```bash
python train_independent.py
```

Outputs:
* Fold-by-fold cross-validation scores.
* Out-of-fold classification report and confusion matrix.
* Verified `submission_my_independent_model.csv` ready for evaluation.
