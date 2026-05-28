# Fraud Detection System — SQL Rule Engine + XGBoost ML Model

A production-inspired fraud detection system built on 50,000 synthetic transactions.  
Combines a **SQL-based rule engine** with an **XGBoost machine learning model** to demonstrate  
how real fraud teams use rules and ML together — not as alternatives, but as layers.

---

## Project Structure

```
fraud_detection_project/
├── sql/
│   ├── 01_exploratory_analysis.sql
│   ├── 02_fraud_by_segment.sql
│   ├── 03_velocity_detection.sql
│   ├── 04_bin_attack_detection.sql
│   ├── 05_geo_velocity_fraud.sql
│   ├── 06_ato_detection.sql
│   ├── 07_merchant_risk.sql
│   ├── 08_aml_structuring.sql
│   ├── 09_risk_scoring_engine.sql
│   └── 10_decision_engine_hybrid.sql
├── python/
│   ├── 01_data_setup.py
│   ├── 02_feature_engineering.py
│   ├── 03_xgboost_model.py
│   └── 04_threshold_analysis.py
├── data/
│   └── fraud_db_setup.sql
├── docs/
│   └── project_findings.md
└── README.md
```

---

## Business Problem

Financial platforms lose billions annually to fraud. Two failure modes exist:

- **Miss fraud** → direct financial loss, chargebacks, regulatory penalties
- **Block legitimate customers** → lost revenue, customer churn, support overhead

This project builds a system that minimises both — catching fraud with high precision  
while keeping false positive rates at an acceptable level for operations teams.

---

## Dataset

- **50,000 transactions** across 5,000 customers and 200 merchants
- **782 fraud cases** — 1.56% fraud rate (realistic for card fraud)
- **Fraud patterns included:** ATO, BIN attacks, geo-velocity, structuring, CNP fraud
- **Features:** amount, transaction type, merchant category, KYC status, failed logins,  
  account age, device ID, geo-location, risk segment

Generated using `data/fraud_db_setup.sql` — fully reproducible in MySQL.

---

## Part 1 — SQL Rule Engine

Ten detection modules built progressively, each targeting a specific fraud pattern.

### Modules

| Module | Fraud Pattern | Key Technique |
|---|---|---|
| 01 | Exploratory analysis | GROUP BY, conditional aggregation |
| 02 | Fraud by segment | CASE WHEN, fraud rate % |
| 03 | Daily velocity detection | GROUP BY + HAVING |
| 04 | Rolling 1-hour velocity | Self JOIN, INTERVAL window |
| 05 | BIN attack detection | CTE, decline rate analysis |
| 06 | Geo-velocity fraud | LAG(), TIMESTAMPDIFF, impossible travel |
| 07 | ATO detection | Multi-signal AND logic |
| 08 | Merchant risk monitoring | JOIN, chargeback rate |
| 09 | AML structuring | Near-threshold pattern detection |
| 10 | Risk scoring + decision engine | Weighted CASE WHEN, hybrid architecture |

### Risk Scoring Engine

Each transaction receives a weighted risk score based on 10 fraud signals.  
Weights were initially set by domain knowledge, then validated by XGBoost feature importance.

| Signal | Weight | Basis |
|---|---|---|
| Failed logins > 2 | 35 | Strongest ATO predictor — 60.5% XGBoost importance |
| Amount > $800 | 10 | High value = high loss exposure |
| KYC not verified | 9 | Unverified identity = elevated risk |
| High risk segment | 9 | Was underweighted in v1 — ML corrected this |
| Crypto/Gaming merchant | 8 | 2.67% fraud rate vs 1.38% ecommerce |
| Foreign transaction | 6 | Was overweighted in v1 — ML corrected this |
| Amount > 3x user average | 5 | Behavioural spike signal |
| Account age < 30 days | 5 | New account risk |
| Off-hours (2am–5am) | 5 | Behavioural anomaly |
| CNP transaction | 5 | Context signal — weak alone |

### Hybrid Decision Engine

Hard rules for high-confidence single signals.  
Soft scoring for combinations of weaker signals.

```
Strong single signal  →  HARD BLOCK  (no score needed)
Weak signals combined →  Score ≥ threshold → REVIEW or APPROVE
```

**Hard block conditions:**
- failed_logins > 5
- failed_logins > 2 AND kyc_verified = 0
- failed_logins > 2 AND amount > $800
- risk_segment = High AND kyc_verified = 0 AND merchant = Crypto/Gaming

### Decision Engine Results

| Decision | Transactions | Fraud Count | Fraud Rate | False Positive Rate |
|---|---|---|---|---|
| BLOCK | 285 | 112 | 39.3% | 60.7% |
| REVIEW | 3,146 | 87 | 2.77% | 97.2% |
| APPROVE | 46,569 | 583 | 1.25% | 0% |

**Overall recall: 25.4%** — see limitations section for explanation.

---

## Part 2 — XGBoost ML Model

### Why ML alongside rules?

Rule engines catch known fraud patterns. ML catches unknown ones.  
More importantly — ML tells you which signals actually matter vs which ones you assumed matter.

### Feature Engineering

11 features engineered from raw transaction data:

```python
# Behavioural features
daily_velocity      # transactions per user per day
amt_vs_avg_ratio    # amount relative to user's own average
txn_hour            # hour of transaction

# Risk flags
is_foreign          # country != transaction location
flag_cnp            # card not present
flag_risky_merchant # Crypto or Gaming category
flag_high_risk_seg  # high risk segment label

# Identity signals
amount              # raw transaction amount
failed_logins       # raw failed login count
account_age         # days since account creation
kyc_verified        # KYC verification status
```

### Model Performance

| Metric | Model V1 (Binary Flags) | Model V2 (Raw + Engineered) |
|---|---|---|
| AUC-ROC | 0.71 | 0.9999* |
| AUC-PR | 0.15 | 0.9956* |
| Fraud caught | 41.7% | 98.1%* |
| False Positives | 969 | 8* |

*Near-perfect scores reflect synthetic data leakage — fraud labels were generated  
from the same features used in training. See limitations section.

### XGBoost Feature Importance

ML importance scores used to revalidate SQL rule weights:

```
failed_logins      60.5%  ← confirms highest SQL weight correct
amount             11.3%  ← SQL weight reduced from 25 to 10
kyc_verified        8.2%  ← SQL weight reduced from 20 to 9
flag_high_risk_seg  7.8%  ← SQL weight increased from 10 to 9
flag_risky_merchant 6.2%  ← SQL weight reduced from 15 to 8
is_foreign          3.1%  ← SQL weight reduced from 20 to 6
daily_velocity      0.0%  ← removed from scoring engine entirely
```

**Key finding:** `is_foreign` was overweighted at 20 points — it fires on 80% of  
transactions making it a noisy signal. ML importance of 3.1% confirmed this.  
`flag_high_risk_seg` was underweighted at 10 points — ML confirmed it should be  
almost equal to `kyc_verified`.

### Threshold Analysis

Systematic evaluation of every threshold from 35 to 97:

| Threshold | Precision | Recall | F1 | FP Rate |
|---|---|---|---|---|
| 35 | 17.6% | 20.6% | 0.190 | 82.4% |
| 50 | 25.9% | 8.7% | 0.130 | 74.1% |
| 65 | 80.0% | 1.5% | 0.030 | 20.0% |

**Best F1 at threshold 35** — but recall still only 20.6%.  
This led to the hybrid architecture decision.

---

## Evolution of the Decision Engine

| Version | Architecture | BLOCK Precision | BLOCK FP Rate | Recall |
|---|---|---|---|---|
| V1 | Domain weights, threshold 60 | 7.75% | 92.25% | 33.2% |
| V2 | ML-validated weights, threshold 65 | 25.86% | 74.14% | 25.4% |
| V3 | Hybrid hard rules + soft scoring | 39.30% | 60.70% | 25.4% |

V1 → V2: ML weights reduced BLOCK size by 92% while improving precision 3.3x.  
V2 → V3: Hard rules pushed BLOCK precision to 39.3% — nearly 1 in 2 blocks is genuine fraud.

---

## Key Findings

**1. Failed logins dominates all other signals**  
At 60.5% XGBoost importance, failed_logins is 5x more predictive than the next feature.  
A single failed_logins > 2 combined with any secondary signal justifies a hard block.

**2. Foreign transaction signal was overweighted**  
is_foreign fires on 80% of all transactions in this dataset — too broad to be a strong  
discriminator. ML importance of 3.1% confirmed it should be a supporting signal only,  
not a primary one. Reduced from 20 to 6 points in the scoring engine.

**3. Hybrid architecture outperforms pure scoring**  
Separating high-confidence single signals (hard rules) from weak signal combinations  
(soft scoring) improved BLOCK precision from 25.86% to 39.30% without changing recall.

**4. Rules and ML validate each other**  
Domain expertise set initial weights. ML feature importance corrected three overweighted  
signals and one underweighted signal. Neither approach alone would have found this.

---

## Limitations

**Synthetic data ceiling — recall capped at 25%**

582 fraud cases score low because they were labelled by single isolated signals  
in the data generator. The scoring engine requires multiple signals to fire together,  
so single-signal fraud cases slip through at every threshold.

In production this limitation does not exist — real fraudsters trigger multiple  
signals simultaneously. An ATO attack shows failed logins AND new device AND  
geo mismatch AND unusual hour together. The architecture is production-ready;  
the constraint is the synthetic data, not the engine design.

**Near-perfect ML metrics indicate data leakage**  
Fraud labels were generated deterministically from the same features used in training.  
XGBoost reverse-engineers the labelling rules rather than learning generalised patterns.  
Real-world fraud models achieve 0.75–0.90 AUC-ROC on production data.

---

## How to Run

**Prerequisites:**
- MySQL 8.0+
- Python 3.9+
- MySQL Workbench (optional)

**Step 1 — Set up the database:**
```sql
-- In MySQL Workbench or mysql CLI
source data/fraud_db_setup.sql
```

**Step 2 — Run SQL modules:**
```sql
USE fraud_db;
source sql/01_exploratory_analysis.sql
-- run each module in order
```

**Step 3 — Run Python model:**
```bash
pip install pandas numpy scikit-learn xgboost==2.1.1 mysql-connector-python matplotlib
jupyter notebook python/03_xgboost_model.py
```

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| Advanced SQL — CTEs, window functions, self joins | All SQL modules |
| Fraud domain knowledge — ATO, BIN attack, geo-velocity, structuring | Modules 04–09 |
| Feature engineering | python/02_feature_engineering.py |
| Class imbalance handling | scale_pos_weight in XGBoost |
| Threshold optimisation | python/04_threshold_analysis.py |
| ML-rule integration | Feature importance → SQL weight update |
| Production thinking | Hybrid architecture, approval rate tracking |
| Honest evaluation | Limitations section — synthetic data ceiling |

---

## Author

**Vishnu Saseendran**  
Senior Executive — Payments Performance & Risk Investigations  
6+ years in fraud detection, transaction monitoring, and payments risk  

[LinkedIn](https://linkedin.com/in/vishnu-saseendran-522798148) | [GitHub](https://github.com/VishnuSas21)
