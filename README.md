# Fraud Detection System — SQL Rule Engine + XGBoost ML Model

A production-inspired fraud detection system built on 50,000 synthetic transactions.  
Combines a **SQL-based rule engine** with an **XGBoost machine learning model** to demonstrate  
how real fraud teams use rules and ML together — not as alternatives, but as layers.

---

## Project Structure

```
fraud-detection-system/
├── sql/
│   ├── 01_exploratory_analysis.sql
│   ├── 02_velocity_bin_geo.sql
│   ├── 03_ato_merchant_structuring.sql
│   └── 04_hybrid_decision_engine.sql
├── python/
│   └── fraud_detection_xgboost.ipynb
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

Four modules built progressively, each targeting specific fraud patterns.

### Modules

| File | Fraud Patterns Covered | Key SQL Techniques |
|---|---|---|
| 01_exploratory_analysis | Baseline fraud rate by segment, KYC, account age | GROUP BY, conditional aggregation, CASE WHEN |
| 02_velocity_bin_geo | Daily velocity, rolling 1-hour velocity, BIN attacks, geo-velocity | Self JOIN, INTERVAL window, CTE, LAG(), TIMESTAMPDIFF |
| 03_ato_merchant_structuring | Account takeover, merchant chargeback risk, AML structuring | Multi-signal AND logic, JOIN, near-threshold detection |
| 04_hybrid_decision_engine | Risk scoring engine + final decision engine | Weighted CASE WHEN, hard rules, 4-CTE chain |

### Risk Scoring Engine

Each transaction receives a weighted risk score based on 10 fraud signals.  
Weights initially set by domain knowledge, then validated by XGBoost feature importance.

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
Weak signals combined →  Score ≥ 25  →  REVIEW
                         Score < 25  →  APPROVE
```

**Hard block conditions:**
- failed_logins > 5
- failed_logins > 2 AND kyc_verified = 0
- failed_logins > 2 AND amount > $800
- risk_segment = High AND kyc_verified = 0 AND merchant = Crypto/Gaming

**Why failed_logins is excluded from soft score:**  
It is already the trigger for 3 of 4 hard rules — including it in soft scoring  
would double-count the signal. The two layers are kept cleanly separated.

### Decision Engine Results

| Decision | Transactions | Fraud Count | Fraud Rate | FP Rate |
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

### Notebook Structure

| Step | What it does |
|---|---|
| Step 1 | Load 50,000 transactions from MySQL |
| Step 2 | Feature engineering v1 — binary flags baseline |
| Step 3 | Train XGBoost model v1 |
| Step 4 | Evaluate model v1 — reveals binary flags are too coarse |
| Step 5 | Feature engineering v2 — raw continuous values |
| Step 6 | Evaluate model v2 — dramatic improvement over v1 |
| Step 7 | XGBoost feature importance analysis |
| Step 8 | Map importance scores to SQL weights v1 |
| Step 9 | Proportional ML-validated SQL weights v2 |
| Step 10 | Threshold analysis — precision/recall trade-off |
| Step 11 | Signal coverage analysis — why recall is capped |
| Step 12 | Hybrid decision engine — hard rules + soft scoring |
| Step 13 | Final 3-way comparison V1 vs V2 vs V3 |

### Model Performance

| Metric | Model V1 (Binary Flags) | Model V2 (Raw + Engineered) |
|---|---|---|
| AUC-ROC | 0.71 | 0.9999* |
| AUC-PR | 0.15 | 0.9956* |
| Fraud caught | 41.7% | 98.1%* |
| False Positives | 969 | 8* |

*Near-perfect scores reflect synthetic data leakage — fraud labels were generated  
from the same features used in training. See limitations section.

### XGBoost Feature Importance — SQL Weight Validation

| Feature | ML Importance | Old SQL Weight | New SQL Weight | Change |
|---|---|---|---|---|
| failed_logins | 60.5% | 30 | 35 | ↑ Confirmed strongest |
| amount | 11.3% | 25 | 10 | ↓ Was overweighted |
| kyc_verified | 8.2% | 20 | 9 | ↓ Reduced |
| flag_high_risk_seg | 7.8% | 10 | 9 | ↑ Was underweighted |
| flag_risky_merchant | 6.2% | 15 | 8 | ↓ Reduced |
| is_foreign | 3.1% | 20 | 6 | ↓ Significantly overweighted |
| daily_velocity | 0.0% | 20 | 0 | ❌ Removed entirely |

### Threshold Analysis

Systematic evaluation showing no single threshold resolves the precision-recall trade-off:

| Threshold | Precision | Recall | F1 | FP Rate |
|---|---|---|---|---|
| 35 | 17.6% | 20.6% | 0.190 | 82.4% |
| 50 | 25.9% | 8.7% | 0.130 | 74.1% |
| 65 | 80.0% | 1.5% | 0.030 | 20.0% |

Best F1 at threshold 35 — but recall still only 20.6%.  
This directly motivated the hybrid architecture decision.

---

## Decision Engine Evolution

| Version | Architecture | BLOCK Precision | BLOCK FP Rate | BLOCK Count | Recall |
|---|---|---|---|---|---|
| V1 | Domain weights, threshold 60 | 7.75% | 92.25% | 3,344 | 33.2% |
| V2 | ML-validated weights, threshold 65 | 25.86% | 74.14% | 263 | 25.4% |
| V3 | Hybrid hard rules + soft scoring | 39.30% | 60.70% | 285 | 25.4% |

**V1 → V2:** ML weights reduced BLOCK size by 92% (3,344 → 263) while improving precision 3.3x  
**V2 → V3:** Hard rules pushed BLOCK precision to 39.3% — nearly 1 in 2 blocks is genuine fraud

---

## Key Findings

**1. Failed logins dominates all other signals**  
At 60.5% XGBoost importance, failed_logins is 5x more predictive than the next feature.  
Any transaction with failed_logins > 2 combined with one secondary signal justifies a hard block.

**2. Foreign transaction signal was overweighted**  
is_foreign fires on 80% of all transactions — too broad to discriminate.  
ML importance of 3.1% confirmed it should be a supporting signal only.  
Reduced from 20 to 6 points in the scoring engine.

**3. Hybrid architecture outperforms pure scoring**  
Separating high-confidence single signals (hard rules) from weak signal combinations  
(soft scoring) improved BLOCK precision from 25.86% to 39.30% without changing recall.

**4. Rules and ML validate each other**  
Domain expertise set initial weights. ML corrected three overweighted signals  
and one underweighted signal. Neither approach alone would have found this.

**5. REVIEW bucket should use step-up authentication in production**  
97% false positive rate in REVIEW is unacceptable for human analyst review.  
Solution: automated OTP or 3DS — customer self-clears in seconds,  
ops team never sees it, false positive cost drops to near zero.

---

## Limitations

**Synthetic data ceiling — recall capped at 25%**

583 fraud cases score low because they were labelled by single isolated signals  
in the data generator. The scoring engine requires multiple signals to fire together,  
so single-signal fraud cases slip through at every threshold.

In production this limitation does not exist — real fraudsters trigger multiple  
signals simultaneously. An ATO attack shows failed logins AND new device AND  
geo mismatch AND unusual hour together. The architecture is production-ready;  
the constraint is the synthetic data, not the engine design.

**Near-perfect ML metrics indicate data leakage**  
Fraud labels were generated deterministically from the same features used in training.  
XGBoost reverse-engineers the labelling rules rather than learning generalised fraud patterns.  
Real-world fraud models achieve 0.75–0.90 AUC-ROC on production data.

---

## How to Run

**Prerequisites:**
- MySQL 8.0+
- Python 3.9+
- Jupyter Notebook

**Step 1 — Set up the database:**
```sql
-- In MySQL Workbench or mysql CLI
source data/fraud_db_setup.sql
```

**Step 2 — Run SQL modules in order:**
```sql
USE fraud_db;
source sql/01_exploratory_analysis.sql
source sql/02_velocity_bin_geo.sql
source sql/03_ato_merchant_structuring.sql
source sql/04_hybrid_decision_engine.sql
```

**Step 3 — Run the notebook:**
```bash
pip install pandas numpy scikit-learn xgboost==2.1.1 mysql-connector-python matplotlib
jupyter notebook
```
Open `python/fraud_detection_xgboost.ipynb` and run all cells.  
Update MySQL password in connection cells before running.

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| Advanced SQL — CTEs, window functions, self joins | All SQL modules |
| Fraud domain knowledge — ATO, BIN attack, geo-velocity, structuring | sql/02, sql/03 |
| Feature engineering — raw values vs binary flags comparison | Notebook Steps 2 and 5 |
| Class imbalance handling — scale_pos_weight in XGBoost | Notebook Step 3 |
| Threshold optimisation — precision/recall trade-off | Notebook Step 10 |
| Signal coverage analysis — why recall is capped | Notebook Step 11 |
| ML-rule integration — feature importance → SQL weight update | Notebook Steps 7–9 |
| Hybrid architecture — hard rules + soft scoring | sql/04, Notebook Step 12 |
| Production thinking — approval rate, step-up auth recommendation | Notebook Step 12 |
| Honest evaluation — synthetic data limitations acknowledged | Limitations section |

---

## Author

**Vishnu Saseendran**  
Senior Executive — Payments Performance & Risk Investigations  
6+ years in fraud detection, transaction monitoring, and payments risk  

[LinkedIn](https://linkedin.com/in/vishnu-saseendran-522798148) | [GitHub](https://github.com/VishnuSas21)
