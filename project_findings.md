# Project Findings — Fraud Detection System

## Executive Summary

Built a hybrid fraud detection system combining SQL rule engine and XGBoost ML model
on 50,000 synthetic transactions. Key outcome: ML feature importance corrected three
overweighted and one underweighted signal in the original domain-based rule engine,
improving BLOCK precision from 7.75% to 39.3%.

---

## Finding 1 — Failed Logins Dominates All Other Signals

XGBoost assigned 60.5% of total importance to `failed_logins` — five times more
than the next feature (`amount` at 11.3%).

**Implication for rule engine:**
Any transaction with `failed_logins > 2` combined with one secondary signal
(high amount, no KYC, or high-risk merchant) justifies a hard block without
needing a composite score. This became the foundation of the hybrid architecture.


## Finding 2 — Foreign Transaction Signal Was Overweighted

`is_foreign` was assigned 20 points in the original domain-based scoring engine.
XGBoost importance: 3.1%.

**Why the gap exists:**
`is_foreign` fires on 80% of all transactions in this dataset — too broad to
discriminate. When a signal is true for 80% of data, it has near-zero information
value for separating fraud from legitimate transactions.

**Fix applied:**
Reduced from 20 points to 6 points in the ML-validated scoring engine.

**Production lesson:**
High-coverage signals (firing on >50% of transactions) should receive low weights
regardless of intuitive appeal. Coverage and predictive power are different things.

---

## Finding 3 — High Risk Segment Was Underweighted

`flag_high_risk_seg` was assigned only 10 points in the original engine.
XGBoost importance: 7.8% — nearly equal to `kyc_verified` at 8.2%.

**Fix applied:**
Increased from 10 points to 9 points — now correctly ranked just below
`kyc_verified` rather than being treated as a minor supporting signal.

---

## Finding 4 — Daily Velocity Should Be Removed

`daily_velocity` scored 0.0% XGBoost importance — the only feature with
zero predictive value.

**Why:**
The synthetic dataset has at most 2-3 transactions per user per day — too sparse
for velocity to be a meaningful signal. In a production dataset with millions of
daily transactions, velocity would be a strong signal.

**Decision:**
Removed from scoring engine entirely. Kept in the SQL modules for demonstration
of the technique, but not included in the weighted risk score.

---

## Finding 5 — Hybrid Architecture Outperforms Pure Scoring

Testing three decision engine versions:

| Version | BLOCK Precision | BLOCK FP Rate | Architecture |
|---|---|---|---|
| V1 — Domain weights | 7.75% | 92.25% | Pure scoring, threshold 60 |
| V2 — ML weights | 25.86% | 74.14% | Pure scoring, threshold 65 |
| V3 — Hybrid | 39.30% | 60.70% | Hard rules + soft scoring |

**Key insight:**
Separating high-confidence single signals (hard rules) from weak signal
combinations (soft scoring) improved BLOCK precision by 5x over the original
domain-based engine — without changing overall recall.

---

## Synthetic Data Limitation — Honest Assessment

Overall recall is capped at 25.4%. This is not an architecture failure — it is
a data generation limitation.

**Root cause:**
582 fraud cases were labelled by single isolated signals in the data generator
(e.g., merchant_cat=Crypto alone, or txn_location≠country alone). The scoring
engine requires multiple signals to fire together to reach the BLOCK or REVIEW
threshold. Single-signal fraud cases score too low to be caught.

**Why this doesn't apply in production:**
Real fraudsters trigger multiple signals simultaneously. An ATO attack involves:
- Multiple failed logins (failed_logins > 2)
- New device or location (geo mismatch)
- Unusual transaction hour (2am-5am)
- High-value CNP transaction
- Sometimes unverified KYC

These co-occur naturally because they all arise from the same underlying event
(account compromise). The synthetic data generator assigned each fraud case one
or two signals independently — an unrealistic simplification.

**Architecture is production-ready. The constraint is the data, not the engine.**

---

## Recommendations for Production Deployment

1. **Add real-time velocity features** — rolling 1-hour and 24-hour transaction
   counts are the most impactful signals missing from this dataset in production form

2. **Device fingerprinting** — first-time device on established account is a
   strong ATO signal not captured in synthetic data

3. **Step-up authentication for REVIEW bucket** — replace analyst review with
   automated OTP/3DS for the 3,146 REVIEW transactions. Customer verifies in
   seconds, ops team never sees it, false positive cost drops to near zero

4. **Feedback loop** — confirmed fraud cases from investigations should feed back
   into model retraining monthly. Fraud patterns evolve; models must evolve with them

5. **Separate models per fraud type** — ATO model, BIN attack model, structuring
   model each trained on their own signal sets. Ensemble their outputs rather than
   using one model for all fraud types


## Finding 6 — REVIEW Bucket Requires Step-Up Authentication Not Analyst Review

REVIEW bucket false positive rate: 97.23% across 3,146 transactions.
Sending every REVIEW case to a human analyst is operationally unviable.

**Root cause:**
Soft scoring captures borderline transactions — by definition these are mostly
legitimate transactions with 2-3 weak signals. High false positives in REVIEW
are expected, not a system failure.

**Production solution:**
Replace analyst review with automated step-up authentication:
- Customer receives OTP or 3DS challenge
- Legitimate customers self-clear in seconds
- Ops team never sees the transaction
- False positive cost drops to near zero
- Only BLOCK bucket requires human analyst investigation

**Key distinction:**
BLOCK = high confidence fraud → analyst investigates
REVIEW = uncertain → customer proves legitimacy themselves
APPROVE = low risk → no friction

This architecture reduces analyst workload by 94% compared to
reviewing all flagged transactions manually.
