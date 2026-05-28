-- ============================================================
-- MODULE 08 — HYBRID DECISION ENGINE (Final Version)
-- Fraud Detection System | Vishnu Saseendran
-- ============================================================
-- Architecture: Hard Rules + Soft Scoring
--
-- Hard rules: high-confidence single signals → BLOCK immediately
-- Soft scoring: weak signals combined → weighted risk score
--
-- Weights validated by XGBoost feature importance:
--   failed_logins = 60.5% importance → highest weight (35)
--   is_foreign    =  3.1% importance → reduced from 20 to 6
--   daily_velocity=  0.0% importance → removed entirely
--
-- Decision thresholds:
--   Hard rule match   → BLOCK
--   Soft score >= 25  → REVIEW
--   Soft score < 25   → APPROVE
-- ============================================================

USE fraud_db;

WITH daily_velocity AS (
    SELECT
        user_id,
        DATE(txn_time)              AS txn_date,
        COUNT(*)                    AS daily_count
    FROM transactions
    GROUP BY user_id, DATE(txn_time)
),
user_avg_amount AS (
    SELECT
        user_id,
        AVG(amount)                 AS avg_amt
    FROM transactions
    GROUP BY user_id
),
scored AS (
    SELECT
        t.txn_id,
        t.user_id,
        t.amount,
        t.txn_type,
        t.merchant_cat,
        t.country,
        t.txn_location,
        t.is_fraud,

        -- ── HARD RULES ──────────────────────────────────────
        -- High confidence single signals — block immediately
        -- No score needed — pattern alone justifies block
        CASE
            WHEN t.failed_logins > 5
                THEN 'HARD_BLOCK'
            WHEN t.failed_logins > 2 AND t.kyc_verified = 0
                THEN 'HARD_BLOCK'
            WHEN t.failed_logins > 2 AND t.amount > 800
                THEN 'HARD_BLOCK'
            WHEN t.risk_segment = 'High'
                 AND t.kyc_verified = 0
                 AND t.merchant_cat IN ('Crypto', 'Gaming')
                THEN 'HARD_BLOCK'
            ELSE 'SCORE'
        END                                                 AS hard_rule,

        -- ── SOFT SCORE ───────────────────────────────────────
        -- ML-validated weights (XGBoost feature importance)
        -- failed_logins excluded — handled by hard rules above
        CASE WHEN t.amount > 800                            THEN 10 ELSE 0 END
      + CASE WHEN t.kyc_verified = 0                        THEN  9 ELSE 0 END
      + CASE WHEN t.risk_segment = 'High'                   THEN  9 ELSE 0 END
      + CASE WHEN t.merchant_cat IN ('Crypto', 'Gaming')    THEN  8 ELSE 0 END
      + CASE WHEN t.txn_location <> t.country               THEN  6 ELSE 0 END
      + CASE WHEN t.amount > 3 * ua.avg_amt                 THEN  5 ELSE 0 END
      + CASE WHEN t.account_age < 30                        THEN  5 ELSE 0 END
      + CASE WHEN HOUR(t.txn_time) BETWEEN 2 AND 5          THEN  5 ELSE 0 END
      + CASE WHEN t.txn_type = 'CNP'                        THEN  5 ELSE 0 END
        AS soft_score

    FROM transactions t
    JOIN daily_velocity dv
        ON  t.user_id = dv.user_id
        AND DATE(t.txn_time) = dv.txn_date
    JOIN user_avg_amount ua
        ON  t.user_id = ua.user_id
),
decisions AS (
    SELECT
        *,
        CASE
            WHEN hard_rule = 'HARD_BLOCK' THEN 'BLOCK'   -- hard rule overrides
            WHEN soft_score >= 25         THEN 'REVIEW'  -- soft score threshold
            ELSE                               'APPROVE'
        END                                               AS decision
    FROM scored
)

-- ── Performance Summary ──────────────────────────────────────
SELECT
    decision,
    COUNT(*)                                                AS total_txns,
    SUM(is_fraud)                                           AS fraud_count,
    ROUND(100.0 * AVG(is_fraud), 2)                        AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1
        THEN amount ELSE 0 END), 2)                         AS fraud_amount,
    SUM(CASE WHEN is_fraud=0
        AND decision != 'APPROVE' THEN 1 ELSE 0 END)        AS false_positives,
    ROUND(100.0 * SUM(CASE WHEN is_fraud=0
        AND decision != 'APPROVE' THEN 1 ELSE 0 END)
        / COUNT(*), 2)                                      AS fp_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN decision='APPROVE'
        THEN 1 ELSE 0 END) / COUNT(*), 2)                  AS approval_rate_pct
FROM decisions
GROUP BY decision
ORDER BY FIELD(decision, 'BLOCK', 'REVIEW', 'APPROVE');

-- ── Expected Results ─────────────────────────────────────────
-- decision | total_txns | fraud_rate | fp_rate | approval_rate
-- BLOCK    |        285 |     39.30% |  60.70% |         0.00%
-- REVIEW   |      3,146 |      2.77% |  97.23% |         0.00%
-- APPROVE  |     46,569 |      1.25% |   0.00% |       100.00%
--
-- Overall recall: 25.4% (limited by synthetic data — see README)
-- BLOCK precision: 39.3% — nearly 1 in 2 hard blocks is genuine fraud
