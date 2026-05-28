-- ============================================================
-- MODULE 05 — ATO DETECTION (Account Takeover)
-- Fraud Detection System | Vishnu Saseendran
-- ============================================================
-- Purpose: Identify account takeover patterns using multiple
-- corroborating signals — single signal is insufficient
-- Requires minimum 3 signals firing together
-- ============================================================

USE fraud_db;

-- ── 5.1 Basic ATO Detection (4 signals) ─────────────────────
SELECT
    txn_id,
    user_id,
    amount,
    txn_time,
    txn_type,
    failed_logins,
    kyc_verified,
    country,
    txn_location,
    'ATO_RISK'                                              AS fraud_flag
FROM transactions
WHERE failed_logins > 2         -- multiple failed attempts before success
  AND txn_type = 'CNP'          -- card not present (attacker doesn't have physical card)
  AND amount > 500               -- high value transaction
  AND kyc_verified = 0           -- unverified account = easier target
ORDER BY amount DESC;
-- Business: ATO pattern = attacker gains access via credential stuffing,
-- then makes high-value CNP transaction immediately
-- The failed_logins signal is the strongest individual ATO indicator

-- ── 5.2 ATO + Geo-Velocity Combined (Highest confidence) ────
WITH geo_velocity AS (
    SELECT
        user_id,
        txn_time                                            AS geo_txn_time,
        LAG(txn_time) OVER (
            PARTITION BY user_id ORDER BY txn_time)        AS prev_txn_time,
        ROUND(TIMESTAMPDIFF(MINUTE,
            LAG(txn_time) OVER (
                PARTITION BY user_id ORDER BY txn_time),
            txn_time) / 60.0, 2)                           AS hours_between,
        country                                             AS current_country,
        LAG(country) OVER (
            PARTITION BY user_id ORDER BY txn_time)        AS prev_country
    FROM transactions
),
ato_base AS (
    SELECT
        txn_id,
        user_id,
        amount,
        txn_time,
        txn_type,
        failed_logins,
        kyc_verified
    FROM transactions
    WHERE failed_logins > 2
      AND txn_type = 'CNP'
      AND amount > 500
      AND kyc_verified = 0
)
SELECT
    a.txn_id,
    a.user_id,
    a.amount,
    a.txn_time                                              AS ato_txn_time,
    a.failed_logins,
    g.current_country,
    g.prev_country,
    g.hours_between,
    'ATO_GEO_RISK'                                          AS fraud_flag
FROM ato_base a
JOIN geo_velocity g
    ON  a.user_id = g.user_id
WHERE g.current_country <> g.prev_country
  AND g.hours_between < 2
  AND g.prev_country IS NOT NULL
ORDER BY a.amount DESC;
-- Business: ATO + impossible travel = near-certain fraud
-- Attacker in foreign country using compromised credentials
-- Recommend: immediate account block, notify customer via registered contact


-- ============================================================
-- MODULE 06 — MERCHANT RISK MONITORING
-- ============================================================
-- Purpose: Identify merchants with abnormally high chargeback
-- rates — Visa/Mastercard threshold is 1%
-- Above 2% = CRITICAL — network penalty risk
-- ============================================================

-- ── 6.1 Merchant Chargeback Rate ────────────────────────────
WITH merchant_stats AS (
    SELECT
        m.merchant_id,
        m.merchant_name,
        m.merchant_cat,
        m.risk_tier,
        COUNT(*)                                            AS total_transactions,
        SUM(t.is_chargeback)                               AS chargeback_count,
        ROUND(100.0 * AVG(t.is_chargeback), 2)            AS chargeback_rate_pct,
        ROUND(100.0 * AVG(t.is_fraud), 2)                 AS fraud_rate_pct
    FROM transactions t
    JOIN merchants m ON t.merchant_id = m.merchant_id
    GROUP BY m.merchant_id, m.merchant_name,
             m.merchant_cat, m.risk_tier
)
SELECT
    *,
    CASE
        WHEN chargeback_rate_pct > 2 THEN 'CRITICAL'
        WHEN chargeback_rate_pct > 1 THEN 'WARNING'
        ELSE 'SAFE'
    END                                                     AS status
FROM merchant_stats
WHERE chargeback_rate_pct > 1
ORDER BY chargeback_rate_pct DESC;
-- Business: Merchants above 2% chargeback rate face Visa/Mastercard
-- program enrollment and potential network penalties
-- CRITICAL merchants need immediate risk tier review


-- ============================================================
-- MODULE 07 — AML STRUCTURING DETECTION
-- ============================================================
-- Purpose: Identify deliberate transaction splitting to avoid
-- automated reporting thresholds
-- Pattern: multiple transactions just below $900 that sum above it
-- ============================================================

-- ── 7.1 Structuring Pattern ──────────────────────────────────
SELECT
    user_id,
    DATE(txn_time)                                          AS txn_date,
    COUNT(*)                                                AS txn_count,
    ROUND(SUM(amount), 2)                                   AS total_amount,
    SUM(CASE WHEN amount BETWEEN 700 AND 899
        THEN 1 ELSE 0 END)                                 AS near_threshold_count,
    MAX(amount)                                             AS max_single_txn,
    'STRUCTURING_ALERT'                                     AS fraud_flag
FROM transactions
WHERE status = 'approved'       -- only approved txns caused loss
GROUP BY user_id, DATE(txn_time)
HAVING SUM(CASE WHEN amount BETWEEN 700 AND 899
               THEN 1 ELSE 0 END) >= 2   -- 2+ near-threshold txns
   AND SUM(amount) > 900                  -- combined exceeds threshold
   AND MAX(amount) < 900                  -- no single txn hit threshold
ORDER BY total_amount DESC;
-- Business: Classic structuring = splitting one large transaction into
-- multiple smaller ones to avoid automated alerts
-- SAR (Suspicious Activity Report) filing may be required
-- Note: filter on approved only — declined transactions caused no loss
-- and approved fraud = cases that slipped through for model training
