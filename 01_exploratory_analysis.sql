-- ============================================================
-- MODULE 01 — EXPLORATORY ANALYSIS
-- Fraud Detection System | Vishnu Saseendran
-- ============================================================
-- Purpose: Understand dataset structure and baseline fraud rate
-- before building detection logic
-- ============================================================

USE fraud_db;

-- ── 1.1 Dataset Overview ─────────────────────────────────────
SELECT
    COUNT(*)                                    AS total_transactions,
    SUM(is_fraud)                               AS total_fraud,
    ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct,
    ROUND(SUM(amount), 2)                       AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud=1
        THEN amount ELSE 0 END), 2)             AS total_fraud_amount
FROM transactions;
-- Business: Baseline fraud rate and total exposure

-- ── 1.2 Fraud by Transaction Type (CP vs CNP) ───────────────
SELECT
    txn_type,
    COUNT(*)                                    AS total_transactions,
    SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1
        THEN amount ELSE 0 END), 2)             AS fraud_amount
FROM transactions
GROUP BY txn_type
ORDER BY fraud_rate_pct DESC;
-- Business: CNP fraud rate is typically higher than CP
-- Card-not-present = no physical card verification

-- ── 1.3 Fraud by Merchant Category ──────────────────────────
SELECT
    merchant_cat,
    COUNT(*)                                    AS total_transactions,
    SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1
        THEN amount ELSE 0 END), 2)             AS fraud_amount
FROM transactions
GROUP BY merchant_cat
ORDER BY fraud_rate_pct DESC;
-- Business: Crypto highest risk — irreversible transactions
-- attract more fraud attempts

-- ── 1.4 Fraud by Account Age Bucket ─────────────────────────
SELECT
    CASE
        WHEN account_age BETWEEN 0  AND 30  THEN '0-30 days (New)'
        WHEN account_age BETWEEN 31 AND 90  THEN '31-90 days'
        WHEN account_age BETWEEN 91 AND 365 THEN '91-365 days'
        ELSE '365+ days (Established)'
    END                                         AS age_bucket,
    COUNT(*)                                    AS total_transactions,
    SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct
FROM transactions
GROUP BY age_bucket
ORDER BY fraud_rate_pct DESC;
-- Business: New accounts (0-30 days) show 3.6x higher fraud rate
-- Recommend step-up authentication for new account high-value txns

-- ── 1.5 Fraud by KYC Status ─────────────────────────────────
SELECT
    CASE WHEN kyc_verified=1 THEN 'KYC Verified'
         ELSE 'KYC Not Verified' END            AS kyc_status,
    COUNT(*)                                    AS total_transactions,
    ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct
FROM transactions
GROUP BY kyc_verified
ORDER BY fraud_rate_pct DESC;
-- Business: Unverified accounts have higher fraud exposure
