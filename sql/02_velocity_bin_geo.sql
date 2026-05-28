-- ============================================================
-- MODULE 02 — VELOCITY DETECTION
-- Fraud Detection System | Vishnu Saseendran
-- ============================================================
-- Purpose: Identify users making abnormally high transaction
-- volume within short time windows — classic bot/automated
-- fraud signal
-- ============================================================

USE fraud_db;

-- ── 2.1 Daily Velocity — Users with 2+ txns same day ────────
SELECT
    user_id,
    DATE(txn_time)              AS txn_date,
    COUNT(*)                    AS daily_txn_count,
    ROUND(SUM(amount), 2)       AS daily_amount,
    'HIGH_VELOCITY'             AS fraud_flag
FROM transactions
GROUP BY user_id, DATE(txn_time)
HAVING COUNT(*) > 1
ORDER BY daily_txn_count DESC;
-- Business: Multiple transactions in one day from same user
-- warrants investigation, especially combined with other signals

-- ── 2.2 Rolling 1-Hour Velocity (Self Join) ─────────────────
-- Most important fraud SQL pattern — catches automated card testing
SELECT
    t1.user_id,
    t1.txn_time,
    COUNT(t2.txn_id)            AS txns_last_1hr,
    ROUND(SUM(t2.amount), 2)    AS amount_last_1hr,
    'HIGH_VELOCITY_1HR'         AS fraud_flag
FROM transactions t1
JOIN transactions t2
    ON  t1.user_id  = t2.user_id
    AND t2.txn_time BETWEEN t1.txn_time - INTERVAL 1 HOUR
                        AND t1.txn_time
GROUP BY t1.user_id, t1.txn_time
HAVING COUNT(t2.txn_id) > 1
ORDER BY txns_last_1hr DESC;
-- Business: Multiple transactions within 60 minutes from same user
-- is the primary signal for automated fraud attacks (bots, card testing)
-- Real production threshold typically 5+ transactions per hour


-- ============================================================
-- MODULE 03 — BIN ATTACK DETECTION
-- ============================================================
-- Purpose: Identify systematic card testing across a BIN range
-- BIN = first 6 digits of card number (identifies issuing bank)
-- Attackers generate card numbers algorithmically and test them
-- ============================================================

-- ── 3.1 BIN Attack Pattern ───────────────────────────────────
WITH bin_stats AS (
    SELECT
        DATE(txn_time)                                      AS txn_date,
        LEFT(card_number, 6)                                AS bin,
        COUNT(DISTINCT card_number)                         AS unique_cards,
        COUNT(*)                                            AS total_attempts,
        SUM(CASE WHEN status='declined' THEN 1 ELSE 0 END)  AS decline_count,
        ROUND(100.0 * SUM(CASE WHEN status='declined'
              THEN 1 ELSE 0 END) / COUNT(*), 2)            AS decline_rate_pct
    FROM transactions
    GROUP BY DATE(txn_time), LEFT(card_number, 6)
)
SELECT
    *,
    'BIN_ATTACK'                                            AS fraud_flag
FROM bin_stats
WHERE unique_cards   > 3        -- multiple cards on same BIN same day
  AND decline_rate_pct > 60     -- high decline = invalid generated numbers
ORDER BY unique_cards DESC;
-- Business: BIN attacks show high unique card count + high decline rate
-- Attacker generates card numbers algorithmically — most are invalid
-- (hence high declines) until they find live cards to exploit
-- Recommend: temp block on BIN when 3+ unique cards + 60%+ declines in 1 day


-- ============================================================
-- MODULE 04 — GEO-VELOCITY FRAUD DETECTION
-- ============================================================
-- Purpose: Find transactions where same user appears in two
-- different countries within 2 hours — physically impossible
-- Strong indicator of cloned card or ATO
-- ============================================================

-- ── 4.1 Impossible Travel Detection ─────────────────────────
WITH travel_check AS (
    SELECT
        user_id,
        txn_time                                            AS current_txn_time,
        LAG(txn_time) OVER (
            PARTITION BY user_id ORDER BY txn_time
        )                                                   AS prev_txn_time,
        ROUND(TIMESTAMPDIFF(MINUTE,
            LAG(txn_time) OVER (
                PARTITION BY user_id ORDER BY txn_time),
            txn_time) / 60.0, 2)                           AS hours_between_txn,
        country                                             AS current_country,
        LAG(country) OVER (
            PARTITION BY user_id ORDER BY txn_time
        )                                                   AS prev_country
    FROM transactions
)
SELECT
    *,
    'GEO_VELOCITY'                                          AS fraud_flag
FROM travel_check
WHERE current_country <> prev_country
  AND hours_between_txn < 2
  AND prev_country IS NOT NULL
  AND prev_txn_time IS NOT NULL
ORDER BY hours_between_txn ASC;
-- Business: User appearing in UAE and UK within 30 minutes is physically
-- impossible — indicates cloned card being used in parallel, or ATO where
-- attacker is in different country to legitimate cardholder
-- Recommend: immediate step-up auth or temp block pending verification
