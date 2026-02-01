-- analysis/sql/queries.sql

-- 1) Outcomes by court (top courts by volume)
SELECT
  o.court,
  COUNT(*) AS n,
  ROUND(AVG(CASE WHEN out.outcome_code = 2 THEN 1.0 ELSE 0.0 END), 3) AS pct_changed_or_mixed,
  ROUND(AVG(CASE WHEN out.needs_review = 1 THEN 1.0 ELSE 0.0 END), 3) AS pct_needs_review
FROM opinions o
JOIN outcomes out USING (case_id)
GROUP BY o.court
ORDER BY n DESC
LIMIT 20;

-- 2) Review rate by court (only courts with enough volume)
SELECT
  o.court,
  COUNT(*) AS n,
  ROUND(AVG(out.needs_review), 3) AS review_rate
FROM opinions o
JOIN outcomes out USING (case_id)
GROUP BY o.court
HAVING n >= 10
ORDER BY review_rate DESC;

-- 3) Confidence bucket distribution
SELECT
  CASE
    WHEN outcome_confidence >= 0.90 THEN "0.90–1.00"
    WHEN outcome_confidence >= 0.80 THEN "0.80–0.89"
    WHEN outcome_confidence >= 0.70 THEN "0.70–0.79"
    WHEN outcome_confidence >= 0.60 THEN "0.60–0.69"
    ELSE "<0.60"
  END AS conf_bucket,
  COUNT(*) AS n,
  ROUND(AVG(needs_review), 3) AS pct_flagged
FROM outcomes
GROUP BY conf_bucket
ORDER BY
  CASE conf_bucket
    WHEN "0.90–1.00" THEN 1
    WHEN "0.80–0.89" THEN 2
    WHEN "0.70–0.79" THEN 3
    WHEN "0.60–0.69" THEN 4
    ELSE 5
  END;

-- 4) Changed-or-mixed rate by court
SELECT
  o.court,
  COUNT(*) AS n,
  SUM(CASE WHEN out.outcome_code = 2 THEN 1 ELSE 0 END) AS changed_or_mixed_n,
  ROUND(AVG(CASE WHEN out.outcome_code = 2 THEN 1.0 ELSE 0.0 END), 3) AS changed_or_mixed_rate
FROM opinions o
JOIN outcomes out USING (case_id)
GROUP BY o.court
HAVING n >= 10
ORDER BY changed_or_mixed_rate DESC;

-- 5) Volume vs review rate (helps make a scatter plot later)
SELECT
  o.court,
  COUNT(*) AS n,
  ROUND(AVG(out.needs_review), 3) AS review_rate,
  ROUND(AVG(out.outcome_confidence), 3) AS avg_confidence
FROM opinions o
JOIN outcomes out USING (case_id)
GROUP BY o.court
ORDER BY n DESC;