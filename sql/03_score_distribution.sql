SELECT round(fraud_probability, 1) AS score_bucket, count(*) AS prediction_count
FROM prediction_log
GROUP BY 1
ORDER BY 1;

