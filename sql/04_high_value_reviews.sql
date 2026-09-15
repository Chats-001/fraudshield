SELECT prediction_id, timestamp, fraud_probability, amount
FROM prediction_log
WHERE decision = 'review'
ORDER BY amount DESC
LIMIT 100;

