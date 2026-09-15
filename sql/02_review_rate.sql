SELECT date(timestamp) AS prediction_date,
       avg(CASE WHEN decision = 'review' THEN 1.0 ELSE 0.0 END) AS review_rate
FROM prediction_log
GROUP BY 1
ORDER BY 1;

