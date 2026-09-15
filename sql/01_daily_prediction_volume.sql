SELECT date(timestamp) AS prediction_date, count(*) AS prediction_count
FROM prediction_log
GROUP BY 1
ORDER BY 1;

