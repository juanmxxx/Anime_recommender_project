-- Añadir campo para tiempo de carga
ALTER TABLE user_metrics ADD COLUMN IF NOT EXISTS load_time_ms INTEGER;

-- Actualizar la vista para incluir el tiempo de carga promedio
CREATE OR REPLACE VIEW conversion_metrics AS
SELECT
    DATE(timestamp) as date,
    COUNT(CASE WHEN event_type = 'search' THEN 1 END) as total_searches,
    COUNT(CASE WHEN event_type = 'click' THEN 1 END) as total_clicks,
    ROUND(
        (COUNT(CASE WHEN event_type = 'click' THEN 1 END)::decimal /
         NULLIF(COUNT(CASE WHEN event_type = 'search' THEN 1 END), 0)) * 100, 2
    ) as conversion_rate_percent,
    ROUND(AVG(CASE WHEN event_type = 'load_time' THEN load_time_ms::decimal ELSE NULL END), 2) as avg_load_time_ms
FROM user_metrics
GROUP BY DATE(timestamp)
ORDER BY date DESC;
