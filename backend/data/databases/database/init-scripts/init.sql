-- Script de inicialización para PostgreSQL
-- Crea la tabla para el dataset de anime de AniList API

DROP TABLE IF EXISTS characters;
DROP TABLE IF EXISTS anime;
DROP TABLE IF EXISTS user_metrics;

-- Tabla para anime
-- Basada en la estructura del dataset de AniList API
CREATE TABLE anime (
    id INTEGER PRIMARY KEY,
    romaji_title VARCHAR(255),
    english_title VARCHAR(255),
    native_title VARCHAR(255),
    genres TEXT,
    description TEXT,
    format VARCHAR(50),
    status VARCHAR(100),
    episodes INTEGER,
    average_score DECIMAL(4,1),
    popularity INTEGER,
    season_year INTEGER,
    cover_image_medium TEXT,
    tags TEXT,
    studios TEXT
);

-- Tabla para personajes
CREATE TABLE characters (
    id INTEGER PRIMARY KEY,
    anime_id INTEGER REFERENCES anime(id),
    name VARCHAR(255),
    image_url TEXT,
    gender VARCHAR(50),
    description TEXT,
    role VARCHAR(50)
);


-- Tabla de métricas para tracking de conversión
CREATE TABLE IF NOT EXISTS user_metrics (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(20) NOT NULL, -- 'search', 'click', 'load_time'
    prompt_text TEXT,
    anime_clicked VARCHAR(255),
    anime_id INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_agent TEXT,
    ip_address INET,
    load_time_ms INTEGER
);







-- Actualización de la tabla de métricas para incluir campos adicionales / -- para mejorar el seguimiento de eventos
-- Índices para optimizar consultas de métricas
CREATE INDEX idx_user_metrics_session ON user_metrics(session_id);
CREATE INDEX idx_user_metrics_event_type ON user_metrics(event_type);
CREATE INDEX idx_user_metrics_timestamp ON user_metrics(timestamp);

-- Vista para calcular tasa de conversión y tiempo de carga promedio
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


