-- Script de inicialización para PostgreSQL
-- Crea la tabla para el dataset de anime
DROP TABLE IF EXISTS anime;
CREATE TABLE anime (
    anime_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    english_name VARCHAR(255),
    other_name VARCHAR(255),
    score DECIMAL(4,2),
    genres TEXT,
    synopsis TEXT,
    type VARCHAR(50),
    episodes DECIMAL(6,1),
    aired VARCHAR(100),
    status VARCHAR(100),
    producers TEXT,
    licensors TEXT,
    studios TEXT,
    source VARCHAR(100),
    duration VARCHAR(100),
    rating VARCHAR(100),
    rank DECIMAL(10,1),
    popularity DECIMAL(10,0),
    favorites INTEGER,
    image_url TEXT
);

-- Importa el CSV a una tabla temporal
CREATE TEMP TABLE tmp_anime AS SELECT * FROM anime WITH NO DATA;
COPY tmp_anime(anime_id, name, english_name, other_name, score, genres, synopsis, type, episodes, aired, status, producers, licensors, studios, source, duration, rating, rank, popularity, favorites, image_url)
FROM '/docker-entrypoint-initdb.d/anime-dataset-2023-cleaned.csv'
DELIMITER ','
CSV HEADER;

-- Inserta en la tabla final, transformando variantes de 'unknown' a NULL
INSERT INTO anime (
    anime_id, name, english_name, other_name, score, genres, synopsis, type, episodes, aired, status, producers, licensors, studios, source, duration, rating, rank, popularity, favorites, image_url
)
SELECT
    anime_id,
    NULLIF(TRIM(name), '') AS name,
    CASE WHEN lower(english_name) LIKE 'unknown%' OR english_name IS NULL OR TRIM(english_name) = '' THEN NULL ELSE english_name END,
    CASE WHEN lower(other_name) LIKE 'unknown%' OR other_name IS NULL OR TRIM(other_name) = '' THEN NULL ELSE other_name END,
    score,
    CASE WHEN lower(genres) LIKE 'unknown%' OR genres IS NULL OR TRIM(genres) = '' THEN NULL ELSE genres END,
    CASE WHEN lower(synopsis) LIKE 'unknown%' OR synopsis IS NULL OR TRIM(synopsis) = '' THEN NULL ELSE synopsis END,
    CASE WHEN lower(type) LIKE 'unknown%' OR type IS NULL OR TRIM(type) = '' THEN NULL ELSE type END,
    episodes,
    CASE WHEN lower(aired) LIKE 'unknown%' OR aired IS NULL OR TRIM(aired) = '' THEN NULL ELSE aired END,
    CASE WHEN lower(status) LIKE 'unknown%' OR status IS NULL OR TRIM(status) = '' THEN NULL ELSE status END,
    CASE WHEN lower(producers) LIKE 'unknown%' OR producers IS NULL OR TRIM(producers) = '' THEN NULL ELSE producers END,
    CASE WHEN lower(licensors) LIKE 'unknown%' OR licensors IS NULL OR TRIM(licensors) = '' THEN NULL ELSE licensors END,
    CASE WHEN lower(studios) LIKE 'unknown%' OR studios IS NULL OR TRIM(studios) = '' THEN NULL ELSE studios END,
    CASE WHEN lower(source) LIKE 'unknown%' OR source IS NULL OR TRIM(source) = '' THEN NULL ELSE source END,
    CASE WHEN lower(duration) LIKE 'unknown%' OR duration IS NULL OR TRIM(duration) = '' THEN NULL ELSE duration END,
    CASE WHEN lower(rating) LIKE 'unknown%' OR rating IS NULL OR TRIM(rating) = '' THEN NULL ELSE rating END,
    rank,
    popularity,
    favorites,
    CASE WHEN lower(image_url) LIKE 'unknown%' OR image_url IS NULL OR TRIM(image_url) = '' THEN NULL ELSE image_url END
FROM tmp_anime;

DROP TABLE tmp_anime;

