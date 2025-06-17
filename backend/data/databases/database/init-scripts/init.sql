-- Script de inicialización para PostgreSQL
-- Crea la tabla para el dataset de anime de AniList API

DROP TABLE IF EXISTS characters;
DROP TABLE IF EXISTS anime;

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
