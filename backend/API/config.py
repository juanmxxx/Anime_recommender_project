# Archivo de configuración para la API
from pydantic_settings import BaseSettings

# Configuración de la base de datos
DB_CONFIG = {
    "host": "localhost",
    "port": "5432", 
    "database": "animes",
    "user": "anime_db",
    "password": "anime_db"
}

# Configuración de CORS
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Otras configuraciones globales
API_VERSION = "2.0.0"
API_TITLE = "Smart Anime Recommender API"
API_DESCRIPTION = "API para el sistema de recomendaciones de anime y análisis de métricas"
