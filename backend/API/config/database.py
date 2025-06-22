# Database configuration with SQLAlchemy
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Determinar si usar SQLite o PostgreSQL - Forzar PostgreSQL para contenedor Docker
USE_SQLITE = os.environ.get("USE_SQLITE", "false").lower() in ("true", "1", "yes")

# Database configuration (PostgreSQL config)
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"), 
    "database": os.environ.get("DB_NAME", "animeDB"),
    "user": os.environ.get("DB_USER", "anime_db"),
    "password": os.environ.get("DB_PASSWORD", "anime_db")
}

# Ruta del archivo SQLite
SQLITE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/anime_db.sqlite"))
os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)

# Create database URL
if USE_SQLITE:
    print(f"⚙️ Usando SQLite como base de datos: {SQLITE_PATH}")
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
else:
    print("⚙️ Usando PostgreSQL como base de datos")
    DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Create engine with appropriate settings
if USE_SQLITE:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

# Create session
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base
Base = declarative_base()

def get_db():
    """Database session generator"""
    db = Session()
    try:
        yield db
    finally:
        db.close()
