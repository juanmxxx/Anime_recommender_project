# Database configuration with SQLAlchemy
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432", 
    "database": "animeDB",
    "user": "anime_db",
    "password": "anime_db"
}

# Create database URL
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Create engine
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
