# Initialize the database and create tables
import sys
import os

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from backend.API.config.database import engine, Base, DB_CONFIG
from backend.API.models.anime import MetricEntry
import psycopg2

def check_table_exists(table_name):
    """Check if a table exists in the database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')")
    exists = cursor.fetchone()[0]
    conn.close()
    return exists

def init_db():
    # Check if tables already exist
    user_metrics_exists = check_table_exists("user_metrics")
    
    if user_metrics_exists:
        print("Table 'user_metrics' already exists, skipping creation.")
    else:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully.")

if __name__ == "__main__":
    init_db()
