# Módulo para manejar conexiones a la base de datos
import psycopg2
from ..config import DB_CONFIG

def get_db_connection():
    """Obtiene conexión a PostgreSQL"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error conectando a DB: {e}")
        return None

def execute_query(query, params=None, fetch=True):
    """
    Ejecuta una consulta SQL y devuelve los resultados
    
    Args:
        query: Consulta SQL a ejecutar
        params: Parámetros para la consulta
        fetch: Si es True, devuelve los resultados
        
    Returns:
        Resultados de la consulta si fetch es True, de lo contrario None
    """
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        
        result = None
        if fetch:
            result = cursor.fetchall()
        else:
            conn.commit()
        
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
