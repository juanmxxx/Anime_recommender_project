# Archivo api_direct.py - Usa las funciones directamente en lugar de subprocess
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import sys
import os
import traceback
import pathlib
import psycopg2
from datetime import datetime
from pydantic import BaseModel

# Modelo para tracking de métricas
class MetricEvent(BaseModel):
    session_id: str
    event_type: str  # 'search', 'click' o 'load_time'
    prompt_text: str = None
    anime_clicked: str = None
    anime_id: int = None
    load_time_ms: int = None  # Tiempo de carga en milisegundos

# Configuración de base de datos
DB_CONFIG = {
    "host": "localhost",
    "port": "5432", 
    "database": "animes",
    "user": "anime_db",
    "password": "anime_db"
}

def get_db_connection():
    """Obtiene conexión a PostgreSQL"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error conectando a DB: {e}")
        return None

def log_metric_event(session_id: str, event_type: str, request: Request, 
                    prompt_text: str = None, anime_clicked: str = None, anime_id: int = None):
    """Registra evento de métrica en la base de datos"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_metrics (session_id, event_type, prompt_text, anime_clicked, anime_id, user_agent, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id, 
            event_type, 
            prompt_text, 
            anime_clicked, 
            anime_id,
            request.headers.get("user-agent", "unknown"),
            request.client.host if request.client else "unknown"
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error logging metric: {e}")
        return False

# Agregar la ruta al directorio AI al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))
# Agregar la ruta base del proyecto para resolver imports de 'backend'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Importar directamente las funciones del recomendador
try:
    from outputAndFormatProcessor import get_recommendations_json, ImprovedAnimeRecommendationSystem
    # Inicializar el sistema una vez para reutilizarlo en todas las peticiones
    anime_system = ImprovedAnimeRecommendationSystem()
except Exception as e:
    print(f"Error al importar outputAndFormatProcessor: {e}")
    traceback.print_exc()
    anime_system = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta para servir el favicon (puede ser cualquier imagen que tengas)
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "frontend", "public", "vite.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    
    # Si no encuentra el favicon, devuelve una respuesta vacía (204)
    return {"status_code": 204}

@app.get("/recommend")
def recommend(keywords: str = Query(...), top_n: int = Query(5)):
    """
    Endpoint para obtener recomendaciones de anime basadas en palabras clave.
    
    Args:
        keywords: Palabras clave para buscar animes
        
    Returns:
        JSON con recomendaciones o mensaje de error
    """
    try:
        if anime_system is None:
            return {"error": "El sistema de recomendación no está inicializado correctamente"}
        
        print(f"Procesando solicitud de recomendación con keywords: {keywords}")
        
        # Llamar directamente a la función de recomendación
        json_result = get_recommendations_json(keywords, anime_system, top_n)
        
        # Convertir de string JSON a objeto Python
        parsed_data = json.loads(json_result)
        return parsed_data
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error en API: {error_msg}")
        print(stack_trace)
        return {
            "error": f"Error en el sistema de recomendación: {error_msg}",
            "details": stack_trace
        }

@app.post("/recommend")
def get_recommendations(keywords: str = Query(..., description="Palabras clave separadas por espacios"), 
                       top_n: int = Query(10, description="Número máximo de recomendaciones")):
    """Endpoint para obtener recomendaciones basadas en palabras clave"""
    try:
        print(f"Recibida solicitud de recomendación: keywords='{keywords}', top_n={top_n}")
        
        if anime_system is None:
            return {"error": "Sistema de recomendación no inicializado"}
        
        # Usar el sistema mejorado
        recommendations = anime_system.get_recommendations(keywords, top_n)
        
        # Obtener keyphrases extraídas
        keyphrases = anime_system.get_last_keyphrases()
        
        print(f"Generadas {len(recommendations)} recomendaciones")
        return {
            "recommendations": recommendations,
            "keyphrases": keyphrases,
            "query": keywords,
            "total_results": len(recommendations)
        }
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error en API: {error_msg}")
        print(stack_trace)
        return {
            "error": f"Error en el sistema de recomendación: {error_msg}",
            "details": stack_trace
        }

# Endpoints para métricas
@app.post("/metrics/search")
def log_search_event(request: Request, metric: MetricEvent):
    """Registra evento de búsqueda"""
    success = log_metric_event(
        session_id=metric.session_id,
        event_type="search", 
        request=request,
        prompt_text=metric.prompt_text
    )
    return {"logged": success}

@app.post("/metrics/click")
def log_click_event(request: Request, metric: MetricEvent):
    """Registra evento de clic en anime"""
    success = log_metric_event(
        session_id=metric.session_id,
        event_type="click",
        request=request, 
        anime_clicked=metric.anime_clicked,
        anime_id=metric.anime_id
    )
    return {"logged": success}

@app.post("/metrics/load_time")
def log_load_time_event(request: Request, metric: MetricEvent):
    """Registra evento de tiempo de carga"""
    try:
        conn = get_db_connection()
        if not conn:
            return {"logged": False}
            
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_metrics (session_id, event_type, prompt_text, user_agent, ip_address, load_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            metric.session_id, 
            "load_time",
            metric.prompt_text,
            request.headers.get("user-agent", "unknown"),
            request.client.host if request.client else "unknown",
            metric.load_time_ms
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        return {"logged": True}
    except Exception as e:
        print(f"Error logging load time metric: {e}")
        return {"logged": False}

@app.get("/metrics/conversion")
def get_conversion_metrics(days: int = Query(30, description="Días a incluir en el análisis")):
    """Obtiene métricas de conversión de los últimos N días"""
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "No se pudo conectar a la base de datos"}
            
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(CASE WHEN event_type = 'search' THEN 1 END) as searches,
                COUNT(CASE WHEN event_type = 'click' THEN 1 END) as clicks,
                ROUND(
                    (COUNT(CASE WHEN event_type = 'click' THEN 1 END)::decimal / 
                     NULLIF(COUNT(CASE WHEN event_type = 'search' THEN 1 END), 0)) * 100, 2
                ) as conversion_rate,
                ROUND(AVG(CASE WHEN event_type = 'load_time' THEN load_time_ms::decimal ELSE NULL END), 2) as avg_load_time_ms
            FROM user_metrics 
            WHERE timestamp >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """, (days,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        metrics = []
        for row in results:
            metrics.append({
                "date": row[0].isoformat() if row[0] else None,
                "searches": row[1],
                "clicks": row[2], 
                "conversion_rate": float(row[3]) if row[3] else 0.0,
                "avg_load_time_ms": float(row[4]) if row[4] else 0.0,
                "avg_load_time_sec": round(float(row[4] or 0) / 1000, 2)  # Convertir a segundos
            })
            
        return {"metrics": metrics, "period_days": days}
        
    except Exception as e:
        return {"error": f"Error obteniendo métricas: {str(e)}"}

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que la API esté funcionando"""
    return {
        "message": "Smart Anime Recommender API está funcionando (modo directo)",
        "usage": "Use /recommend?keywords=your_keywords&top_n=5 para obtener recomendaciones",
        "status": "OK" if anime_system is not None else "Sistema no inicializado"
    }
