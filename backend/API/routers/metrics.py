# Router para las métricas de uso
from fastapi import APIRouter, Request, Query, HTTPException
import traceback

# Importaciones relativas
from ..db.models import MetricEvent
from ..db.connection import get_db_connection, execute_query

router = APIRouter()

def log_metric_event(session_id: str, event_type: str, request: Request, 
                    prompt_text: str = None, anime_clicked: str = None, anime_id: int = None,
                    load_time_ms: int = None):
    """
    Registra evento de métrica en la base de datos
    
    Args:
        session_id: ID de sesión del usuario
        event_type: Tipo de evento (search, click, load_time)
        request: Objeto Request de FastAPI
        prompt_text: Texto de búsqueda (para eventos search)
        anime_clicked: Nombre del anime clicado (para eventos click)
        anime_id: ID del anime clicado (para eventos click)
        load_time_ms: Tiempo de carga en ms (para eventos load_time)
        
    Returns:
        True si se registró correctamente, False en caso contrario
    """
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Query base para la inserción
        query = """
            INSERT INTO user_metrics 
                (session_id, event_type, prompt_text, anime_clicked, anime_id, user_agent, ip_address{})
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s{})
        """
        
        # Parámetros base
        params = [
            session_id, 
            event_type, 
            prompt_text, 
            anime_clicked, 
            anime_id,
            request.headers.get("user-agent", "unknown"),
            request.client.host if request.client else "unknown"
        ]
        
        # Añadir load_time_ms si está presente
        if load_time_ms is not None:
            query = query.format(", load_time_ms", ", %s")
            params.append(load_time_ms)
        else:
            query = query.format("", "")
        
        cursor.execute(query, params)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error logging metric: {e}")
        return False

@router.post("/search", summary="Registrar evento de búsqueda")
def log_search_event(request: Request, metric: MetricEvent):
    """Registra evento de búsqueda"""
    success = log_metric_event(
        session_id=metric.session_id,
        event_type="search", 
        request=request,
        prompt_text=metric.prompt_text
    )
    return {"logged": success}

@router.post("/click", summary="Registrar evento de clic en anime")
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

@router.post("/load_time", summary="Registrar evento de tiempo de carga")
def log_load_time_event(request: Request, metric: MetricEvent):
    """Registra evento de tiempo de carga"""
    success = log_metric_event(
        session_id=metric.session_id,
        event_type="load_time",
        request=request,
        prompt_text=metric.prompt_text,
        load_time_ms=metric.load_time_ms
    )
    return {"logged": success}

@router.get("/conversion", summary="Obtener métricas de conversión")
def get_conversion_metrics(days: int = Query(30, description="Días a incluir en el análisis")):
    """Obtiene métricas de conversión de los últimos N días"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="No se pudo conectar a la base de datos")
            
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
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error obteniendo métricas: {error_msg}")
        print(stack_trace)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas: {error_msg}"
        )
