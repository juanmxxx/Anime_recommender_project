# Modelos de datos para la API
from pydantic import BaseModel
from typing import Optional

# Modelo para tracking de métricas
class MetricEvent(BaseModel):
    session_id: str
    event_type: str  # 'search', 'click' o 'load_time'
    prompt_text: Optional[str] = None
    anime_clicked: Optional[str] = None
    anime_id: Optional[int] = None
    load_time_ms: Optional[int] = None  # Tiempo de carga en milisegundos

# Modelo para solicitudes de recomendación
class RecommendationRequest(BaseModel):
    keywords: str
    top_n: int = 5
