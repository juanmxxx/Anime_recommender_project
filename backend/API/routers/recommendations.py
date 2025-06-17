# Router para las recomendaciones de anime
from fastapi import APIRouter, Query, HTTPException
import json
import traceback
import sys
import os

# Importaciones relativas para mantener una estructura limpia
from ..db.models import RecommendationRequest

# Añadir la ruta al directorio AI para importar las funciones necesarias
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Importaciones desde el módulo AI
from backend.AI.tools.APIprocessor import get_recommendations_from_prompt, CustomJSONEncoder

# Crear el router
router = APIRouter()

@router.get("/", summary="Obtener recomendaciones de anime")
def recommend(keywords: str = Query(..., description="Palabras clave o descripción para buscar animes"), 
             top_n: int = Query(5, description="Número máximo de recomendaciones")):
    """
    Endpoint para obtener recomendaciones de anime basadas en palabras clave o descripciones.
    
    Args:
        keywords: Palabras clave o descripción para buscar animes
        top_n: Número máximo de recomendaciones a devolver
        
    Returns:
        JSON con recomendaciones o mensaje de error
    """
    try:
        print(f"Procesando solicitud de recomendación con keywords: {keywords}")
        
        # Llamar directamente a la función de recomendación
        results = get_recommendations_from_prompt(keywords, top_n)
        json_result = json.dumps(results, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        # Convertir de string JSON a objeto Python
        parsed_data = json.loads(json_result)
        
        print(f"Generadas {len(parsed_data.get('recommendations', []))} recomendaciones")
        return parsed_data
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error en API: {error_msg}")
        print(stack_trace)
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error en el sistema de recomendación: {error_msg}",
                "details": stack_trace
            }
        )

# Endpoint POST para recomendaciones (mantener compatibilidad con API anterior)
@router.post("/", summary="Obtener recomendaciones de anime (POST)")
def get_recommendations(request: RecommendationRequest):
    """Endpoint POST para obtener recomendaciones basadas en palabras clave"""
    return recommend(keywords=request.keywords, top_n=request.top_n)
