# Archivo api_direct.py - Usa las funciones directamente en lugar de subprocess
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import sys
import os
import traceback
import pathlib

# Agregar la ruta al directorio AI al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))

# Importar directamente las funciones del recomendador
try:
    from recommendHandler import get_recommendations_json, ImprovedAnimeRecommendationSystem
    # Inicializar el sistema una vez para reutilizarlo en todas las peticiones
    anime_system = ImprovedAnimeRecommendationSystem()
except Exception as e:
    print(f"Error al importar recommendHandler: {e}")
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

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que la API esté funcionando"""
    return {
        "message": "Smart Anime Recommender API está funcionando (modo directo)",
        "usage": "Use /recommend?keywords=your_keywords&top_n=5 para obtener recomendaciones",
        "status": "OK" if anime_system is not None else "Sistema no inicializado"
    }
