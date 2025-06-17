# Punto de entrada principal para la API de S.A.R
# Este archivo ahora es un wrapper que importa la API modular desde api_new.py

# Importamos la app FastAPI principal desde api_new.py

# Esto nos permite mantener compatibilidad con código existente que importa 'app' desde api.py
# mientras internamente usamos la implementación modular más organizada

# Nota:
# La estructura modular permite una mejor organización del código:
# 1. routers/recommendations.py - Contiene la lógica de recomendaciones
# 2. routers/metrics.py - Contiene la lógica de métricas
# 3. db/connection.py - Manejo de la conexión a la base de datos
# 4. db/models.py - Modelos de datos
# 5. config.py - Configuración centralizada

# Si necesitas inicializar algún recurso específico a nivel global, aquí es donde deberías hacerlo.
# Por ejemplo:
# import os
# os.environ["DEBUG"] = "True"


# Punto de entrada principal para la API de S.A.R
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import traceback
import sys
import importlib.util

# Asegurarse que la carpeta raíz del proyecto esté en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Importar configuraciones
from backend.API.config import CORS_CONFIG, API_TITLE, API_DESCRIPTION, API_VERSION

# Importar routers
from backend.API.routers import recommendations, metrics

# Crear la aplicación FastAPI
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    **CORS_CONFIG
)

# Ruta para servir el favicon
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "frontend", "public", "vite.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    
    # Si no encuentra el favicon, devuelve una respuesta vacía (204)
    return {"status_code": 204}

# Incluir los routers
# Mantenemos la compatibilidad con la API anterior usando las mismas rutas
app.include_router(recommendations.router, prefix="/recommend", tags=["recommendations"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])

@app.get("/", include_in_schema=True)
def read_root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "message": f"Smart Anime Recommender API v{API_VERSION} está funcionando",
        "endpoints": {
            "recommendations": "/recommend",
            "metrics": "/metrics"
        },
        "documentation": "/docs",
        "status": "OK"
    }

# Middleware para manejo global de errores
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        error_detail = {
            "error": str(exc),
            "path": request.url.path
        }
        
        # Si estamos en modo depuración, añadimos el traceback
        if os.environ.get("DEBUG", "False").lower() in ("true", "1", "t"):
            error_detail["traceback"] = traceback.format_exc()
            
        return {
            "status_code": 500,
            "detail": error_detail
        }


# Para ejecutar esta API:
# uvicorn backend.API.api:app --reload
# O desde la raíz del proyecto: python -m backend.API.api

if __name__ == "__main__":
    import uvicorn
    # Asegurarse que estamos ejecutando desde la raíz del proyecto
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)