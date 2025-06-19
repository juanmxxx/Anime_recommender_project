from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import traceback

# Import routers
from routers.recommendations import router as recommendations_router
from routers.metrics import router as metrics_router

# Import config
from config import CORS_CONFIG, API_TITLE, API_DESCRIPTION, API_VERSION

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    **CORS_CONFIG
)

# Favicon route
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "frontend", "public", "vite.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    
    # Return empty response (204) if favicon not found
    return {"status_code": 204}

# Include routers
app.include_router(recommendations_router, prefix="/recommend", tags=["recommendations"])
app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])

@app.get("/", include_in_schema=True)
def read_root():
    """Root endpoint to verify the API is working"""
    return {
        "message": f"Smart Anime Recommender API v{API_VERSION} is running",
        "endpoints": {
            "recommendations": "/recommend",
            "metrics": "/metrics"
        },
        "documentation": "/docs",
        "status": "OK"
    }

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        error_detail = {
            "error": str(exc),
            "path": request.url.path
        }
        
        # Add traceback in debug mode
        if os.environ.get("DEBUG", "False").lower() in ("true", "1", "t"):
            error_detail["traceback"] = traceback.format_exc()
            
        return {
            "status_code": 500,
            "detail": error_detail
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
