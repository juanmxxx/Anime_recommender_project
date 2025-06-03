# Nuevo archivo api.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import pandas as pd
import io
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(keywords: str = Query(...), top_n: int = Query(5)):
    try:
        # Usamos el python executable del entorno virtual
        python_executable = r"c:\proyectoIA\.venv\Scripts\python.exe"        # Usamos el anime_search_engine_fixed.py con formato JSON
        result = subprocess.run(
            [python_executable, "anime_search_engine.py", keywords, "-n", str(top_n), "-f", "json"],
            capture_output=True, text=False, cwd=None
        )
        
        # Handle output encoding safely
        try:
            output = result.stdout.decode('utf-8', errors='replace') if result.stdout else None
            error = result.stderr.decode('utf-8', errors='replace') if result.stderr else None
        except Exception as e:
            return {"error": f"Encoding error: {str(e)}"}
          # Output and error have been decoded in the previous step

        # Verificar si hay salida y manejar casos problemáticos
        if result.returncode != 0:
            return {"error": f"Process returned error code {result.returncode}", "stderr": error}
        
        if output is None:
            return {"error": "No output received from anime_search_engine.py"}
            
        if not output:
            return {"error": "Empty output from anime_search_engine.py", "stderr": error}

        # Output is now JSON, so just parse and return it
        try:
            # Asegurarnos de obtener solo la parte JSON de la salida
            json_output = output.strip()
            # Encontrar el inicio del JSON (primer carácter '{' o '[')
            json_start = json_output.find('{')
            
            # Si encontramos un inicio de JSON, extraer esa parte
            if json_start >= 0:
                json_output = json_output[json_start:]
            
            # Intentar cargar el JSON
            parsed_data = json.loads(json_output)
            return parsed_data
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON: {str(e)}", 
                "raw_output": output[:500] + ("..." if len(output) > 500 else ""), 
                "stderr": error
            }
    except Exception as e:
        return {"error": f"Exception in API: {str(e)}"}

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que la API esté funcionando"""
    return {
        "message": "Smart Anime Recommender API está funcionando",
        "usage": "Use /recommend?keywords=your_keywords&top_n=5 para obtener recomendaciones"
    }
