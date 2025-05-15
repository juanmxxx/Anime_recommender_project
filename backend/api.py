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
        # Use the python executable from the .venv environment
        python_executable = r"c:\proyectoIA\.venv\Scripts\python.exe"

        result = subprocess.run(
        [python_executable, "modelFormer.py", keywords, str(top_n)],
        capture_output=True, text=True, encoding="utf-8", cwd=None
        )
        
        output = result.stdout
        error = result.stderr

        # If there is an error, return it as a response
        if result.returncode != 0 or not output.strip():
            return {"error": error or "No output from modelFormer.py"}

        # Output is now JSON, so just parse and return it
        try:
            return json.loads(output)
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw_output": output, "stderr": error}
    except Exception as e:
        return {"error": f"Exception in API: {str(e)}"}