#!/usr/bin/env python3
# filepath: c:\proyectoIA\S.A.R_Launcher.py
"""
S.A.R Launcher - Script para iniciar automáticamente el sistema completo
- Activa el entorno virtual
- Inicia el backend (FastAPI)
- Inicia el frontend (Vite)
- Abre el navegador en la aplicación
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Muestra un banner para el launcher"""
    print("\n" + "="*60)
    print("            S.A.R. LAUNCHER - SISTEMA DE ANIME RECOMENDATION           ")
    print("="*60)
    print("\nIniciando todos los componentes del sistema...\n")

def check_python_env():
    """Verifica la existencia del entorno virtual"""
    venv_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv"))
    python_executable = venv_path / "Scripts" / "python.exe"
    
    if not python_executable.exists():
        print(f"❌ Error: No se encontró el entorno virtual en {venv_path}")
        print("Por favor, crea el entorno virtual con: python -m venv .venv")
        return False
    
    print(f"✅ Entorno virtual encontrado en: {venv_path}")
    return True

def start_backend():
    """Inicia el backend con FastAPI usando uvicorn"""
    print("\n🔄 Iniciando el backend (FastAPI)...")
    
    # Ruta al directorio del backend
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    
    # Comando para iniciar uvicorn (servidor para FastAPI)
    # Usamos el Python del entorno virtual
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    
    # Iniciamos el servidor en un proceso separado
    backend_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "api:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        cwd=backend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE  # Abre en una nueva ventana de consola
    )
    
    print("✅ Backend iniciado en http://127.0.0.1:8000")
    
    # Esperamos un momento para asegurarnos de que el backend se inicie correctamente
    time.sleep(3)
    
    return backend_process

def start_frontend():
    """Inicia el frontend con npm run dev"""
    print("\n🔄 Iniciando el frontend (Vite)...")
    
    # Ruta al directorio del frontend
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    
    # Verificamos que estamos en el directorio correcto
    if not os.path.exists(os.path.join(frontend_dir, "package.json")):
        print(f"❌ Error: No se encontró package.json en {frontend_dir}")
        return None
    
    # Iniciamos npm run dev en un proceso separado
    frontend_process = subprocess.Popen(
        ["powershell.exe", "-Command", "npm run dev"],
        cwd=frontend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE  # Abre en una nueva ventana de consola
    )
    
    print("✅ Frontend iniciado - Vite estará disponible en http://localhost:5173")
    
    # Esperamos un momento para que el frontend se inicie correctamente
    time.sleep(5)
    
    return frontend_process

def open_in_browser():
    """Abre la aplicación en el navegador predeterminado"""
    frontend_url = "http://localhost:5173"
    print(f"\n🌐 Abriendo la aplicación en el navegador: {frontend_url}")

def main():
    """Función principal que coordina todo el proceso de inicio"""
    print_banner()
    
    # Verificar entorno virtual
    if not check_python_env():
        input("\nPresiona Enter para salir...")
        sys.exit(1)
        
    # Iniciar backend
    backend_process = start_backend()
    if not backend_process:
        input("\nPresiona Enter para salir...")
        sys.exit(1)
        
    # Iniciar frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Error al iniciar el frontend")
        backend_process.terminate()  # Cerramos el backend si el frontend falla
        input("\nPresiona Enter para salir...")
        sys.exit(1)
        
    # Abrir la aplicación en el navegador
    open_in_browser()
    
    print("\n✨ Sistema S.A.R. iniciado completamente")
    print("\nPresiona Ctrl+C en esta ventana para detener todos los servicios cuando termines")
    
    try:
        # Mantener el script en ejecución hasta que el usuario presione Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Deteniendo todos los servicios...")
        
        # Detener los procesos
        if frontend_process:
            frontend_process.terminate()
        if backend_process:
            backend_process.terminate()
            
        print("✅ Servicios detenidos correctamente")

if __name__ == "__main__":
    main()