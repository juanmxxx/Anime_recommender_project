#!/usr/bin/env python3
"""
S.A.R Launcher - Script de inicio para Smart Anime Recommender
- Inicia los contenedores Docker si no están activos
- Inicia el backend FastAPI y el frontend React
- Gestiona procesos y terminación limpia
"""

import os
import subprocess
import sys
import time
import signal
import json
import tempfile
import threading
import atexit
from pathlib import Path

# Control para evitar terminación múltiple
is_terminating = threading.Event()

def check_docker_running():
    """Verifica si Docker está ejecutándose"""
    try:
        subprocess.run(
            ["powershell.exe", "docker info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def start_docker_containers():
    """Inicia los contenedores Docker necesarios para la aplicación"""
    print("🔄 Verificando contenedores Docker...")
    
    # Ruta al archivo docker-compose.yml
    docker_compose_path = Path(__file__).parent / "backend" / "data"
    
    if not (docker_compose_path / "docker-compose.yml").exists():
        print("❌ Error: No se encontró docker-compose.yml")
        return False
    
    # Verificar si Docker está en ejecución
    if not check_docker_running():
        print("❌ Error: Docker no está en ejecución. Por favor, inicie Docker Desktop.")
        return False
    
    # Verificar si los contenedores ya están activos
    result = subprocess.run(
        ["powershell.exe", "docker ps -q --filter 'name=anime-db'"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print("✅ Contenedores Docker ya están activos")
        return True
    
    # Iniciar contenedores
    print("🔄 Iniciando contenedores Docker...")
    try:
        subprocess.run(
            ["powershell.exe", "docker-compose up -d"],
            cwd=docker_compose_path,
            check=True
        )
        print("✅ Contenedores Docker iniciados correctamente")
        time.sleep(5)  # Dar tiempo para que la base de datos esté lista
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al iniciar contenedores Docker: {e}")
        return False

def start_backend():
    """Inicia el backend con FastAPI"""
    print("🔄 Iniciando backend...")
    
    # Rutas y configuración
    backend_dir = Path(__file__).parent / "backend" / "API"
    venv_python = Path(__file__).parent / ".venv311" / "Scripts" / "python.exe"
    
    # Verificar entorno virtual
    if not venv_python.exists():
        print("❌ Error: No se encontró el entorno virtual")
        return None
    
    # Iniciar proceso
    backend_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "api:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
        cwd=backend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print("✅ Backend iniciado (http://127.0.0.1:8000)")
    time.sleep(2)
    return backend_process

def start_frontend():
    """Inicia el frontend con Vite"""
    print("🔄 Iniciando frontend...")
    
    # Ruta y verificación
    frontend_dir = Path(__file__).parent / "frontend"
    if not (frontend_dir / "package.json").exists():
        print("❌ Error: No se encontró package.json")
        return None
    
    # Iniciar proceso
    frontend_process = subprocess.Popen(
        ["powershell.exe", "-Command", "npm run dev"],
        cwd=frontend_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print("✅ Frontend iniciado (http://localhost:5173)")
    time.sleep(3)
    return frontend_process

def terminate_processes(processes):
    """Termina todos los procesos iniciados"""
    if is_terminating.is_set():
        return
    
    is_terminating.set()
    print("\nDeteniendo el sistema...")
    
    for name, process in processes.items():
        if process:
            try:
                print(f"Terminando {name} (PID: {process.pid})...")
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Error: {e}")
    
    print("✅ Sistema detenido")

def main():
    """Función principal"""
    print("\n=== S.A.R LAUNCHER - SMART ANIME RECOMMENDER ===\n")
    
    # Inicializar procesos
    processes = {'backend': None, 'frontend': None}
    
    # Manejar señales
    signal.signal(signal.SIGINT, lambda sig, frame: terminate_processes(processes))
    atexit.register(lambda: None if is_terminating.is_set() else terminate_processes(processes))
    
    # Iniciar servicios
    if not start_docker_containers():
        return 1
    
    processes['backend'] = start_backend()
    if not processes['backend']:
        return 1
    
    processes['frontend'] = start_frontend()
    if not processes['frontend']:
        terminate_processes({'backend': processes['backend']})
        return 1
    
    # Guardar PIDs para recuperación
    pid_file = os.path.join(tempfile.gettempdir(), "sar_processes.json")
    with open(pid_file, 'w') as f:
        json.dump({name: process.pid for name, process in processes.items() if process}, f)
    
    print("\n=== SISTEMA INICIADO COMPLETAMENTE ===")
    print(f"Backend: PID {processes['backend'].pid}")
    print(f"Frontend: PID {processes['frontend'].pid}")
    print("\n✅ Sistema listo | Ctrl+C para detener")
    
    try:
        # Esperar hasta que un proceso termine o se reciba Ctrl+C
        while all(process.poll() is None for process in processes.values() if process) and not is_terminating.is_set():
            time.sleep(1)
        
        if not is_terminating.is_set():
            print("\n⚠️ Un proceso ha terminado inesperadamente")
            terminate_processes(processes)
            
    except KeyboardInterrupt:
        terminate_processes(processes)
    
    # Dar tiempo para que los mensajes se impriman
    if is_terminating.is_set():
        time.sleep(0.5)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
