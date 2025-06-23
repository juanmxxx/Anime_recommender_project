#!/usr/bin/env python3
"""
S.A.R Launcher - Script de inicio para Smart Anime Recommender
-----------------------------------------------------------------
Este launcher ofrece un menú simplificado con tres opciones:

1. Iniciar sistema con bases de datos vacías
   - Inicia contenedores Docker de PostgreSQL
   - Crea bases de datos vacías listas para ser usadas
   - Inicia el backend FastAPI y el frontend React

2. Iniciar sistema con datos precargados desde backup
   - Inicia contenedores Docker de PostgreSQL
   - Restaura los datos desde los archivos de backup
   - Inicia el backend FastAPI y el frontend React

3. Salir
   - Termina el programa sin iniciar nada

Requisitos:
- Docker Desktop debe estar en ejecución
- Los archivos del modelo de IA deben existir en /model
- Archivos de backup (opcional) en /backend/data/backup

Autor: Smart Anime Recommender Team
Última actualización: 2024
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
    print("🔄 Verificando entorno Docker...")
    
    # Rutas a los archivos docker-compose.yml
    db_main_path = Path(__file__).parent / "backend" / "data" / "database"
    db_embeddings_path = Path(__file__).parent / "backend" / "data" / "databaseForEmbeddings"
    
    # Verificar existencia de archivos docker-compose.yml
    if not (db_main_path / "docker-compose.yml").exists():
        print("❌ Error: No se encontró docker-compose.yml para base de datos principal")
        return False
        
    if not (db_embeddings_path / "docker-compose.yml").exists():
        print("❌ Error: No se encontró docker-compose.yml para base de datos de embeddings")
        return False
    
    # Verificar si Docker está en ejecución
    if not check_docker_running():
        print("\n❌ Error: Docker no está en ejecución.")
        print("ℹ️ Por favor, inicie Docker Desktop y vuelva a intentarlo.")
        return False
    
    # Verificar si existen los volúmenes de backup montados en los archivos docker-compose.yml
    backup_dir = Path(__file__).parent / "backend" / "data" / "backup"
    if not backup_dir.exists():
        print(f"⚠️ Advertencia: No se encontró la carpeta de backups: {backup_dir}")
        print("ℹ️ La restauración desde backup podría no funcionar correctamente.")
        
        # Crear directorio de backup si no existe
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Se ha creado el directorio de backup.")
        except Exception as e:
            print(f"❌ Error al crear el directorio de backup: {e}")
    
    # Detener contenedores activos para asegurar un estado limpio
    print("\n🔄 Preparando entorno Docker limpio...")
    try:
        # Detener contenedores específicos por nombre en lugar de usar docker-compose
        subprocess.run(
            ["powershell.exe", "docker stop anime_postgres anime_postgres_embeddings 2>$null"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["powershell.exe", "docker rm anime_postgres anime_postgres_embeddings 2>$null"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ Contenedores anteriores detenidos y eliminados")
    except Exception:
        pass  # Ignorar errores si los contenedores no existían

    # Iniciar base de datos principal
    print("\n🔄 Iniciando contenedor de base de datos principal...")
    try:
        result = subprocess.run(
            ["powershell.exe", "docker-compose up -d"],
            cwd=db_main_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Error al iniciar base de datos principal:")
            print(result.stderr.strip())
            return False
            
        print("✅ Base de datos principal iniciada")
        
        # Esperar a que PostgreSQL esté listo - verificación activa
        print("⏳ Esperando a que la base de datos principal esté lista...")
        for i in range(20):  # Intentar por 20 segundos
            time.sleep(1)
            check = subprocess.run(
                ["powershell.exe", 'docker exec -i anime_postgres pg_isready -h localhost -U anime_db 2>$null'],
                capture_output=True,
                text=True
            )
            if check.returncode == 0:
                print("✅ Base de datos principal lista para conexiones")
                break
                
            # Al final del bucle, mostrar progreso
            if i == 19:
                print("⚠️ Tiempo de espera agotado, continuando de todas formas...")
    except Exception as e:
        print(f"❌ Error al iniciar base de datos principal: {e}")
        return False

    # Iniciar base de datos de embeddings
    print("\n🔄 Iniciando contenedor de base de datos de embeddings...")
    try:
        result = subprocess.run(
            ["powershell.exe", "docker-compose up -d"],
            cwd=db_embeddings_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Error al iniciar base de datos de embeddings:")
            print(result.stderr.strip())
            return False
            
        print("✅ Base de datos de embeddings iniciada")
        
        # Esperar a que PostgreSQL esté listo - verificación activa
        print("⏳ Esperando a que la base de datos de embeddings esté lista...")
        for i in range(20):  # Intentar por 20 segundos
            time.sleep(1)
            check = subprocess.run(
                ["powershell.exe", 'docker exec -i anime_postgres_embeddings pg_isready -h localhost -U anime_db 2>$null'],
                capture_output=True,
                text=True
            )
            if check.returncode == 0:
                print("✅ Base de datos de embeddings lista para conexiones")
                break
                
            # Al final del bucle, mostrar progreso
            if i == 19:
                print("⚠️ Tiempo de espera agotado, continuando de todas formas...")
    except Exception as e:
        print(f"❌ Error al iniciar base de datos de embeddings: {e}")
        return False
    
    # Finalización y verificación final de disponibilidad
    print("\n✅ Contenedores Docker iniciados correctamente")
    return True

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
        [venv_python, "-m", "uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"],
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

def restore_databases_from_backup():
    """Restaura las bases de datos desde los archivos de backup"""
    print("\n🔄 Restaurando bases de datos desde los archivos de backup...")
    
    try:
        # Verificar si existen los archivos de backup
        backup_main = Path(__file__).parent / "backend" / "data" / "backup" / "animeDB.sql"
        backup_embeddings = Path(__file__).parent / "backend" / "data" / "backup" / "animeDBEmbeddings.sql"
        
        success = True
        
        # Restaurar la base de datos principal
        if backup_main.exists():
            print("✓ Archivo de backup encontrado para la base de datos principal")
            print("⏳ Restaurando base de datos principal (puede tardar varios minutos)...")
            
            try:
                # Verificar tamaño y formato del archivo SQL
                file_size_mb = backup_main.stat().st_size / (1024 * 1024)
                print(f"   - Tamaño del archivo: {file_size_mb:.2f} MB")
                
                if file_size_mb < 0.1:
                    print("⚠️ El archivo de backup parece estar vacío o es demasiado pequeño")
                    success = False
                else:
                    # Limpiar el esquema actual
                    subprocess.run(
                        ["powershell.exe", 'docker exec anime_postgres psql -U anime_db -d animeDB -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                    
                    # Verificar que el archivo SQL esté disponible dentro del contenedor
                    check_file = subprocess.run(
                        ["powershell.exe", "docker exec anime_postgres ls -la /backup/animeDB.sql"],
                        capture_output=True,
                        text=True
                    )
                    
                    if "No such file or directory" in check_file.stderr:
                        print("❌ El archivo de backup no está disponible dentro del contenedor")
                        print("   Asegúrate de que el volumen está montado correctamente en docker-compose.yml")
                        success = False
                    else:
                        # Restaurar desde backup
                        print("   - Ejecutando restauración (psql)...")
                        result = subprocess.run(
                            ["powershell.exe", 'docker exec anime_postgres psql -U anime_db -d animeDB -f /backup/animeDB.sql'],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            print("✅ Base de datos principal restaurada correctamente")
                        else:
                            print(f"❌ Error al restaurar la base de datos principal:")
                            print(f"   {result.stderr.strip()[:200]}...")
                            success = False
                    
            except Exception as e:
                print(f"❌ Error al restaurar la base de datos principal: {str(e)}")
                success = False
        else:
            print(f"❌ No se encontró el archivo de backup: {backup_main}")
            success = False
              # Restaurar la base de datos de embeddings
        if backup_embeddings.exists():
            print("\n✓ Archivo de backup encontrado para la base de datos de embeddings")
            print("⏳ Restaurando base de datos de embeddings (puede tardar varios minutos)...")
            
            try:
                # Verificar tamaño y formato del archivo SQL
                file_size_mb = backup_embeddings.stat().st_size / (1024 * 1024)
                print(f"   - Tamaño del archivo: {file_size_mb:.2f} MB")
                
                if file_size_mb < 0.1:
                    print("⚠️ El archivo de backup parece estar vacío o es demasiado pequeño")
                    print("   Se procederá a crear una estructura vacía")
                    will_create_structure = True
                else:                    # Verificar si el archivo tiene contenido válido
                    print("   - Verificando formato del archivo SQL...")
                    valid_content = False
                    try:
                        with open(backup_embeddings, 'r', encoding='utf-8', errors='ignore') as f:
                            content_sample = f.read(1000)
                            if "CREATE TABLE" in content_sample or "INSERT INTO" in content_sample:
                                valid_content = True
                            elif content_sample.strip() == "" or "PDO::query(): Argument #1" in content_sample:
                                print("⚠️ El archivo SQL está vacío o contiene errores")
                                will_create_structure = True
                                
                                # Intentar reparar el archivo SQL
                                print("\n🔄 Se detectaron problemas en el archivo SQL de embeddings")
                                repair_response = input("¿Intentar reparar automáticamente el archivo? (s/n): ")
                                if repair_response.lower() in ['s', 'si', 'sí', 'y', 'yes']:
                                    if attempt_sql_repair(backup_embeddings):
                                        print("✅ Archivo reparado, intentando restaurar nuevamente...")
                                        valid_content = True
                                        will_create_structure = False
                    except Exception as e:
                        print(f"⚠️ No se pudo leer el archivo SQL: {str(e)}")
                        will_create_structure = True
                
                # Limpiar el esquema actual
                print("   - Limpiando esquema existente...")
                subprocess.run(
                    ["powershell.exe", 'docker exec anime_postgres_embeddings psql -U anime_db -d animeDBEmbeddings -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;"'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Verificar que el archivo SQL esté disponible dentro del contenedor
                check_file = subprocess.run(
                    ["powershell.exe", "docker exec anime_postgres_embeddings ls -la /backup/animeDBEmbeddings.sql"],
                    capture_output=True,
                    text=True
                )
                
                if "No such file or directory" in check_file.stderr:
                    print("❌ El archivo de backup no está disponible dentro del contenedor")
                    print("   Revisa la configuración del volumen en docker-compose.yml:")
                    print("   Debe contener una línea como:")
                    print("   volumes:")
                    print("     - ../backup:/backup")
                    will_create_structure = True
                elif not valid_content and file_size_mb >= 0.1:
                    print("⚠️ El archivo SQL existe pero podría no tener un formato válido")
                    will_create_structure = True
                
                # Crear estructura mínima para la DB de embeddings
                print("   - Creando estructura básica de tablas...")
                create_table_result = subprocess.run(
                    ["powershell.exe", '''docker exec anime_postgres_embeddings psql -U anime_db -d animeDBEmbeddings -c "
                    CREATE TABLE IF NOT EXISTS anime_unified_embeddings (
                        id SERIAL PRIMARY KEY,
                        anime_id INTEGER NOT NULL,
                        embedding FLOAT[] NOT NULL
                    );"'''],
                    capture_output=True,
                    text=True
                )
                
                if create_table_result.returncode != 0:
                    print("❌ Error al crear la estructura básica de tablas:")
                    print(create_table_result.stderr)
                    success = False
                else:
                    print("✅ Estructura básica creada correctamente")
                
                # Intentar restaurar desde backup solo si el archivo parece válido
                if valid_content and not will_create_structure:
                    print("   - Ejecutando restauración desde backup...")
                    result = subprocess.run(
                        ["powershell.exe", 'docker exec anime_postgres_embeddings psql -U anime_db -d animeDBEmbeddings -f /backup/animeDBEmbeddings.sql'],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        print("✅ Base de datos de embeddings restaurada correctamente")
                        
                        # Verificar si realmente se importaron datos
                        check_data = subprocess.run(
                            ["powershell.exe", '''docker exec anime_postgres_embeddings psql -U anime_db -d animeDBEmbeddings -t -c "SELECT COUNT(*) FROM anime_unified_embeddings;"'''],
                            capture_output=True,
                            text=True
                        )
                        
                        if check_data.returncode == 0 and check_data.stdout.strip() and int(check_data.stdout.strip()) > 0:
                            print(f"   - Se importaron {check_data.stdout.strip()} registros de embeddings")
                        else:
                            print("⚠️ Se creó la estructura pero no se importaron datos")
                            will_create_structure = True
                    else:
                        print(f"❌ Error al restaurar la base de datos de embeddings:")
                        print(f"   {result.stderr.strip()[:200]}...")
                        will_create_structure = True
                
                # Si no se pudieron importar datos, ofrecer alternativa
                if will_create_structure:
                    print("\n🔄 Implementando solución alternativa para embeddings...")
                    print("   - Se ha creado una estructura básica de tablas vacía")
                    print("   - Para generar los embeddings, necesitarás ejecutar manualmente:")
                    print("     python backend/AI/generate-save-embeddings/generate_improved_embeddings.py")
                    print("     (después de que el sistema esté en funcionamiento y la DB principal tenga datos)")
                    
                    # Verificar que la tabla esté accesible
                    check_access = subprocess.run(
                        ["powershell.exe", '''docker exec anime_postgres_embeddings psql -U anime_db -d animeDBEmbeddings -c "SELECT to_regclass('public.anime_unified_embeddings');"'''],
                        capture_output=True,
                        text=True
                    )
                    
                    if check_access.returncode == 0 and "anime_unified_embeddings" in check_access.stdout:
                        print("✅ Base de datos de embeddings preparada (vacía) y lista para usar")
                    else:
                        print("❌ No se pudo configurar correctamente la base de datos de embeddings")
                        success = False
                    
            except Exception as e:
                print(f"❌ Error al restaurar la base de datos de embeddings: {str(e)}")
                success = False
        else:
            print(f"❌ No se encontró el archivo de backup: {backup_embeddings}")
            success = False
            
        return success
    except Exception as e:
        print(f"❌ Error general al restaurar las bases de datos: {str(e)}")
        return False

def check_database_status():
    """Verifica el estado de las bases de datos y si los datos están cargados"""
    main_db_status = "❓ Desconocido"
    main_db_count = 0
    emb_db_status = "❓ Desconocido"
    emb_db_count = 0
    
    # Verificar base de datos principal
    try:
        # Verificar si el contenedor está en ejecución
        container_check = subprocess.run(
            ["powershell.exe", "docker ps -q --filter 'name=anime_postgres'"],
            capture_output=True,
            text=True
        )
        
        if not container_check.stdout.strip():
            main_db_status = "⚠️ Contenedor no activo"
        else:
            # Usar psql para consultar el número de registros en la tabla anime
            result = subprocess.run(
                ["powershell.exe", 'docker exec anime_postgres psql -h localhost -p 5432 -U anime_db -d animeDB -t -c "SELECT COUNT(*) FROM anime;"'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                count = result.stdout.strip()
                if count and count.isdigit():
                    main_db_count = int(count)
                    if main_db_count > 0:
                        main_db_status = "✅ Activa con datos"
                    else:
                        main_db_status = "ℹ️ Activa sin datos"
                else:
                    main_db_status = "⚠️ Estructura vacía"
            else:
                main_db_status = "⚠️ Error de conexión"
    except Exception:
        main_db_status = "❌ Error"
    
    # Verificar base de datos de embeddings
    try:
        # Verificar si el contenedor está en ejecución
        container_check = subprocess.run(
            ["powershell.exe", "docker ps -q --filter 'name=anime_postgres_embeddings'"],
            capture_output=True,
            text=True
        )
        
        if not container_check.stdout.strip():
            emb_db_status = "⚠️ Contenedor no activo"
        else:
            # Verificar número de embeddings
            emb_result = subprocess.run(
                ["powershell.exe", 'docker exec anime_postgres_embeddings psql -h localhost -p 5432 -U anime_db -d animeDBEmbeddings -t -c "SELECT COUNT(*) FROM anime_unified_embeddings;"'],
                capture_output=True,
                text=True
            )
            
            if emb_result.returncode == 0:
                count = emb_result.stdout.strip()
                if count and count.isdigit():
                    emb_db_count = int(count)
                    if emb_db_count > 0:
                        emb_db_status = "✅ Activa con datos"
                    else:
                        emb_db_status = "ℹ️ Activa sin datos"
                else:
                    # La tabla puede no existir, comprobar
                    table_check = subprocess.run(
                        ["powershell.exe", 'docker exec anime_postgres_embeddings psql -h localhost -p 5432 -U anime_db -d animeDBEmbeddings -t -c "SELECT COUNT(*) FROM pg_tables WHERE tablename = \'anime_unified_embeddings\';"'],
                        capture_output=True,
                        text=True
                    )
                    
                    if table_check.returncode == 0 and table_check.stdout.strip() == "0":
                        emb_db_status = "⚠️ Tabla no encontrada"
                    else:
                        emb_db_status = "⚠️ Estructura vacía"
            else:
                emb_db_status = "⚠️ Error de conexión"
    except Exception:
        emb_db_status = "❌ Error"
    
    # Mostrar resumen
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                   ESTADO DE BASES DE DATOS                  │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ Base de datos principal:    {main_db_status.ljust(40)} │")
    if main_db_count > 0:
        print(f"│ Animes cargados:           {str(main_db_count).ljust(40)} │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ Base de datos embeddings:   {emb_db_status.ljust(40)} │")
    if emb_db_count > 0:
        print(f"│ Vectores cargados:          {str(emb_db_count).ljust(40)} │")
    print("└─────────────────────────────────────────────────────────────┘")
    
    # Mostrar recomendación según estado
    if main_db_status != "✅ Activa con datos" or emb_db_status != "✅ Activa con datos":
        print("\nℹ️ NOTA: Una o ambas bases de datos están vacías o presentan problemas.")
        print("     Algunas funcionalidades del sistema pueden estar limitadas.")

def check_model_exists():
    """Verifica si existen los archivos necesarios del modelo de IA"""
    print("🔄 Verificando archivos del modelo de IA...")
    
    model_dir = Path(__file__).parent / "model"
    
    required_files = [
        "anime_nn_model.pkl",
        "anime_data.pkl",
        "combined_embeddings.npy",
        "anime_id_to_index.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = model_dir / file
        if not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Error: No se encontraron los siguientes archivos del modelo:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nDebes obtener estos archivos antes de iniciar el sistema.")
        print("Opciones para obtener el modelo:")
        print("  1. Descarga los archivos del modelo desde el repositorio externo")
        print("  2. Ejecuta el script de entrenamiento para crear un nuevo modelo")
        print("     python backend/AI/trainer/modelKNN.py")
        return False
    
    print("✅ Modelo de IA verificado correctamente")
    return True

def check_docker_compose_config():
    """Verifica que los archivos docker-compose.yml estén configurados correctamente para la restauración desde backup"""
    print("🔄 Verificando la configuración de Docker Compose...")
    
    # Rutas a los archivos docker-compose.yml
    db_main_path = Path(__file__).parent / "backend" / "data" / "database"
    db_embeddings_path = Path(__file__).parent / "backend" / "data" / "databaseForEmbeddings"
    backup_path = Path(__file__).parent / "backend" / "data" / "backup"
    
    # Verificar que la carpeta de backup exista
    if not backup_path.exists():
        print(f"❌ No se encontró la carpeta de backups: {backup_path}")
        print("   Creando directorio de backup...")
        backup_path.mkdir(parents=True, exist_ok=True)
        print("✅ Directorio de backup creado.")
    
    # Verificar docker-compose.yml de base de datos principal
    if (db_main_path / "docker-compose.yml").exists():
        with open(db_main_path / "docker-compose.yml", "r") as f:
            content = f.read()
            if "/backup:/backup" not in content:
                print("⚠️ La configuración de volúmenes en docker-compose.yml de la base de datos principal no incluye la carpeta de backup")
                print("   Los backups podrían no ser accesibles desde el contenedor.")
                print("   Se recomienda añadir el siguiente volumen en el archivo docker-compose.yml:")
                print("   - ../backup:/backup")
    
    # Verificar docker-compose.yml de base de datos de embeddings
    if (db_embeddings_path / "docker-compose.yml").exists():
        with open(db_embeddings_path / "docker-compose.yml", "r") as f:
            content = f.read()
            if "/backup:/backup" not in content:
                print("⚠️ La configuración de volúmenes en docker-compose.yml de la base de datos de embeddings no incluye la carpeta de backup")
                print("   Los backups podrían no ser accesibles desde el contenedor.")
                print("   Se recomienda añadir el siguiente volumen en el archivo docker-compose.yml:")
                print("   - ../backup:/backup")

def attempt_sql_repair(sql_file_path):
    """Intenta reparar un archivo SQL corrupto o con errores de formato"""
    print(f"🔄 Intentando reparar archivo SQL: {sql_file_path.name}")
    
    if not sql_file_path.exists():
        print("❌ El archivo no existe")
        return False
    
    # Crear un respaldo antes de modificar el archivo
    backup_path = sql_file_path.with_suffix('.sql.bak')
    try:
        import shutil
        shutil.copy2(sql_file_path, backup_path)
        print(f"✓ Backup creado: {backup_path.name}")
    except Exception as e:
        print(f"⚠️ No se pudo crear backup: {e}")
    
    try:
        # Leer el contenido del archivo
        with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Verificar problemas comunes
        if not content.strip():
            print("❌ El archivo está vacío, no se puede reparar")
            return False
        
        # Eliminar caracteres problemáticos al inicio del archivo
        if not content.lstrip().startswith(("--", "/*", "CREATE", "INSERT", "BEGIN", "SET")):
            # Buscar el primer comando SQL válido
            valid_starts = ["--", "/*", "CREATE", "INSERT", "BEGIN", "SET"]
            start_pos = -1
            for marker in valid_starts:
                pos = content.find(marker)
                if pos >= 0 and (start_pos == -1 or pos < start_pos):
                    start_pos = pos
            
            if start_pos > 0:
                print(f"✓ Eliminando {start_pos} caracteres no válidos al inicio")
                content = content[start_pos:]
        
        # Agregar instrucciones necesarias al inicio si faltan
        if not any(content.lstrip().startswith(prefix) for prefix in ["BEGIN", "SET", "CREATE"]):
            print("✓ Agregando encabezado SQL estándar")
            header = "-- Archivo SQL reparado automáticamente\n"
            header += "SET client_encoding = 'UTF8';\n"
            header += "BEGIN;\n\n"
            content = header + content
        
        # Asegurar que hay un COMMIT al final si hay BEGIN
        if "BEGIN" in content and "COMMIT" not in content:
            print("✓ Agregando COMMIT al final del archivo")
            content = content.rstrip() + "\n\nCOMMIT;\n"
        
        # Escribir el contenido reparado
        with open(sql_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Archivo SQL reparado y guardado")
        return True
    
    except Exception as e:
        print(f"❌ Error al intentar reparar el archivo: {str(e)}")
        
        # Intentar restaurar desde backup si existe
        if backup_path.exists():
            try:
                import shutil
                shutil.copy2(backup_path, sql_file_path)
                print("✓ Archivo restaurado desde backup")
            except Exception as e2:
                print(f"❌ No se pudo restaurar desde backup: {e2}")
        
        return False

# Eliminamos la función install_pgvector_in_containers 
# ya que no es necesaria para el flujo simplificado

# Eliminamos las funciones diagnose_embeddings_db y repair_embeddings_database
# ya que no son necesarias para el flujo simplificado

def main():
    """Función principal"""
    print("\n=== S.A.R LAUNCHER - SMART ANIME RECOMMENDER ===\n")
    
    # Inicializar procesos
    processes = {'backend': None, 'frontend': None}
    
    # Manejar señales
    signal.signal(signal.SIGINT, lambda sig, frame: terminate_processes(processes))
    atexit.register(lambda: None if is_terminating.is_set() else terminate_processes(processes))
    
    # Verificar existencia del modelo de IA
    if not check_model_exists():
        print("\n❌ ERROR: No se puede iniciar el sistema sin los archivos del modelo de IA")
        input("\nPresiona Enter para salir...")
        return 1
    
    # Menú simplificado con 3 opciones
    print("\n=== OPCIONES DE INICIO ===")
    print("1. Iniciar sistema con bases de datos vacías")
    print("2. Iniciar sistema con datos precargados desde backup")
    print("3. Salir")
    
    while True:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        # Validar la entrada
        if choice not in ["1", "2", "3"]:
            print("❌ Opción no válida. Por favor, selecciona 1, 2 o 3.")
            continue
            
        # Salir inmediatamente si esa fue la opción elegida
        if choice == "3":
            print("✅ Saliendo del sistema")
            return 0
        
        break  # Salir del bucle si la opción es válida (1 o 2)
      # Verificar la configuración de Docker Compose
    if choice == "2":
        print("\n🔄 Verificando configuración para restauración desde backup...")
        check_docker_compose_config()    # Iniciar los contenedores Docker para ambas opciones (1 y 2)
    print("\n🔄 Iniciando contenedores Docker...")
    if not start_docker_containers():
        print("❌ Error al iniciar los contenedores Docker.")
        input("\nPresiona Enter para salir...")
        return 1
    
    # Si la opción es 2, cargar datos desde backup
    if choice == "2":
        print("\n🔄 Cargando datos desde archivos de backup...")
        restore_result = restore_databases_from_backup()
        
        if not restore_result:
            print("\n⚠️ No se pudieron restaurar completamente los datos desde backup.")
            print("ℹ️ Es posible que algunas funcionalidades no estén disponibles.")
            response = input("\n¿Deseas continuar con el inicio del sistema de todas formas? (s/n): ")
            if response.lower() not in ['s', 'si', 'sí', 'y', 'yes']:
                print("❌ Inicio del sistema cancelado por el usuario.")
                input("\nPresiona Enter para salir...")
                return 1
    
    # Verificar el estado de las bases de datos para información del usuario
    print("\n=== ESTADO DE LAS BASES DE DATOS ===")
    check_database_status()
    
    # Iniciar el backend y frontend
    print("\n=== INICIANDO SERVICIOS ===")
      # Iniciar backend
    print("🔄 Iniciando backend (API FastAPI)...")
    processes['backend'] = start_backend()
    if not processes['backend']:
        print("❌ Error al iniciar el backend")
        input("\nPresiona Enter para salir...")
        return 1
    
    print("\n⏳ Esperando a que el backend esté listo...")
    time.sleep(5)  # Dar tiempo para que el backend se inicie completamente
    
    # Iniciar frontend
    print("🔄 Iniciando frontend (React)...")
    processes['frontend'] = start_frontend()
    if not processes['frontend']:
        print("❌ Error al iniciar el frontend")
        print("🛑 Deteniendo el backend...")
        terminate_processes({'backend': processes['backend']})
        input("\nPresiona Enter para salir...")
        return 1
    
    # Guardar PIDs para recuperación
    pid_file = os.path.join(tempfile.gettempdir(), "sar_processes.json")
    with open(pid_file, 'w') as f:
        json.dump({name: process.pid for name, process in processes.items() if process}, f)
    
    # Mostrar resumen final
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                SISTEMA INICIADO CORRECTAMENTE                │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 🔹 Backend (API):         http://127.0.0.1:8000             │")
    print("│ 🔹 Frontend (Interfaz):   http://localhost:5173             │")
    print("│ 🔹 Documentación API:     http://127.0.0.1:8000/docs        │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("\n✅ Todos los servicios están en ejecución")
    print("✅ Puedes cerrar esta ventana para detener el sistema (Ctrl+C)")
    
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
