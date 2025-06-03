# Smart Anime Recommender (S.A.R.) System

Este sistema proporciona recomendaciones de anime basadas en consultas de lenguaje natural, utilizando tecnologías modernas de procesamiento de lenguaje y búsqueda.

## Componentes del Sistema

1. **Frontend**: Interfaz web construida con React + Vite
2. **Backend API**: API REST construida con FastAPI
3. **Motor de Búsqueda**: Procesador de consultas y recomendador basado en ML
4. **Launcher**: Script para iniciar todo el sistema de forma coordinada

## Mejoras Implementadas

En la última versión se han implementado las siguientes mejoras:

1. **Gestión de estado en frontend**:
   - Almacenamiento en localStorage para persistencia entre sesiones
   - Pantalla de bienvenida para nuevos usuarios
   - Botón "Nueva Búsqueda" para limpiar resultados anteriores

2. **Motor de búsqueda mejorado**:
   - Soporte para múltiples formatos de salida (texto y JSON)
   - Manejo de argumentos por línea de comandos
   - Extracción de palabras clave mejorada

3. **API con manejo robusto de errores**:
   - Endpoint raíz para información de API
   - Detección y manejo de problemas de codificación
   - Respuestas estructuradas en JSON

4. **Sistema de diagnóstico**:
   - Herramienta para identificar y solucionar problemas en consultas

5. **Launcher con soporte para versiones mejoradas**:
   - Inicia los componentes fijos del sistema
   - Manejo mejorado de errores y procesos

## Cómo usar el sistema

### Método 1: Launcher automático (recomendado)

Ejecuta el script de launcher para iniciar todo el sistema:

```
python S.A.R_Launcher_v2.py
```

### Método 2: Inicio manual de componentes

1. **Iniciar el backend**:
   ```
   cd backend
   python -m uvicorn api_fixed:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Iniciar el frontend**:
   ```
   cd frontend
   npm run dev
   ```

3. **Abrir en navegador**:
   - Visita http://localhost:5173 en tu navegador

## Solución de problemas

Si encuentras problemas con el sistema:

1. **Diagnóstico del motor de búsqueda**:
   ```
   cd backend
   python diagnostico.py "tu consulta aquí"
   ```

2. **Prueba del sistema completo**:
   ```
   python system_test.py
   ```

3. **Verificación de la API**:
   - Visita http://127.0.0.1:8000/ para comprobar que la API está funcionando
   - Prueba una recomendación directa: http://127.0.0.1:8000/recommend?keywords=action

## Notas importantes

- El sistema requiere un entorno virtual de Python con todas las dependencias instaladas
- Es necesario tener Node.js instalado para el frontend
- Los problemas de codificación en textos con caracteres especiales están siendo manejados, pero pueden aparecer símbolos "�" en algunas descripciones

## Versiones

- **S.A.R_Launcher_v2.py**: Versión actualizada que usa los componentes corregidos
- **S.A.R_Launcher.py**: Versión original (puede contener errores)
- **anime_search_engine_fixed.py**: Motor de búsqueda corregido
- **api_fixed.py**: API con manejo mejorado de errores

---

Desarrollado como proyecto de IA y Procesamiento de Lenguaje Natural
