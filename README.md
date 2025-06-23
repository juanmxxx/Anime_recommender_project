# Manual de Usuario - S.A.R. (Smart Anime Recommender)

## Índice

1. [Introducción](#introducción)
2. [Visión General](#visión-general)
3. [Manual de Usuario](#manual-de-usuario)
   1. [Requisitos del sistema](#31-requisitos-del-sistema)
   2. [Instalación y puesta en marcha](#32-instalación-y-puesta-en-marcha)
   3. [Guía para desarrolladores](#33-guía-para-desarrolladores)
   4. [Uso de la interfaz web](#34-uso-de-la-interfaz-web)
   5. [Realización de búsquedas y filtros](#35-realización-de-búsquedas-y-filtros)
   6. [Interpretación de resultados](#36-interpretación-de-resultados)
   7. [Funcionalidades adicionales](#37-funcionalidades-adicionales)
   8. [Resolución de problemas comunes](#38-resolución-de-problemas-comunes)

## Introducción

S.A.R. (Smart Anime Recommender) es un sistema avanzado de recomendación de anime basado en inteligencia artificial. Este sistema utiliza procesamiento de lenguaje natural y algoritmos de machine learning para ofrecer recomendaciones personalizadas de anime basadas en palabras clave, géneros o descripciones proporcionadas por el usuario.

## Visión General

S.A.R. proporciona una interfaz web intuitiva que permite a los usuarios descubrir nuevos animes basados en sus intereses. El sistema analiza una base de datos con más de 17,000 títulos de anime, cada uno con metadatos detallados como género, sinopsis, calificación, año de emisión y otros atributos relevantes.

## Manual de Usuario

### 1.1. Requisitos del sistema

#### Requisitos mínimos:

* **Sistema operativo**: Windows 10/11, macOS 10.15+, o Linux (distribución moderna)
* **Memoria RAM**: 4GB mínimo, 8GB recomendado
* **Procesador**: Intel Core i3/AMD Ryzen 3 o superior
* **Espacio en disco**: 2GB libres mínimo
* **Navegador**: Chrome 90+, Firefox 90+, Edge 90+ o Safari 14+
* **Conexión a Internet**: Requerida para cargar imágenes de anime

#### Software requerido:

Para ejecutar S.A.R. desde el código fuente, necesitarás:

* **Python**: Versión 3.8+ (preferiblemente 3.11)
* **Node.js**: Versión 14+ (recomendado 16+)
* **npm**: Versión 6+
* **Docker y Docker Compose**: Si deseas utilizar la configuración containerizada
* **PostgreSQL**: Versión 12+ (solo si no utilizas Docker)

### 1.2. Instalación y puesta en marcha

#### A) Instalación manual (para desarrollo)

1. **Clonar el repositorio** (si aplica):
   ```bash
   git clone <url-repositorio>
   cd proyectoIA
   ```

2. **Configurar el entorno virtual de Python**:
   ```bash
   py -3.11 -m venv .venv    
   .venv\Scripts\activate  # En Windows
   # source .venv311/bin/activate  # En macOS/Linux
   pip install -r requirements.txt
   ```

3. **Configurar la base de datos PostgreSQL**:
   * Opción 1: Usar Docker (recomendado)
     ```bash
     cd backend/data
     docker-compose up -d
     ```
   
   * Opción 2: Instalar y configurar PostgreSQL manualmente
     - Instalar PostgreSQL 12+
     - Crear una base de datos llamada `animes`
     - Crear un usuario `anime_db` con contraseña `anime_db`
     - Ejecutar los scripts en `backend/data/init-scripts/`

4. **Instalar dependencias del frontend**:
   ```bash
   cd frontend
   npm install
   ```

5. **Iniciar el sistema completo** con el script launcher:
   ```bash
   python S.A.R_Launcher.py
   ```



Si recibes errores con la base de datos de embeddings, puedes regenerar los embeddings ejecutando:

<h2>Ojo: el funcionamiento no se vera afectado, solo en caso de querer entrenar un nuevo modelo, que se emanara de los embeddings en la base de datos</h2>

Continuamos con la generacion de embeddings para la base de datos

```bash
cd backend/AI/generate-save-embeddings
python generate_improved_embeddings.py
```

Este script verificará la instalación de pgvector y te ofrecerá opciones para instalarlo o continuar sin él.

**Opción: Instalar pgvector manualmente en el contenedor**

Si prefieres instalar pgvector manualmente en el contenedor:

```bash
# Conectarse al contenedor
docker exec -it anime_postgres_embeddings bash

# Instalar dependencias necesarias
apt-get update && apt-get install -y postgresql-server-dev-all gcc make git

# Clonar repositorio pgvector
git clone https://github.com/pgvector/pgvector.git /tmp/pgvector

# Compilar e instalar
cd /tmp/pgvector && make && make install

# Crear extensión en la base de datos
psql -U anime_db -d animeDBEmbeddings -c 'CREATE EXTENSION vector;'
```

##### Opción 3: Usar almacenamiento alternativo sin pgvector

Si prefieres no instalar pgvector, el sistema puede funcionar con almacenamiento alternativo basado en JSON. Sin embargo, algunas funcionalidades de búsqueda semántica podrían ser más lentas.



6. **Acceder a la interfaz web**:
   - Abre tu navegador y visita `http://localhost:5173`




### 1.3. Uso de la interfaz web

La interfaz de S.A.R. ha sido diseñada para ser intuitiva y fácil de usar:

#### Página principal


1. **Barra de navegación**: Contiene el título del proyecto "Smart Anime Recommender".
2. **Área de búsqueda**: Un campo de texto donde puedes ingresar tus palabras clave o descripciones.
3. **Botón de búsqueda**: Presiona "Recommend" para iniciar la búsqueda.
4. **Selector de cantidad**: Elige cuántas recomendaciones quieres ver (Top 5, 10, 20, 50 o 100).
5. **Sugerencias rápidas**: Botones con categorías predefinidas para búsquedas instantáneas.
6. **Fondo decorativo**: Imágenes de anime sutiles que dan ambientación a la interfaz.

### 1.4. Realización de búsquedas y filtros

#### Métodos de búsqueda

1. **Búsqueda por palabras clave**:
   - Escribe términos como "romance comedy", "action adventure", "cyberpunk dystopia"
   - Haz clic en "Recommend" o presiona Enter

2. **Uso de sugerencias rápidas**:
   - Haz clic en cualquiera de las categorías predefinidas:
     - romance comedy
     - action adventure
     - sports
     - fantasy magic
     - slice of life
     - psychological drama

3. **Búsqueda avanzada (por conceptos)**:
   - Puedes describir el tipo de contenido que buscas:
     - "high school students with supernatural powers"
     - "medieval fantasy with strong female protagonist"
     - "dystopian future with philosophical themes"

#### Filtrado de resultados

Después de realizar una búsqueda, puedes:

1. **Ajustar la cantidad de resultados**:
   - Utiliza el selector "Show:" para cambiar entre Top 5, 10, 20, 50 o 100 resultados
   - Los resultados se actualizarán automáticamente

2. **Nueva búsqueda**:
   - Usa el botón "New Search" para limpiar los resultados y realizar una nueva consulta
   - También puedes modificar tu búsqueda anterior y volver a presionar "Recommend"

### 1.5. Interpretación de resultados

Después de realizar una búsqueda, S.A.R. mostrará una lista de animes recomendados:

Resultados de busqueda

Cada tarjeta de anime contiene:

1. **Imagen del anime**: Portada o imagen representativa
2. **Título**: Nombre del anime
3. **Puntuación**: Rating en una escala de 1 a 10 (ejemplo: "Score: 8.76")
4. **Ranking**: Posición en la clasificación general (ejemplo: "Ranking #24")
5. **Porcentaje de coincidencia**: Qué tan bien coincide con tu búsqueda (ejemplo: "Match: 95.2%")
6. **Tipo**: TV, Película, OVA, etc.
7. **Año**: Año de emisión o lanzamiento
8. **Estado**: Si está en emisión o finalizado
9. **Episodios**: Número total de episodios
10. **Géneros**: Etiquetas de género representadas con colores
11. **Sinopsis**: Breve descripción del contenido

#### Detalles adicionales

Al hacer clic en cualquier tarjeta de anime, se abrirá una ventana modal con:

![Detalle de anime](docs/images/anime_detail.png)

1. **Confirmación de visualización**: Pregunta si deseas ver este anime
2. **Botón "Sí"**: Te redirigirá a AnimeFlv para buscar este título
3. **Botón "No"**: Cerrará el modal y volverás a los resultados de búsqueda

### 1.6. Funcionalidades adicionales

#### Modo desarrollador

S.A.R. incluye un modo de depuración para desarrolladores o usuarios avanzados:

1. **Activar modo depuración**:
   - En la esquina inferior derecha hay una opción "Debug Mode"
   - Activa la casilla para habilitar las funciones de depuración

2. **Funciones disponibles en modo depuración**:
   - Ver datos: Muestra la estructura completa del objeto de anime en la consola del navegador
   - Botón Debug: Aparece en las tarjetas de anime para inspeccionar sus datos


#### Persistencia de datos

S.A.R. conserva tu última búsqueda entre sesiones:

1. **Almacenamiento local**: La última búsqueda y resultados se guardan en tu navegador
2. **Reanudación**: Al volver a cargar la página, se recuperará tu última búsqueda

### 1.8. Resolución de problemas comunes

#### Problemas con el desarrollo

##### Problema: Error de pgvector al importar el archivo SQL de embeddings

**Síntomas**:
- Error al restaurar la base de datos de embeddings
- Mensaje de error: "extension 'vector' is not available"
- Error de Adminer: "Uncaught ValueError: PDO::query(): Argument #1 ($query) must not be empty"

**Solución**:
1. Instala pgvector manualmente en el contenedor:
   ```bash
   docker exec -it anime_postgres_embeddings bash
   apt-get update && apt-get install -y postgresql-server-dev-all gcc make git
   git clone https://github.com/pgvector/pgvector.git /tmp/pgvector
   cd /tmp/pgvector && make && make install
   psql -U anime_db -d animeDBEmbeddings -c 'CREATE EXTENSION vector;'
   ```
2. Regenera los embeddings ejecutando:
   ```bash
   python backend/AI/generate-save-embeddings/generate_improved_embeddings.py
   ```

##### Problema: Error al instalar dependencias del frontend

**Síntomas**:
- Mensajes de error en la consola de npm
- El frontend no inicia correctamente

**Solución**:
1. Verifica que tienes Node.js v14+ y npm v6+ instalados
2. Intenta borrar la carpeta `node_modules` y volver a instalar:
   ```bash
   cd frontend
   rm -rf node_modules
   npm install
   ```
3. Si persisten los errores, verifica la compatibilidad de versiones en `package.json`

##### Problema: Los contenedores Docker no inician

**Síntomas**:
- Error al iniciar los contenedores con el launcher
- Mensaje "Docker no está en ejecución"

**Solución**:
1. Asegúrate de que Docker Desktop está iniciado
2. Verifica que no hay otros servicios usando los mismos puertos (5432, 5433, 8000, 5173)
3. Intenta reiniciar Docker
4. Ejecuta `docker system prune` para limpiar recursos no utilizados

#### Problema: El sistema no muestra resultados

**Posibles soluciones**:
- Verifica tu conexión a internet
- Asegúrate de haber escrito al menos una palabra clave en el campo de búsqueda
- Intenta con términos más generales (ejemplo: "action" en lugar de un término muy específico)
- Revisa que el backend esté funcionando correctamente visitando `http://localhost:8000/health`

#### Problema: Las imágenes no se cargan

**Posibles soluciones**:
- Verifica tu conexión a internet
- Las imágenes provienen de servicios externos que podrían estar caídos temporalmente
- S.A.R. mostrará una imagen predeterminada cuando la original no pueda cargarse

#### Problema: Error de conexión al backend

**Posibles soluciones**:
- Verifica que el servidor backend esté en ejecución (debería estar disponible en `http://localhost:8000`)
- Reinicia el backend ejecutando nuevamente `uvicorn api:app --reload` en el directorio `backend/API`
- Comprueba si hay errores en la consola donde se ejecuta el backend
- Asegúrate de que PostgreSQL esté funcionando (especialmente si no usas Docker)

#### Problema: Tiempos de respuesta lentos

**Posibles soluciones**:
- Reduce el número de resultados solicitados (ejemplo: usa "Top 5" en lugar de "Top 100")
- Reinicia el servidor backend
- Si estás usando Docker, asegúrate de que tu sistema tenga recursos suficientes asignados a Docker
- Verifica la conexión a la base de datos ejecutando `http://localhost:8000/health` en tu navegador

#### Problema: El frontend no responde

**Posibles soluciones**:
- Actualiza la página del navegador
- Borra la caché del navegador
- Reinicia el servidor frontend con `npm run dev` en el directorio `frontend`
- Verifica la consola de desarrollador en el navegador para identificar errores específicos

