
# README - S.A.R. (Smart Anime Recommender)

<div align="center">
  <img src="frontend/images/ruby.webp" alt="S.A.R. Logo" width="150" />
  <br>
  <h3>Sistema inteligente de recomendación de anime basado en IA</h3>
</div>

## 📖 Índice

- [Acerca del Proyecto](#-acerca-del-proyecto)
- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Manual de Usuario](#-manual-de-usuario)
- [KPIs y Métricas](#-kpis-y-métricas)
- [Base de Datos](#-base-de-datos)
- [Licencia](#-licencia)

## 🌟 Acerca del Proyecto

S.A.R. (Smart Anime Recommender) es un sistema avanzado de recomendación de anime que utiliza técnicas de procesamiento de lenguaje natural (NLP) y machine learning para ofrecer recomendaciones personalizadas de anime basadas en descripciones textuales de preferencias del usuario.

El sistema analiza una base de datos con más de 17,000 títulos de anime, cada uno con metadatos detallados como género, sinopsis, calificación, año de emisión y otros atributos relevantes.

## ✨ Características

- **Búsqueda por descripción**: Introduce frases como "cyberpunk dystopia" o "slice of life romance" y obtén recomendaciones relevantes
- **Interfaz web intuitiva**: Diseñada para ser atractiva y fácil de usar
- **Sugerencias rápidas**: Categorías predefinidas para exploración inmediata
- **Visualización detallada**: Información completa de cada anime incluyendo sinopsis, puntuación, ranking, episodios y géneros
- **Tracking de métricas**: Sistema integrado de seguimiento de uso y rendimiento
- **Persistencia de datos**: Tus búsquedas recientes se conservan entre sesiones
- **Modo desarrollador**: Herramientas avanzadas de depuración y análisis

## 🏗️ Arquitectura

S.A.R. utiliza una arquitectura moderna dividida en tres componentes principales:

1. **Frontend**: Interfaz web construida con React + Vite
2. **API Backend**: Servidor FastAPI que proporciona endpoints para recomendaciones y métricas
3. **Motor de IA**: Modelo de machine learning para procesamiento de texto y recomendaciones

### Diagrama de Flujo

```
┌──────────┐     ┌────────┐     ┌───────────────┐     ┌─────────────┐
│  Usuario │────►│ React  │────►│ FastAPI       │────►│ PostgreSQL  │
│          │◄────│ (Vite) │◄────│ + Módulos IA  │◄────│   DB        │
└──────────┘     └────────┘     └───────────────┘     └─────────────┘
```

## 🔧 Requisitos

### Requisitos mínimos:

* **Sistema operativo**: Windows 10/11, macOS 10.15+, o Linux (distribución moderna)
* **Memoria RAM**: 4GB mínimo, 8GB recomendado
* **Procesador**: Intel Core i3/AMD Ryzen 3 o superior
* **Espacio en disco**: 2GB libres mínimo
* **Navegador**: Chrome 90+, Firefox 90+, Edge 90+ o Safari 14+

### Software requerido:

* **Python**: Versión 3.8+ (preferiblemente 3.11)
* **Node.js**: Versión 14+ (recomendado 16+)
* **Docker** y **Docker Compose**: Para la configuración containerizada
* **PostgreSQL**: Versión 12+ (solo si no utilizas Docker)

## 🚀 Instalación

### Método 1: Usando S.A.R. Launcher (recomendado)

1. **Configurar el entorno virtual de Python**:
   ```bash
   python -m venv .venv311
   .venv311\Scripts\activate  # En Windows
   # source .venv311/bin/activate  # En macOS/Linux
   pip install -r requirements.txt
   ```

2. **Configurar la base de datos**:
   ```bash
   cd backend/data
   docker-compose up -d
   ```

3. **Instalar dependencias del frontend**:
   ```bash
   cd frontend
   npm install
   ```

4. **Iniciar el sistema completo**:
   ```bash
   python S.A.R_Launcher.py
   ```

### Método 2: Iniciar componentes manualmente

1. **Iniciar la base de datos**:
   ```bash
   cd backend/data
   docker-compose up -d
   ```

2. **Iniciar el backend**:
   ```bash
   cd backend/API
   uvicorn api:app --reload --host 127.0.0.1 --port 8000
   ```

3. **Iniciar el frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

Una vez completados estos pasos, podrás acceder a S.A.R. a través de tu navegador en `http://localhost:5173`.

## 📘 Manual de Usuario

Para obtener información detallada sobre cómo utilizar S.A.R., consulta el [Manual de Usuario](docs/MANUAL_DE_USUARIO.md) completo, que incluye:

- [Requisitos del sistema](docs/MANUAL_DE_USUARIO.md#31-requisitos-del-sistema)
- [Instalación y puesta en marcha](docs/MANUAL_DE_USUARIO.md#32-instalación-y-puesta-en-marcha)
- [Uso de la interfaz web](docs/MANUAL_DE_USUARIO.md#33-uso-de-la-interfaz-web)
- [Realización de búsquedas y filtros](docs/MANUAL_DE_USUARIO.md#34-realización-de-búsquedas-y-filtros)
- [Interpretación de resultados](docs/MANUAL_DE_USUARIO.md#35-interpretación-de-resultados)
- [Funcionalidades adicionales](docs/MANUAL_DE_USUARIO.md#36-funcionalidades-adicionales)
- [Resolución de problemas comunes](docs/MANUAL_DE_USUARIO.md#37-resolución-de-problemas-comunes)

## 📊 KPIs y Métricas

S.A.R. incluye un sistema completo de tracking de métricas para evaluar su rendimiento:

- **Tasa de conversión**: Porcentaje de búsquedas que resultan en clics sobre animes
- **Tiempos de carga**: Medición del rendimiento del sistema
- **Volumen de búsquedas**: Cantidad de consultas realizadas
- **Tasa de clics**: Cantidad de animes seleccionados

Para más información sobre la implementación de KPIs, consulta la [documentación específica](docs/KPI_Implementation.md).

## 💾 Base de Datos

S.A.R. utiliza PostgreSQL para el almacenamiento y gestión de:

- **Dataset de anime**: Más de 17,000 títulos con metadatos completos
- **Métricas de uso**: Datos de interacción de usuarios
- **Estadísticas de rendimiento**: Tiempos de respuesta y carga

La elección de PostgreSQL como sistema de gestión de base de datos está [justificada en detalle aquí](docs/justificacion_postgresql.md).

## 📜 Licencia

Este proyecto está distribuido bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`.

---

<div align="center">
  <p>Desarrollado por Equipo S.A.R. | 2025</p>
</div>