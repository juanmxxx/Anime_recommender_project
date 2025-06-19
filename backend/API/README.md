# Smart Anime Recommender API

This directory contains the API structure for the Smart Anime Recommender (S.A.R.) system. The structure follows a standard FastAPI organization pattern.

## Project Structure

```
backend/API/
├── config/             # Configuration files
│   ├── __init__.py
│   ├── database.py     # Database connection configuration
│   └── settings.py     # API settings (CORS, versions, etc.)
├── models/             # SQLAlchemy database models
│   ├── __init__.py
│   └── anime.py        # Database models for animes and metrics
├── schemas/            # Pydantic schemas/validation models
│   ├── __init__.py
│   └── anime.py        # Request/response schemas
├── routers/            # API route definitions
│   ├── __init__.py
│   ├── metrics.py      # Metrics endpoints
│   └── recommendations.py # Recommendation endpoints
├── services/           # Business logic services
│   ├── __init__.py
│   └── anime.py        # Recommendation and metrics services
├── __init__.py
├── api.py              # Legacy entry point (for compatibility)
├── init_db.py          # Database initialization script
└── main.py             # Main application entry point
```

## Running the API

To run the API in development mode:

```
cd c:\proyectoIA
python -m backend.API.main
```

To initialize the database:

```
python -m backend.API.init_db
```

## API Endpoints

- `/recommend` - Anime recommendations
  - `GET /recommend/?keywords=...&top_n=5` - Get recommendations based on keywords
  - `POST /recommend/` - Same as GET but with request body

- `/metrics` - Usage metrics
  - `POST /metrics/search` - Record search event
  - `POST /metrics/click` - Record click event
  - `POST /metrics/load_time` - Record load time event
  - `GET /metrics/summary` - Get metrics summary

## Database Models

The API uses SQLAlchemy ORM for database access. The main models are:

- `MetricEntry` - For tracking usage metrics

## Dependencies

Make sure to install all dependencies from the main project's requirements.txt file.
