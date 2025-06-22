import sys
import os
import json
import traceback
import numpy as np
import joblib
from sqlalchemy import func
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import requests
from decimal import Decimal

# Add AI directory path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import other modules
from models.anime import MetricEntry
from schemas.anime import MetricEvent, RecommendationRequest

# Custom JSON Encoder for handling special types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

class TextEmbedder:
    """
    Clase para convertir texto en embeddings compatibles con el modelo de recomendación
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Inicializa el embedder de texto
        
        Args:
            model_name: El modelo de SentenceTransformers a utilizar
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Modelo de embeddings de texto '{model_name}' cargado correctamente")
            self.use_local_model = True
        except Exception as e:
            print(f"⚠️ No se pudo cargar el modelo local: {e}")
            print("ℹ️ Se utilizará una API alternativa para embeddings")
            self.use_local_model = False
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Convierte un texto en un embedding
        
        Args:
            text: El texto a convertir en embedding
            
        Returns:
            El vector embedding resultante
        """
        if self.use_local_model:
            # Usar el modelo local de SentenceTransformers
            return self.model.encode([text])[0]
        else:
            # Intentar usar una API de embedding alternativa
            try:
                # Ejemplo usando una API ficticia (reemplazar con una real según necesidad)
                response = requests.post(
                    "https://api.embedding-service.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {os.environ.get('EMBEDDING_API_KEY')}"},
                    json={"input": text, "model": "text-embedding-ada-002"}
                )
                
                if response.status_code == 200:
                    embedding = response.json()["data"][0]["embedding"]
                    return np.array(embedding, dtype=np.float32)
                else:
                    raise Exception(f"Error en la API: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Error obteniendo embedding: {e}")
                # Devolver un vector aleatorio como último recurso (no recomendado para producción)
                print("⚠️ Usando embedding aleatorio como fallback")
                return np.random.randn(384)  # dimension común para modelos de embedding

class AnimeRecommender:
    def __init__(self, model_dir="model"):
        """
        Inicializa el recomendador de animes cargando los modelos guardados
        
        Args:
            model_dir: Directorio donde están guardados los modelos
        """
        self.model_dir = model_dir
        self.initialized = False
        
        try:
            # Verificar si existen los archivos necesarios
            required_files = [
                'anime_nn_model.pkl',
                'anime_data.pkl',
                'combined_embeddings.npy',
                'anime_id_to_index.pkl'
            ]
            
            # Buscar el directorio del modelo
            if model_dir is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                possible_locations = [
                    os.path.abspath(os.path.join(script_dir, "../../model")),  # Ruta relativa desde el script
                    os.path.abspath(os.path.join(script_dir, "../model")),     # Una carpeta arriba
                    os.path.abspath(os.path.join(script_dir, "model")),        # En la misma carpeta
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(script_dir)), "model")), # Root del proyecto
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))), "model")), # Un nivel más arriba
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(script_dir)), "AI/model")),
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))), "AI/model"))
                ]
                
                # Encontrar la primera ubicación que exista
                self.model_dir = None
                for location in possible_locations:
                    if os.path.exists(location) and os.path.isdir(location):
                        # Verificar si contiene los archivos necesarios
                        if all(os.path.exists(os.path.join(location, file)) for file in required_files):
                            self.model_dir = location
                            print(f"✓ Encontrado directorio de modelos en: {self.model_dir}")
                            break
                
                if self.model_dir is None:
                    raise FileNotFoundError("No se pudo encontrar el directorio de modelos en ninguna ubicación esperada")
            
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"El archivo {file_path} no existe")
            
            # Cargar modelos y datos
            print("🔄 Cargando modelos y datos...")
            self.nn_model = joblib.load(os.path.join(self.model_dir, 'anime_nn_model.pkl'))
            self.anime_data = joblib.load(os.path.join(self.model_dir, 'anime_data.pkl'))
            self.combined_embeddings = np.load(os.path.join(self.model_dir, 'combined_embeddings.npy'))
            self.anime_id_to_index = joblib.load(os.path.join(self.model_dir, 'anime_id_to_index.pkl'))
            
            # Crear índice inverso
            self.index_to_anime_id = {idx: anime_id for anime_id, idx in self.anime_id_to_index.items()}
            
            # Crear mapeo de título a índice para búsquedas rápidas
            self.title_to_index = {}
            for i, anime in enumerate(self.anime_data):
                if anime.get('romaji_title'):
                    self.title_to_index[anime['romaji_title'].lower()] = i
                if anime.get('english_title') and anime['english_title'] not in ('null', None):
                    self.title_to_index[anime['english_title'].lower()] = i
            
            # Inicializar el embedder de texto
            self.text_embedder = TextEmbedder()
            self.initialized = True
            
            print(f"✅ Modelo cargado correctamente. Datos de {len(self.anime_data)} animes disponibles.")
        except Exception as e:
            import traceback
            print(f"❌ Error al inicializar el recomendador: {e}")
            traceback.print_exc()
            self.initialized = False
    
    def get_recommendations_by_prompt(self, prompt: str, num_recommendations: int = 10) -> List[Dict]:
        """
        Obtiene recomendaciones basadas en un prompt de texto
        
        Args:
            prompt: El texto descriptivo para buscar animes similares
            num_recommendations: Número de recomendaciones a devolver
            
        Returns:
            Lista de animes recomendados
        """
        if not self.initialized:
            raise Exception("El recomendador no está inicializado correctamente")
        
        print(f"🔄 Procesando prompt: '{prompt}'")
        
        # Convertir el prompt a un embedding
        prompt_embedding = self.text_embedder.get_embedding(prompt)
        
        # Normalizar el embedding para que sea compatible con el modelo
        prompt_embedding = normalize(prompt_embedding.reshape(1, -1), norm='l2')[0]
        
        # Asegurar que tiene la misma dimensión que los embeddings del modelo
        model_embedding_dim = self.combined_embeddings.shape[1]
        if prompt_embedding.shape[0] != model_embedding_dim:
            print(f"⚠️ Dimensionalidad incorrecta. Ajustando de {prompt_embedding.shape[0]} a {model_embedding_dim}")
            # Si las dimensiones no coinciden, redimensionar (esto es una aproximación)
            if prompt_embedding.shape[0] > model_embedding_dim:
                prompt_embedding = prompt_embedding[:model_embedding_dim]
            else:
                # Completar con ceros si el embedding es más pequeño
                new_embedding = np.zeros(model_embedding_dim)
                new_embedding[:prompt_embedding.shape[0]] = prompt_embedding
                prompt_embedding = new_embedding
            
            # Renormalizar después del ajuste
            prompt_embedding = normalize(prompt_embedding.reshape(1, -1), norm='l2')[0]
        
        # Buscar los animes más similares
        distances, indices = self.nn_model.kneighbors(
            prompt_embedding.reshape(1, -1), 
            n_neighbors=num_recommendations
        )
        
        # Convertir a lista de recomendaciones
        recommendations = []
        for i, idx in enumerate(indices.flatten()):
            anime = self.anime_data[idx]
            similarity = 1 - distances.flatten()[i]  # Convertir distancia a similitud
            
            # Procesar géneros si es necesario
            genres = anime.get('genres', [])
            if isinstance(genres, str):
                try:
                    genres = json.loads(genres.replace("'", '"'))
                except:
                    genres = [g.strip() for g in genres.strip('[]').split(',')]
            
            # Crear copia limpia del anime
            clean_anime = {
                "id": int(anime.get("id")) if anime.get("id") is not None else None,
                "romaji_title": anime.get("romaji_title"),
                "english_title": anime.get("english_title"),
                "average_score": float(anime.get("average_score")) if anime.get("average_score") is not None else None,
                "popularity": int(anime.get("popularity")) if anime.get("popularity") is not None else None,
                "similarity": float(similarity),
                "description": anime.get("description", "").replace('<br>', ' ').replace('<i>', '').replace('</i>', ''),
                "format": anime.get("format"),
                "episodes": int(anime.get("episodes")) if anime.get("episodes") not in (None, 'N/A') else None,
                "season_year": int(anime.get("season_year")) if anime.get("season_year") not in (None, 'N/A') else None,
                "image_url": anime.get("cover_image_medium"),
                "status": anime.get("status", "UNKNOWN"),
                "genres": genres
            }
            
            recommendations.append(clean_anime)
        
        return recommendations
    
    def get_random_anime(self):
        """
        Obtiene un anime aleatorio de la base de datos
        
        Returns:
            Datos del anime seleccionado
        """
        if not self.initialized:
            raise Exception("El recomendador no está inicializado correctamente")
            
        import random
        random_idx = random.randint(0, len(self.anime_data) - 1)
        return self.anime_data[random_idx]

class AnimeFetcher:
    """
    Class to handle anime recommendations using the AI model
    
    This is a wrapper around the AnimeRecommender to provide a consistent API
    """
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the anime fetcher and recommender system
        
        Args:
            model_dir: Optional directory where models are stored
        """
        self.recommender = AnimeRecommender(model_dir)
        self.model_initialized = self.recommender.initialized
    
    def get_recommendations_by_prompt(self, prompt: str, num_recommendations: int = 10) -> Dict:
        """
        Get anime recommendations based on a text prompt
        
        Args:
            prompt: The descriptive text to search for anime
            num_recommendations: Maximum number of results to return
            
        Returns:
            Dictionary with recommendations
        """
        try:
            if not self.model_initialized:
                return {
                    "success": False,
                    "error": "El modelo de recomendación no está inicializado correctamente"
                }
            
            recommendations = self.recommender.get_recommendations_by_prompt(prompt, num_recommendations)
            
            return {
                "success": True,
                "query": prompt,
                "recommendations": recommendations
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Error getting recommendations by prompt: {str(e)}"
            }
    
    @staticmethod
    def serialize_response(data: Dict) -> str:
        """
        Serialize the recommendation data to JSON
        
        Args:
            data: The recommendation data dictionary
        
        Returns:
            JSON string with serialized data
        """
        return json.dumps(data, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

class AnimeRecommendationService:
    def __init__(self, db=None, model_dir=None):
        """
        Inicializa el servicio de recomendación
        
        Args:
            db: Sesión de base de datos para registrar métricas
            model_dir: Directorio donde están guardados los modelos
        """
        self.db = db
        
        # Intentar ubicar los modelos si no se especifica un directorio
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_model_dirs = [
                os.path.abspath(os.path.join(script_dir, "../../AI/model")),
                os.path.abspath(os.path.join(script_dir, "../../../AI/model")),
                os.path.abspath(os.path.join(script_dir, "../../backend/AI/model")),
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(script_dir)), "AI/model")),
            ]
            
            for dir_path in possible_model_dirs:
                if os.path.exists(dir_path):
                    model_dir = dir_path
                    print(f"✓ Se usará el directorio de modelos: {model_dir}")
                    break
        
        self.fetcher = AnimeFetcher(model_dir)
        if not self.fetcher.model_initialized:
            print(f"⚠️ El servicio de recomendación se ha inicializado pero el modelo no está disponible.")
    
    def get_recommendations(self, keywords: str, top_n: int = 5):
        """
        Get anime recommendations based on keywords
        
        Args:
            keywords: Keywords or description to search animes
            top_n: Maximum number of recommendations to return
        
        Returns:
            Parsed recommendations data
        """
        try:
            # Call recommendation function
            results = self.fetcher.get_recommendations_by_prompt(keywords, top_n)
            
            # Log para debugging
            print(f"✓ Se obtuvieron {len(results.get('recommendations', []))} recomendaciones para: '{keywords}'")
            
            return results
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            print(f"❌ Error in recommendation service: {error_msg}")
            print(stack_trace)
            raise Exception(f"Error in recommendation system: {error_msg}")

class MetricsService:
    def __init__(self, db):
        self.db = db
    
    def record_metric(self, metric: MetricEvent, request=None):
        """
        Record a metric event in the database
        
        Args:
            metric: MetricEvent object with the data to record
            request: FastAPI Request object to extract user_agent and IP
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract request information if available
            user_agent = None
            ip_address = None
            if request:
                user_agent = request.headers.get("user-agent", None)
                # Las IPs se pasan directamente al tipo INET de PostgreSQL
                ip_address = request.client.host if request.client else None
            
            # Verificar y convertir tipos de datos
            anime_id = int(metric.anime_id) if metric.anime_id is not None else None
            load_time_ms = int(metric.load_time_ms) if metric.load_time_ms is not None else None
            
            # Log para debugging
            print(f"⏺️ Registrando métrica: {metric.event_type} - ID de sesión: {metric.session_id}")
            print(f"  - IP: {ip_address}")
            print(f"  - Anime ID: {anime_id}")
            print(f"  - Tiempo de carga: {load_time_ms}ms")
            
            # Create database entry from schema
            db_metric = MetricEntry(
                session_id=metric.session_id,
                event_type=metric.event_type,
                prompt_text=metric.prompt_text,
                anime_clicked=metric.anime_clicked,
                anime_id=anime_id,
                load_time_ms=load_time_ms,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            # Add to database
            self.db.add(db_metric)
            self.db.commit()
            self.db.refresh(db_metric)
            print(f"✓ Métrica registrada correctamente con ID: {db_metric.id}")
            return True
        except Exception as e:
            self.db.rollback()
            print(f"❌ Error recording metric: {e}")
            traceback.print_exc()  # Print full stack trace for debugging
            return False
    
    def get_metrics_summary(self):
        """
        Get summary of recorded metrics
        
        Returns:
            Dictionary with metrics summary
        """
        try:
            total_searches = self.db.query(MetricEntry).filter(MetricEntry.event_type == "search").count()
            total_clicks = self.db.query(MetricEntry).filter(MetricEntry.event_type == "click").count()
            avg_load_time = self.db.query(func.avg(MetricEntry.load_time_ms)).filter(
                MetricEntry.event_type == "load_time"
            ).scalar() or 0
            
            return {
                "total_searches": total_searches,
                "total_clicks": total_clicks,
                "average_load_time_ms": round(float(avg_load_time), 2),
            }
        except Exception as e:
            print(f"Error getting metrics summary: {e}")
            return {
                "error": str(e)
            }
