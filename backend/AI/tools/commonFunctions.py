import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

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
