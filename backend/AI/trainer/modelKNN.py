#!/usr/bin/env python3
"""
Anime Embeddings Model Trainer

Este script entrena un modelo de recomendación utilizando los embeddings
de anime almacenados en la base de datos. El modelo entrenado se guarda
para ser utilizado posteriormente en recomendaciones basadas en prompts.

Comandos disponibles:
- train: Entrena un nuevo modelo de recomendación (python hybrid_recommender_fixed.py train)
- delete: Elimina modelos existentes (python hybrid_recommender_fixed.py delete)
"""

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor
import joblib
import time
from tqdm import tqdm
import os
import argparse
import sys
import psutil
from typing import List, Dict, Tuple, Any, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import json

class AnimeEmbeddingModelTrainer:
    def __init__(self, model_dir="AI/model", batch_size=500):
        """
        Inicializa el entrenador de modelos basado en embeddings de anime
        
        Args:
            model_dir: Directorio donde se guardarán los modelos entrenados
            batch_size: Número de animes a procesar por lote
        """
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.conn = None
        
        # Asegurarnos de que el directorio de modelos existe
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Conectar a la base de datos
        self._setup_connection()
        
        # Verificar disponibilidad de pgvector
        self.vector_available = self._is_vector_available()
        if not self.vector_available:
            print("⚠️ pgvector extension not available. Using JSON fallback method.")
        else:
            print("✅ pgvector extension available. Using vector operations for optimal performance.")
    
    def _setup_connection(self):
        """Establece conexión a la base de datos de embeddings"""
        try:
            # Conexión a la BD de embeddings (puerto 5433)
            self.conn = psycopg2.connect(
                host="localhost",
                port="5433",
                database="animeDBEmbeddings",
                user="anime_db",
                password="anime_db"
            )
            print("✅ Connection established with embeddings database (port 5433)")
            
        except Exception as e:
            print(f"❌ Error setting up database connection: {e}")
            sys.exit(1)
    
    def _is_vector_available(self):
        """Comprueba si la extensión vector está disponible"""
        if not self.conn:
            return False
            
        cur = self.conn.cursor()
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone() is not None
        cur.close()
        return result
    
    def get_anime_count(self) -> int:
        """
        Obtiene el número total de animes en la base de datos
        
        Returns:
            Número total de animes
        """
        if not self.conn:
            return 0
            
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM anime")
            count = cur.fetchone()[0]
            cur.close()
            return count
        except Exception as e:
            print(f"❌ Error getting anime count: {e}")
            return 0
    
    def _get_embedding(self, table: str, id_col: str, id_val: int) -> Optional[np.ndarray]:
        """
        Función genérica para recuperar embeddings de cualquier tabla
        
        Args:
            table: Nombre de la tabla que contiene embeddings
            id_col: Nombre de la columna de ID
            id_val: Valor del ID a buscar
            
        Returns:
            Array numpy con el embedding, o None si no se encuentra
        """
        cur = self.conn.cursor()
        
        try:
            if self.vector_available:
                cur.execute(f"""
                    SELECT embedding 
                    FROM {table} 
                    WHERE {id_col} = %s
                """, (id_val,))
                result = cur.fetchone()
                if result:
                    # Ensure we're working with numpy array of floats
                    emb = result[0]
                    if isinstance(emb, (list, np.ndarray)):
                        # Convert to numpy array and ensure float64 type
                        return np.array(emb, dtype=np.float64)
                    elif isinstance(emb, str):
                        # Handle case where embedding is a string
                        try:
                            # Try to parse as JSON first
                            emb_data = json.loads(emb)
                            return np.array(emb_data, dtype=np.float64)
                        except json.JSONDecodeError:
                            # If not JSON, try to parse as space-separated values
                            try:
                                emb_data = [float(x) for x in emb.strip().split()]
                                return np.array(emb_data, dtype=np.float64)
                            except ValueError:
                                print(f"⚠️ Warning: Cannot convert string embedding to floats for {id_col}={id_val}")
                                return None
                    else:
                        print(f"⚠️ Warning: Unexpected embedding format for {id_col}={id_val}: {type(emb)}")
                        return None
                return None
            else:
                cur.execute(f"""
                    SELECT embedding_json 
                    FROM {table} 
                    WHERE {id_col} = %s
                """, (id_val,))
                result = cur.fetchone()
                if result and result[0]:
                    try:
                        # Parse JSON and ensure we get a float array
                        emb_data = json.loads(result[0])
                        if isinstance(emb_data, list):
                            return np.array(emb_data, dtype=np.float64)
                        else:
                            print(f"⚠️ Warning: JSON embedding is not a list for {id_col}={id_val}")
                            return None
                    except (json.JSONDecodeError, TypeError, ValueError) as e:
                        print(f"⚠️ Warning: Invalid embedding data for {id_col}={id_val}: {e}")
                        return None
                return None
        except Exception as e:
            print(f"❌ Error getting embedding from {table} for {id_col}={id_val}: {e}")
            return None
        finally:
            cur.close()
    
    def fetch_anime_embeddings_batch(self, offset: int) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
        """
        Obtiene un lote de embeddings de anime desde la base de datos
        """
        if not self.conn:
            return [], np.array([]), np.array([]), np.array([])
        
        try:
            cur = self.conn.cursor(cursor_factory=DictCursor)
            
            # Obtener información básica de animes en este lote
            cur.execute("""
                SELECT id, romaji_title, english_title, genres, 
                       description, format, average_score, popularity,
                       cover_image_medium, episodes, season_year
                FROM anime
                ORDER BY id
                LIMIT %s OFFSET %s
            """, (self.batch_size, offset))
            
            anime_data = [dict(row) for row in cur.fetchall()]
            
            if not anime_data:
                return [], np.array([]), np.array([]), np.array([])
            
            # Obtener los tres tipos de embeddings
            description_embeddings = []
            genre_embeddings = []
            metadata_embeddings = []
            
            for anime in anime_data:
                anime_id = anime['id']
                
                desc_emb = self._get_embedding('anime_description_embeddings', 'anime_id', anime_id)
                genre_emb = self._get_embedding('anime_genre_embeddings', 'anime_id', anime_id)
                meta_emb = self._get_embedding('anime_metadata_embeddings', 'anime_id', anime_id)
                
                description_embeddings.append(desc_emb)
                genre_embeddings.append(genre_emb)
                metadata_embeddings.append(meta_emb)
            
            # Filtrar animes que no tienen todos los embeddings o contienen embeddings no válidos
            valid_indices = [i for i, (d, g, m) in enumerate(zip(description_embeddings, 
                                                               genre_embeddings, 
                                                               metadata_embeddings)) 
                            if d is not None and g is not None and m is not None 
                            and isinstance(d, np.ndarray) and isinstance(g, np.ndarray) and isinstance(m, np.ndarray)]
            
            if not valid_indices:
                return [], np.array([]), np.array([]), np.array([])
            
            filtered_anime_data = [anime_data[i] for i in valid_indices]
            filtered_desc_embeddings = np.vstack([description_embeddings[i] for i in valid_indices])
            filtered_genre_embeddings = np.vstack([genre_embeddings[i] for i in valid_indices])
            filtered_metadata_embeddings = np.vstack([metadata_embeddings[i] for i in valid_indices])
            
            # Debugging info
            for i in range(min(3, len(valid_indices))):
                idx = valid_indices[i]
                print(f"DEBUG - Anime {anime_data[idx]['id']} embeddings shapes: " + 
                      f"desc={description_embeddings[idx].shape} (type: {description_embeddings[idx].dtype}), " +
                      f"genre={genre_embeddings[idx].shape} (type: {genre_embeddings[idx].dtype}), " +
                      f"meta={metadata_embeddings[idx].shape} (type: {metadata_embeddings[idx].dtype})")
            
            return (filtered_anime_data, 
                   filtered_desc_embeddings, 
                   filtered_genre_embeddings, 
                   filtered_metadata_embeddings)
            
        except Exception as e:
            print(f"❌ Error fetching anime embeddings batch: {e}")
            import traceback
            print(traceback.format_exc())
            return [], np.array([]), np.array([]), np.array([])
    
    def fetch_character_embeddings_for_anime(self, anime_id: int) -> List[np.ndarray]:
        """
        Obtiene los embeddings de personajes para un anime específico
        
        Args:
            anime_id: ID del anime
            
        Returns:
            Lista de embeddings de personajes
        """
        if not self.conn:
            return []
        
        try:
            cur = self.conn.cursor()
            
            # Obtener IDs de personajes asociados a este anime
            cur.execute("""
                SELECT id FROM characters
                WHERE anime_id = %s
            """, (anime_id,))
            
            character_ids = [row[0] for row in cur.fetchall()]
            
            if not character_ids:
                return []
            
            character_embeddings = []
            
            # Obtener embedding para cada personaje
            for char_id in character_ids:
                if self.vector_available:
                    cur.execute("""
                        SELECT embedding 
                        FROM character_embeddings
                        WHERE character_id = %s
                    """, (char_id,))
                    result = cur.fetchone()
                    if result:
                        character_embeddings.append(np.array(result[0]))
                else:
                    cur.execute("""
                        SELECT embedding_json 
                        FROM character_embeddings
                        WHERE character_id = %s
                    """, (char_id,))
                    result = cur.fetchone()
                    if result and result[0]:
                        character_embeddings.append(np.array(json.loads(result[0])))
            
            return character_embeddings
            
        except Exception as e:
            print(f"❌ Error fetching character embeddings for anime {anime_id}: {e}")
            return []
    
    def enrich_anime_data_with_character_embeddings(self, anime_data: List[Dict]) -> Dict[int, List[np.ndarray]]:
        """
        Enriquece los datos de anime con embeddings de personajes
        
        Args:
            anime_data: Lista de metadatos de anime
            
        Returns:
            Diccionario de ID de anime a lista de embeddings de personajes
        """
        anime_to_character_embeddings = {}
        
        for anime in tqdm(anime_data, desc="Fetching character embeddings"):
            anime_id = anime['id']
            char_embeddings = self.fetch_character_embeddings_for_anime(anime_id)
            
            if char_embeddings:
                anime_to_character_embeddings[anime_id] = char_embeddings
        
        return anime_to_character_embeddings
    
    def create_combined_embeddings(self, 
                                  desc_embeddings: np.ndarray,
                                  genre_embeddings: np.ndarray,
                                  metadata_embeddings: np.ndarray,
                                  anime_to_character_embeddings: Dict[int, List[np.ndarray]],
                                  anime_data: List[Dict]) -> np.ndarray:
        """
        Crea embeddings combinados para cada anime
        """
        num_animes = len(anime_data)
        
        # Print debug info for the input arrays
        print(f"DEBUG - Input embeddings shapes and types:")
        print(f"Description: {desc_embeddings.shape}, type: {desc_embeddings.dtype}")
        print(f"Genre: {genre_embeddings.shape}, type: {genre_embeddings.dtype}")
        print(f"Metadata: {metadata_embeddings.shape}, type: {metadata_embeddings.dtype}")
        
        # Ensure all embeddings are float arrays
        if desc_embeddings.dtype != np.float64:
            try:
                desc_embeddings = desc_embeddings.astype(np.float64)
            except ValueError as e:
                print(f"❌ Error converting description embeddings to float64: {e}")
                print(f"First problematic value sample: {desc_embeddings[0][0]}")
                raise
                
        if genre_embeddings.dtype != np.float64:
            genre_embeddings = genre_embeddings.astype(np.float64)
            
        if metadata_embeddings.dtype != np.float64:
            metadata_embeddings = metadata_embeddings.astype(np.float64)
        
        # Ensure all arrays have the same dimensionality
        embedding_dim = desc_embeddings.shape[1]
        combined_embeddings = np.zeros((num_animes, embedding_dim), dtype=np.float64)
        
        # Pesos para combinar diferentes tipos de embeddings
        weights = {
            'description': 0.4,  # Mayor importancia a la descripción
            'genre': 0.3,        # Peso medio para géneros
            'metadata': 0.1,     # Menor peso para metadatos
            'characters': 0.2    # Peso medio para personajes
        }
        
        # Crear embeddings combinados
        for i, anime in enumerate(anime_data):
            # Combinar embeddings con sus respectivos pesos
            combined_embeddings[i] = (
                weights['description'] * desc_embeddings[i] +
                weights['genre'] * genre_embeddings[i] +
                weights['metadata'] * metadata_embeddings[i]
            )
              # Añadir embeddings de personajes si están disponibles
            anime_id = anime['id']
            if anime_id in anime_to_character_embeddings and anime_to_character_embeddings[anime_id]:
                try:
                    # Manejo de diferentes formatos de embeddings de personajes
                    char_embeddings_list = []
                    for char_emb in anime_to_character_embeddings[anime_id]:
                        if isinstance(char_emb, np.ndarray):
                            char_embeddings_list.append(char_emb)
                        elif isinstance(char_emb, str):
                            # Intentar procesar cadenas como '[0.123, 0.456, ...]'
                            if char_emb.startswith('[') and char_emb.endswith(']'):
                                # Eliminar los corchetes y dividir por comas
                                values_str = char_emb[1:-1].split(',')
                                # Convertir a float, manejo de notación científica
                                values = []
                                for val in values_str:
                                    try:
                                        val = val.strip()
                                        if 'e' in val:  # notación científica
                                            values.append(float(val))
                                        else:
                                            values.append(float(val))
                                    except ValueError:
                                        # Ignorar valores que no se pueden convertir
                                        continue
                                char_embeddings_list.append(np.array(values, dtype=np.float64))
                    
                    if char_embeddings_list:
                        char_embeddings = np.array(char_embeddings_list, dtype=np.float64)
                        avg_char_embedding = np.mean(char_embeddings, axis=0)
                        combined_embeddings[i] += weights['characters'] * avg_char_embedding
                except Exception as e:
                    print(f"⚠️ Error al procesar embeddings de personajes para anime {anime_id}: {e}")
                    # Continuar sin embeddings de personajes si hay error
        
        # Normalizar embeddings para que tengan norma unitaria
        return normalize(combined_embeddings, norm='l2', axis=1)
    
    def train_nearest_neighbors_model(self, combined_embeddings: np.ndarray) -> NearestNeighbors:
        """
        Entrena un modelo de vecinos más cercanos para recomendaciones
        
        Args:
            combined_embeddings: Matriz de embeddings combinados
            
        Returns:
            Modelo NearestNeighbors entrenado
        """
        print("🔄 Training nearest neighbors model...")
        
        # Crear y entrenar el modelo de vecinos más cercanos usando similitud del coseno
        nn_model = NearestNeighbors(
            n_neighbors=25,  # Número de vecinos a considerar
            algorithm='auto',
            metric='cosine'  # Similitud del coseno
        )
        
        nn_model.fit(combined_embeddings)
        
        print("✅ Nearest neighbors model trained successfully")
        return nn_model
    
    def save_model_and_data(self, 
                           nn_model: NearestNeighbors, 
                           anime_data: List[Dict], 
                           combined_embeddings: np.ndarray):
        """
        Guarda el modelo entrenado y los datos asociados para uso futuro
        
        Args:
            nn_model: Modelo de vecinos más cercanos entrenado
            anime_data: Lista de metadatos de anime
            combined_embeddings: Matriz de embeddings combinados
        """
        # Crear un diccionario de ID a índice para referencia rápida
        anime_id_to_index = {anime['id']: i for i, anime in enumerate(anime_data)}
        
        # Guardar el modelo entrenado
        model_path = os.path.join(self.model_dir, 'anime_nn_model.pkl')
        joblib.dump(nn_model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Guardar los datos de anime para referencia
        anime_data_path = os.path.join(self.model_dir, 'anime_data.pkl')
        joblib.dump(anime_data, anime_data_path)
        print(f"✅ Anime data saved to {anime_data_path}")
        
        # Guardar los embeddings combinados
        embeddings_path = os.path.join(self.model_dir, 'combined_embeddings.npy')
        np.save(embeddings_path, combined_embeddings)
        print(f"✅ Combined embeddings saved to {embeddings_path}")
        
        # Guardar el mapeo de ID a índice
        mapping_path = os.path.join(self.model_dir, 'anime_id_to_index.pkl')
        joblib.dump(anime_id_to_index, mapping_path)
        print(f"✅ Anime ID to index mapping saved to {mapping_path}")
    
    def optimize_batch_size(self) -> int:
        """
        Determina el tamaño de lote óptimo basado en memoria disponible
        
        Returns:
            Tamaño de lote recomendado
        """
        try:
            # Obtener memoria disponible en bytes y convertir a GB
            available_memory = psutil.virtual_memory().available
            available_memory_gb = available_memory / (1024**3)
            
            # Calcular un tamaño de lote basado en la memoria disponible
            estimated_batch_size = int((available_memory_gb * 0.25 * 1024**2) / 10240)
            
            # Limitar dentro de rangos razonables
            if estimated_batch_size < 100:
                return 100  # Mínimo 100 animes por lote
            elif estimated_batch_size > 2000:
                return 2000  # Máximo 2000 animes por lote
            else:
                return estimated_batch_size
        except:
            return 500  # Valor por defecto si no se puede calcular
    
    def delete_models(self, force=False) -> bool:
        """
        Elimina todos los modelos entrenados en el directorio configurado
        
        Args:
            force: Si es True, elimina sin confirmación adicional
        
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        if not os.path.exists(self.model_dir):
            print(f"❌ No existe el directorio de modelos: {self.model_dir}")
            return False
        
        model_files = [f for f in os.listdir(self.model_dir) 
                      if f.endswith(('.pkl', '.npy'))]
        
        if not model_files:
            print(f"❌ No se encontraron archivos de modelo en {self.model_dir}")
            return False
        
        if not force:
            confirmation = input(f"⚠️ ¿Estás seguro de eliminar {len(model_files)} archivos de modelo? (s/N): ")
            if confirmation.lower() != 's':
                print("❌ Operación cancelada")
                return False
        
        try:
            for file in model_files:
                file_path = os.path.join(self.model_dir, file)
                os.remove(file_path)
                print(f"🗑️ Archivo eliminado: {file}")
            
            print(f"✅ Se eliminaron {len(model_files)} archivos de modelo de {self.model_dir}")
            return True
        except Exception as e:
            print(f"❌ Error al eliminar modelos: {e}")
            return False

    def train_model(self):
        """
        Proceso principal para entrenar el modelo de recomendación
        """
        if not self.conn:
            print("❌ No database connection available")
            return False
        
        try:
            # Obtener el número total de animes para planificar lotes
            total_animes = self.get_anime_count()
            if total_animes == 0:
                print("❌ No animes found in the database")
                return False
            
            print(f"📊 Processing {total_animes} animes in batches of {self.batch_size}")
            
            # Inicializar listas para acumular los datos de todos los lotes
            all_anime_data = []
            all_desc_embeddings = []
            all_genre_embeddings = []
            all_metadata_embeddings = []
            
            # Procesar animes en lotes
            for offset in tqdm(range(0, total_animes, self.batch_size), desc="Processing anime batches"):
                anime_data, desc_embeddings, genre_embeddings, metadata_embeddings = self.fetch_anime_embeddings_batch(offset)
                
                if anime_data:
                    all_anime_data.extend(anime_data)
                    
                    if len(all_desc_embeddings) == 0:
                        all_desc_embeddings = desc_embeddings
                        all_genre_embeddings = genre_embeddings
                        all_metadata_embeddings = metadata_embeddings
                    else:
                        all_desc_embeddings = np.vstack((all_desc_embeddings, desc_embeddings))
                        all_genre_embeddings = np.vstack((all_genre_embeddings, genre_embeddings))
                        all_metadata_embeddings = np.vstack((all_metadata_embeddings, metadata_embeddings))
            
            print(f"✅ Processed {len(all_anime_data)} animes with valid embeddings")
            
            # Enriquecer con embeddings de personajes
            print("🔄 Enriching anime data with character embeddings...")
            anime_to_character_embeddings = self.enrich_anime_data_with_character_embeddings(all_anime_data)
            print(f"✅ Found character embeddings for {len(anime_to_character_embeddings)} animes")
            
            # Crear embeddings combinados
            print("🔄 Creating combined embeddings...")
            combined_embeddings = self.create_combined_embeddings(
                all_desc_embeddings,
                all_genre_embeddings,
                all_metadata_embeddings,
                anime_to_character_embeddings,
                all_anime_data
            )
            print(f"✅ Created combined embeddings with shape {combined_embeddings.shape}")
            
            # Entrenar modelo de vecinos más cercanos
            nn_model = self.train_nearest_neighbors_model(combined_embeddings)
            
            # Guardar modelo y datos
            self.save_model_and_data(nn_model, all_anime_data, combined_embeddings)
            
            print("✅ Model training complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        finally:
            if self.conn:
                self.conn.close()
                print("✅ Database connection closed")

def train_command(args):
    """
    Ejecuta el comando de entrenamiento del modelo
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "🌟 ANIME MODEL TRAINER 🌟")
    print("=" * 80)
    
    start_time = time.time()
    
    trainer = AnimeEmbeddingModelTrainer(
        model_dir=args.model_dir,
        batch_size=args.batch_size
    )
    
    if args.auto:
        optimal_batch = trainer.optimize_batch_size()
        print(f"🔍 Auto-detected optimal batch size: {optimal_batch}")
        trainer.batch_size = optimal_batch
    
    success = trainer.train_model()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total training time: {elapsed:.2f} seconds")
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"📂 Model and data saved to {args.model_dir}/")
        print("📝 You can now use anime_ai_recommender.py to get recommendations")
        return 0
    else:
        print("\n❌ Training failed")
        return 1

def delete_command(args):
    """
    Ejecuta el comando de eliminación de modelos
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "🗑️ ANIME MODEL DELETION 🗑️")
    print("=" * 80)
    
    trainer = AnimeEmbeddingModelTrainer(model_dir=args.model_dir)
    success = trainer.delete_models(force=args.force)
    
    return 0 if success else 1

def main():
    parser = argparse.ArgumentParser(description='Anime Embeddings Model Manager')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: train
    train_parser = subparsers.add_parser('train', help='Entrenar un nuevo modelo de recomendación')
    train_parser.add_argument('--model-dir', type=str, default='AI/model',
                             help='Directorio para guardar modelos entrenados (default: AI/model)')
    train_parser.add_argument('--batch-size', type=int, default=500,
                             help='Cantidad de animes a procesar por lote (default: 500)')
    train_parser.add_argument('--auto', action='store_true',
                             help='Detectar automáticamente parámetros óptimos para entrenar')
    
    # Comando: delete
    delete_parser = subparsers.add_parser('delete', help='Eliminar modelos existentes')
    delete_parser.add_argument('--model-dir', type=str, default='AI/model',
                              help='Directorio de modelos a eliminar (default: AI/model)')
    delete_parser.add_argument('--force', action='store_true',
                              help='Forzar eliminación sin confirmación')
    
    args = parser.parse_args()
    
    # Manejar comandos
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'delete':
        return delete_command(args)
    else:
        # Si no se especifica comando, asumir 'train' para compatibilidad
        print("⚠️ No command specified. Using 'train' command by default.")
        print("💡 Tip: Use 'python hybrid_recommender_fixed.py train --auto' for optimal parameters")
        args.command = 'train'
        args.model_dir = 'AI/model'
        args.batch_size = 500
        args.auto = True
        return train_command(args)

if __name__ == "__main__":
    sys.exit(main())
