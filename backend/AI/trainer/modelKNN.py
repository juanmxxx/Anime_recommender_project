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
    def __init__(self, model_dir="../../../model", batch_size=500):
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
            
        # Verificar tablas de personajes
        try:
            self.characters_available = (self._check_table_exists('characters') and 
                                        self._check_table_exists('character_embeddings'))
            if not self.characters_available:
                print("⚠️ Character tables not found. Character embeddings will not be used.")
            else:
                print("✅ Character tables available. Character embeddings will be used if found.")        
                
        except Exception as e:
            print(f"⚠️ Error checking character tables: {e}")
            self.characters_available = False
            print("⚠️ Assuming character tables are not available due to error.")
            
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
            # Configurar autocommit después de establecer la conexión
            self.conn.autocommit = True
            print("✅ Connection established with embeddings database (port 5433)")
            
            # Verificar que la conexión funciona ejecutando una consulta simple
            try:
                cur = self.conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            except Exception as query_e:
                print(f"⚠️ Conexión establecida pero hay un problema al ejecutar consultas: {query_e}")
                self.conn.rollback()
                print("🔄 Se intentó recuperar la conexión con rollback")
            
        except Exception as e:
            print(f"❌ Error setting up database connection: {e}")
            self.conn = None
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
            print("❌ No hay conexión a la base de datos")
            return 0
            
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM anime")
            count = cur.fetchone()[0]
            cur.close()
            return count
        except Exception as e:
            print(f"❌ Error getting anime count: {e}")
            try:
                # Intentar recuperar la conexión si está en un estado de transacción abortada
                print("🔄 Intentando recuperar la conexión...")
                self.conn.rollback()
                
                # Intentar nuevamente con la conexión recuperada
                cur = self.conn.cursor()
                cur.execute("SELECT COUNT(*) FROM anime")
                count = cur.fetchone()[0]
                cur.close()
                print("✅ Consulta recuperada exitosamente")
                return count
            except Exception as inner_e:
                print(f"❌ No se pudo recuperar la consulta: {inner_e}")
                print("🔍 Diagnóstico de tablas:")
                self._diagnose_database_tables()
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
                    emb = result[0]
                    if isinstance(emb, (list, np.ndarray)):
                        return np.array(emb, dtype=np.float64)
                    elif isinstance(emb, str):
                        try:
                            emb_data = json.loads(emb)
                            return np.array(emb_data, dtype=np.float64)
                        except json.JSONDecodeError:
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
            
        if not hasattr(self, 'characters_available') or not self.characters_available:
            # Solo log para el anime 1 como muestra
            if anime_id == 1:
                print(f"ℹ️ Tablas de personajes no disponibles. No se usarán embeddings de personajes.")
            return []
        
        try:
            # Verificar una vez más que las tablas existen (doble verificación)
            if not self._check_table_exists('characters') or not self._check_table_exists('character_embeddings'):
                if anime_id == 1:  # Solo para el primer anime para evitar spam
                    print(f"⚠️ Tablas de personajes no encontradas en una segunda verificación.")
                return []
            
            cur = self.conn.cursor()
            
            # Obtener IDs de personajes asociados a este anime
            try:
                cur.execute("""
                    SELECT id FROM characters
                    WHERE anime_id = %s
                """, (anime_id,))
                
                character_ids = [row[0] for row in cur.fetchall()]
                
                if not character_ids:
                    # Log para algunos animes si no tienen personajes asociados
                    if anime_id % 1000 == 0:
                        print(f"👤 Anime {anime_id}: No tiene personajes registrados")
                    return []
                
                character_embeddings = []
                problematic_chars = 0
                
                for char_id in character_ids:
                    try:
                        if self.vector_available:
                            cur.execute("""
                                SELECT embedding 
                                FROM character_embeddings
                                WHERE character_id = %s
                            """, (char_id,))
                            result = cur.fetchone()
                            if result and result[0] is not None:
                                # Verificar que el embedding no esté vacío
                                emb = result[0]
                                if isinstance(emb, (list, np.ndarray)) and len(emb) > 0:
                                    np_emb = np.array(emb, dtype=np.float64)
                                    if np_emb.size > 0:  # Verificar que no esté vacío
                                        character_embeddings.append(np_emb)
                                    else:
                                        problematic_chars += 1
                        else:
                            cur.execute("""
                                SELECT embedding_json 
                                FROM character_embeddings
                                WHERE character_id = %s
                            """, (char_id,))
                            result = cur.fetchone()
                            if result and result[0]:
                                try:
                                    # Parse JSON and ensure we get a non-empty float array
                                    emb_data = json.loads(result[0])
                                    if isinstance(emb_data, list) and len(emb_data) > 0:
                                        np_emb = np.array(emb_data, dtype=np.float64)
                                        if np_emb.size > 0:  # Verificar que no esté vacío
                                            character_embeddings.append(np_emb)
                                        else:
                                            problematic_chars += 1
                                except (json.JSONDecodeError, TypeError, ValueError) as e:
                                    problematic_chars += 1
                    except Exception as char_e:
                        if anime_id in [163133, 163134, 185880]:  # Log especial para animes problemáticos
                            print(f"⚠️ Error procesando embedding para personaje {char_id} del anime {anime_id}: {char_e}")
                        problematic_chars += 1
                        continue
                
                if anime_id in [163133, 163134, 185880] and problematic_chars > 0:
                    print(f"📊 Anime {anime_id}: {len(character_embeddings)} embeddings de personajes válidos, {problematic_chars} problemáticos")
                
                return character_embeddings
                
            except Exception as query_e:
                print(f"❌ Error en consulta de personajes para anime {anime_id}: {query_e}")
                self.conn.rollback()
                return []
                
        except Exception as e:
            print(f"❌ Error fetching character embeddings for anime {anime_id}: {e}")
            return []
        finally:
            if 'cur' in locals() and cur:
                cur.close()
    
    def enrich_anime_data_with_character_embeddings(self, anime_data: List[Dict]) -> Dict[int, List[np.ndarray]]:
        """
        Enriquece los datos de anime con embeddings de personajes
        
        Args:
            anime_data: Lista de metadatos de anime
            
        Returns:
            Diccionario de ID de anime a lista de embeddings de personajes
        """
        if not hasattr(self, 'characters_available') or not self.characters_available:
            print("ℹ️ Las tablas de personajes no están disponibles. Saltando procesamiento de embeddings de personajes.")
            return {}
            
        # Verificar una vez más directamente si las tablas existen
        if not self._check_table_exists('characters') or not self._check_table_exists('character_embeddings'):
            print("⚠️ Verificación adicional: Las tablas de personajes no existen. Saltando procesamiento.")
            return {}
            
        anime_to_character_embeddings = {}
        animes_with_chars = 0
        animes_without_chars = 0
        total_characters = 0
        sample_checks = 0
        
        # Verificar una muestra antes de procesar todo
        if anime_data:
            print("🔍 Verificando muestra de personajes antes de procesar...")
            try:
                # Probar con el primer anime
                sample_anime_id = anime_data[0]['id']
                cur = self.conn.cursor()
                
                # Verificar si hay personajes para este anime
                cur.execute("SELECT COUNT(*) FROM characters WHERE anime_id = %s", (sample_anime_id,))
                sample_chars_count = cur.fetchone()[0]
                
                # Verificar si hay embeddings para algún personaje de este anime
                if sample_chars_count > 0:
                    cur.execute("""
                        SELECT COUNT(*) FROM character_embeddings ce
                        JOIN characters c ON ce.character_id = c.id
                        WHERE c.anime_id = %s
                    """, (sample_anime_id,))
                    sample_embs_count = cur.fetchone()[0]
                    
                    print(f"📋 Muestra anime {sample_anime_id}: {sample_chars_count} personajes, {sample_embs_count} con embeddings")
                    
                    # Si hay embeddings, obtener una muestra para verificar formato
                    if sample_embs_count > 0:
                        if self.vector_available:
                            cur.execute("""
                                SELECT ce.embedding FROM character_embeddings ce
                                JOIN characters c ON ce.character_id = c.id
                                WHERE c.anime_id = %s
                                LIMIT 1
                            """, (sample_anime_id,))
                        else:
                            cur.execute("""
                                SELECT ce.embedding_json FROM character_embeddings ce
                                JOIN characters c ON ce.character_id = c.id
                                WHERE c.anime_id = %s
                                LIMIT 1
                            """, (sample_anime_id,))
                            
                        sample_emb = cur.fetchone()
                        if sample_emb and sample_emb[0]:
                            print(f"✅ Se encontró un embedding de muestra. Tipo: {type(sample_emb[0]).__name__}")
                        else:
                            print("⚠️ No se pudo obtener un embedding de muestra.")
                cur.close()
            except Exception as e:
                print(f"❌ Error al verificar muestra de personajes: {e}")
                return {}
        
        for anime in tqdm(anime_data, desc="Fetching character embeddings"):
            anime_id = anime['id']
            char_embeddings = self.fetch_character_embeddings_for_anime(anime_id)
            
            if char_embeddings:
                anime_to_character_embeddings[anime_id] = char_embeddings
                animes_with_chars += 1
                total_characters += len(char_embeddings)
            else:
                animes_without_chars += 1
                
            # Hacer verificaciones más detalladas para un subconjunto de animes
            if sample_checks < 20 and (animes_with_chars + animes_without_chars) % 200 == 0:
                sample_checks += 1
                # Verificar si existen personajes en la tabla de characters
                try:
                    cur = self.conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM characters WHERE anime_id = %s", (anime_id,))
                    chars_count = cur.fetchone()[0]
                    
                    # Verificar si existen embeddings para esos personajes
                    char_emb_count = 0
                    if chars_count > 0:
                        cur.execute("""
                            SELECT COUNT(*) FROM character_embeddings e
                            JOIN characters c ON e.character_id = c.id
                            WHERE c.anime_id = %s
                        """, (anime_id,))
                        char_emb_count = cur.fetchone()[0]
                    
                    print(f"🔍 Revisión anime {anime_id}: {chars_count} personajes en DB, {char_emb_count} con embedding, {len(char_embeddings)} embeddings válidos")
                    cur.close()
                except Exception as e:
                    print(f"❌ Error al verificar personajes para anime {anime_id}: {e}")
                    try:
                        self.conn.rollback()  # Recuperar de posibles errores de transacción
                    except:
                        pass
        
        print(f"\n📊 Resumen de personajes:")
        print(f"   - {animes_with_chars} animes con embeddings de personajes ({total_characters} personajes en total)")
        print(f"   - {animes_without_chars} animes sin embeddings de personajes")
        
        if animes_with_chars == 0:
            print("\n⚠️ ADVERTENCIA: No se encontraron embeddings de personajes. Ejecutando diagnóstico completo...")
            self.diagnose_character_embeddings()
        
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
                weights['metadata'] * metadata_embeddings[i]            )
              # Añadir embeddings de personajes si están disponibles
            anime_id = anime['id']
            # Saltar directamente animes problemáticos conocidos para evitar errores
            if anime_id in [163133, 163134]:
                print(f"⚠️ Saltando anime problemático conocido {anime_id}")
                continue
                
            if anime_id in anime_to_character_embeddings and anime_to_character_embeddings[anime_id] and len(anime_to_character_embeddings[anime_id]) > 0:
                try:
                    # Manejo de diferentes formatos de embeddings de personajes
                    char_embeddings_list = []
                    valid_embeddings_count = 0
                    invalid_embeddings_count = 0
                    
                    for char_emb in anime_to_character_embeddings[anime_id]:
                        try:
                            # Saltar embeddings nulos o vacíos
                            if char_emb is None:
                                invalid_embeddings_count += 1
                                continue
                                
                            if isinstance(char_emb, np.ndarray):
                                # Verificar que el array tenga la dimensión correcta y no esté vacío
                                if char_emb.size > 0 and char_emb.size == embedding_dim:
                                    char_embeddings_list.append(char_emb)
                                    valid_embeddings_count += 1
                                else:
                                    print(f"⚠️ Embeddings de personaje con dimensión incorrecta para anime {anime_id}: {char_emb.shape}, tamaño: {char_emb.size}")
                                    invalid_embeddings_count += 1
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
                                            # Ignorar valores vacíos
                                            if not val:
                                                continue
                                            values.append(float(val))
                                        except ValueError as e:
                                            # Log detallado para depurar problemas de conversión
                                            if anime_id in [163133, 163134, 185880]:  # Log especial para animes problemáticos
                                                print(f"⚠️ Valor problemático para anime {anime_id}: '{val}' - {e}")
                                            # Ignorar valores que no se pueden convertir
                                            continue
                                    
                                    # Solo crear el array si hay valores válidos y suficientes
                                    if values and len(values) == embedding_dim:
                                        char_embeddings_list.append(np.array(values, dtype=np.float64))
                                        valid_embeddings_count += 1
                                    elif values:
                                        print(f"⚠️ Dimensión incorrecta en embedding de personaje para anime {anime_id}: {len(values)} vs {embedding_dim} esperados")
                                        invalid_embeddings_count += 1
                                else:
                                    invalid_embeddings_count += 1
                            else:
                                print(f"⚠️ Tipo de embedding no soportado para anime {anime_id}: {type(char_emb)}")
                                invalid_embeddings_count += 1
                        
                        except Exception as inner_e:
                            print(f"⚠️ Error al procesar embedding individual para anime {anime_id}: {inner_e}")
                            invalid_embeddings_count += 1
                            continue
                    
                    # Registrar estadísticas de embeddings para animes problemáticos
                    if anime_id in [163133, 163134, 185880] or invalid_embeddings_count > 0:
                        print(f"📊 Estadísticas para anime {anime_id}: {valid_embeddings_count} embeddings válidos, {invalid_embeddings_count} inválidos")
                    
                    if char_embeddings_list:
                        # Verificar que todas las listas tengan la misma dimensión
                        dims = [emb.shape[0] for emb in char_embeddings_list]
                        if len(set(dims)) > 1:
                            print(f"⚠️ Dimensiones inconsistentes en embeddings de personajes para anime {anime_id}: {dims}")
                            # Usar solo embeddings con la dimensión correcta
                            char_embeddings_list = [emb for emb in char_embeddings_list if emb.shape[0] == embedding_dim]
                        
                        if char_embeddings_list:  # Verificar de nuevo si quedan embeddings válidos
                            char_embeddings = np.array(char_embeddings_list, dtype=np.float64)
                            avg_char_embedding = np.mean(char_embeddings, axis=0)
                            # Solo añadir la contribución si el embedding tiene la dimensión correcta
                            if avg_char_embedding.shape[0] == embedding_dim:
                                combined_embeddings[i] += weights['characters'] * avg_char_embedding
                            else:
                                print(f"⚠️ Dimensión incorrecta en embedding promedio de personajes para anime {anime_id}")
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
            
    def _check_table_exists(self, table_name):
        """Verifica si una tabla existe en la base de datos"""
        if not self.conn:
            return False
            
        cur = self.conn.cursor()
        try:
            cur.execute("""
                SELECT to_regclass('public.%s')
            """, (table_name,))
            exists = cur.fetchone()[0] is not None
            cur.close()
            return exists
        except Exception as e:
            print(f"⚠️ Error al verificar si la tabla '{table_name}' existe: {e}")
            try:
                # Intentar recuperar la conexión si está en un estado de transacción abortada
                self.conn.rollback()
                print("🔄 Se intentó recuperar la conexión con rollback")
            except Exception:
                pass
            cur.close()
            return False

    def _diagnose_database_tables(self):
        """
        Diagnostica problemas con las tablas de la base de datos
        
        Verifica si existen las tablas necesarias y prueba consultas simples
        """
        if not self.conn:
            print("❌ No hay conexión a la base de datos")
            return
        
        # Lista de tablas principales a verificar
        tables = ['anime', 'description_embeddings', 'genre_embeddings', 'metadata_embeddings', 
                 'characters', 'character_embeddings']
        
        print("\n🔍 DIAGNÓSTICO DE TABLAS DE BASE DE DATOS")
        print("=" * 50)
        
        cursor = self.conn.cursor()
        
        # Verificar cada tabla
        for table in tables:
            try:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    )
                """, (table,))
                exists = cursor.fetchone()[0]
                print(f"Tabla '{table}': {'✅ Existe' if exists else '❌ No existe'}")
                
                # Si la tabla existe, cuenta registros
                if exists:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        print(f"  - Registros: {count}")
                    except Exception as e:
                        print(f"  - Error al contar registros: {e}")
            except Exception as e:
                print(f"❌ Error al verificar tabla '{table}': {e}")
                
        cursor.close()
        print("=" * 50)

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
            
            # Si no se encontraron embeddings de personajes, ejecutar diagnóstico
            if len(anime_to_character_embeddings) == 0:
                self.diagnose_character_embeddings()
            
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

    def diagnose_character_embeddings(self):
        """
        Diagnostica problemas con los embeddings de personajes
        
        Verifica si existen las tablas necesarias y cuenta registros
        """
        if not self.conn:
            print("❌ No hay conexión a la base de datos")
            return
        
        try:
            self.conn.rollback()  # Asegurar que no hay transacciones pendientes
            cur = self.conn.cursor()
            tables_to_check = ['characters', 'character_embeddings', 'anime']
            tables_exist = {}
            
            print("\n🔍 DIAGNÓSTICO DE EMBEDDINGS DE PERSONAJES")
            print("=" * 50)
            
            # Verificar qué tablas existen
            for table in tables_to_check:
                try:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        )
                    """, (table,))
                    exists = cur.fetchone()[0]
                    tables_exist[table] = exists
                    print(f"Tabla '{table}': {'✅ Existe' if exists else '❌ No existe'}")
                except Exception as e:
                    tables_exist[table] = False
                    print(f"Tabla '{table}': ❌ Error al verificar: {e}")
            
            # Si no existen las tablas necesarias, terminar el diagnóstico
            if not tables_exist.get('characters') or not tables_exist.get('character_embeddings'):
                print("\n❌ Las tablas necesarias no existen. No se pueden obtener embeddings de personajes.")
                return
            
            # Contar registros en las tablas
            for table in tables_to_check:
                if tables_exist.get(table):
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        print(f"Registros en '{table}': {count}")
                    except Exception as e:
                        print(f"Error al contar registros en '{table}': {e}")
            
            # Verificar relaciones entre tablas
            try:
                cur.execute("""
                    SELECT COUNT(*) FROM characters c
                    JOIN anime a ON c.anime_id = a.id
                """)
                related_chars = cur.fetchone()[0]
                print(f"Personajes relacionados con animes: {related_chars}")
            except Exception as e:
                print(f"Error al verificar relación characters-anime: {e}")
            
            try:
                cur.execute("""
                    SELECT COUNT(*) FROM character_embeddings ce
                    JOIN characters c ON ce.character_id = c.id
                """)
                related_embs = cur.fetchone()[0]
                print(f"Embeddings relacionados con personajes: {related_embs}")
            except Exception as e:
                print(f"Error al verificar relación character_embeddings-characters: {e}")
            
            # Verificar la estructura de las columnas
            print("\n🔍 ESTRUCTURA DE TABLAS:")
            try:
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'characters'
                """)
                print("\nColumnas de la tabla 'characters':")
                for col in cur.fetchall():
                    print(f"  - {col[0]}: {col[1]}")
            except Exception as e:
                print(f"Error al verificar columnas de 'characters': {e}")
                
            try:
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'character_embeddings'
                """)
                print("\nColumnas de la tabla 'character_embeddings':")
                for col in cur.fetchall():
                    print(f"  - {col[0]}: {col[1]}")
            except Exception as e:
                print(f"Error al verificar columnas de 'character_embeddings': {e}")
            
            # Revisar estructura de los embeddings
            try:
                if self.vector_available:
                    column_name = "embedding"
                else:
                    column_name = "embedding_json"
                    
                cur.execute(f"""
                    SELECT character_id, {column_name} FROM character_embeddings 
                    WHERE character_id IN (
                        SELECT id FROM characters LIMIT 3
                    )
                    LIMIT 3
                """)
                sample_rows = cur.fetchall()
                print(f"\nMuestra de embeddings ({len(sample_rows)} filas):")
                
                for row in sample_rows:
                    char_id = row[0]
                    emb = row[1]
                    emb_type = type(emb).__name__
                    emb_len = len(emb) if hasattr(emb, "__len__") else "N/A"
                    print(f"  Character ID {char_id}: Tipo {emb_type}, Longitud {emb_len}")
            except Exception as e:
                print(f"Error al revisar estructura de embeddings: {e}")
            
            print("\n📋 CONCLUSIÓN:")
            if tables_exist.get('characters') and tables_exist.get('character_embeddings'):
                if related_embs > 0:
                    print("✅ La estructura de la base de datos parece correcta para obtener embeddings de personajes.")
                    print("   Si no se encuentran embeddings durante el entrenamiento, revisa los mensajes de error anteriores.")
                else:
                    print("⚠️ Las tablas existen pero no hay relaciones entre character_embeddings y characters.")
                    print("   Es posible que la tabla character_embeddings esté vacía o que las claves no coincidan.")
            else:
                print("❌ Faltan tablas necesarias. No se pueden obtener embeddings de personajes.")
            
            print("=" * 50)
        except Exception as e:
            print(f"❌ Error general durante el diagnóstico: {e}")
        finally:
            if 'cur' in locals() and cur:
                cur.close()
    
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

def evaluate_command(args):
    """
    Ejecuta el comando de evaluación del modelo
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "📊 ANIME MODEL EVALUATION 📊")
    print("=" * 80)
    
    try:
        import sys
        import os
        
        # Ruta al script de evaluación de métricas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        evaluate_script_path = os.path.join(script_dir, "..", "evaluate_metrics", "enhanced_evaluate_metrics.py")
        
        if not os.path.exists(evaluate_script_path):
            print(f"❌ Script de evaluación no encontrado en {evaluate_script_path}")
            return 1
        
        print(f"🔄 Ejecutando script de evaluación desde {evaluate_script_path}")
        
        # Crear el comando para ejecutar el script de evaluación
        from subprocess import run
        
        cmd = [
            sys.executable,
            evaluate_script_path,
            "--num-prompts", str(args.num_prompts),
            "--output-dir", "enhanced_metrics_results"
        ]
        
        # Ejecutar el script de evaluación como subproceso
        result = run(cmd, check=False)
        
        if result.returncode == 0:
            print("✅ Evaluación completada con éxito")
            return 0
        else:
            print(f"❌ Evaluación fallida con código de salida {result.returncode}")
            return result.returncode
    
    except Exception as e:
        print(f"❌ Error al ejecutar la evaluación: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Anime Embeddings Model Manager')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: train
    train_parser = subparsers.add_parser('train', help='Entrenar un nuevo modelo de recomendación')
    train_parser.add_argument('--model-dir', type=str, default='../../../model',
                             help='Directorio para guardar modelos entrenados (default: /model)')
    train_parser.add_argument('--batch-size', type=int, default=500,
                             help='Cantidad de animes a procesar por lote (default: 500)')
    train_parser.add_argument('--auto', action='store_true',
                             help='Detectar automáticamente parámetros óptimos para entrenar')
    
    # Comando: delete
    delete_parser = subparsers.add_parser('delete', help='Eliminar modelos existentes')
    delete_parser.add_argument('--model-dir', type=str, default='../../../model',
                              help='Directorio de modelos a eliminar (default: /model)')
    delete_parser.add_argument('--force', action='store_true',
                              help='Forzar eliminación sin confirmación')
        # Comando: evaluate
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluar el rendimiento del modelo')
    evaluate_parser.add_argument('--model-dir', type=str, default='../../../model',
                               help='Directorio que contiene el modelo a evaluar (default: /model)')
    evaluate_parser.add_argument('--num-prompts', type=int, default=30,
                               help='Número de prompts de prueba a evaluar (default: 30)')
    
    args = parser.parse_args()
    
    # Manejar comandos
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'delete':
        return delete_command(args)
    elif args.command == 'evaluate':
        return evaluate_command(args)
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
