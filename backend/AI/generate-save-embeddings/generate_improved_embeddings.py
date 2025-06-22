#!/usr/bin/env python3
"""
Script para generar embeddings de anime mejorados desde la base de datos original
y guardarlos en la base de datos de embeddings.

Este script:
1. Se conecta a la BD original (5432) y obtiene datos de anime y personajes
2. Genera embeddings de alta calidad para descripción, género, personajes y metadatos
3. Se conecta a la BD de embeddings (5433) y guarda los embeddings asociados a cada anime
4. Crea índices vectoriales para búsqueda por similitud rápida

Mejoras sobre la versión original:
- Usa modelos de embeddings más potentes (all-mpnet-base-v2 en lugar de all-MiniLM-L6-v2)
- Preprocesamiento de texto más sofisticado para mejor calidad semántica
- Optimizado para reconocimiento de personajes y recomendaciones basadas en prompt
- Añade campos adicionales y más contexto para mejorar la precisión de recomendaciones
"""
import psycopg2
import argparse
import time
import sys
import re
import traceback
import numpy as np
import json
import subprocess
import os
import nltk
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class AnimeEmbeddingsGenerator:    
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=50, use_cuda=True, max_animes=None, limit_by_popularity=True):
        """
        Inicializa el generador de embeddings para datos de anime.
        
        Args:
            model_name: Modelo de Sentence Transformers a utilizar
                Opciones recomendadas:
                - "all-mpnet-base-v2": Mayor calidad pero más pesado (768 dim)
                - "multi-qa-mpnet-base-dot-v1": Optimizado para búsqueda semántica (768 dim)
                - "all-MiniLM-L12-v2": Equilibrio entre calidad y rendimiento (384 dim)
                - "all-MiniLM-L6-v2": Más rápido pero menos preciso (384 dim)
            batch_size: Tamaño del lote para procesamiento por lotes
            use_cuda: Si es True, intenta usar la GPU para acelerar el proceso
            max_animes: Limitar el número de animes a procesar (None=procesar todos)
            limit_by_popularity: Si es True y max_animes está definido, procesa los más populares
        """
        # Descargar recursos necesarios de NLTK
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("🔄 Descargando recursos de NLTK...")
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Verificar disponibilidad de CUDA
        import torch
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        
        if self.use_cuda:
            print(f"🚀 CUDA disponible! Usando GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("ℹ️ CUDA no disponible o desactivado. Usando CPU.")
        
        print(f"🔄 Cargando modelo {model_name} en {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.stop_words = set(stopwords.words('english'))
        self.max_animes = max_animes
        self.limit_by_popularity = limit_by_popularity
        
        print(f"✅ Modelo cargado con dimensión de embeddings: {self.embedding_dim}")
        if self.max_animes:
            print(f"⚠️ Modo limitado: Procesando solo {self.max_animes} animes {'' if not self.limit_by_popularity else 'más populares'}")
        print(f"ℹ️ Usando tamaño de lote (batch): {self.batch_size}")
        
        # Conexiones a bases de datos
        self.source_conn = None  # Conexión a BD original (texto plano)
        self.target_conn = None  # Conexión a BD de embeddings
        
        # Configurar conexiones
        self._setup_connections()
        
    def _setup_connections(self):
        """Establece conexiones a ambas bases de datos"""
        try:
            # Conexión a la BD original (puerto 5432)
            self.source_conn = psycopg2.connect(
                host="localhost",
                port="5432",
                database="animeDB",
                user="anime_db",
                password="anime_db"
            )
            print("✅ Conexión establecida con la base de datos original (puerto 5432)")
            
            # Conexión a la BD de embeddings (puerto 5433)
            self.target_conn = psycopg2.connect(
                host="localhost",
                port="5433",
                database="animeDBEmbeddings2",
                user="anime_db",
                password="anime_db"
            )
            print("✅ Conexión establecida con la base de datos de embeddings (puerto 5433)")
            
            # Verificar si la extensión pgvector está instalada
            self._check_pgvector()
            
            # Crear tablas en la BD de embeddings
            self._create_target_tables()
            
        except Exception as e:
            print(f"❌ Error al configurar conexiones: {e}")
            if self.source_conn:
                self.source_conn.close()
            if self.target_conn:
                self.target_conn.close()
            sys.exit(1)
    
    def _check_pgvector(self):
        """Verifica si pgvector está instalado y sugiere cómo instalarlo si no lo está"""
        if not self.target_conn:
            return False
            
        cur = self.target_conn.cursor()
        try:
            # Verificar si la extensión ya está instalada
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                print("✅ Extensión pgvector ya está instalada")
                cur.close()
                return True
            
            # Intentar instalar la extensión
            print("⚠️ La extensión pgvector no está instalada. Intentando instalarla...")
            cur.execute("CREATE EXTENSION vector")
            self.target_conn.commit()
            print("✅ Extensión pgvector instalada exitosamente")
            cur.close()
            return True
            
        except Exception as e:
            self.target_conn.rollback()
            print(f"❌ No se pudo instalar pgvector: {e}")
            print("\nPara instalar pgvector en su contenedor PostgreSQL:")
            print("1. Reinicie Docker Compose con la configuración actualizada:")
            print("   docker-compose down && docker-compose up -d")
            print("2. O conéctese manualmente al contenedor e instale pgvector:")
            print("   docker exec -it anime_postgres_embeddings bash")
            print("   apt-get update && apt-get install -y postgresql-server-dev-all gcc make git")
            print("   git clone https://github.com/pgvector/pgvector.git /tmp/pgvector")
            print("   cd /tmp/pgvector && make && make install")
            print("   psql -U anime_db -d animeDBEmbeddings2 -c 'CREATE EXTENSION vector;'")
            
            response = input("\n¿Desea continuar intentando crear las tablas sin vector? (s/n): ")
            if response.lower() == 's':
                return False
            else:
                cur.close()
                sys.exit(1)
    
    def _create_target_tables(self):
        """Crea las tablas necesarias en la BD de embeddings"""
        if not self.target_conn:
            return
            
        cur = self.target_conn.cursor()
        try:
            # Tabla de anime (replica de la original para mantener integridad referencial)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anime (
                    id INTEGER PRIMARY KEY,
                    romaji_title VARCHAR(255),
                    english_title VARCHAR(255),
                    native_title VARCHAR(255),
                    genres TEXT,
                    description TEXT,
                    format VARCHAR(50),
                    status VARCHAR(100),
                    episodes INTEGER,
                    average_score DECIMAL(5,2),
                    popularity INTEGER,
                    season_year INTEGER,
                    cover_image_medium TEXT,
                    tags TEXT,
                    studios TEXT
                )
            """)
            
            # Tabla de personajes (replica de la original)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS characters (
                    id INTEGER PRIMARY KEY,
                    anime_id INTEGER REFERENCES anime(id),
                    name VARCHAR(255),
                    image_url TEXT,
                    gender VARCHAR(50),
                    description TEXT,
                    role VARCHAR(50)
                )
            """)
            
            # Nueva tabla para tags individuales
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Nueva tabla para la relación muchos a muchos entre animes y tags
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anime_tags (
                    id SERIAL PRIMARY KEY,
                    anime_id INTEGER REFERENCES anime(id),
                    tag_id INTEGER REFERENCES tags(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(anime_id, tag_id)
                )
            """)
            
            # Crear índices para búsqueda rápida por tags
            cur.execute("""
                CREATE INDEX IF NOT EXISTS tags_name_idx
                ON tags (name);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS anime_tags_anime_id_idx
                ON anime_tags (anime_id);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS anime_tags_tag_id_idx
                ON anime_tags (tag_id);
            """)
            
            # Verificar si pgvector está disponible
            vector_available = True
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            if not cur.fetchone():
                vector_available = False
                print("⚠️ La extensión pgvector no está disponible, los embeddings serán almacenados como JSON")
            
            # Tablas de embeddings (usando vector o text dependiendo de disponibilidad)
            if vector_available:
                # Tabla para embeddings de descripción
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS anime_description_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla para embeddings de géneros
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS anime_genre_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla para embeddings de tags individuales
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS tag_embeddings (
                        tag_id INTEGER PRIMARY KEY REFERENCES tags(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla para embeddings combinados de tags por anime
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS anime_tags_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla para embeddings de personajes
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS character_embeddings (
                        character_id INTEGER PRIMARY KEY REFERENCES characters(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                  # Tabla para embeddings de metadata
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS anime_metadata_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla para embeddings unificados (combinación de todos los tipos de información)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS anime_unified_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Crear índices para búsquedas más rápidas
                try:
                    print("🔄 Creando índices HNSW para búsqueda vectorial rápida...")
                    
                    # Índice HNSW para búsqueda por similitud en descripción
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS anime_description_embeddings_hnsw_idx
                        ON anime_description_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    # Índice HNSW para búsqueda por similitud en géneros
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS anime_genre_embeddings_hnsw_idx
                        ON anime_genre_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    # Índice HNSW para búsqueda por similitud en tags individuales
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS tag_embeddings_hnsw_idx
                        ON tag_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    # Índice HNSW para búsqueda por similitud en tags por anime
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS anime_tags_embeddings_hnsw_idx
                        ON anime_tags_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    # Índice HNSW para búsqueda por similitud en personajes
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS character_embeddings_hnsw_idx
                        ON character_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)                    # Índice HNSW para búsqueda por similitud en metadatos
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS anime_metadata_embeddings_hnsw_idx
                        ON anime_metadata_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    # Índice HNSW para búsqueda por similitud en embeddings unificados
                    print("🔄 Creando índice HNSW para búsqueda por similitud en embeddings unificados...")
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS anime_unified_embeddings_hnsw_idx
                        ON anime_unified_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    # Índice HNSW para búsqueda por similitud en embeddings unificados
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS anime_unified_embeddings_hnsw_idx
                        ON anime_unified_embeddings
                        USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    print("✅ Índices HNSW creados exitosamente")
                    
                except Exception as e:
                    print(f"⚠️ No se pudieron crear índices HNSW: {e}")
                    print("Las consultas seguirán funcionando pero pueden ser más lentas.")
                    
            else:
                # Versión alternativa usando JSON para almacenar embeddings temporalmente
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS anime_description_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding_json JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS anime_genre_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding_json JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tag_embeddings (
                        tag_id INTEGER PRIMARY KEY REFERENCES tags(id),
                        embedding_json JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS anime_tags_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding_json JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS character_embeddings (
                        character_id INTEGER PRIMARY KEY REFERENCES characters(id),
                        embedding_json JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS anime_metadata_embeddings (
                        anime_id INTEGER PRIMARY KEY REFERENCES anime(id),
                        embedding_json JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            # Tabla adicional para almacenar nombres alternativos de personajes para mejorar la búsqueda
            cur.execute("""
                CREATE TABLE IF NOT EXISTS character_alternative_names (
                    id SERIAL PRIMARY KEY,
                    character_id INTEGER REFERENCES characters(id),
                    alternative_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Crear índice para búsqueda rápida por nombre alternativo
            cur.execute("""
                CREATE INDEX IF NOT EXISTS character_alternative_names_name_idx
                ON character_alternative_names (alternative_name);
            """)
            
            # Crear índice para búsqueda rápida por character_id en nombres alternativos
            cur.execute("""
                CREATE INDEX IF NOT EXISTS character_alternative_names_char_id_idx
                ON character_alternative_names (character_id);
            """)
            
            self.target_conn.commit()
            print("✅ Tablas creadas correctamente en la base de datos de embeddings")
            
        except Exception as e:
            print(f"❌ Error al crear tablas: {e}")
            self.target_conn.rollback()
        finally:
            cur.close()
            
    def _clean_text(self, text):
        """Limpia y normaliza texto para mejorar calidad de embeddings"""
        if not text:
            return ""
            
        # Eliminar citas de fuentes
        text = re.sub(r'\s*\(Source:.*?\)', '', text)
        text = re.sub(r'\s*\[Source:.*?\]', '', text)
        
        # Eliminar etiquetas HTML
        text = re.sub(r'<.*?>', '', text)
        
        # Eliminar saltos de línea y reemplazarlos con espacios
        text = re.sub(r'\n+|\r+|\t+', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres no alfanuméricos excepto espacios y puntuación básica
        text = re.sub(r'[^\w\s.,!?:;()\-]', '', text)
        
        return text.strip()
        
    def _preprocess_for_embedding(self, text):
        """
        Preprocesa texto para generar embeddings de mayor calidad
        Aplica tokenización y mantiene palabras significativas
        """
        if not text:
            return ""
            
        # Limpiar el texto
        clean_text = self._clean_text(text)
        
        # Tokenizar y filtrar stopwords para análisis
        # (no eliminamos stopwords del texto original para mantener semántica)
        tokens = word_tokenize(clean_text.lower())
        important_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        
        # Identificar palabras clave para dar énfasis (repetir)
        if len(important_tokens) > 5:
            keywords = important_tokens[:10]  # Tomar las primeras 10 palabras importantes
            emphasis = " " + " ".join(keywords)  # Repetir palabras clave
            clean_text = clean_text + emphasis
            
        return clean_text
    
    def _is_vector_available(self):
        """Comprueba si la extensión vector está disponible"""
        if not self.target_conn:
            return False
            
        cur = self.target_conn.cursor()
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone() is not None
        cur.close()
        return result
    
    def _copy_anime_data(self):
        """
        Copia los datos de anime de la BD original a la BD de embeddings
        para mantener la integridad referencial
        """
        if not self.source_conn or not self.target_conn:
            print("❌ No hay conexiones disponibles")
            return False
            
        try:
            print("⚠️ Este proceso puede tomar varios minutos dependiendo de la cantidad de datos...")
            source_cur = self.source_conn.cursor()
            target_cur = self.target_conn.cursor()
            
            # Comprobar si hay datos en la tabla de anime
            target_cur.execute("SELECT COUNT(*) FROM anime")
            count = target_cur.fetchone()[0]
            
            # Si ya hay datos, preguntar si se desean sobrescribir
            if count > 0:
                print(f"⚠️ La base de datos de embeddings ya contiene {count} animes.")
                response = input("¿Desea borrar todos los datos y reemplazarlos? (s/n): ")
                if response.lower() != 's':
                    print("🛑 Operación cancelada por el usuario.")
                    return False
            
            # 1. Limpiar tablas existentes            is_vector = self._is_vector_available()
            # Truncar tablas de embeddings
            target_cur.execute("TRUNCATE anime_description_embeddings, anime_genre_embeddings, character_embeddings, anime_metadata_embeddings, tag_embeddings, anime_tags_embeddings, anime_unified_embeddings CASCADE")
                
            target_cur.execute("TRUNCATE character_alternative_names CASCADE")
            target_cur.execute("TRUNCATE characters CASCADE")
            target_cur.execute("TRUNCATE anime CASCADE")
            self.target_conn.commit()
            
            # 2. Copiar datos de anime
            print("🔄 Copiando datos de anime...")
            source_cur.execute("SELECT * FROM anime")
            animes = source_cur.fetchall()
            
            # Usar executemany para mejor rendimiento
            target_cur.executemany("""
                INSERT INTO anime VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, animes)
            
            # 3. Copiar datos de personajes
            print("🔄 Copiando datos de personajes...")
            source_cur.execute("SELECT * FROM characters")
            characters = source_cur.fetchall()
            
            # Copiar en lotes de 1000 para mejor rendimiento
            batch_size = 1000
            for i in range(0, len(characters), batch_size):
                batch = characters[i:i+batch_size]
                target_cur.executemany("""
                    INSERT INTO characters VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, batch)
                self.target_conn.commit()
            
            self.target_conn.commit()
            print(f"✅ Copiados {len(animes)} animes y {len(characters)} personajes a la BD de embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error al copiar datos: {e}")
            self.target_conn.rollback()
            return False
    
    def generate_description_embedding(self, anime):
        """Genera embedding para la descripción de un anime"""
        romaji_title = anime[1] or ''  # romaji_title
        eng_title = anime[2] or ''  # english_title
        native_title = anime[3] or ''  # native_title
        description = anime[5] or ''  # description
        
        # Combinar toda la información con énfasis en los títulos
        text = f"Anime: {romaji_title}. "
        if eng_title and eng_title != romaji_title:
            text += f"English title: {eng_title}. "
        if native_title and native_title != romaji_title and native_title != eng_title:
            text += f"Native title: {native_title}. "
        text += f"Description: {description}"
        
        # Aplicar preprocesamiento avanzado
        text = self._preprocess_for_embedding(text)
        
        if not text:
            return np.zeros(self.embedding_dim)
        
        # Generar embedding con el modelo más potente
        return self.model.encode(text)
    
    def generate_genre_embedding(self, anime):
        """Genera embedding para géneros, tags y estudios"""
        # Extraer información relevante
        genres = anime[4] or ''  # genres
        tags = anime[13] or ''  # tags
        studios = anime[14] or ''  # studios
        
        # Expandir tags para darles mayor relevancia semántica
        expanded_tags = tags.replace(", ", ", anime with ").replace(",", ", anime with")
        if expanded_tags and not expanded_tags.startswith("anime with"):
            expanded_tags = "anime with " + expanded_tags
        
        # Construir texto enriquecido
        text = f"Anime genres: {genres}. "
        if expanded_tags:
            text += f"{expanded_tags}. "
        if studios:
            text += f"Produced by: {studios}."
        
        # Aplicar preprocesamiento avanzado
        text = self._preprocess_for_embedding(text)
        
        if not text or text == "Anime genres: . . Produced by: .":
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text)
    
    def generate_metadata_embedding(self, anime):
        """Genera embedding para metadatos como popularidad, año, etc."""
        format_type = anime[6] or ''  # format
        status = anime[7] or ''  # status
        episodes = anime[8] or 0  # episodes
        score = anime[9] or 0  # average_score
        popularity = anime[10] or 0  # popularity
        year = anime[11] or 0  # season_year
        
        # Normalizar popularidad a escala del 1-10
        normalized_popularity = min(10, max(1, popularity / 10000)) if popularity else 0
        
        # Construir texto con información contextual
        text = f"This is a {format_type} anime "
        
        if year:
            text += f"from {year} "
        
        if episodes == 1:
            text += f"with a single episode. "
        elif episodes > 1:
            text += f"with {episodes} episodes. "
        else:
            text += ". "
            
        if score:
            text += f"Rating: {score}/100. "
            
        if popularity:
            text += f"Popularity level: {normalized_popularity}/10. "
            
        if status:
            text += f"Status: {status}."
        
        # Aplicar preprocesamiento avanzado
        text = self._preprocess_for_embedding(text)
        
        if not text:
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text)
    
    def generate_character_embedding(self, character):
        """Genera embedding para un personaje optimizado para búsqueda de personajes"""
        anime_id = character[1] or 0   # anime_id
        name = character[2] or ''      # name
        gender = character[4] or ''    # gender
        description = character[5] or ''  # description
        role = character[6] or ''      # role
        
        # Buscar información del anime asociado al personaje
        anime_title = ""
        if anime_id:
            try:
                if self.source_conn:
                    cur = self.source_conn.cursor()
                    cur.execute("SELECT romaji_title, english_title FROM anime WHERE id = %s", (anime_id,))
                    anime_info = cur.fetchone()
                    if anime_info:
                        anime_title = anime_info[0] or anime_info[1] or ""
                    cur.close()
            except Exception as e:
                print(f"⚠️ No se pudo obtener el anime para el personaje {name}: {e}")
        
        # Construir texto enriquecido dando prioridad al nombre y anime
        text = f"Character name: {name} "
        
        # Repetir el nombre para dar énfasis en la búsqueda
        if name:
            text += f"({name}) "
            
        if anime_title:
            text += f"from anime {anime_title}. "
            
        if role:
            text += f"Role: {role}. "
            
        if gender:
            text += f"Gender: {gender}. "
            
        if description:
            text += f"Description: {description}"
        
        # Aplicar preprocesamiento avanzado
        text = self._preprocess_for_embedding(text)
        
        if not text or text == "Character name:  () from anime . Role: . Gender: . Description: ":
            return np.zeros(self.embedding_dim)
        
        # Generar nombres alternativos para el personaje y guardarlos
        self._extract_and_store_alternative_names(character[0], name, description)
        
        return self.model.encode(text)
        
    def _extract_and_store_alternative_names(self, character_id, name, description):
        """
        Extrae nombres alternativos del personaje (nicknames, apodos)
        desde su descripción y los almacena en la BD para mejorar búsquedas
        """
        if not name or not character_id or not self.target_conn:
            return
            
        # Guardar el nombre principal
        try:
            cur = self.target_conn.cursor()
            
            # Buscar posibles alias en la descripción
            alternative_names = set()
            
            # Añadir el nombre principal primero
            alternative_names.add(name)
            
            # Buscar nombres entre comillas
            if description:
                # Buscar patrones como '"NickName"' o "'NickName'" o "conocido como NickName"
                quote_patterns = [r'"([^"]{2,30})"', r"'([^']{2,30})'", r"known as ([A-Z][a-zA-Z]{2,30})"]
                for pattern in quote_patterns:
                    matches = re.findall(pattern, description)
                    for match in matches:
                        if match and match != name and len(match) >= 2:
                            alternative_names.add(match)
            
            # Guardar todos los nombres alternativos encontrados
            for alt_name in alternative_names:
                cur.execute("""
                    INSERT INTO character_alternative_names (character_id, alternative_name)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (character_id, alt_name))
                
            self.target_conn.commit()
            cur.close()
            
        except Exception as e:
            print(f"⚠️ Error al guardar nombres alternativos para {name}: {e}")
            if self.target_conn:
                self.target_conn.rollback()
    
    def generate_tag_embedding(self, tag_name):
        """Genera embedding para un tag individual"""
        if not tag_name:
            return np.zeros(self.embedding_dim)
        
        # Construir texto enriquecido para el tag
        # Expandir el contexto del tag para mejorar la semántica
        text = f"Anime with {tag_name} tag. Shows featuring {tag_name}. Animation related to {tag_name}."
        
        # Aplicar preprocesamiento avanzado
        text = self._preprocess_for_embedding(text)
        
        return self.model.encode(text)
        
    def generate_anime_tags_embedding(self, anime):
        """Genera embedding combinado para todos los tags de un anime"""
        tags = anime[13] or ''  # tags
        
        if not tags:
            return np.zeros(self.embedding_dim)
            
        # Construir texto enriquecido dando mayor contexto a los tags
        tag_list = tags.split(", ")
        
        # Construir texto completo con todos los tags
        text = "Anime with the following themes and elements: "
        text += ", ".join(tag_list)
        text += ". "
        
        # Añadir contexto adicional para los primeros 5 tags más importantes
        if tag_list:
            important_tags = tag_list[:5]
            for tag in important_tags:
                text += f"Features {tag}. "
                
        # Aplicar preprocesamiento avanzado
        text = self._preprocess_for_embedding(text)
        
        if not text or text == "Anime with the following themes and elements: . ":
            return np.zeros(self.embedding_dim)
            
        return self.model.encode(text)
        
    def _process_and_store_tags(self, anime_id, tags_str):
        """
        Procesa los tags de un anime, los almacena en la BD y genera sus embeddings
        
        Args:
            anime_id: ID del anime
            tags_str: String con tags separados por comas
        """
        if not tags_str or not self.target_conn:
            return
            
        # Separar los tags
        tag_list = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        if not tag_list:
            return
            
        try:
            cur = self.target_conn.cursor()
            
            # Diccionario para almacenar los ID de los tags
            tag_ids = {}
            
            # Para cada tag
            for tag in tag_list:
                # Verificar si el tag ya existe
                cur.execute("SELECT id FROM tags WHERE name = %s", (tag,))
                result = cur.fetchone()
                
                if result:
                    # El tag ya existe, obtener su ID
                    tag_id = result[0]
                else:
                    # El tag no existe, insertarlo y obtener su ID
                    cur.execute("INSERT INTO tags (name) VALUES (%s) RETURNING id", (tag,))
                    tag_id = cur.fetchone()[0]
                    
                    # Generar embedding para el tag individual
                    tag_embedding = self.generate_tag_embedding(tag)
                    
                    # Guardar embedding del tag
                    if self._is_vector_available():
                        cur.execute(
                            "INSERT INTO tag_embeddings (tag_id, embedding) VALUES (%s, %s) ON CONFLICT (tag_id) DO UPDATE SET embedding = %s",
                            (tag_id, tag_embedding.tolist(), tag_embedding.tolist())
                        )
                    else:
                        cur.execute(
                            "INSERT INTO tag_embeddings (tag_id, embedding_json) VALUES (%s, %s) ON CONFLICT (tag_id) DO UPDATE SET embedding_json = %s",
                            (tag_id, json.dumps(tag_embedding.tolist()), json.dumps(tag_embedding.tolist()))
                        )
                
                tag_ids[tag] = tag_id
                
                # Relacionar el tag con el anime
                cur.execute(
                    "INSERT INTO anime_tags (anime_id, tag_id) VALUES (%s, %s) ON CONFLICT (anime_id, tag_id) DO NOTHING",
                    (anime_id, tag_id)
                )
            
            # Generar embedding combinado para todos los tags del anime
            anime_data = cur.execute("SELECT * FROM anime WHERE id = %s", (anime_id,))
            anime = cur.fetchone()
            
            if anime:
                tags_embedding = self.generate_anime_tags_embedding(anime)
                
                # Guardar embedding combinado de tags
                if self._is_vector_available():
                    cur.execute(
                        "INSERT INTO anime_tags_embeddings (anime_id, embedding) VALUES (%s, %s) ON CONFLICT (anime_id) DO UPDATE SET embedding = %s",
                        (anime_id, tags_embedding.tolist(), tags_embedding.tolist())
                    )
                else:
                    cur.execute(
                        "INSERT INTO anime_tags_embeddings (anime_id, embedding_json) VALUES (%s, %s) ON CONFLICT (anime_id) DO UPDATE SET embedding_json = %s",
                        (anime_id, json.dumps(tags_embedding.tolist()), json.dumps(tags_embedding.tolist()))
                    )
            
            self.target_conn.commit()
            
        except Exception as e:
            print(f"❌ Error al procesar tags para anime {anime_id}: {e}")
            self.target_conn.rollback()
        finally:
            cur.close()
    
    def process_all_data(self):
        """
        Procesa todos los datos de anime y personajes, generando
        embeddings y almacenándolos en la BD de embeddings
        """
        if not self._copy_anime_data():
            print("❌ No se pudieron copiar los datos base. Abortando.")
            return False
        
        try:
            source_cur = self.source_conn.cursor()
            target_cur = self.target_conn.cursor()
            is_vector = self._is_vector_available()
            
            # 1. Procesar animes
            print("\n🔄 Procesando embeddings de animes...")
            source_cur.execute("SELECT * FROM anime ORDER BY popularity DESC")
            animes = source_cur.fetchall()
            
            for anime in tqdm(animes, desc="Generando embeddings de anime"):
                anime_id = anime[0]
                
                # 1.1 Embedding de descripción
                desc_embedding = self.generate_description_embedding(anime)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO anime_description_embeddings (anime_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, desc_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_description_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding_json = EXCLUDED.embedding_json,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(desc_embedding.tolist())))
                
                # 1.2 Embedding de géneros
                genre_embedding = self.generate_genre_embedding(anime)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO anime_genre_embeddings (anime_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, genre_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_genre_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding_json = EXCLUDED.embedding_json,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(genre_embedding.tolist())))
                
                # 1.3 Embedding de metadata
                meta_embedding = self.generate_metadata_embedding(anime)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO anime_metadata_embeddings (anime_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, meta_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_metadata_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding_json = EXCLUDED.embedding_json,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(meta_embedding.tolist())))
                
                # 1.4 Procesar tags individuales y generar sus embeddings
                tags = anime[13]
                if tags:
                    self._process_and_store_tags(anime_id, tags)
                
                # Commit cada 100 animes para evitar transacciones muy largas
                if anime_id % 100 == 0:
                    self.target_conn.commit()
            
            # 2. Procesar personajes
            print("\n🔄 Procesando embeddings de personajes...")
            source_cur.execute("SELECT * FROM characters")
            characters = source_cur.fetchall()
            
            for character in tqdm(characters, desc="Generando embeddings de personajes"):
                character_id = character[0]
                
                # 2.1 Embedding de personaje
                char_embedding = self.generate_character_embedding(character)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO character_embeddings (character_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (character_id) DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """, (character_id, char_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO character_embeddings (character_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (character_id) DO UPDATE SET 
                            embedding_json = EXCLUDED.embedding_json,
                            created_at = CURRENT_TIMESTAMP
                    """, (character_id, json.dumps(char_embedding.tolist())))
                  # Commit cada 100 personajes para evitar transacciones muy largas
                if character_id % 100 == 0:
                    self.target_conn.commit()
            
            # 3. Generar embeddings unificados para cada anime
            print("\n🔄 Generando embeddings unificados para cada anime...")
            target_cur.execute("SELECT id FROM anime ORDER BY popularity DESC")
            anime_ids = [row[0] for row in target_cur.fetchall()]
            
            for anime_id in tqdm(anime_ids, desc="Generando embeddings unificados"):
                # Generar embedding unificado
                unified_embedding = self.generate_unified_embedding(anime_id)
                
                # Guardar embedding unificado
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO anime_unified_embeddings (anime_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, unified_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_unified_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE SET 
                            embedding_json = EXCLUDED.embedding_json,
                            created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(unified_embedding.tolist())))
                
                # Commit cada 100 animes para evitar transacciones muy largas
                if anime_id % 100 == 0:
                    self.target_conn.commit()
            
            # Commit final
            self.target_conn.commit()
            print("\n✅ Proceso completado exitosamente")
            return True
            
        except Exception as e:
            print(f"\n❌ Error al procesar los datos: {e}")
            self.target_conn.rollback()
            traceback.print_exc()
            return False
    
    def generate_unified_embedding(self, anime_id):
        """
        Genera un embedding unificado que combina todos los tipos de información del anime:
        - Descripción
        - Géneros
        - Tags
        - Metadatos
        - Personajes relacionados
        
        Este embedding es ideal para búsquedas generales y recomendaciones basadas en prompts.
        """
        if not self.target_conn:
            return np.zeros(self.embedding_dim)
            
        try:
            cur = self.target_conn.cursor()
            vector_available = self._is_vector_available()
            embeddings_to_combine = []
            
            # Obtener embedding de descripción
            if vector_available:
                cur.execute("SELECT embedding FROM anime_description_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result:
                    desc_embedding = np.array(result[0])
                    embeddings_to_combine.append(desc_embedding)
            else:
                cur.execute("SELECT embedding_json FROM anime_description_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result and result[0]:
                    desc_embedding = np.array(json.loads(result[0]))
                    embeddings_to_combine.append(desc_embedding)
            
            # Obtener embedding de géneros
            if vector_available:
                cur.execute("SELECT embedding FROM anime_genre_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result:
                    genre_embedding = np.array(result[0])
                    embeddings_to_combine.append(genre_embedding)
            else:
                cur.execute("SELECT embedding_json FROM anime_genre_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result and result[0]:
                    genre_embedding = np.array(json.loads(result[0]))
                    embeddings_to_combine.append(genre_embedding)
                    
            # Obtener embedding de tags
            if vector_available:
                cur.execute("SELECT embedding FROM anime_tags_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result:
                    tags_embedding = np.array(result[0])
                    embeddings_to_combine.append(tags_embedding)
            else:
                cur.execute("SELECT embedding_json FROM anime_tags_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result and result[0]:
                    tags_embedding = np.array(json.loads(result[0]))
                    embeddings_to_combine.append(tags_embedding)
                    
            # Obtener embedding de metadatos
            if vector_available:
                cur.execute("SELECT embedding FROM anime_metadata_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result:
                    meta_embedding = np.array(result[0])
                    embeddings_to_combine.append(meta_embedding)
            else:
                cur.execute("SELECT embedding_json FROM anime_metadata_embeddings WHERE anime_id = %s", (anime_id,))
                result = cur.fetchone()
                if result and result[0]:
                    meta_embedding = np.array(json.loads(result[0]))
                    embeddings_to_combine.append(meta_embedding)
            
            # Obtener embedding promedio de personajes principales
            if vector_available:
                cur.execute("""
                    SELECT AVG(e.embedding) 
                    FROM character_embeddings e
                    JOIN characters c ON e.character_id = c.id
                    WHERE c.anime_id = %s AND c.role IN ('MAIN', 'PROTAGONIST')
                """, (anime_id,))
                result = cur.fetchone()
                if result and result[0]:
                    char_embedding = np.array(result[0])
                    embeddings_to_combine.append(char_embedding)
            else:
                # Para bases de datos sin soporte vector, esto es más complejo
                # Obtener los personajes y calcular manualmente el promedio
                cur.execute("""
                    SELECT e.embedding_json 
                    FROM character_embeddings e
                    JOIN characters c ON e.character_id = c.id
                    WHERE c.anime_id = %s AND c.role IN ('MAIN', 'PROTAGONIST')
                """, (anime_id,))
                results = cur.fetchall()
                if results:
                    char_embeddings = [np.array(json.loads(r[0])) for r in results if r[0]]
                    if char_embeddings:
                        char_embedding = np.mean(char_embeddings, axis=0)
                        embeddings_to_combine.append(char_embedding)
            
            # Combinar embeddings con diferentes pesos según la importancia
            if not embeddings_to_combine:
                return np.zeros(self.embedding_dim)
                
            # Pesos: descripción (0.35), géneros (0.2), tags (0.25), metadatos (0.1), personajes (0.1)
            weights = [0.35, 0.2, 0.25, 0.1, 0.1]
            
            # Ajustar los pesos si no tenemos todos los embeddings
            if len(embeddings_to_combine) < len(weights):
                weights = weights[:len(embeddings_to_combine)]
                weights = [w/sum(weights) for w in weights]  # Normalizar
                
            # Combinar embeddings ponderados
            weighted_embeddings = [emb * weight for emb, weight in zip(embeddings_to_combine, weights)]
            combined_embedding = np.sum(weighted_embeddings, axis=0)
            
            # Normalizar para mantener la norma L2=1
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            return combined_embedding
            
        except Exception as e:
            print(f"❌ Error al generar embedding unificado para anime {anime_id}: {e}")
            return np.zeros(self.embedding_dim)
        finally:
            cur.close()
    
    def close(self):
        """Cierra las conexiones a las bases de datos"""
        if self.source_conn:
            self.source_conn.close()
        if self.target_conn:
            self.target_conn.close()
        print("✅ Conexiones cerradas")

    def search_anime_by_prompt(self, prompt, top_k=10, use_character_matching=True):
        """
        Busca animes similares basados en un prompt del usuario.
        
        Args:
            prompt: El prompt del usuario (texto)
            top_k: Número de resultados a devolver
            use_character_matching: Si es True, busca menciones de personajes en el prompt
                                    y prioriza sus animes relacionados
            
        Returns:
            Lista de diccionarios con información de animes recomendados
        """
        if not self.target_conn or not prompt:
            return []
            
        try:
            print(f"🔍 Buscando animes basados en: '{prompt}'")
            cur = self.target_conn.cursor()
            vector_available = self._is_vector_available()
            
            # Generar embedding para el prompt del usuario
            prompt_embedding = self.model.encode(self._preprocess_for_embedding(prompt))
            
            # 1. Verificar si se menciona algún personaje específico
            character_matches = []
            if use_character_matching:
                # Buscar en nombres alternativos de personajes
                cur.execute("""
                    SELECT c.id, c.name, c.anime_id, a.romaji_title, a.english_title, a.native_title
                    FROM character_alternative_names can
                    JOIN characters c ON can.character_id = c.id
                    JOIN anime a ON c.anime_id = a.id
                    WHERE LOWER(can.alternative_name) LIKE LOWER(%s)
                    LIMIT 5
                """, (f"%{prompt.lower()}%",))
                
                results = cur.fetchall()
                for row in results:
                    character_id, char_name, anime_id, romaji, english, native = row
                    anime_title = english or romaji or native
                    
                    # Verificar similitud semántica con el nombre del personaje
                    char_embedding = None
                    if vector_available:
                        cur.execute("SELECT embedding FROM character_embeddings WHERE character_id = %s", (character_id,))
                        result = cur.fetchone()
                        if result:
                            char_embedding = np.array(result[0])
                    else:
                        cur.execute("SELECT embedding_json FROM character_embeddings WHERE character_id = %s", (character_id,))
                        result = cur.fetchone()
                        if result and result[0]:
                            char_embedding = np.array(json.loads(result[0]))
                    
                    if char_embedding is not None:
                        # Calcular la similitud del prompt con el embedding del personaje
                        similarity = util.cos_sim(prompt_embedding, char_embedding).item()
                        if similarity > 0.4:  # Umbral de similitud significativa
                            character_matches.append({
                                'anime_id': anime_id,
                                'title': anime_title,
                                'character_name': char_name,
                                'character_id': character_id,
                                'similarity': float(similarity)
                            })
            
            # Ordenar por similitud
            character_matches = sorted(character_matches, key=lambda x: x['similarity'], reverse=True)
            
            # 2. Buscar animes similares basados en embeddings unificados
            anime_matches = []
            
            if vector_available:
                # Buscar usando pgvector (más rápido y preciso)
                cur.execute(f"""
                    SELECT a.id, a.romaji_title, a.english_title, a.native_title, a.genres,
                           a.description, a.format, a.episodes, a.average_score, a.popularity,
                           1 - (e.embedding <=> %s::vector) as similarity
                    FROM anime_unified_embeddings e
                    JOIN anime a ON e.anime_id = a.id
                    ORDER BY similarity DESC
                    LIMIT {top_k}
                """, (prompt_embedding.tolist(),))
            else:
                # Alternativa sin pgvector (más lento)
                cur.execute("SELECT anime_id, embedding_json FROM anime_unified_embeddings")
                all_embeddings = []
                anime_ids = []
                
                for row in cur.fetchall():
                    anime_id, emb_json = row
                    if emb_json:
                        anime_ids.append(anime_id)
                        all_embeddings.append(np.array(json.loads(emb_json)))
                
                if anime_ids and all_embeddings:
                    # Calcular similitudes
                    similarities = util.cos_sim(prompt_embedding, all_embeddings)[0].tolist()
                    
                    # Crear pares (id, similitud) y ordenar
                    id_sim_pairs = sorted(zip(anime_ids, similarities), key=lambda x: x[1], reverse=True)
                    
                    # Tomar los top_k animes
                    top_anime_ids = [pair[0] for pair in id_sim_pairs[:top_k]]
                    top_similarities = [pair[1] for pair in id_sim_pairs[:top_k]]
                    
                    # Consultar información detallada de los animes top
                    placeholders = ','.join(['%s'] * len(top_anime_ids))
                    cur.execute(f"""
                        SELECT a.id, a.romaji_title, a.english_title, a.native_title, a.genres,
                               a.description, a.format, a.episodes, a.average_score, a.popularity
                        FROM anime a
                        WHERE a.id IN ({placeholders})
                    """, top_anime_ids)
                    
                    anime_info = {}
                    for row in cur.fetchall():
                        anime_id = row[0]
                        anime_info[anime_id] = row
                    
                    for anime_id, similarity in zip(top_anime_ids, top_similarities):
                        if anime_id in anime_info:
                            row = anime_info[anime_id]
                            anime_matches.append({
                                'id': row[0],
                                'title': row[2] or row[1] or row[3],  # english, romaji, o native
                                'genres': row[4],
                                'description': row[5],
                                'format': row[6],
                                'episodes': row[7],
                                'score': row[8],
                                'popularity': row[9],
                                'similarity': float(similarity)
                            })
            
            # Procesar resultados si usamos pgvector
            if vector_available and cur.description:
                for row in cur.fetchall():
                    anime_matches.append({
                        'id': row[0],
                        'title': row[2] or row[1] or row[3],  # english, romaji, o native
                        'genres': row[4], 
                        'description': row[5],
                        'format': row[6],
                        'episodes': row[7],
                        'score': row[8],
                        'popularity': row[9],
                        'similarity': float(row[10])
                    })
            
            # 3. Combinar resultados de personajes y embeddings
            # Si encontramos personajes mencionados, priorizarlos
            final_results = []
            seen_anime_ids = set()
            
            # Primero añadir animes con personajes mencionados
            for match in character_matches[:3]:  # Máximo 3 personajes
                anime_id = match['anime_id']
                if anime_id not in seen_anime_ids:
                    # Buscar información completa del anime
                    cur.execute("""
                        SELECT a.id, a.romaji_title, a.english_title, a.native_title, a.genres,
                               a.description, a.format, a.episodes, a.average_score, a.popularity
                        FROM anime a
                        WHERE a.id = %s
                    """, (anime_id,))
                    
                    row = cur.fetchone()
                    if row:
                        final_results.append({
                            'id': row[0],
                            'title': row[2] or row[1] or row[3],  # english, romaji, o native
                            'genres': row[4],
                            'description': row[5],
                            'format': row[6],
                            'episodes': row[7],
                            'score': row[8],
                            'popularity': row[9],
                            'similarity': match['similarity'],
                            'character_match': match['character_name']
                        })
                        seen_anime_ids.add(anime_id)
            
            # Luego añadir el resto de animes similares
            for match in anime_matches:
                anime_id = match['id']
                if anime_id not in seen_anime_ids:
                    final_results.append(match)
                    seen_anime_ids.add(anime_id)
                    
                    # Limitar resultados totales
                    if len(final_results) >= top_k:
                        break
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error al buscar animes por prompt: {e}")
            traceback.print_exc()
            return []
        finally:
            cur.close()
def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(description='Generador de Embeddings de Anime Mejorado')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                        help='Modelo a utilizar (default: all-mpnet-base-v2)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamaño de lote para procesamiento (default: 32)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERADOR DE EMBEDDINGS DE ANIME MEJORADO")
    print("=" * 80)
    print("Este script conecta a dos bases de datos:")
    print("1. Base de datos original (puerto 5432): Contiene datos de anime en texto plano")
    print("2. Base de datos de embeddings (puerto 5433): Donde se guardarán los embeddings")
    print("\nMejoras en esta versión:")
    print("• Utiliza el modelo all-mpnet-base-v2 (768 dim) para mayor calidad de embeddings")
    print("• Preprocesamiento de texto optimizado para recomendaciones por prompt")
    print("• Optimizado para reconocimiento de personajes mencionados en consultas")
    print("• Almacenamiento de nombres alternativos de personajes para mejor búsqueda")
    print("• Creación de índices vectoriales HNSW para búsqueda más rápida")
    print("• Generación de embeddings unificados optimizados para búsqueda por prompt")
    print("• Reconocimiento inteligente de personajes mencionados en consultas")
    print("=" * 80)
    
    try:
        import json
    except ImportError:
        print("❌ El módulo 'json' no está disponible. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "json"])

    
    generator = AnimeEmbeddingsGenerator(model_name=args.model, batch_size=args.batch_size)
    
    try:
        start_time = time.time()
        generator.process_all_data()
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"✅ Proceso completado en {elapsed:.2f} segundos")
        print("=" * 80)
        
    finally:
        generator.close()

if __name__ == "__main__":
    main()
