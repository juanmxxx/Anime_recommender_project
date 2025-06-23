#!/usr/bin/env python3
"""
Script para generar embeddings de anime desde la base de datos original
y guardarlos en la base de datos de embeddings.

Este script:
1. Se conecta a la BD original (5432) y obtiene datos de anime y personajes
2. Genera embeddings para descripción, género, personajes y metadatos
3. Se conecta a la BD de embeddings (5433) y guarda los embeddings asociados a cada anime
"""

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import time
import re
import argparse
from tqdm import tqdm
import sys
import os
import subprocess

class AnimeEmbeddingsGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=50):
        """
        Inicializa el generador de embeddings para datos de anime.
        
        Args:
            model_name: Modelo de Sentence Transformers a utilizar
            batch_size: Tamaño del lote para procesamiento por lotes
        """
        print(f"🔄 Cargando modelo {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        print(f"✅ Modelo cargado con dimensión de embeddings: {self.embedding_dim}")
        
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
                port="5435",
                database="animeDBEmbeddings",
                user="anime_db",
                password="anime_db"
            )
            print("✅ Conexión establecida con la base de datos de embeddings (puerto 5435)")
            
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
            print("   psql -U anime_db -d animeDBEmbeddings -c 'CREATE EXTENSION vector;'")
            
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
            
            # Verificar si pgvector está disponible
            vector_available = True
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            if not cur.fetchone():
                print("⚠️ Extensión vector no disponible. Usando tipo TEXT para embeddings temporalmente.")
                vector_available = False
            
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
                
                # Crear índices para búsquedas más rápidas
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS anime_description_embedding_idx 
                        ON anime_description_embeddings USING ivfflat (embedding vector_cosine_ops)
                    """)
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS anime_genre_embedding_idx 
                        ON anime_genre_embeddings USING ivfflat (embedding vector_cosine_ops)
                    """)
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS character_embedding_idx 
                        ON character_embeddings USING ivfflat (embedding vector_cosine_ops)
                    """)
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS anime_metadata_embedding_idx 
                        ON anime_metadata_embeddings USING ivfflat (embedding vector_cosine_ops)
                    """)
                    
                    print("✅ Índices de búsqueda creados correctamente")
                    
                except Exception as e:
                    print(f"⚠️ No se pudieron crear índices optimizados: {e}")
                    print("⚠️ Esto no afectará la funcionalidad básica, continuando...")
                    self.target_conn.rollback()
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
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
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
                    print("✅ Usando datos existentes en la base de datos de embeddings.")
                    return True
            
            # 1. Limpiar tablas existentes
            is_vector = self._is_vector_available()
            if is_vector:
                target_cur.execute("TRUNCATE anime_description_embeddings, anime_genre_embeddings, character_embeddings, anime_metadata_embeddings CASCADE")
            else:
                target_cur.execute("TRUNCATE anime_description_embeddings, anime_genre_embeddings, character_embeddings, anime_metadata_embeddings CASCADE")
                
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
        title = anime[1] or ''  # romaji_title
        eng_title = anime[2] or ''  # english_title
        description = anime[5] or ''  # description
        
        text = f"{title} {eng_title} {description}"
        text = self._clean_text(text)
        
        if not text:
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text)
    
    def generate_genre_embedding(self, anime):
        """Genera embedding para géneros, tags y estudios"""
        genres = anime[4] or ''  # genres
        tags = anime[13] or ''  # tags
        studios = anime[14] or ''  # studios
        
        text = f"Géneros: {genres}. Tags: {tags}. Estudios: {studios}"
        text = self._clean_text(text)
        
        if not text or text == "Géneros: . Tags: . Estudios: ":
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
        
        text = (f"Anime {format_type} del año {year} con {episodes} episodios. "
                f"Popularidad: {normalized_popularity}/10. Puntuación: {score}/100. "
                f"Estado: {status}.")
                
        text = self._clean_text(text)
        
        if not text:
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text)
    
    def generate_character_embedding(self, character):
        """Genera embedding para un personaje"""
        name = character[2] or ''  # name
        gender = character[4] or ''  # gender
        description = character[5] or ''  # description
        role = character[6] or ''  # role
        
        text = f"Nombre: {name}. Rol: {role}. Género: {gender}. {description}"
        text = self._clean_text(text)
        
        if not text or text == "Nombre: . Rol: . Género: . ":
            return np.zeros(self.embedding_dim)
        
        return self.model.encode(text)
    
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
                        ON CONFLICT (anime_id) DO UPDATE 
                        SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP
                    """, (anime_id, desc_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_description_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE 
                        SET embedding_json = EXCLUDED.embedding_json, created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(desc_embedding.tolist())))
                
                # 1.2 Embedding de géneros
                genre_embedding = self.generate_genre_embedding(anime)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO anime_genre_embeddings (anime_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE 
                        SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP
                    """, (anime_id, genre_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_genre_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE 
                        SET embedding_json = EXCLUDED.embedding_json, created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(genre_embedding.tolist())))
                
                # 1.3 Embedding de metadata
                meta_embedding = self.generate_metadata_embedding(anime)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO anime_metadata_embeddings (anime_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE 
                        SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP
                    """, (anime_id, meta_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO anime_metadata_embeddings (anime_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (anime_id) DO UPDATE 
                        SET embedding_json = EXCLUDED.embedding_json, created_at = CURRENT_TIMESTAMP
                    """, (anime_id, json.dumps(meta_embedding.tolist())))
                
                # Commit cada 50 animes para evitar transacciones demasiado largas
                if anime_id % 50 == 0:
                    self.target_conn.commit()
            
            # Commit final para animes
            self.target_conn.commit()
            print(f"✅ Procesados embeddings para {len(animes)} animes")
            
            # 2. Procesar personajes
            print("\n🔄 Procesando embeddings de personajes...")
            source_cur.execute("""
                SELECT c.* FROM characters c
                JOIN anime a ON c.anime_id = a.id
                ORDER BY a.popularity DESC
            """)
            characters = source_cur.fetchall()
            
            for character in tqdm(characters, desc="Generando embeddings de personajes"):
                character_id = character[0]
                
                char_embedding = self.generate_character_embedding(character)
                if is_vector:
                    target_cur.execute("""
                        INSERT INTO character_embeddings (character_id, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (character_id) DO UPDATE 
                        SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP
                    """, (character_id, char_embedding.tolist()))
                else:
                    target_cur.execute("""
                        INSERT INTO character_embeddings (character_id, embedding_json)
                        VALUES (%s, %s)
                        ON CONFLICT (character_id) DO UPDATE 
                        SET embedding_json = EXCLUDED.embedding_json, created_at = CURRENT_TIMESTAMP
                    """, (character_id, json.dumps(char_embedding.tolist())))
                
                # Commit cada 100 personajes
                if character_id % 100 == 0:
                    self.target_conn.commit()
            
            # Commit final para personajes
            self.target_conn.commit()
            print(f"✅ Procesados embeddings para {len(characters)} personajes")
            
            if not is_vector:
                print("\n⚠️ Los embeddings se han guardado como JSON debido a que la extensión")
                print("   pgvector no está disponible. Para usar búsquedas de similitud")
                print("   eficientes, instale pgvector y ejecute de nuevo este script.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error al procesar datos: {e}")
            import traceback
            print(traceback.format_exc())
            self.target_conn.rollback()
            return False
    
    def close(self):
        """Cierra las conexiones a las bases de datos"""
        if self.source_conn:
            self.source_conn.close()
        if self.target_conn:
            self.target_conn.close()
        print("✅ Conexiones cerradas")

def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(description='Generador de Embeddings de Anime')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Modelo a utilizar (default: all-MiniLM-L6-v2)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Tamaño de lote para procesamiento (default: 50)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERADOR DE EMBEDDINGS DE ANIME")
    print("=" * 80)
    print("Este script conecta a dos bases de datos:")
    print("1. Base de datos original (puerto 5432): Contiene datos de anime en texto plano")
    print("2. Base de datos de embeddings (puerto 5433): Donde se guardarán los embeddings")
    print("=" * 80)
    
    try:
        import json  # Import needed for JSON fallback
    except ImportError:
        print("❌ El módulo 'json' no está disponible. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "json"])
        import json
    
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