import requests
import json
import time
import os
import re
import psycopg2
from psycopg2.extras import Json
import sys

def fetch_anime_data(limit):
    """
    Rescata los animes desde la API de Anilist con filtros específicos.
    Args:
        limit (int): Número de animes a recuperar (se recuperarán en lotes)

    Returns:
        list: Lista de animes con sus detalles
    """
    url = 'https://graphql.anilist.co'
      # GraphQL consulta para obtener animes populares
    query = '''
    query ($page: Int, $perPage: Int) {
      Page(page: $page, perPage: $perPage) {
        pageInfo {
          total
          currentPage
          lastPage
          hasNextPage
          perPage
        }
        media(
          type: ANIME, 
          sort: POPULARITY_DESC, 
          format_in: [TV, MOVIE, OVA], 
          status_not: NOT_YET_RELEASED
        ) {
            id
            title {
                romaji
                english
                native
            }
            genres
            description
            format
            status
            episodes
            averageScore
            popularity
            seasonYear
            coverImage {
                medium
            }
            tags {
                name
            }
            studios {
                nodes {
                    name
                }
            }
            characters(sort: [ROLE, RELEVANCE, ID], perPage: 6) {
                edges {
                    role
                    node {
                        id
                        name {
                            full
                        }
                        image {
                            medium
                        }
                        gender
                        description
                    }
                }
            }
        }
      }
    }
    '''

    animes = []
    per_page = 50 # nUmero de animes por página
    total_pages = (limit + per_page - 1) // per_page  # Calcula el número total de páginas necesarias
    
    consecutive_low_popularity_pages = 0
    # Páginas consecutivas máximas con baja popularidad antes de detenerse
    MAX_CONSECUTIVE_LOW_PAGES = 3
    POPULARITY_THRESHOLD = 3000

    for page in range(1, total_pages + 1):
        variables = {
            'page': page,
            'perPage': per_page
        }
        
        response = requests.post(url, json={'query': query, 'variables': variables})
        if response.status_code == 200:
            data = response.json()
            page_info = data['data']['Page']['pageInfo']
            media = data['data']['Page']['media']
            
            # Mostrar información de la paginación
            if page == 1:  # Solo mostramos esta información en la primera página
                total_animes = page_info['total']
                last_page = page_info['lastPage']
                print(f"Total de animes disponibles en la API: {total_animes}")
                print(f"Número total de páginas: {last_page}")
                print(f"Animes por página: {page_info['perPage']}")
            
            # Filtrar animes con popularidad baja (menos de 3000)
            filtered_media = [anime for anime in media if anime.get('popularity', 0) >= POPULARITY_THRESHOLD]
            

            if len(filtered_media) == 0:
                consecutive_low_popularity_pages += 1
                print(f"Page {page} has no animes above popularity threshold ({POPULARITY_THRESHOLD})")
                print(f"Consecutive low popularity pages: {consecutive_low_popularity_pages}/{MAX_CONSECUTIVE_LOW_PAGES}")

                # PARAR si alcanzamos el número máximo de páginas consecutivas con baja popularidad
                if consecutive_low_popularity_pages >= MAX_CONSECUTIVE_LOW_PAGES:
                    print(f"Stopping fetch: {MAX_CONSECUTIVE_LOW_PAGES} consecutive pages with all animes below popularity threshold")
                    break
            else:
                # Reset counter if we found animes above threshold
                consecutive_low_popularity_pages = 0
            
            print(f"Filtered out {len(media) - len(filtered_media)} animes with popularity below {POPULARITY_THRESHOLD}")
            media = filtered_media

            # Procesar cada anime para simplificar los datos
            for anime in media:
                # Convertir etiquetas de objetos a una cadena separada por comas
                if 'tags' in anime:
                    tag_names = [tag['name'] for tag in anime['tags']]
                    anime['tags'] = ", ".join(tag_names) if tag_names else ""
                
                # Convertir estudios de objetos anidados a una cadena separada por comas
                if 'studios' in anime and 'nodes' in anime['studios']:
                    studio_names = [studio['name'] for studio in anime['studios']['nodes']]
                    anime['studios'] = ", ".join(studio_names) if studio_names else ""
                
                # Convertir géneros a una cadena separada por comas si es una lista
                if 'genres' in anime and isinstance(anime['genres'], list):
                    anime['genres'] = ", ".join(anime['genres']) if anime['genres'] else ""

                # Limpiar la descripción eliminando la parte "(Source: ...)"
                if 'description' in anime and anime['description']:
                    anime['description'] = re.sub(r'\s*\(Source:.*?\)', '', anime['description'])
                    anime['description'] = re.sub(r'\s*\[Source:.*?\]', '', anime['description'])
                    # Trim any resulting extra whitespace
                    anime['description'] = anime['description'].strip()

                # Procesar personajes
                if 'characters' in anime and 'edges' in anime['characters']:
                    # Extraer datos de personajes
                    characters = []
                    for edge in anime['characters']['edges']:
                        char = edge['node']
                        # Limpiar descripción del personaje
                        if 'description' in char and char['description']:
                            char['description'] = re.sub(r'\s*\(Source:.*?\)', '', char['description'])
                            char['description'] = re.sub(r'\s*\[Source:.*?\]', '', char['description'])
                            char['description'] = char['description'].strip()
                        
                        characters.append({
                            'id': char['id'],
                            'name': char['name']['full'],
                            'image': char['image']['medium'] if 'image' in char and char['image'] else None,
                            'gender': char['gender'],
                            'description': char['description'],
                            'role': edge['role']
                        })
                    
                    # Reemplazar la lista de personajes en el anime
                    anime['characters'] = characters
            
            animes.extend(media)
            print(f"Fetched page {page} with {len(media)} anime(s), total animes so far: {len(animes)}")
        else:
            print(f"Failed to fetch page {page}: {response.status_code} - {response.text}")

        time.sleep(1)  # Espera de 1 segundo entre solicitudes para evitar sobrecargar la API

    # Limitar el número total de animes a la cantidad solicitada
    return animes[:limit]

def save_to_database(animes):
    """
    Save anime data directly to the PostgreSQL database
    Args:
        animes (list): List of anime data
    """
    # Inicializar conexión a la base de datos
    conn = None
    cur = None
    try:
        # Conectar a la base de datos PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            port="5432",  # Usualmente el puerto por defecto de PostgreSQL
            database="animeDB",  # Nombre original de la base de datos
            user="anime_db",
            password="anime_db"
        )

        # Crear un cursor
        cur = conn.cursor()
        # Primero limpiar datos existentes (si es necesario)
        cur.execute("TRUNCATE characters CASCADE;")
        cur.execute("TRUNCATE anime CASCADE;")
        conn.commit()

        # Insertar cada anime en la base de datos
        for anime in animes:
            # Insert anime data
            cur.execute("""
                INSERT INTO anime (
                    id, romaji_title, english_title, native_title, genres, description,
                    format, status, episodes, average_score, popularity, season_year,
                    cover_image_medium, tags, studios
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                anime['id'],
                anime['title']['romaji'],
                anime['title']['english'],
                anime['title']['native'],
                anime['genres'],
                anime['description'],
                anime['format'],
                anime['status'],
                anime['episodes'],
                anime['averageScore'],
                anime['popularity'],
                anime['seasonYear'],
                anime['coverImage']['medium'],
                anime['tags'],
                anime['studios']
            ))
            # Insertar personajes asociados al anime
            if 'characters' in anime:
                for character in anime['characters']:
                    cur.execute("""
                        INSERT INTO characters (
                            id, anime_id, name, image_url, gender, description, role
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        character['id'],
                        anime['id'],
                        character['name'],
                        character['image'],
                        character['gender'],
                        character['description'],
                        character['role']
                    ))
        
        # Guardar los cambios en la base de datos
        conn.commit()
        print(f"Successfully saved {len(animes)} animes to the database")
        
    except Exception as e:
        print(f"Database error: {e}")
    
    finally:
        # Cerrar cursor y conexión
        if cur:
            cur.close()
        if conn:
            conn.close()

def main(limit=1000):  # Cambiado para obtener 1000 animes por defecto
    """
    Función principal para obtener y guardar datos de anime.
    
    Args:
        limit (int, opcional): Número de animes a obtener. Por defecto es 1000.
    """
    # Intenta convertir el valor de limit a entero en caso de que sea una cadena
    try:
        limit = int(limit)
    except (ValueError, TypeError):
        print(f"El valor de limit no es válido: {limit}. Usando 5000 como valor por defecto.")
        limit = 5000
    
    print(f"Obteniendo datos de {limit} animes...")
    animes = fetch_anime_data(limit)
    if animes:
        # Also save to database
        save_to_database(animes)
    else:
        print("No anime data fetched.")

if __name__ == "__main__":
    # Si se pasa un argumento en la línea de comandos, úsalo como límite
    if len(sys.argv) > 1:
        try:
            limit_arg = int(sys.argv[1])
            main(limit_arg)
        except ValueError:
            print(f"Error: El argumento '{sys.argv[1]}' no es un número válido. Usando el valor por defecto.")
            main()
    else:
        main()
