"""
APIprocessor - Procesador para la API de recomendaciones de anime

Este módulo proporciona funciones para procesar las recomendaciones de animes
y devolverlas en un formato adecuado para la API.
"""

import os
import sys
import json
import numpy as np
from decimal import Decimal
from typing import List, Dict, Any, Optional
from pathlib import Path

# Definir un encoder JSON personalizado para manejar tipos especiales como Decimal
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

# Importar ruta para encontrar el módulo AnimeRecommender
sys.path.append(str(Path(__file__).parent))
from anime_recommender_cli import AnimeRecommender

# Inicializar el recomendador como una variable global para reutilizarlo
recommender = None

def init_recommender(model_dir=None) -> bool:
    """
    Inicializa el recomendador de animes si aún no está inicializado.
    
    Args:
        model_dir: Directorio donde están guardados los modelos
    
    Returns:
        True si se inicializó correctamente, False en caso contrario
    """
    global recommender
    
    if recommender is not None:
        return True
    
    try:
        # Si no se especifica un directorio, buscar en varias ubicaciones posibles
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_locations = [
                os.path.abspath(os.path.join(script_dir, "../../model")),  # Ruta relativa desde el script
                os.path.abspath(os.path.join(script_dir, "../model")),     # Una carpeta arriba
                os.path.abspath(os.path.join(script_dir, "model")),        # En la misma carpeta
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(script_dir)), "model")), # Root del proyecto
                os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))), "model")) # Un nivel más arriba
            ]
            
            # Encontrar la primera ubicación que exista
            model_dir = None
            for location in possible_locations:
                if os.path.exists(location) and os.path.isdir(location):
                    # Verificar si contiene los archivos necesarios (al menos el .pkl)
                    pkl_files = [f for f in os.listdir(location) if f.endswith('.pkl')]
                    if pkl_files:  # Si hay al menos un archivo .pkl
                        model_dir = location
                        print(f"✓ Encontrado directorio de modelos en: {model_dir}")
                        break
            
            if model_dir is None:
                print("❌ No se pudo encontrar el directorio de modelos en ninguna ubicación esperada.")
                print(f"Ubicaciones buscadas: {possible_locations}")
                print("Por favor, especifica la ruta correcta o ejecuta el entrenamiento primero.")
                return False
        
        # Verificar si el directorio existe antes de inicializar
        if not os.path.exists(model_dir):
            print(f"❌ El directorio de modelos no existe: {model_dir}")
            print("Por favor, ejecuta primero 'python hybrid_recommender_fixed.py train' para crear los modelos.")
            return False
        
        recommender = AnimeRecommender(model_dir=model_dir)
        print(f"✓ Recomendador inicializado correctamente usando modelos en: {model_dir}")
        return True
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error al inicializar el recomendador: {e}")
        print(f"Detalles del error: {error_details}")
        return False

def get_recommendations_from_prompt(prompt: str, num_recommendations: int = 10) -> Dict:
    """
    Obtiene recomendaciones de animes basadas en un prompt de texto y las devuelve en formato JSON.
    
    Args:
        prompt: El texto descriptivo para buscar animes similares
        num_recommendations: Número de recomendaciones a devolver
    
    Returns:
        Diccionario con las recomendaciones en formato JSON adecuado para la API
    """
    global recommender
    
    # Inicializar el recomendador si aún no está inicializado
    if not init_recommender():
        return {
            "success": False,
            "error": "Error al inicializar el recomendador de animes"
        }
    
    try:
        # Obtener recomendaciones
        recommendations = recommender.get_recommendations_by_prompt(prompt, num_recommendations)
        
        # Preparar la respuesta
        clean_recommendations = []
        for anime in recommendations:
            print(f"Debug anime data: {anime.keys()}")
            # Crear un objeto limpio con solo los campos necesarios
            clean_anime = {
                "id": int(anime.get("id")) if anime.get("id") is not None else None,
                "romaji_title": anime.get("romaji_title"),
                "english_title": anime.get("english_title"),
                "average_score": float(anime.get("average_score")) if anime.get("average_score") is not None else None,
                "popularity": int(anime.get("popularity")) if anime.get("popularity") is not None else None,
                "similarity": float(anime.get("similarity", 0)),
                "description": anime.get("description", "").replace('<br>', ' ').replace('<i>', '').replace('</i>', ''),
                "format": anime.get("format"),
                "episodes": int(anime.get("episodes")) if anime.get("episodes") not in (None, 'N/A') else None,
                "season_year": int(anime.get("season_year")) if anime.get("season_year") not in (None, 'N/A') else None,
                "image_url": anime.get("cover_image_medium"),
                "status": anime.get("status", "UNKNOWN")
            }
            
            # Procesar géneros
            genres = anime.get("genres")
            if genres:
                if isinstance(genres, str):
                    try:
                        genres = json.loads(genres.replace("'", '"'))
                    except:
                        genres = [g.strip() for g in genres.strip('[]').split(',')]
                clean_anime["genres"] = genres
            else:
                clean_anime["genres"] = []
                
            clean_recommendations.append(clean_anime)
        
        return {
            "success": True,
            "query": prompt,
            "recommendations": clean_recommendations
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "details": error_details
        }

def get_recommendations_from_title(title: str, num_recommendations: int = 10) -> Dict:
    """
    Obtiene recomendaciones de animes basadas en un título existente y las devuelve en formato JSON.
    
    Args:
        title: Título del anime base
        num_recommendations: Número de recomendaciones a devolver
    
    Returns:
        Diccionario con las recomendaciones en formato JSON adecuado para la API
    """
    global recommender
    
    # Inicializar el recomendador si aún no está inicializado
    if not init_recommender():
        return {
            "success": False,
            "error": "Error al inicializar el recomendador de animes"
        }
    
    try:
        # Obtener recomendaciones
        base_anime, recommendations = recommender.get_recommendations_by_title(title, num_recommendations)
        
        if not base_anime:
            return {
                "success": False,
                "error": f"No se encontró ningún anime con el título: '{title}'"
            }
        
        # Preparar la respuesta con el mismo formato que la función anterior
        clean_base = {
            "id": int(base_anime.get("id")) if base_anime.get("id") is not None else None,
            "romaji_title": base_anime.get("romaji_title"),
            "english_title": base_anime.get("english_title"),
            "average_score": float(base_anime.get("average_score")) if base_anime.get("average_score") is not None else None,
            "popularity": int(base_anime.get("popularity")) if base_anime.get("popularity") is not None else None,
            "description": base_anime.get("description", "").replace('<br>', ' ').replace('<i>', '').replace('</i>', ''),
            "format": base_anime.get("format"),
            "episodes": int(base_anime.get("episodes")) if base_anime.get("episodes") not in (None, 'N/A') else None,
            "season_year": int(base_anime.get("season_year")) if base_anime.get("season_year") not in (None, 'N/A') else None
        }
        
        # Procesar géneros del anime base
        genres = base_anime.get("genres")
        if genres:
            if isinstance(genres, str):
                try:
                    genres = json.loads(genres.replace("'", '"'))
                except:
                    genres = [g.strip() for g in genres.strip('[]').split(',')]
            clean_base["genres"] = genres
        else:
            clean_base["genres"] = []
        
        # Procesar recomendaciones
        clean_recommendations = []
        for anime in recommendations:
            clean_anime = {
                "id": int(anime.get("id")) if anime.get("id") is not None else None,
                "romaji_title": anime.get("romaji_title"),
                "english_title": anime.get("english_title"),
                "average_score": float(anime.get("average_score")) if anime.get("average_score") is not None else None,
                "popularity": int(anime.get("popularity")) if anime.get("popularity") is not None else None,
                "similarity": float(anime.get("similarity", 0)),
                "description": anime.get("description", "").replace('<br>', ' ').replace('<i>', '').replace('</i>', ''),
                "format": anime.get("format"),
                "episodes": int(anime.get("episodes")) if anime.get("episodes") not in (None, 'N/A') else None,
                "season_year": int(anime.get("season_year")) if anime.get("season_year") not in (None, 'N/A') else None,
                "image_url": anime.get("cover_image", {}).get("large") if isinstance(anime.get("cover_image"), dict) else anime.get("image_url"),
                "status": anime.get("status", "UNKNOWN")
            }
            
            # Procesar géneros
            genres = anime.get("genres")
            if genres:
                if isinstance(genres, str):
                    try:
                        genres = json.loads(genres.replace("'", '"'))
                    except:
                        genres = [g.strip() for g in genres.strip('[]').split(',')]
                clean_anime["genres"] = genres
            else:
                clean_anime["genres"] = []
                
            clean_recommendations.append(clean_anime)
        
        return {
            "success": True,
            "query": title,
            "base_anime": clean_base,
            "recommendations": clean_recommendations
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "details": error_details
        }

def get_random_recommendations(num_recommendations: int = 10) -> Dict:
    """
    Obtiene recomendaciones basadas en un anime aleatorio y las devuelve en formato JSON.
    
    Args:
        num_recommendations: Número de recomendaciones a devolver
    
    Returns:
        Diccionario con las recomendaciones en formato JSON adecuado para la API
    """
    global recommender
    
    # Inicializar el recomendador si aún no está inicializado
    if not init_recommender():
        return {
            "success": False,
            "error": "Error al inicializar el recomendador de animes"
        }
    
    try:
        # Obtener anime aleatorio
        random_anime = recommender.get_random_anime()
        # Obtener recomendaciones
        recommendations = recommender.get_recommendations_by_anime_id(random_anime['id'], num_recommendations)
        
        # Convertir a formato compatible con la API usando la misma lógica que las otras funciones
        return get_recommendations_from_title(random_anime.get("romaji_title"), num_recommendations)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "details": error_details
        }

if __name__ == "__main__":
    """
    Sección principal para probar las funcionalidades del módulo.
    
    Ejemplos de uso:
    - python APIprocessor.py prompt "Un anime sobre ninjas adolescentes con poderes"
    - python APIprocessor.py title "Naruto"
    - python APIprocessor.py random
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Prueba de APIprocessor para recomendaciones de animes')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando para recomendaciones por prompt
    prompt_parser = subparsers.add_parser('prompt', help='Recomendar animes basados en una descripción')
    prompt_parser.add_argument('text', type=str, help='Texto descriptivo para buscar animes similares')
    prompt_parser.add_argument('--count', '-c', type=int, default=5, help='Número de recomendaciones (default: 5)')
    
    # Comando para recomendaciones por título
    title_parser = subparsers.add_parser('title', help='Recomendar animes similares a un título existente')
    title_parser.add_argument('text', type=str, help='Título del anime base para recomendaciones')
    title_parser.add_argument('--count', '-c', type=int, default=5, help='Número de recomendaciones (default: 5)')
    
    # Comando para recomendaciones aleatorias
    random_parser = subparsers.add_parser('random', help='Recomendar animes basados en un anime aleatorio')
    random_parser.add_argument('--count', '-c', type=int, default=5, help='Número de recomendaciones (default: 5)')
    
    # Procesar argumentos
    args = parser.parse_args()
    
    # Ejecutar el comando correspondiente
    results = None
    
    if args.command == 'prompt':
        print(f"📝 Obteniendo recomendaciones para el prompt: '{args.text}'")
        results = get_recommendations_from_prompt(args.text, args.count)
    elif args.command == 'title':
        print(f"🔍 Obteniendo recomendaciones para el título: '{args.text}'")
        results = get_recommendations_from_title(args.text, args.count)
    elif args.command == 'random':
        print("🎲 Obteniendo recomendaciones aleatorias")
        results = get_random_recommendations(args.count)
    else:
        parser.print_help()
        sys.exit(1)
    
    # Imprimir resultados formateados
    if results:
        print("\n" + "=" * 80)
        print(" " * 20 + "📊 RESULTADOS DE LA RECOMENDACIÓN 📊")
        print("=" * 80 + "\n")
        
        # Imprimir el JSON formateado usando el encoder personalizado
        formatted_json = json.dumps(results, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        print(formatted_json)
        
        # También guardar en un archivo para revisar más fácilmente
        output_file = f"recommendation_{args.command}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            # Usar el encoder personalizado también al escribir en el archivo
            json.dump(results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            
        print(f"\n✅ Resultados guardados también en el archivo: {output_file}")
        
        # Mostrar un resumen si la recomendación fue exitosa
        if results.get("success"):
            print("\n" + "=" * 80)
            print(" " * 20 + "📌 RESUMEN DE RECOMENDACIONES 📌")
            print("=" * 80)
            
            recommendations = results.get("recommendations", [])
            for i, anime in enumerate(recommendations, 1):
                similarity = anime.get("similarity", 0) * 100
                print(f"{i}. {anime.get('romaji_title')} - Similitud: {similarity:.1f}%")
