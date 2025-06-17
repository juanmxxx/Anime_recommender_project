#!/usr/bin/env python3
"""
Anime Recommender CLI

Este script proporciona una interfaz de línea de comandos para probar
el modelo de recomendación de animes. Permite realizar recomendaciones
basadas en títulos de animes existentes, prompts de texto o selección aleatoria.

Uso:
    python anime_recommender_cli.py --by-title "Naruto"
    python anime_recommender_cli.py --by-prompt "Un anime sobre ninjas adolescentes con poderes sobrenaturales"
    python anime_recommender_cli.py --random
"""

import os
import sys
import argparse
import joblib
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import requests
from sentence_transformers import SentenceTransformer
from commonFunctions import TextEmbedder


class AnimeRecommender:
    def __init__(self, model_dir="model"):
        """
        Inicializa el recomendador de animes cargando los modelos guardados
        
        Args:
            model_dir: Directorio donde están guardados los modelos
        """
        self.model_dir = model_dir
        
        # Verificar si existen los archivos necesarios
        required_files = [
            'anime_nn_model.pkl',
            'anime_data.pkl',
            'combined_embeddings.npy',
            'anime_id_to_index.pkl'
        ]
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"El archivo {file_path} no existe. Ejecuta primero 'python hybrid_recommender_fixed.py train' para crear los modelos.")
        
        # Cargar modelos y datos
        print("🔄 Cargando modelos y datos...")
        self.nn_model = joblib.load(os.path.join(model_dir, 'anime_nn_model.pkl'))
        self.anime_data = joblib.load(os.path.join(model_dir, 'anime_data.pkl'))
        self.combined_embeddings = np.load(os.path.join(model_dir, 'combined_embeddings.npy'))
        self.anime_id_to_index = joblib.load(os.path.join(model_dir, 'anime_id_to_index.pkl'))
        
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
        
        print(f"✅ Modelo cargado correctamente. Datos de {len(self.anime_data)} animes disponibles.")

    def find_anime_by_title(self, title: str) -> Optional[Dict]:
        """
        Busca un anime por título (exacto o aproximado)
        
        Args:
            title: Título del anime a buscar
            
        Returns:
            Datos del anime si se encuentra, None en caso contrario
        """
        title_lower = title.lower()
        
        # Buscar coincidencia exacta
        if title_lower in self.title_to_index:
            idx = self.title_to_index[title_lower]
            return self.anime_data[idx]
        
        # Buscar coincidencias parciales
        possible_matches = []
        for db_title, idx in self.title_to_index.items():
            if title_lower in db_title:
                possible_matches.append((db_title, idx))
        
        if possible_matches:
            # Ordenar por longitud para priorizar títulos más cortos (coincidencias más precisas)
            possible_matches.sort(key=lambda x: len(x[0]))
            best_match_idx = possible_matches[0][1]
            return self.anime_data[best_match_idx]
        
        return None
    
    def get_random_anime(self) -> Dict:
        """
        Obtiene un anime aleatorio de la base de datos
        
        Returns:
            Datos de un anime aleatorio
        """
        return random.choice(self.anime_data)
    
    def get_recommendations_by_anime_id(self, anime_id: int, num_recommendations: int = 10) -> List[Dict]:
        """
        Obtiene recomendaciones basadas en un ID de anime
        
        Args:
            anime_id: ID del anime base
            num_recommendations: Número de recomendaciones a devolver
            
        Returns:
            Lista de animes recomendados
        """
        if anime_id not in self.anime_id_to_index:
            print(f"❌ Error: Anime con ID {anime_id} no encontrado")
            return []
        
        anime_index = self.anime_id_to_index[anime_id]
        query_vector = self.combined_embeddings[anime_index].reshape(1, -1)
        
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=num_recommendations + 1)
        
        # Eliminar el anime de consulta (debería ser el primer resultado)
        recommended_indices = indices.flatten()[1:]
        recommended_distances = distances.flatten()[1:]
        
        recommendations = []
        for i, idx in enumerate(recommended_indices):
            anime = self.anime_data[idx]
            similarity = 1 - recommended_distances[i]  # Convertir distancia a similitud
            recommendations.append({
                **anime,
                'similarity': similarity
            })
        
        return recommendations

    def get_recommendations_by_title(self, title: str, num_recommendations: int = 10) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Obtiene recomendaciones basadas en un título de anime
        
        Args:
            title: Título del anime base
            num_recommendations: Número de recomendaciones a devolver
            
        Returns:
            Tupla de (anime base, lista de recomendaciones)
        """
        base_anime = self.find_anime_by_title(title)
        if not base_anime:
            return None, []
        
        recommendations = self.get_recommendations_by_anime_id(base_anime['id'], num_recommendations)
        return base_anime, recommendations

    def get_recommendations_by_prompt(self, prompt: str, num_recommendations: int = 10) -> List[Dict]:
        """
        Obtiene recomendaciones basadas en un prompt de texto
        
        Args:
            prompt: El texto descriptivo para buscar animes similares
            num_recommendations: Número de recomendaciones a devolver
            
        Returns:
            Lista de animes recomendados
        """
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
            recommendations.append({
                **anime,
                'similarity': similarity
            })
        
        return recommendations

    def print_anime_info(self, anime: Dict, similarity: float = None):
        """
        Imprime información de un anime con formato
        
        Args:
            anime: Datos del anime a imprimir
            similarity: Similitud opcional para mostrar
        """
        print("\n" + "=" * 80)
        print(f"📺 {anime.get('romaji_title')}")
        
        if anime.get('english_title') and anime['english_title'] not in ('null', None):
            print(f"🇺🇸 {anime['english_title']}")
        
        print(f"🏆 Puntuación: {anime.get('average_score', 'N/A')}/100")
        print(f"👁️  Popularidad: {anime.get('popularity', 'N/A')}")
        
        if similarity is not None:
            print(f"🔍 Similitud: {similarity:.2%}")
        
        # Géneros
        if anime.get('genres'):
            genres = anime['genres']
            if isinstance(genres, str):
                try:
                    genres = json.loads(genres.replace("'", '"'))
                except:
                    genres = [g.strip() for g in genres.strip('[]').split(',')]
            print(f"🏷️  Géneros: {', '.join(genres)}")
        
        # Formato y episodios
        format_str = anime.get('format', 'N/A')
        episodes = anime.get('episodes', 'N/A')
        year = anime.get('season_year', 'N/A')
        print(f"📊 Formato: {format_str} ({episodes} episodios) - Año: {year}")
        
        # Descripción
        if anime.get('description'):
            print("\n📝 Descripción:")
            desc = anime['description'].replace('<br>', '\n').replace('<i>', '').replace('</i>', '')
            # Limitar a 200 caracteres para no saturar la pantalla
            if len(desc) > 200:
                print(f"{desc[:200]}...")
            else:
                print(desc)
        
        print("=" * 80)

def print_recommendations(base_anime, recommendations):
    """
    Imprime el anime base y sus recomendaciones
    
    Args:
        base_anime: Anime base
        recommendations: Lista de animes recomendados
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "🌟 ANIME BASE PARA RECOMENDACIONES 🌟")
    print("=" * 80)
    
    recommender.print_anime_info(base_anime)
    
    print("\n" + "=" * 80)
    print(" " * 25 + "✨ ANIMES RECOMENDADOS ✨")
    print("=" * 80)
    
    for i, anime in enumerate(recommendations, 1):
        print(f"\n{i}.", end=" ")
        recommender.print_anime_info(anime, anime.get('similarity'))

def main():
    parser = argparse.ArgumentParser(description='Anime Recommender CLI')
    
    # Grupo de argumentos mutuamente excluyentes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--by-title', type=str,
                      help='Recomendar animes similares a un título específico')
    group.add_argument('--by-prompt', type=str,
                      help='Recomendar animes basados en una descripción textual')
    group.add_argument('--random', action='store_true',
                      help='Recomendar animes similares a un anime aleatorio')
    
    # Otros argumentos
    parser.add_argument('--count', type=int, default=10,
                       help='Número de recomendaciones a mostrar (default: 10)')
    parser.add_argument('--model-dir', type=str, default='../model',
                       help='Directorio de los modelos entrenados (default: ../model)')
    
    args = parser.parse_args()
    
    try:
        # Inicializar el recomendador
        global recommender
        recommender = AnimeRecommender(model_dir=args.model_dir)
        
        if args.by_title:
            # Recomendar por título
            base_anime, recommendations = recommender.get_recommendations_by_title(args.by_title, args.count)
            if not base_anime:
                print(f"❌ No se encontró ningún anime con el título: '{args.by_title}'")
                return 1
            
            print_recommendations(base_anime, recommendations)
            
        elif args.by_prompt:
            # Recomendar basado en un prompt de texto
            print("\n" + "=" * 80)
            print(" " * 20 + "🔍 BUSCANDO ANIMES POR DESCRIPCIÓN 🔍")
            print("=" * 80)
            
            print(f"\n📝 Prompt: {args.by_prompt}")
            recommendations = recommender.get_recommendations_by_prompt(args.by_prompt, args.count)
            
            print("\n" + "=" * 80)
            print(" " * 25 + "✨ ANIMES RECOMENDADOS ✨")
            print("=" * 80)
            
            for i, anime in enumerate(recommendations, 1):
                print(f"\n{i}.", end=" ")
                recommender.print_anime_info(anime, anime.get('similarity'))
            
        elif args.random:
            # Recomendar basado en un anime aleatorio
            random_anime = recommender.get_random_anime()
            recommendations = recommender.get_recommendations_by_anime_id(random_anime['id'], args.count)
            
            print_recommendations(random_anime, recommendations)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
