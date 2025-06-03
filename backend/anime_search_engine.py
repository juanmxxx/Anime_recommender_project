"""
Anime Search Engine - Motor de búsqueda y recomendación de anime

Este script implementa un motor de búsqueda para animes que:
1. Procesa consultas en lenguaje natural (prompts)
2. Extrae palabras clave relevantes
3. Utiliza el módulo modelFormer para generar recomendaciones
4. Devuelve resultados en formato texto o JSON

Puede usarse como:
- Módulo importable desde la API (api.py)
- Herramienta de línea de comandos independiente
- Intermediario entre la interfaz de usuario y el modelo de recomendación

Ejemplos de uso:
  python anime_search_engine.py "animes de romance y comedia" -n 10
  python anime_search_engine.py "action with strong female protagonist" -f json -o results.json
"""

import re
import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
from collections import Counter
import os
import sys
import time
import json
import pandas as pd

# Añadimos el directorio padre a la ruta para poder importar modelFormer
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import modelFormer

# Constantes y configuración global
MAX_PROMPT_LENGTH = 500  # Máximo de caracteres para procesar sin truncar
MAX_KEYWORDS = 10        # Número máximo de palabras clave
PROCESSING_TIMEOUT = 30  # Tiempo máximo para procesamiento (segundos)

# Cargamos el modelo de spaCy
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Descargando modelo de spaCy...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

class PromptProcessor:
    """
    Clase para procesar prompts en lenguaje natural y extraer palabras clave
    para recomendaciones de anime, con soporte mejorado para prompts largos.
    """
    
    def __init__(self, model_name='all-mpnet-base-v2'):
        # Inicializamos los modelos
        self.kw_model = KeyBERT(model_name)
        self.st_model = SentenceTransformer(model_name)
        
        # Lista de stopwords y términos importantes
        self.custom_stopwords = {
            'want', 'need', 'looking', 'something', 'which', 'the', 'be', 'with', 'and', 
            'if', 'can', 'is', 'a', 'of', 'to', 'it', 'some', 'what', 'where', 'when',
            'who', 'how', 'why', 'would', 'could', 'should', 'an', 'for', 'that', 'this',
            'these', 'those', 'my', 'our', 'your', 'anime', 'similar', 'like', 'recommend',
            'recommendation', 'about', 'has', 'have', 'had', 'me', 'i', 'show', 'please',
            'thanks', 'thank', 'you', 'there', 'their', 'they', 'series', 'film', 'movie',
            'episode', 'season', 'character', 'characters', 'plot', 'story'
        }
        
        # Términos importantes por categorías
        self.genre_terms = {
            'action', 'adventure', 'comedy', 'drama', 'fantasy', 'horror', 'magic',
            'mecha', 'mystery', 'romance', 'sci-fi', 'thriller', 'supernatural', 
            'school', 'space', 'historical', 'war', 'sports', 'psychological'
        }
        
        self.visual_terms = {
            'hair', 'eyes', 'blonde', 'brunette', 'redhead', 'bald', 'glasses',
            'tall', 'short', 'blue', 'red', 'green', 'black', 'white', 'pink', 'purple',
            'blood', 'bloody', 'gore', 'violent'
        }
        
        self.character_terms = {
            'protagonist', 'hero', 'heroine', 'villain', 'antagonist', 'main',
            'girl', 'boy', 'man', 'woman', 'female', 'male', 'kid', 'adult', 'teen'
        }
        
        self.setting_terms = {
            'school', 'academy', 'city', 'village', 'future', 'past', 'medieval',
            'modern', 'ancient', 'space', 'planet', 'kingdom', 'empire', 'dystopia',
            'utopia', 'forest', 'ocean', 'mountain', 'reality', 'parallel', 'alternate'
        }
    
    def extract_keywords_simple(self, prompt):
        """
        Método simplificado para extraer palabras clave, especialmente útil
        para prompts largos o cuando el procesamiento complejo falla.
        """
        keywords = []
        
        # Categorías de términos importantes a buscar
        important_categories = [
            # Términos específicos del dominio
            ['blood', 'bloody', 'violent', 'gore', 'monster', 'monsters', 'survival', 'reality', 'parallel'],
            
            # Colores y visuales
            ['red', 'blue', 'green', 'black', 'white', 'blonde', 'hair', 'eyes'],
            
            # Géneros principales
            ['action', 'adventure', 'horror', 'fantasy', 'sci-fi', 'mystery', 'thriller'],
            
            # Características de personajes
            ['strong', 'hero', 'protagonist', 'villain', 'warrior', 'fighter']
        ]
        
        # Frases clave específicas
        key_phrases = [
            "parallel reality", "parallel world", "alternate reality", "alternate world",
            "blood", "bloody", "gore", "violent", "survival", "monster", "demons",
            "fighting monsters", "kill monsters", "delete monster", "destroy monster"
        ]
        
        # Convertimos a minúsculas para búsqueda case-insensitive
        prompt_lower = prompt.lower()
        
        # Primero buscamos frases completas de alta prioridad
        for phrase in key_phrases:
            if phrase in prompt_lower and phrase not in keywords and len(keywords) < MAX_KEYWORDS:
                keywords.append(phrase)
        
        # Luego buscamos términos individuales por categoría
        for category in important_categories:
            found_in_category = 0
            for term in category:
                if term in prompt_lower and term not in keywords:
                    keywords.append(term)
                    found_in_category += 1
                if found_in_category >= 2 or len(keywords) >= MAX_KEYWORDS:
                    break
        
        # Detectamos si hay negaciones ("no OVA", "no 1 capítulo")
        negative_terms = ['ova', 'one episode', '1 episode', 'single episode', 'short']
        for term in negative_terms:
            if term in prompt_lower:
                # Si se detectan términos negativos, añadimos "series" y "multiple episodes"
                if "series" not in keywords and len(keywords) < MAX_KEYWORDS:
                    keywords.append("series")
                if "multiple episodes" not in keywords and len(keywords) < MAX_KEYWORDS:
                    keywords.append("multiple episodes")
                break
        
        # Si no tenemos suficientes palabras clave, extraemos las más frecuentes
        if len(keywords) < 3:
            words = [w for w in re.findall(r'\b\w{4,}\b', prompt_lower) 
                    if w not in self.custom_stopwords]
            word_counts = Counter(words)
            
            for word, count in word_counts.most_common(5):
                if word not in keywords and len(keywords) < MAX_KEYWORDS:
                    keywords.append(word)
        
        return keywords
    
    def process_prompt(self, prompt):
        """
        Procesa un prompt y extrae palabras clave relevantes.
        Para prompts largos, usa un enfoque simplificado.
        """
        start_time = time.time()
        
        # Para prompts muy largos, usamos directamente el método simple
        if len(prompt) > 800:
            print(f"Prompt largo detectado ({len(prompt)} caracteres). Usando método optimizado.")
            return self.extract_keywords_simple(prompt)
        
        try:
            # Intentamos el método KeyBERT para prompts de longitud razonable
            prompt_truncated = prompt[:500] if len(prompt) > 500 else prompt
            
            # Extraemos palabras clave con KeyBERT
            keywords = self.kw_model.extract_keywords(
                prompt_truncated,
                keyphrase_ngram_range=(1, 3),
                stop_words=list(self.custom_stopwords),
                top_n=10,
                use_mmr=True,
                diversity=0.7
            )
            
            # Filtramos y limpiamos resultados
            cleaned_keywords = []
            for kw, score in keywords:
                # Eliminamos stopwords
                words = [w for w in re.findall(r'\w+', kw.lower()) 
                        if w not in self.custom_stopwords]
                cleaned_kw = ' '.join(words)
                
                if cleaned_kw and cleaned_kw not in cleaned_keywords:
                    cleaned_keywords.append(cleaned_kw)
            
            # Si el tiempo excede 10 segundos o no encontramos palabras clave,
            # recurrimos al método simple
            if time.time() - start_time > 10 or not cleaned_keywords:
                return self.extract_keywords_simple(prompt)
            
            return cleaned_keywords
            
        except Exception as e:
            print(f"Error en procesamiento de prompt: {e}")
            return self.extract_keywords_simple(prompt)
            
    def recommend_from_prompt(self, prompt, top_n=5, output_format="string"):
        """
        Procesa un prompt y devuelve recomendaciones de anime.
        
        Args:
            prompt (str): El prompt del usuario para buscar animes
            top_n (int): Número de recomendaciones a devolver
            output_format (str): Formato de salida: "string" (texto para CLI) o "json" (para API)
        
        Returns:
            Según el formato seleccionado:
            - "string": Objeto DataFrame de pandas con las recomendaciones
            - "json": Diccionario JSON con las recomendaciones formateadas
        """
        start_time = time.time()
        
        try:
            # Extraemos palabras clave del prompt
            print("Procesando prompt...")
            keywords = self.process_prompt(prompt)
            
            # Si no se encontraron palabras clave
            if not keywords:
                print("No se pudieron extraer palabras clave significativas.")
                # En lugar de devolver None, devolvemos un mensaje de error estructurado
                if output_format.lower() == "json":
                    return {
                        "success": False,
                        "error": "No se pudieron extraer palabras clave significativas del prompt."
                    }
                return None
            
            # Convertimos las palabras clave a cadena
            keywords_str = " ".join(keywords)
            print(f"Palabras clave extraídas: {keywords_str}")
            
            # Obtenemos recomendaciones
            print("Buscando recomendaciones...")
            recommendations = modelFormer.recommend_anime(keywords_str, top_n=top_n)
            
            # Si no se encontraron recomendaciones
            if recommendations is None or len(recommendations) == 0:
                print("No se encontraron animes que coincidan con la búsqueda.")
                if output_format.lower() == "json":
                    return {
                        "success": False,
                        "keywords": keywords,
                        "error": "No se encontraron animes que coincidan con la búsqueda.",
                        "results": []
                    }
                return None
            
            # Formatear la salida según el formato solicitado
            if output_format.lower() == "json":
                # Convertir el DataFrame a formato JSON
                # Convertimos a diccionario para formatear correctamente
                result = {
                    "success": True,
                    "keywords": keywords,
                    "total_results": len(recommendations),
                    "results": []
                }
                
                # Agregar cada anime al resultado
                for idx, anime in recommendations.iterrows():
                    anime_dict = anime.to_dict()
                    
                    # Asegurar que todos los campos estén presentes
                    anime_json = {
                        "id": idx,
                        "name": anime.get('Name', ''),
                        "score": float(anime.get('Score', 0)),
                        "image_url": anime.get('Image URL', ''),
                        "synopsis": anime.get('Synopsis', ''),
                        "type": anime.get('Type', ''),
                        "episodes": anime.get('Episodes', ''),
                        "genres": anime.get('Genres', ''),
                        "rank": anime.get('Rank', 0),
                        "aired": anime.get('Aired', ''),
                        "status": anime.get('Status', '')
                    }
                    
                    # Incluir explicación si está disponible
                    if 'explanation' in anime:
                        anime_json['explanation'] = anime['explanation']
                        
                    result["results"].append(anime_json)
                
                return result
            else:
                # Para formato string (o cualquier valor por defecto), devolvemos el DataFrame original
                return recommendations
            
        except Exception as e:
            error_msg = f"Error al obtener recomendaciones: {e}"
            print(error_msg)
            if output_format.lower() == "json":
                return {"success": False, "error": error_msg}
            return None

# Función principal para usar desde línea de comandos
if __name__ == "__main__":
    import argparse
    
    # Configuramos el parser de argumentos
    parser = argparse.ArgumentParser(description='Motor de búsqueda y recomendación de anime')
    parser.add_argument('prompt', nargs='?', default="Anime with monsters in parallel reality", 
                      help='Prompt de búsqueda (texto descriptivo)')
    parser.add_argument('-n', '--num', type=int, default=5, 
                      help='Número de resultados a mostrar (default: 5)')
    parser.add_argument('-f', '--format', choices=['string', 'json'], default='string', 
                      help='Formato de salida: string (texto legible) o json')
    parser.add_argument('-o', '--output', type=str, 
                      help='Archivo de salida para guardar resultados (opcional)')
    
    args = parser.parse_args()
    
    # Procesamos el prompt
    processor = PromptProcessor()
    print(f"\nProcesando: \"{args.prompt}\"")
    
    keywords = processor.process_prompt(args.prompt)
    print(f"Palabras clave: {keywords}")
    
    # Obtenemos recomendaciones con el formato especificado
    result = processor.recommend_from_prompt(args.prompt, top_n=args.num, output_format=args.format)
    
    # Mostramos o guardamos resultados según el formato
    if result is not None:
        if args.format.lower() == "json":
            # Formato JSON: mostramos el resultado como JSON formateado
            json_output = json.dumps(result, ensure_ascii=False, indent=2)
            
            if args.output:
                # Guardar en archivo si se especificó
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"\nResultados guardados en: {args.output}")
            else:
                # Mostrar en consola
                print(json_output)
        else:
            # Formato de texto legible
            recommendations = result  # Para claridad
            print("\n" + "="*80)
            print(f"TOP {len(recommendations)} RECOMENDACIONES")
            print("="*80)
            
            for idx, anime in recommendations.iterrows():
                print(f"{idx+1}. {anime['Name']} - Puntuación: {anime.get('Score', 'N/A')}")
                if "explanation" in anime:
                    print(f"   Por qué: {anime['explanation']}")
                print(f"   Tipo: {anime.get('Type', 'N/A')} | Episodios: {anime.get('Episodes', 'N/A')}")
                print(f"   Géneros: {anime.get('Genres', 'N/A')}")
                if "Synopsis" in anime:
                    synopsis = anime["Synopsis"]
                    if isinstance(synopsis, str) and len(synopsis) > 150:
                        synopsis = synopsis[:147] + "..."
                    print(f"   Sinopsis: {synopsis}")
                print("-"*80)
            
            # Si se especificó archivo de salida para formato string
            if args.output:
                try:
                    # Intentar guardar como CSV
                    recommendations.to_csv(args.output, index=False)
                    print(f"\nResultados guardados en: {args.output}")
                except Exception as e:
                    print(f"Error al guardar resultados: {e}")
    else:
        error_msg = "\nNo se encontraron recomendaciones."
        print(error_msg)
        
        # Si se solicitó guardar en archivo y hubo error
        if args.output and args.format.lower() == "json":
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({"success": False, "error": "No se encontraron recomendaciones"}, f, ensure_ascii=False, indent=2)
            print(f"Mensaje de error guardado en: {args.output}")
