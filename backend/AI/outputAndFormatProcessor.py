#!/usr/bin/env python3
"""
Manejador de recomendaciones de anime mejorado
============================================

Sistema de recomendación basado en contenido que utiliza:
- Búsqueda por palabras clave en sinopsis, géneros y nombres
- Ranking por popularidad (favoritos)
- Factor de actualidad (fecha de emisión)
- Búsqueda especializada para animes de ídolos

Uso:
    python outputAndFormatProcessor_improved.py "aspiring idols who wants to be the best"
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import re
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine

# Agregar directorio actual al path para importar modelTokenizerHandler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el tokenizador
from phraseTokernizer import extract_keyphrases, remove_prepositions

# Configuración de rutas
BASE_DIR = Path(__file__).parent

# Configuración de base de datos
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'animes',
    'user': 'anime_db',
    'password': 'anime_db'
}

class ImprovedAnimeRecommendationSystem:
    """Sistema de recomendación de animes mejorado basado en contenido"""
    
    def __init__(self):
        """Inicializa el sistema de recomendación cargando los datos"""
        self.df = None
        
        # Cargar datos de anime
        self._load_anime_data()
    
    def _get_db_connection(self):
        """Crea conexión a la base de datos PostgreSQL"""
        try:
            conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            engine = create_engine(conn_string)
            return engine
        except Exception as e:
            print(f"Error conectando a la base de datos: {e}")
            raise
        
    def _load_anime_data(self):
        """Carga los datos de anime desde PostgreSQL y filtra 'Not Yet Aired'"""
        try:
            engine = self._get_db_connection()
            
            # Query para obtener datos de animes, excluyendo 'Not Yet Aired'
            query = """
            SELECT 
                anime_id,
                name,
                english_name,
                other_name,
                score,
                genres,
                synopsis,
                type,
                episodes,
                aired,
                status,
                producers,
                licensors,
                studios,
                source,
                duration,
                rating,
                rank,
                popularity,
                favorites,
                image_url
            FROM anime
            WHERE synopsis IS NOT NULL 
            AND synopsis != ''
            AND aired != 'Not available'
            AND aired IS NOT NULL
            ORDER BY anime_id
            """
            
            print("Cargando datos desde PostgreSQL...")
            self.df = pd.read_sql_query(query, engine)
            
            # Verificar que tenemos datos
            if self.df.empty:
                raise ValueError("No se encontraron datos en la base de datos")
            
            # Limpiar y procesar datos
            self.df = self.df.dropna(subset=['synopsis'])
            
            # Llenar valores NaN
            self.df['favorites'] = pd.to_numeric(self.df['favorites'], errors='coerce').fillna(0)
            self.df['aired'] = self.df['aired'].fillna('Unknown')
            
            # Asegurar que todas las columnas están en minúsculas para consistencia interna
            self.df.columns = [col.lower() for col in self.df.columns]
            
            print(f"Dataset cargado desde PostgreSQL: {len(self.df)} animes")
            print(f"Registros con 'Not Yet Aired' excluidos")
            
        except Exception as e:
            print(f"Error al cargar los datos de la base de datos: {e}")
            raise
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    def process_prompt(self, prompt: str) -> str:
        """Procesa el prompt para extraer keyphrases mejoradas"""
        # Extraer keyphrases usando el tokenizador existente
        keyphrases = extract_keyphrases(prompt)
        
        # Si no obtenemos keyphrases suficientes, extraer palabras clave manualmente
        if not keyphrases or len(keyphrases.split()) < 2:
            # Palabras clave importantes del prompt original
            important_words = []
            words = prompt.lower().split()
            
            # Lista de palabras clave relevantes para anime
            anime_keywords = {
                'action', 'adventure', 'comedy', 'drama', 'romance', 'fantasy', 'sci-fi', 'thriller',
                'mystery', 'horror', 'slice of life', 'supernatural', 'magic', 'school', 'superhero',
                'superheroes', 'hero', 'heroes', 'idol', 'idols', 'music', 'singing', 'dance',
                'fighting', 'battle', 'war', 'mecha', 'robot', 'space', 'future', 'past',
                'demon', 'vampire', 'ninja', 'samurai', 'sports', 'cooking', 'medical'
            }
            
            for word in words:
                # Limpiar palabra de puntuación
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word in anime_keywords and len(clean_word) > 2:
                    important_words.append(clean_word)
            
            if important_words:
                keyphrases = " ".join(important_words)
        
        # Expandir con palabras clave adicionales para búsqueda
        keyword_expansions = {
            'idol': ['idol', 'idols', 'singer', 'performer', 'entertainment'],
            'aspiring': ['aspiring', 'dreams', 'goals', 'ambitions', 'wants to be'],
            'best': ['best', 'top', 'number one', 'greatest', 'champion'],
            'music': ['music', 'song', 'singing', 'performance', 'concert'],
            'school': ['school', 'academy', 'students', 'high school'],
            'group': ['group', 'team', 'unit', 'band', 'troupe'],
            'superhero': ['superhero', 'superheroes', 'hero', 'heroes', 'powers'],
            'adventure': ['adventure', 'adventurous', 'journey', 'quest', 'exploration'],
            'comedy': ['comedy', 'funny', 'humor', 'humorous', 'comedic']
        }
        
        # Expandir keyphrases si contienen palabras relacionadas
        expanded_phrases = keyphrases.lower()
        for key, synonyms in keyword_expansions.items():
            if key in expanded_phrases:
                for synonym in synonyms:
                    if synonym not in expanded_phrases:
                        expanded_phrases += f" {synonym}"
        
        return expanded_phrases
    
    def get_recommendations(self, keyphrases: str, top_n: int = 10) -> List[Dict]:
        """Genera recomendaciones basadas en las keyphrases procesadas"""
        # Verificar que los datos estén cargados
        if self.df is None:
            raise ValueError("Datos no cargados correctamente")
        
        # Crear una copia del dataframe para trabajar
        df_work = self.df.copy()
        
        # Inicializar scores
        df_work['content_score'] = 0.0
        df_work['popularity_score'] = 0.0
        df_work['recency_score'] = 0.0
        df_work['recommendation_score'] = 0.0
        
        # 1. BÚSQUEDA POR CONTENIDO (Synopsis, Géneros, Nombre)
        keywords = keyphrases.lower().split()
        
        # Búsqueda especial para ídolos
        idol_related = any(word in keyphrases.lower() for word in ['idol', 'singer', 'performer', 'music', 'aspiring'])
        
        for keyword in keywords:
            if len(keyword) > 2:  # Ignorar palabras muy cortas
                # Buscar en sinopsis
                synopsis_match = df_work['synopsis'].str.lower().str.contains(keyword, na=False)
                df_work.loc[synopsis_match, 'content_score'] += 3.0
                
                # Buscar en géneros (más peso)
                genre_match = df_work['genres'].str.lower().str.contains(keyword, na=False)
                df_work.loc[genre_match, 'content_score'] += 5.0
                
                # Buscar en nombre (más peso)
                name_match = df_work['name'].str.lower().str.contains(keyword, na=False)
                df_work.loc[name_match, 'content_score'] += 4.0
                
                # Buscar en nombre en inglés
                if 'english_name' in df_work.columns:
                    english_match = df_work['english_name'].str.lower().str.contains(keyword, na=False)
                    df_work.loc[english_match, 'content_score'] += 4.0
        
        # 2. SCORE DE POPULARIDAD (basado en favoritos)
        if 'favorites' in df_work.columns:
            max_favorites = df_work['favorites'].max()
            if max_favorites > 0:
                df_work['popularity_score'] = (df_work['favorites'] / max_favorites) * 10.0
        
        # 3. SCORE DE ACTUALIDAD (fecha de emisión)
        if 'aired' in df_work.columns:
            # Extraer año de la fecha de emisión
            def extract_year(aired_str):
                try:
                    if pd.isna(aired_str):
                        return 1990
                    # Buscar el primer año en la cadena
                    years = re.findall(r'\b(19|20)\d{2}\b', str(aired_str))
                    if years:
                        return int(years[0])
                    return 1990
                except:
                    return 1990
            
            df_work['year'] = df_work['aired'].apply(extract_year)
            current_year = 2024
            
            # Score de actualidad: más puntos para animes más recientes
            max_year_score = 5.0
            df_work['recency_score'] = ((df_work['year'] - 1990) / (current_year - 1990)) * max_year_score
        
        # 4. SCORE FINAL COMBINADO
        # Pesos: contenido 60%, popularidad 30%, actualidad 10%
        df_work['recommendation_score'] = (
            df_work['content_score'] * 0.6 +
            df_work['popularity_score'] * 0.3 +
            df_work['recency_score'] * 0.1
        )
        
        # 5. FILTROS ADICIONALES
        # Filtrar animes con score MAL muy bajo (opcional)
        if 'score' in df_work.columns:
            df_work = df_work[df_work['score'].isna() | (df_work['score'] >= 5.0)]
        
        # 6. ORDENAMIENTO FINAL
        # Primero por score de recomendación, luego por favoritos, luego por score MAL
        sort_columns = ['recommendation_score']
        sort_ascending = [False]
        
        if 'favorites' in df_work.columns:
            sort_columns.append('favorites')
            sort_ascending.append(False)
        
        if 'score' in df_work.columns:
            sort_columns.append('score')
            sort_ascending.append(False)
        
        result = df_work.sort_values(by=sort_columns, ascending=sort_ascending)
        
        # Seleccionar top_n recomendaciones
        top_recommendations = result.head(top_n)
        
        # Formatear resultados
        recommendations = []
        for _, row in top_recommendations.iterrows():
            rec = {}
            # Incluir todos los campos disponibles
            for col in row.index:
                if pd.notna(row[col]):
                    # Convertir campos numéricos a float/int cuando sea necesario
                    if col == 'recommendation_score':
                        rec[col] = float(row[col])
                    elif col in ['anime_id', 'rank', 'popularity', 'favorites', 'episodes']:
                        try:
                            rec[col] = int(row[col]) if pd.notna(row[col]) else None
                        except:
                            rec[col] = row[col]
                    else:
                        rec[col] = row[col]
            
            recommendations.append(rec)
        
        return recommendations

def get_recommendations_text(prompt: str, system: Optional[ImprovedAnimeRecommendationSystem] = None, 
                           top_n: int = 10) -> str:
    """Genera recomendaciones en formato texto"""
    try:
        # Crear sistema si no se proporcionó uno
        if system is None:
            system = ImprovedAnimeRecommendationSystem()
        
        # Procesar prompt y obtener keyphrases
        keyphrases = system.process_prompt(prompt)
        
        if not keyphrases:
            return "No se pudieron extraer keyphrases del prompt."
        
        # Obtener recomendaciones
        recommendations = system.get_recommendations(keyphrases, top_n)
        
        # Formatear salida en texto
        output = [f"🔍 Búsqueda: \"{prompt}\"",
                 f"🔑 Keyphrases extraídas: {keyphrases}",
                 f"\n✨ Top {len(recommendations)} Recomendaciones:"]
        
        for i, rec in enumerate(recommendations, 1):
            output.append(f"\n{i}. {rec.get('name', 'Sin título')}")
            
            if 'english_name' in rec and rec['english_name']:
                output.append(f"   Título en inglés: {rec['english_name']}")
                
            if 'recommendation_score' in rec:
                output.append(f"   Puntuación de recomendación: {rec['recommendation_score']:.3f}")
                
            if 'score' in rec:
                output.append(f"   Puntuación MAL: {rec['score']}")
                
            if 'favorites' in rec:
                output.append(f"   Favoritos: {rec['favorites']:,}")
                
            if 'genres' in rec:
                output.append(f"   Géneros: {rec['genres']}")
                
            if 'episodes' in rec:
                output.append(f"   Episodios: {rec['episodes']}")
                
            if 'aired' in rec:
                output.append(f"   Emitido: {rec['aired']}")
                
            if 'synopsis' in rec:
                synopsis = rec['synopsis']
                if len(synopsis) > 200:
                    synopsis = synopsis[:200] + "..."
                output.append(f"   Sinopsis: {synopsis}")
                
            output.append("   " + "-"*50)
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error al generar recomendaciones: {str(e)}"

def get_recommendations_json(prompt: str, system: Optional[ImprovedAnimeRecommendationSystem] = None,
                           top_n: int = 10) -> str:
    """Genera recomendaciones en formato JSON"""
    try:
        # Crear sistema si no se proporcionó uno
        if system is None:
            system = ImprovedAnimeRecommendationSystem()
        
        # Procesar prompt y obtener keyphrases
        keyphrases = system.process_prompt(prompt)
        
        if not keyphrases:
            return json.dumps({"error": "No se pudieron extraer keyphrases del prompt."})
        
        # Obtener recomendaciones
        recommendations = system.get_recommendations(keyphrases, top_n)
        
        # Crear estructura JSON
        result = {
            "prompt": prompt,
            "keyphrases": keyphrases,
            "recommendations": recommendations
        }
        
        # Convertir a JSON
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    """Función principal para usar desde línea de comandos"""
    # Prompt de prueba por defecto
    default_prompt = "aspiring idols who wants to be the best"
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Sistema de recomendación de anime mejorado")
    parser.add_argument("prompt", nargs="?", default=default_prompt,
                       help=f"Prompt de recomendación (default: {default_prompt})")
    parser.add_argument("output_format", nargs="?", choices=["text", "json"], default="text",
                       help="Formato de salida: 'text' o 'json' (default: text)")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Número de recomendaciones a mostrar (default: 10)")
    
    # Procesar argumentos
    args = parser.parse_args()
    
    # Inicializar sistema
    system = ImprovedAnimeRecommendationSystem()
    
    # Obtener recomendaciones en el formato solicitado
    if args.output_format.lower() == "json":
        result = get_recommendations_json(args.prompt, system, args.top_n)
    else:
        result = get_recommendations_text(args.prompt, system, args.top_n)
    
    # Mostrar resultado
    print(result)
    
    return result

if __name__ == "__main__":
    main()
