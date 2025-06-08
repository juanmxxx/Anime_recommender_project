#!/usr/bin/env python3
"""
Manejador de recomendaciones de anime
==================================

Este script procesa prompts de usuario, los tokeniza para extraer keyphrases,
y luego utiliza el modelo entrenado para generar recomendaciones de anime.

Uso:
    python recommendHandler.py "I want to watch an action anime with magic"
    python recommendHandler.py "I want to watch an action anime with magic" json
"""

import sys
import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple

# Agregar directorio actual al path para importar modelTokenizerHandler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el tokenizador
from modelTokenizerHandler import extract_keyphrases, remove_prepositions

# Configuración de rutas
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATASET_CSV = BASE_DIR.parent / "data" / "init-scripts" / "anime-dataset-2023-cleaned.csv"

class AnimeRecommendationSystem:
    """Sistema de recomendación de animes basado en un modelo pre-entrenado"""
    
    def __init__(self):
        """Inicializa el sistema de recomendación cargando el modelo y los datos"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.df = None
        self.model = None
        self.anime_embeddings = None
        self.embedding_model = None
        
        # Cargar modelo y datos
        self._load_model()
        self._load_anime_data()
        
    def _load_model(self):
        """Carga el modelo de recomendación entrenado"""
        try:
            # Cargar información del modelo
            model_info_path = MODEL_DIR / "model_info.json"
            if not model_info_path.exists():
                raise FileNotFoundError(f"Información del modelo no encontrada en {model_info_path}")
                
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            # Cargar SentenceTransformer
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_info['embedding_model'])
            embedding_dim = model_info['embedding_dim']
            
            # Cargar modelo de recomendación
            model_path = MODEL_DIR / "anime_recommender.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
                
            # Definir la arquitectura del modelo
            from torch import nn
            class RecommendationMLP(nn.Module):
                def __init__(self, embedding_dim=384):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(embedding_dim * 2, 512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, keyphrase_emb, anime_emb):
                    x = torch.cat([keyphrase_emb, anime_emb], dim=1)
                    return self.fc(x)            # Crear y cargar el modelo
            # Nota: No usamos weights_only=True porque el modelo contiene otras estructuras de datos
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = RecommendationMLP(embedding_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Cargar embeddings de anime
            embeddings_path = MODEL_DIR / "anime_embeddings.npy"
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings no encontrados en {embeddings_path}")
                
            self.anime_embeddings = np.load(embeddings_path)            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise
    
    def _load_anime_data(self):
        """Carga los datos de anime desde CSV"""
        try:
            # Cargar desde CSV
            if not DATASET_CSV.exists():
                raise FileNotFoundError(f"Dataset no encontrado en {DATASET_CSV}")
                
            self.df = pd.read_csv(DATASET_CSV)
            self.df = self.df.dropna(subset=['Synopsis'])
            
            # Renombrar columnas para consistencia
            column_mapping = {
                'Name': 'name',
                'English name': 'english_name', 
                'Other name': 'other_name',
                'Score': 'score',
                'Genres': 'genres',
                'Synopsis': 'synopsis',
                'Type': 'type',
                'Episodes': 'episodes',
                'Aired': 'aired',
                'Status': 'status',
                'Producers': 'producers',
                'Licensors': 'licensors',
                'Studios': 'studios',
                'Source': 'source',
                'Duration': 'duration',
                'Rating': 'rating',
                'Rank': 'rank',
                'Popularity': 'popularity',
                'Favorites': 'favorites',
                'Image URL': 'image_url'
            }
            
            self.df.rename(columns={k: v for k, v in column_mapping.items() if k in self.df.columns}, inplace=True)
            
            # Asegurar que el tamaño del DataFrame coincida con los embeddings
            if self.anime_embeddings is not None:
                num_embeddings = len(self.anime_embeddings)
                if len(self.df) != num_embeddings:
                    print(f"Ajustando tamaño del DataFrame de {len(self.df)} a {num_embeddings} registros para coincidir con embeddings")
                    if len(self.df) > num_embeddings:
                        self.df = self.df.head(num_embeddings)
                    else:
                        # Si hay más embeddings que filas en el DataFrame, truncamos los embeddings
                        self.anime_embeddings = self.anime_embeddings[:len(self.df)]
            
        except Exception as e:
            print(f"Error al cargar los datos de anime: {e}")
            raise
    
    def process_prompt(self, prompt: str) -> str:
        """Procesa el prompt para extraer keyphrases"""
        # Extraer keyphrases
        keyphrases = extract_keyphrases(prompt)
        
        return keyphrases
    
    def get_recommendations(self, keyphrases: str, top_n: int = 10) -> List[Dict]:
        """Genera recomendaciones basadas en las keyphrases procesadas"""
        # Verificar que el modelo esté cargado
        if self.model is None or self.anime_embeddings is None or self.df is None:
            raise ValueError("Modelo o datos no cargados correctamente")
        
        # Obtener embedding de las keyphrases
        keyphrase_emb = self.embedding_model.encode([keyphrases], convert_to_numpy=True)
        keyphrase_emb = torch.tensor(keyphrase_emb, dtype=torch.float32).to(self.device)
        
        # Convertir embeddings de anime a tensor
        anime_embs = torch.tensor(self.anime_embeddings, dtype=torch.float32).to(self.device)
        
        # Replicar embedding de keyphrases para todos los animes
        keyphrase_emb_repeated = keyphrase_emb.repeat(len(self.anime_embeddings), 1)
        
        # Calcular scores usando el modelo
        with torch.no_grad():
            scores = self.model(keyphrase_emb_repeated, anime_embs).cpu().numpy().flatten()
        
        # Crear una copia del dataframe y añadir scores
        df_with_scores = self.df.copy()
        df_with_scores['recommendation_score'] = scores
        
        # Ordenar por score de recomendación y popularidad
        result = df_with_scores.sort_values(by=['recommendation_score', 'popularity'], 
                                           ascending=[False, True])
        
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

def get_recommendations_text(prompt: str, system: Optional[AnimeRecommendationSystem] = None, 
                           top_n: int = 10) -> str:
    """Genera recomendaciones en formato texto"""
    try:
        # Crear sistema si no se proporcionó uno
        if system is None:
            system = AnimeRecommendationSystem()
        
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

def get_recommendations_json(prompt: str, system: Optional[AnimeRecommendationSystem] = None,
                           top_n: int = 10) -> str:
    """Genera recomendaciones en formato JSON"""
    try:
        # Crear sistema si no se proporcionó uno
        if system is None:
            system = AnimeRecommendationSystem()
        
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
    default_prompt = "I want to watch an action anime with magic powers and strong characters in a fantasy world"
    
    # Configurar argumentos
    parser = argparse.ArgumentParser(description="Sistema de recomendación de anime")
    parser.add_argument("prompt", nargs="?", default=default_prompt,
                       help=f"Prompt de recomendación (default: {default_prompt})")
    parser.add_argument("output_format", nargs="?", choices=["text", "json"], default="text",
                       help="Formato de salida: 'text' o 'json' (default: text)")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Número de recomendaciones a mostrar (default: 10)")
    
    # Procesar argumentos
    args = parser.parse_args()
    
    # Inicializar sistema
    system = AnimeRecommendationSystem()
      # Obtener recomendaciones en el formato solicitado
    if args.output_format.lower() == "json":
        result = get_recommendations_json(args.prompt, system, args.top_n)
    else:
        result = get_recommendations_text(args.prompt, system, args.top_n)
    
    # Mostrar resultado (con manejo de codificación para Windows)
    try:
        print(result)
    except UnicodeEncodeError:
        # En Windows, usar sys.stdout.buffer para manejar caracteres Unicode
        import sys
        sys.stdout.buffer.write(result.encode('utf-8'))
        print("\n")  # Agregar salto de línea después
    
    return result

if __name__ == "__main__":
    main()