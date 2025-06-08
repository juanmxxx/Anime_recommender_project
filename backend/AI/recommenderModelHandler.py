#!/usr/bin/env python3
"""
Sistema de Recomendación de Animes con Deep Learning Mejorado
=============================================================

Combina las mejoras del sistema basado en contenido con un modelo de Deep Learning
que incorpora:
- Scoring de popularidad (favoritos)
- Factor de recency (fecha de emisión)
- Detección especializada de ídolos
- Expansión de keyphrases
- Arquitectura de red neuronal mejorada

Uso:
    python modelRecommenderImproved.py train
    python modelRecommenderImproved.py recommend "aspiring idols who wants to be the best"
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import os
import json
import random
import re
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import psycopg2
from sqlalchemy import create_engine

# Agregar directorio actual al path para importar modelTokenizerHandler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phraseTokernizer import extract_keyphrases


# Configuración de rutas
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# Configuración de base de datos
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'animes',
    'user': 'anime_db',
    'password': 'anime_db'
}

class ImprovedRecommendationMLP(nn.Module):
    """Red neuronal mejorada para recomendación con features adicionales"""
    
    def __init__(self, embedding_dim=384, additional_features=4):
        super().__init__()
        
        # Capa de procesamiento de embeddings con conexiones residuales
        self.embedding_input = nn.Linear(embedding_dim * 2, 512)
        self.embedding_norm1 = nn.LayerNorm(512)
        self.embedding_block1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.embedding_block2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capa de procesamiento de features adicionales mejorada
        self.feature_processor = nn.Sequential(
            nn.Linear(additional_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism simple para features
        self.attention = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Capas de combinación con conexiones residuales y normalización
        self.combiner_input = nn.Linear(512 + 64, 256)
        self.combiner_norm1 = nn.LayerNorm(256)
        
        self.combiner_block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.combiner_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        # Capa final con menos dropout para mejor precisión
        self.final_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Inicialización de pesos Xavier/Glorot
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa los pesos usando Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, keyphrase_emb, anime_emb, additional_features):
        # Procesar embeddings con conexiones residuales
        combined_emb = torch.cat([keyphrase_emb, anime_emb], dim=1)
        emb_out = self.embedding_input(combined_emb)
        emb_out = self.embedding_norm1(emb_out)
        emb_out = nn.ReLU()(emb_out)
        
        # Bloques residuales para embeddings
        residual = emb_out
        emb_out = self.embedding_block1(emb_out)
        emb_out = emb_out + residual  # Conexión residual
        
        residual = emb_out
        emb_out = self.embedding_block2(emb_out)
        emb_out = emb_out + residual  # Conexión residual
        
        # Procesar features adicionales
        add_features = self.feature_processor(additional_features)
        
        # Combinar todas las features
        combined = torch.cat([emb_out, add_features], dim=1)
        
        # Aplicar atención simple (opcional, para dar más importancia a ciertas features)
        combined_out = self.combiner_input(combined)
        combined_out = self.combiner_norm1(combined_out)
        combined_out = nn.ReLU()(combined_out)
        
        # Bloques con conexiones residuales en el combinador
        residual = combined_out
        combined_out = self.combiner_block1(combined_out)
        if combined_out.shape == residual.shape:
            combined_out = combined_out + residual
        
        combined_out = self.combiner_block2(combined_out)
        
        # Capa final
        output = self.final_layers(combined_out)
        
        return output

class ImprovedAnimeRecommender:
    """Sistema de recomendación de animes mejorado con Deep Learning"""
    
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Crear directorio de modelos si no existe
        MODEL_DIR.mkdir(exist_ok=True)
        
        self.df = None
        self.anime_embeddings = None
        self.deep_model = None
        self.scaler = StandardScaler()
    
        # Expansión de keyphrases por categorías
        self.keyword_expansion = {
            'idol': ['singer', 'performer', 'entertainment', 'dreams', 'goals', 'ambitions'],
            'action': ['fight', 'battle', 'combat', 'fighting', 'martial arts', 'warrior'],
            'romance': ['love', 'romantic', 'dating', 'relationship', 'couple'],
            'comedy': ['funny', 'humor', 'humorous', 'comedic', 'hilarious'],
            'fantasy': ['magic', 'magical', 'supernatural', 'mystical', 'wizard'],
            'sci-fi': ['science fiction', 'future', 'technology', 'space', 'robot'],
            'drama': ['emotional', 'serious', 'tragic', 'melodrama'],
            'horror': ['scary', 'frightening', 'terror', 'supernatural'],
            'mystery': ['detective', 'investigation', 'puzzle', 'suspense']
        }
        
        # Keywords específicos para ídolos
        self.idol_keywords = ['idol', 'singer', 'performer', 'entertainment', 'dreams', 'goals', 'ambitions']
    
    def get_db_connection(self):
        """Crea conexión a la base de datos PostgreSQL"""
        try:
            conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            engine = create_engine(conn_string)
            return engine
        except Exception as e:
            print(f"Error conectando a la base de datos: {e}")
            raise
    
    def load_data_from_database(self):
        """Carga datos desde PostgreSQL con filtrado de 'Not Yet Aired'"""
        try:
            engine = self.get_db_connection()
            
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
            
            # Resetear índice para usar como anime_id interno
            self.df = self.df.reset_index(drop=True)
            
            print(f"Dataset cargado desde PostgreSQL: {len(self.df)} animes")
            print(f"Registros con 'Not Yet Aired' excluidos")
            
            # Mostrar estadísticas básicas
            # Extraer años y calcular rango de manera más segura
            years = self.df['aired'].str.extract(r'(\d{4})').dropna()
            if not years.empty:
                min_year = int(years.astype(int).min())
                max_year = int(years.astype(int).max())
                year_range = f"{min_year} - {max_year}"
            else:
                year_range = "N/A"
                
            print(f"Rango de años: {year_range}")
            print(f"Promedio de favoritos: {self.df['favorites'].mean():.0f}")
            
        except Exception as e:
            print(f"Error al cargar los datos de la base de datos: {e}")
            raise
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    def load_data_from_csv(self):
        """Método legacy - mantener por compatibilidad pero usar load_data_from_database"""
        print("Método CSV deshabilitado. Usando base de datos PostgreSQL...")
        self.load_data_from_database()
    
    def extract_year_from_aired(self, aired_str: str) -> int:
        """Extrae el año de la fecha de emisión"""
        if pd.isna(aired_str) or aired_str == 'Not available':
            return 1990  # Año por defecto para animes sin fecha
        
        # Buscar años de 4 dígitos en el string
        years = re.findall(r'\b(19|20)\d{2}\b', str(aired_str))
        if years:
            return int(years[0])
        return 1990
    
    def calculate_additional_features(self, anime_row, keyphrase: str) -> np.ndarray:
        """Calcula features adicionales para el modelo"""
        
        # 1. Popularity score (normalizado)
        favorites = float(anime_row['favorites']) if pd.notna(anime_row['favorites']) else 0
        max_favorites = self.df['favorites'].max() if len(self.df) > 0 else 1
        popularity_score = favorites / max_favorites if max_favorites > 0 else 0
        
        # 2. Recency score (basado en año de emisión)
        year = self.extract_year_from_aired(anime_row['aired'])
        current_year = 2025
        years_diff = current_year - year
        recency_score = max(0, 1 - years_diff / 35) if years_diff >= 0 else 0
        
        # 3. Content relevance score (búsqueda en texto)
        synopsis = str(anime_row['synopsis']).lower()
        genres = str(anime_row['genres']).lower()
        name = str(anime_row['name']).lower()
        
        keyphrase_lower = keyphrase.lower()
        content_matches = 0
        
        # Contar coincidencias en synopsis
        for word in keyphrase_lower.split():
            if word in synopsis:
                content_matches += 2
            if word in genres:
                content_matches += 3
            if word in name:
                content_matches += 1
        
        content_score = min(1.0, content_matches / 10)
        
        # 4. Idol detection score
        idol_score = 0
        if any(keyword in keyphrase_lower for keyword in self.idol_keywords):
            # Buscar series conocidas de ídolos
            idol_series = ['love live', 'idolmaster', 'aikatsu', 'pretty rhythm', 'idol school']
            for series in idol_series:
                if series in name or series in synopsis:
                    idol_score = 1.0
                    break
            
            # Buscar keywords de ídolos en géneros y synopsis
            if 'idol' in genres or 'music' in genres:
                idol_score = max(idol_score, 0.8)
            
            idol_keywords_in_text = sum(1 for keyword in self.idol_keywords 
                                      if keyword in synopsis or keyword in genres)
            idol_score = max(idol_score, min(1.0, idol_keywords_in_text / 3))
        
        return np.array([popularity_score, recency_score, content_score, idol_score], dtype=np.float32)
    
    def process_keyphrase(self, keyphrase: str) -> str:
        """Procesa y expande keyphrases"""
        # Extraer keyphrases usando el tokenizador existente
        extracted = extract_keyphrases(keyphrase)
        
        # Si no se extraen suficientes keyphrases, usar el prompt original
        if not extracted or len(extracted.split()) < 2:
            extracted = keyphrase
        
        # Expandir keywords
        expanded_words = set(extracted.lower().split())
        
        for word in list(expanded_words):
            if word in self.keyword_expansion:
                expanded_words.update(self.keyword_expansion[word])
        
        # Agregar expansiones específicas para ídolos
        if any(keyword in keyphrase.lower() for keyword in ['idol', 'aspiring', 'singer']):
            expanded_words.update(['singer', 'performer', 'entertainment', 'dreams', 
                                 'goals', 'ambitions', 'wants to be'])
        
        return ' '.join(expanded_words)
    
    def generate_training_data(self, num_samples_per_anime=5):
        """Genera datos de entrenamiento con etiquetas sintéticas"""
        training_data = []
        
        print("Generando datos de entrenamiento...")
        
        # Templates de queries para diferentes géneros
        query_templates = {
            'action': [
                "action anime with fights and battles",
                "martial arts and combat scenes",
                "warriors fighting against evil",
                "epic battles and action sequences"
            ],
            'romance': [
                "romantic love story between characters",
                "cute couple developing relationship", 
                "romantic comedy with dating",
                "love triangle and romantic drama"
            ],
            'comedy': [
                "funny and hilarious anime",
                "comedy with humor and jokes",
                "lighthearted and entertaining show",
                "comedic situations and funny characters"
            ],
            'idol': [
                "aspiring idols who wants to be the best",
                "singers and performers on stage",
                "idol group with dreams and goals",
                "music and entertainment industry"
            ],
            'fantasy': [
                "magical world with wizards and magic",
                "fantasy adventure with supernatural powers",
                "mystical creatures and magic spells",
                "fantasy realm with magical abilities"
            ],
            'sci-fi': [
                "science fiction with advanced technology",
                "future world with robots and AI",
                "space adventure and exploration",
                "cyberpunk and futuristic setting"
            ]
        }
        
        for idx, anime_row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"Procesando anime {idx}/{len(self.df)}")
            
            genres = str(anime_row['genres']).lower()
            synopsis = str(anime_row['synopsis']).lower()
            
            # Generar muestras positivas (relevantes)
            positive_queries = []
            
            # Basado en géneros
            for genre, templates in query_templates.items():
                if genre in genres:
                    positive_queries.extend(random.sample(templates, min(2, len(templates))))
            
            # Basado en palabras clave del synopsis
            if 'idol' in synopsis or 'singer' in synopsis or 'music' in synopsis:
                positive_queries.extend(query_templates['idol'])
            
            # Si no hay coincidencias específicas, usar queries genéricas
            if not positive_queries:
                all_templates = [t for templates in query_templates.values() for t in templates]
                positive_queries = random.sample(all_templates, 2)
            
            # Agregar muestras positivas
            for query in positive_queries[:num_samples_per_anime//2]:
                training_data.append({
                    'keyphrase': query,
                    'anime_id': anime_row.name,
                    'label': 1.0  # Relevante
                })
            
            # Generar muestras negativas (irrelevantes)
            negative_genres = []
            if 'action' not in genres:
                negative_genres.extend(query_templates['action'])
            if 'romance' not in genres:
                negative_genres.extend(query_templates['romance'])
            if 'comedy' not in genres:
                negative_genres.extend(query_templates['comedy'])
            
            # Seleccionar queries negativas
            if negative_genres:
                negative_queries = random.sample(negative_genres, 
                                               min(num_samples_per_anime//2, len(negative_genres)))
                
                for query in negative_queries:
                    training_data.append({
                        'keyphrase': query,
                        'anime_id': anime_row.name,
                        'label': 0.0  # No relevante
                    })
        
        print(f"Generados {len(training_data)} samples de entrenamiento")
        return training_data
    
    def compute_anime_embeddings(self):
        """Calcula embeddings para todos los animes"""
        print("Calculando embeddings de animes...")
        
        anime_texts = []
        for _, row in self.df.iterrows():
            # Combinar información relevante del anime
            text_parts = []
            
            if pd.notna(row['name']):
                text_parts.append(str(row['name']))
            if pd.notna(row['english_name']):
                text_parts.append(str(row['english_name']))
            if pd.notna(row['genres']):
                text_parts.append(str(row['genres']))
            if pd.notna(row['synopsis']):
                synopsis = str(row['synopsis'])[:500]  # Limitar longitud
                text_parts.append(synopsis)
            
            anime_text = ' '.join(text_parts)
            anime_texts.append(anime_text)
        
        self.anime_embeddings = self.model.encode(anime_texts, 
                                                convert_to_tensor=True, 
                                                device=self.device)
        
        print(f"Embeddings calculados: {self.anime_embeddings.shape}")
        
        # Guardar embeddings
        embeddings_path = MODEL_DIR / "anime_embeddings_improved.npy"
        np.save(embeddings_path, self.anime_embeddings.cpu().numpy())
        print(f"Embeddings guardados en {embeddings_path}")
    
    def train_model(self, epochs=50, batch_size=32, learning_rate=0.001):
        """Entrena el modelo de deep learning"""
        if self.df is None:
            self.load_data_from_database()  # Cambiar a cargar desde DB
        
        if self.anime_embeddings is None:
            self.compute_anime_embeddings()
        
        # Generar datos de entrenamiento
        training_data = self.generate_training_data()
        
        # Preparar datos para entrenamiento
        print("Preparando datos de entrenamiento...")
        
        keyphrase_embeddings = []
        anime_embeddings = []
        additional_features_list = []
        labels = []
        
        for sample in training_data:
            keyphrase = self.process_keyphrase(sample['keyphrase'])
            anime_idx = sample['anime_id']
            
            # Embedding de keyphrase
            keyphrase_emb = self.model.encode([keyphrase], convert_to_tensor=True, device=self.device)
            keyphrase_embeddings.append(keyphrase_emb[0])
            
            # Embedding de anime
            anime_emb = self.anime_embeddings[anime_idx]
            anime_embeddings.append(anime_emb)
            
            # Features adicionales
            anime_row = self.df.iloc[anime_idx]
            additional_features = self.calculate_additional_features(anime_row, keyphrase)
            additional_features_list.append(additional_features)
            
            labels.append(sample['label'])
        
        # Convertir a tensores
        keyphrase_embeddings = torch.stack(keyphrase_embeddings)
        anime_embeddings = torch.stack(anime_embeddings)
        additional_features_tensor = torch.tensor(additional_features_list, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Normalizar features adicionales
        additional_features_normalized = self.scaler.fit_transform(additional_features_list)
        additional_features_tensor = torch.tensor(additional_features_normalized, dtype=torch.float32, device=self.device)
        
        # Split de datos
        indices = list(range(len(labels)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        
        # Inicializar modelo
        self.deep_model = ImprovedRecommendationMLP(self.embedding_dim).to(self.device)
        
        # Optimizador y función de pérdida
        optimizer = optim.Adam(self.deep_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        print(f"Iniciando entrenamiento por {epochs} épocas...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entrenamiento
            self.deep_model.train()
            train_loss = 0
            num_train_batches = 0
            
            # Shuffle training data
            random.shuffle(train_indices)
            
            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i+batch_size]
                
                batch_keyphrase_emb = keyphrase_embeddings[batch_indices]
                batch_anime_emb = anime_embeddings[batch_indices]
                batch_features = additional_features_tensor[batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                predictions = self.deep_model(batch_keyphrase_emb, batch_anime_emb, batch_features)
                loss = criterion(predictions, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_train_batches += 1
            
            # Validación
            self.deep_model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_indices), batch_size):
                    batch_indices = val_indices[i:i+batch_size]
                    
                    batch_keyphrase_emb = keyphrase_embeddings[batch_indices]
                    batch_anime_emb = anime_embeddings[batch_indices]
                    batch_features = additional_features_tensor[batch_indices]
                    batch_labels = labels_tensor[batch_indices]
                    
                    predictions = self.deep_model(batch_keyphrase_emb, batch_anime_emb, batch_features)
                    loss = criterion(predictions, batch_labels)
                    
                    val_loss += loss.item()
                    num_val_batches += 1
            
            avg_train_loss = train_loss / num_train_batches
            avg_val_loss = val_loss / num_val_batches
            
            print(f"Época {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Guardar mejor modelo
                torch.save({
                    'model_state_dict': self.deep_model.state_dict(),
                    'scaler': self.scaler,
                    'embedding_dim': self.embedding_dim
                }, MODEL_DIR / "anime_recommender_improved.pt")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping activado")
                    break
            
            scheduler.step(avg_val_loss)
        
        print("Entrenamiento completado!")
    
    def load_model(self):
        """Carga el modelo entrenado"""
        model_path = MODEL_DIR / "anime_recommender_improved.pt"
        embeddings_path = MODEL_DIR / "anime_embeddings_improved.npy"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        
        if not embeddings_path.exists():
            raise FileNotFoundException(f"Embeddings no encontrados en {embeddings_path}")
        
        # Cargar datos si no están cargados
        if self.df is None:
            self.load_data_from_database()  # Cambiar a cargar desde DB
        
        # Cargar embeddings
        self.anime_embeddings = torch.tensor(np.load(embeddings_path), device=self.device)
        
        # Cargar modelo
        checkpoint = torch.load(model_path, map_location=self.device)
        self.deep_model = ImprovedRecommendationMLP(self.embedding_dim).to(self.device)
        self.deep_model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        self.deep_model.eval()
        print("Modelo cargado exitosamente")
    
    def recommend(self, keyphrase: str, top_n: int = 10) -> List[Dict]:
        """Genera recomendaciones usando el modelo de deep learning"""
        if self.deep_model is None:
            self.load_model()
        
        # Procesar keyphrase
        processed_keyphrase = self.process_keyphrase(keyphrase)
        
        # Obtener embedding de keyphrase
        keyphrase_embedding = self.model.encode([processed_keyphrase], 
                                              convert_to_tensor=True, 
                                              device=self.device)
        
        # Calcular scores para todos los animes
        scores = []
        
        print(f"Calculando scores para {len(self.df)} animes...")
        
        with torch.no_grad():
            for idx, anime_row in self.df.iterrows():
                # Embedding del anime
                anime_embedding = self.anime_embeddings[idx].unsqueeze(0)
                
                # Features adicionales
                additional_features = self.calculate_additional_features(anime_row, processed_keyphrase)
                additional_features_normalized = self.scaler.transform([additional_features])
                additional_features_tensor = torch.tensor(additional_features_normalized, 
                                                        dtype=torch.float32, 
                                                        device=self.device)
                
                # Predicción del modelo
                prediction = self.deep_model(keyphrase_embedding, 
                                           anime_embedding, 
                                           additional_features_tensor)
                
                score = prediction.item()
                scores.append((idx, score))
        
        # Ordenar por score descendente
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Preparar recomendaciones
        recommendations = []
        for i, (anime_idx, score) in enumerate(scores[:top_n]):
            anime_row = self.df.iloc[anime_idx]
            
            recommendation = {
                'rank': i + 1,
                'anime_id': int(anime_idx),
                'name': anime_row['name'],
                'english_name': anime_row.get('english_name', ''),
                'genres': anime_row['genres'],
                'synopsis': anime_row['synopsis'],
                'score': float(score),
                'favorites': int(anime_row['favorites']) if pd.notna(anime_row['favorites']) else 0,
                'aired': anime_row['aired'],
                'type': anime_row.get('type', ''),
                'episodes': anime_row.get('episodes', ''),
                'image_url': anime_row.get('image_url', '')
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def delete_models(self):
        """Elimina todos los modelos y embeddings guardados"""
        try:
            # Lista ampliada de archivos y patrones a eliminar
            files_to_delete = [
                MODEL_DIR / "anime_recommender_improved.pt",
                MODEL_DIR / "anime_embeddings_improved.npy",
                MODEL_DIR / "*.pt",  # Cualquier otro checkpoint de modelo
                MODEL_DIR / "*.npy",  # Cualquier otro archivo de embeddings
                MODEL_DIR / "*.bin",  # Posibles archivos binarios
                MODEL_DIR / "model_info.json",  # Configuraciones o metadatos
                MODEL_DIR / "*.log"   # Archivos de log de entrenamiento
            ]
            
            deleted_files = []
            
            # Procesar archivos específicos primero
            for i in range(2):
                file_path = files_to_delete[i]
                if file_path.exists():
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    print(f"✓ Eliminado: {file_path}")
                else:
                    print(f"✗ No encontrado: {file_path}")
            
            # Procesar patrones con glob
            for pattern in files_to_delete[2:]:
                for file_path in MODEL_DIR.glob(pattern.name):
                    if file_path.exists() and file_path.is_file():
                        file_path.unlink()
                        deleted_files.append(str(file_path))
                        print(f"✓ Eliminado: {file_path}")
            
            # Limpiar variables de clase
            self.anime_embeddings = None
            self.deep_model = None
            self.scaler = StandardScaler()  # Reiniciar scaler
            
            # Mostrar resumen
            if deleted_files:
                print(f"\n🗑️  Se eliminaron {len(deleted_files)} archivos:")
                for file in deleted_files:
                    print(f"   - {file}")
            else:
                print("📂 No se encontraron archivos de modelo para eliminar")
            
            print("\n💾 Memoria liberada y modelos eliminados")
                
        except Exception as e:
            print(f"❌ Error al eliminar modelos: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Sistema de Recomendación de Animes Mejorado')
    parser.add_argument('action', choices=['train', 'recommend', 'delete'], 
                       help='Acción a realizar: train, recommend o delete')
    parser.add_argument('query', nargs='?', 
                       help='Query para recomendación (requerido para recommend)')
    parser.add_argument('--top_n', type=int, default=10, 
                       help='Número de recomendaciones (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Número de épocas para entrenamiento (default: 50)')
    
    args = parser.parse_args()
    
    recommender = ImprovedAnimeRecommender()
    
    if args.action == 'train':
        print("Iniciando entrenamiento del modelo mejorado...")
        recommender.train_model(epochs=args.epochs)
        print("Entrenamiento completado!")
    
    elif args.action == 'recommend':
        if not args.query:
            print("Error: Se requiere una query para generar recomendaciones")
            parser.print_help()
            return
        
        print(f"Generando recomendaciones para: '{args.query}'")
        recommendations = recommender.recommend(args.query, args.top_n)
        
        print(f"\n🔍 Top {len(recommendations)} Recomendaciones para: '{args.query}'")
        print("=" * 80)
        
        for rec in recommendations:
            print(f"\n{rec['rank']}. {rec['name']}")
            if rec['english_name']:
                print(f"   Título en inglés: {rec['english_name']}")
            print(f"   Score de ML: {rec['score']:.4f}")
            print(f"   Favoritos: {rec['favorites']:,}")
            print(f"   Géneros: {rec['genres']}")
            print(f"   Tipo: {rec['type']} | Episodios: {rec['episodes']}")
            print(f"   Emitido: {rec['aired']}")
            synopsis = rec['synopsis'][:200] + "..." if len(rec['synopsis']) > 200 else rec['synopsis']
            print(f"   Sinopsis: {synopsis}")
            print("-" * 80)
    
    elif args.action == 'delete':
        print("🗑️  Eliminando modelos existentes...")
        recommender.delete_models()
        print("✅ Proceso de eliminación completado!")

if __name__ == "__main__":
    main()
