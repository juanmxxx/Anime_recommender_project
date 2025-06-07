#!/usr/bin/env python3
"""
Recomendador de Animes con Deep Learning
========================================

Sistema de recomendación basado en keyphrases que usa un modelo de deep learning
para encontrar animes relacionados semánticamente.

Uso:
    python modelRecommenderPreviousTokenizer.py --help
    python modelRecommenderPreviousTokenizer.py train
    python modelRecommenderPreviousTokenizer.py retrain
    python modelRecommenderPreviousTokenizer.py recommend "musical group aspiring idols, personal challenges"
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
import os
import json
import psycopg2
import pandas.io.sql as psql
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# Configuración de rutas
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR.parent / "data"
DATASET_CSV = DATA_DIR / "init-scripts" / "anime-dataset-2023-cleaned.csv"

# Configuración de la base de datos
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'anime_db',
    'password': 'anime_db',
    'dbname': 'animes'
}

class RecommendationMLP(nn.Module):
    """Red neuronal para predicir similitud entre keyphrases y animes"""
    
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
        return self.fc(x)

class AnimeRecommender:
    """Sistema de recomendación de animes basado en keyphrases"""
    
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
        
    def load_data_from_db(self, limit=3000):
        """Carga datos desde PostgreSQL"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            query = """
                SELECT anime_id, name, english_name, other_name, score, genres, 
                       synopsis, type, episodes, aired, status, producers, 
                       licensors, studios, source, duration, rating, rank, 
                       popularity, favorites, image_url
                FROM anime
                WHERE synopsis IS NOT NULL AND synopsis != ''
                ORDER BY anime_id ASC
                LIMIT %s
            """
            self.df = psql.read_sql(query, conn, params=[limit])
            conn.close()
            print(f"Cargados {len(self.df)} animes desde la base de datos")
        except Exception as e:
            print(f"Error conectando a la base de datos: {e}")
            print("Cargando desde CSV como respaldo...")
            self.load_data_from_csv(limit)
    
    def load_data_from_csv(self, limit=3000):
        """Carga datos desde CSV"""
        self.df = pd.read_csv(DATASET_CSV)
        self.df = self.df.head(limit)
        self.df = self.df.dropna(subset=['Synopsis'])
        self.df.rename(columns={'Synopsis': 'synopsis', 'Name': 'name', 'English name': 'english_name', 
                               'Popularity': 'popularity', 'Aired': 'aired', 'Genres': 'genres'}, inplace=True)
        print(f"Cargados {len(self.df)} animes desde CSV")
    
    def generate_training_data(self):
        """Genera datos de entrenamiento etiquetados"""
        
        # Datos etiquetados manualmente basados en géneros y sinopsis
        labeled_data = {
            "musical group aspiring idols, personal challenges, biggest concert year": [
                "Love Live! School Idol Project", "The Idolmaster", "AKB0048", 
                "Wake Up, Girls!", "Aikatsu!"
            ],
            "supernatural mystery, detective, crimes, ancient folklore": [
                "Death Note", "Psycho-Pass", "Monster", "Detective Conan", "Hell Girl"
            ],
            "mecha robots, giant battles, post apocalyptic world": [
                "Neon Genesis Evangelion", "Mobile Suit Gundam", "Code Geass", 
                "Darling in the FranXX", "Aldnoah.Zero"
            ],
            "slice of life, school, friendship, daily struggles": [
                "K-On!", "Clannad", "Azumanga Daioh", "Lucky Star", "Nichijou"
            ],
            "action adventure, martial arts, tournaments, power levels": [
                "Dragon Ball Z", "Naruto", "One Piece", "Bleach", "My Hero Academia"
            ],
            "romance comedy, high school, misunderstandings, confessions": [
                "Toradora!", "Lovely Complex", "Nisekoi", "School Rumble", "Kaguya-sama"
            ],
            "fantasy magic, wizards, epic quests, medieval world": [
                "Fairy Tail", "Magi", "Re:Zero", "Overlord", "Konosuba"
            ],
            "psychological thriller, mind games, manipulation, dark themes": [
                "Death Note", "Future Diary", "Psycho-Pass", "Monster", "Liar Game"
            ],
            "sports competition, teamwork, training, championships": [
                "Haikyuu!!", "Kuroko no Basket", "Free!", "Ace of Diamond", "Prince of Tennis"
            ],
            "horror supernatural, ghosts, demons, scary atmosphere": [
                "Another", "Corpse Party", "Hell Girl", "Shiki", "Ghost Hunt"
            ]
        }
        
        # Guardar datos etiquetados
        with open(MODEL_DIR / "training_labels.json", 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, ensure_ascii=False, indent=2)
        
        # Generar pares de entrenamiento
        training_pairs = []
        
        for keyphrase, anime_names in labeled_data.items():
            for anime_name in anime_names:
                # Buscar el anime en el dataset
                matches = self.df[self.df['name'].str.contains(anime_name, case=False, na=False)]
                if not matches.empty:
                    anime_idx = matches.index[0]
                    training_pairs.append((keyphrase, anime_idx, 1.0))  # Positivo
        
        # Generar ejemplos negativos
        negative_pairs = []
        for _ in range(len(training_pairs)):
            random_keyphrase = random.choice(list(labeled_data.keys()))
            random_idx = random.choice(self.df.index)
            
            # Verificar que no sea un ejemplo positivo
            is_positive = False
            for keyphrase, anime_names in labeled_data.items():
                if keyphrase == random_keyphrase:
                    anime_name = self.df.loc[random_idx, 'name']
                    if any(name.lower() in anime_name.lower() for name in anime_names):
                        is_positive = True
                        break
            
            if not is_positive:
                negative_pairs.append((random_keyphrase, random_idx, 0.0))
        
        training_pairs.extend(negative_pairs)
        random.shuffle(training_pairs)
        
        print(f"Generados {len(training_pairs)} pares de entrenamiento")
        print(f"Positivos: {len([p for p in training_pairs if p[2] == 1.0])}")
        print(f"Negativos: {len([p for p in training_pairs if p[2] == 0.0])}")
        
        return training_pairs
    
    def compute_embeddings(self):
        """Calcula embeddings para todas las sinopsis"""
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecuta load_data_from_db() primero.")
        
        # Limpiar sinopsis
        synopses = self.df['synopsis'].fillna('').astype(str).tolist()
        
        print("Calculando embeddings de animes...")
        self.anime_embeddings = self.model.encode(
            synopses, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # Guardar embeddings
        np.save(MODEL_DIR / "anime_embeddings.npy", self.anime_embeddings)
        print(f"Embeddings guardados: {self.anime_embeddings.shape}")
    
    def train_model(self, epochs=50, lr=1e-3, batch_size=32):
        """Entrena el modelo de deep learning"""
        
        # Cargar datos y calcular embeddings
        self.load_data_from_db()
        self.compute_embeddings()
        
        # Generar datos de entrenamiento
        training_data = self.generate_training_data()
        
        # Dividir en train/val
        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
        
        # Inicializar modelo
        self.deep_model = RecommendationMLP(self.embedding_dim).to(self.device)
        optimizer = optim.Adam(self.deep_model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Convertir anime_embeddings a tensor
        anime_embs_tensor = torch.tensor(self.anime_embeddings, dtype=torch.float32).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Entrenamiento
            self.deep_model.train()
            train_losses = []
            
            # Mezclar datos de entrenamiento
            random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Preparar batch
                keyphrases = [item[0] for item in batch]
                anime_indices = [item[1] for item in batch]
                labels = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(self.device)
                
                # Embeddings de keyphrases
                keyphrase_embs = self.model.encode(keyphrases, convert_to_numpy=True)
                keyphrase_embs = torch.tensor(keyphrase_embs, dtype=torch.float32).to(self.device)
                
                # Embeddings de animes
                anime_embs = anime_embs_tensor[anime_indices]
                
                # Forward pass
                predictions = self.deep_model(keyphrase_embs, anime_embs).squeeze()
                loss = criterion(predictions, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validación
            self.deep_model.eval()
            val_losses = []
            
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    
                    keyphrases = [item[0] for item in batch]
                    anime_indices = [item[1] for item in batch]
                    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32).to(self.device)
                    
                    keyphrase_embs = self.model.encode(keyphrases, convert_to_numpy=True)
                    keyphrase_embs = torch.tensor(keyphrase_embs, dtype=torch.float32).to(self.device)
                    anime_embs = anime_embs_tensor[anime_indices]
                    
                    predictions = self.deep_model(keyphrase_embs, anime_embs).squeeze()
                    val_loss = criterion(predictions, labels)
                    val_losses.append(val_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Guardar mejor modelo
                torch.save({
                    'model_state_dict': self.deep_model.state_dict(),
                    'embedding_dim': self.embedding_dim,
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, MODEL_DIR / "anime_recommender.pt")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping activado")
                    break
            
            scheduler.step(avg_val_loss)
        
        print("Entrenamiento completado")
        self.save_model_info()
    
    def save_model_info(self):
        """Guarda información del modelo"""
        info = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dim': self.embedding_dim,
            'dataset_size': len(self.df),
            'device': str(self.device),
            'model_path': str(MODEL_DIR / "anime_recommender.pt"),
            'embeddings_path': str(MODEL_DIR / "anime_embeddings.npy")
        }
        
        with open(MODEL_DIR / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_model(self):
        """Carga el modelo entrenado"""
        model_path = MODEL_DIR / "anime_recommender.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Entrena el modelo primero.")
        
        # Cargar información del modelo
        info_path = MODEL_DIR / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                self.embedding_dim = model_info['embedding_dim']
        
        # Cargar modelo
        checkpoint = torch.load(model_path, map_location=self.device)
        self.deep_model = RecommendationMLP(self.embedding_dim).to(self.device)
        self.deep_model.load_state_dict(checkpoint['model_state_dict'])
        self.deep_model.eval()
        
        # Cargar embeddings y datos
        self.anime_embeddings = np.load(MODEL_DIR / "anime_embeddings.npy")
        
        # Cargar datos (desde DB o CSV)
        try:
            self.load_data_from_db()
        except:
            self.load_data_from_csv()
        
        print("Modelo cargado exitosamente")
    
    def recommend(self, keyphrases: str, top_n: int = 10) -> List[Dict]:
        """Genera recomendaciones usando el modelo entrenado"""
        if self.deep_model is None:
            self.load_model()
        
        # Procesar keyphrases
        keyphrase_emb = self.model.encode([keyphrases], convert_to_numpy=True)
        keyphrase_emb = torch.tensor(keyphrase_emb, dtype=torch.float32).to(self.device)
        
        # Repetir para todos los animes
        keyphrase_emb_repeated = keyphrase_emb.repeat(len(self.anime_embeddings), 1)
        anime_embs = torch.tensor(self.anime_embeddings, dtype=torch.float32).to(self.device)
        
        # Calcular scores
        with torch.no_grad():
            scores = self.deep_model(keyphrase_emb_repeated, anime_embs).cpu().numpy().flatten()
        
        # Añadir scores al dataframe
        self.df['score'] = scores
        
        # Ordenar por score y popularidad
        result = self.df.sort_values(by=['score', 'popularity'], ascending=[False, False])
        top_recommendations = result.head(top_n)
        
        # Formatear resultados
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendations.append({
                'name': row['name'],
                'english_name': row.get('english_name', ''),
                'score': float(row['score']),
                'genres': row.get('genres', ''),
                'synopsis': row.get('synopsis', '')[:200] + '...' if len(str(row.get('synopsis', ''))) > 200 else row.get('synopsis', ''),
                'popularity': row.get('popularity', 0),
                'aired': row.get('aired', '')
            })
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(
        description="Recomendador de Animes con Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python modelRecommenderPreviousTokenizer.py train
  python modelRecommenderPreviousTokenizer.py retrain
  python modelRecommenderPreviousTokenizer.py recommend "musical group aspiring idols, personal challenges"
  python modelRecommenderPreviousTokenizer.py recommend "supernatural mystery, detective, crimes"
        """
    )
    
    parser.add_argument('command', choices=['train', 'retrain', 'recommend'],
                       help='Comando a ejecutar')
    parser.add_argument('keyphrases', nargs='?', default='',
                       help='Keyphrases para recomendación (solo para comando recommend)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Número de recomendaciones a mostrar (default: 10)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas de entrenamiento (default: 50)')
    
    args = parser.parse_args()
    
    recommender = AnimeRecommender()
    
    if args.command in ['train', 'retrain']:
        print(f"Iniciando {args.command}...")
        recommender.train_model(epochs=args.epochs)
        print("Modelo entrenado y guardado exitosamente")
        
    elif args.command == 'recommend':
        if not args.keyphrases:
            print("Error: Debes proporcionar keyphrases para la recomendación")
            parser.print_help()
            return
        
        print(f"Generando recomendaciones para: '{args.keyphrases}'")
        recommendations = recommender.recommend(args.keyphrases, args.top_n)
        
        print(f"\n🎯 Top {len(recommendations)} Recomendaciones:")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            if rec['english_name']:
                print(f"   Inglés: {rec['english_name']}")
            print(f"   Score: {rec['score']:.3f}")
            print(f"   Géneros: {rec['genres']}")
            print(f"   Popularidad: {rec['popularity']}")
            print(f"   Sinopsis: {rec['synopsis']}")
            print("-" * 80)

if __name__ == "__main__":
    main()