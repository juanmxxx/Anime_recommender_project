import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import json

# Load dataset
DATASET_PATH = './dataset/small2.csv'
df = pd.read_csv(DATASET_PATH)

# Fill missing values
for col in ["Score", "Rank", "Popularity", "Favorites"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

def preprocess_text(row):
    # Combine genres and synopsis for text input
    genres = row["Genres"] if isinstance(row["Genres"], str) else ""
    synopsis = row["Synopsis"] if isinstance(row["Synopsis"], str) else ""
    return f"{genres} {synopsis}"

df["text"] = df.apply(preprocess_text, axis=1)

# Normalize numeric features
numeric_cols = [col for col in ["Score", "Rank", "Popularity", "Favorites"] if col in df.columns]
if numeric_cols:
    scaler = MinMaxScaler()
    norm_cols = [f"{col}_norm" for col in numeric_cols]
    df[norm_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save the scaler for later use
    os.makedirs("./model", exist_ok=True)
    with open("./model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Load transformer model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

# Detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Function to get BERT embeddings for a batch of texts (batched for efficiency)
def get_bert_embeddings(texts, tokenizer, model, device=DEVICE, max_length=128, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Prepare anime embeddings (cache for efficiency)
if not hasattr(df, "bert_emb"):
    print("Generating BERT embeddings for anime dataset...")
    anime_embeddings = get_bert_embeddings(df["text"].tolist(), tokenizer, bert_model, device=DEVICE)
    df["bert_emb"] = list(anime_embeddings)
    
    # Save embeddings for later use
    np.save("./model/anime_embeddings.npy", anime_embeddings)
    
    # Save anime data (excluding embeddings)
    df_to_save = df.drop(columns=["bert_emb"])
    df_to_save.to_pickle("./model/anime_data.pkl")
else:
    anime_embeddings = np.stack(df["bert_emb"].values)

# Build the recommendation model
class AnimeRecommender(nn.Module):
    def __init__(self, emb_dim, num_features):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim + num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# Funci�n para evaluar la relevancia de un anime con respecto a las palabras clave
def calculate_relevance(anime_row, keywords, embeddings):
    """
    Calcula la relevancia de un anime con respecto a las palabras clave de b�squeda.
    Devuelve True si el anime es relevante, False en caso contrario.
    """
    # Convertir palabras clave a min�sculas para comparaci�n sin distinci�n de may�sculas
    keywords_lower = keywords.lower()
    keyword_list = keywords_lower.split()
    
    # Verificar coincidencias directas en los campos importantes
    genres = str(anime_row.get("Genres", "")).lower()
    name = str(anime_row.get("Name", "")).lower()
    synopsis = str(anime_row.get("Synopsis", "")).lower()
    
    # Contar coincidencias de palabras clave
    keyword_matches = sum(1 for kw in keyword_list if kw in genres or kw in name or kw in synopsis)
    
    # Si hay al menos una coincidencia directa, considerar relevante
    if keyword_matches > 0:
        return True
    
    # Si no hay coincidencias directas, verificar similitud sem�ntica
    if "bert_emb" in anime_row:
        anime_emb = anime_row["bert_emb"]
    else:
        anime_idx = anime_row.name
        anime_emb = embeddings[anime_idx]
    
    # Obtener embedding de las palabras clave
    kw_emb = get_bert_embeddings([keywords], tokenizer, bert_model, device=DEVICE)[0]
    
    # Calcular similitud coseno
    sim = np.sum(anime_emb * kw_emb) / (np.linalg.norm(anime_emb) * np.linalg.norm(kw_emb) + 1e-8)
    
    # Umbral de similitud para considerar relevante
    # (Este valor puede ajustarse seg�n los resultados)
    return sim > 0.15  # Umbral conservador

# Function to train the model
def train_model(model, anime_embeddings, df, device=DEVICE, epochs=50, batch_size=64):
    print("Training recommendation model...")
    model = model.to(device)
    
    # Prepare training data
    # Use cosine similarity to create positive and negative examples
    n_samples = len(anime_embeddings)
    X_train = []
    y_train = []
    
    # Create a simple dataset for demonstration
    # In a real scenario, you'd want to use actual user preferences
    for i in range(n_samples):
        # Get embedding and numeric features for current anime
        emb = anime_embeddings[i]
        
        norm_cols = [col for col in df.columns if col.endswith("_norm")]
        num_features = df.iloc[i][norm_cols].values.astype(np.float32)
        
        # Combine embedding and numeric features
        features = np.concatenate([emb, num_features])
        X_train.append(features)
        
        # For demonstration purpose, we'll use popularity as our target
        # Higher popularity (lower rank) means better anime (according to general consensus)
        if "Popularity_norm" in df.columns:
            y = 1.0 - df.iloc[i]["Popularity_norm"]  # Invert so higher values are better
        else:
            y = 0.5  # Default middle value if no popularity info
        y_train.append([y])
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if epoch > 0 and hasattr(optimizer, "param_groups"):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./model/anime_recommender.pt")
        
        # Adjust learning rate
        scheduler.step(val_loss)
    
    print("Model training complete!")
    return model

# Instantiate model
emb_dim = anime_embeddings.shape[1]
norm_cols = [col for col in df.columns if col.endswith("_norm")]
num_features = len(norm_cols)
model = AnimeRecommender(emb_dim, num_features)

# Check if the model already exists, if not, train it
model_path = "./model/anime_recommender.pt"
if not os.path.exists(model_path):
    model = train_model(model, anime_embeddings, df, device=DEVICE)
else:
    print(f"Loading existing model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

# --- DEEP LEARNING RECOMMENDATION MODEL ---
def recommend_anime(keywords: str, top_n: int = 10) -> pd.DataFrame:
    """
    Deep learning anime recommendation model using BERT embeddings and a trained neural network.
    INPUT:  keywords (str) - e.g. 'Comedy magic romance'
    OUTPUT: DataFrame with ALL columns for the top N recommended anime (most famous and relevant first)
    """
    # 1. Get BERT embedding for the input keywords
    kw_emb = get_bert_embeddings([keywords], tokenizer, bert_model, device=DEVICE)[0]
    
    # 2. Compute similarity scores for each anime using our trained model
    model.eval()  # Set model to evaluation mode
    predictions = []
    similarities = []
    explanations = []
    
    # Lista de t�rminos de b�squeda
    search_terms = [term.lower() for term in keywords.split()]
    
    with torch.no_grad():
        # First calculate semantic similarities for all animes with the query
        for i in range(len(df)):
            # Get embedding for current anime
            anime_emb = anime_embeddings[i]
            
            # Calculate cosine similarity between input keywords and anime
            sim = np.sum(anime_emb * kw_emb) / (np.linalg.norm(anime_emb) * np.linalg.norm(kw_emb) + 1e-8)
            similarities.append(sim)
        
        # Normalize similarities to [0, 1] range
        similarities = np.array(similarities)
        min_sim, max_sim = similarities.min(), similarities.max()
        if max_sim > min_sim:
            similarities = (similarities - min_sim) / (max_sim - min_sim)
        
        # Now compute final scores with higher weight on similarity
        for i in range(len(df)):
            # Get embedding for current anime
            anime_emb = anime_embeddings[i]
            
            # Get numeric features
            norm_cols = [col for col in df.columns if col.endswith("_norm")]
            num_features = df.iloc[i][norm_cols].values.astype(np.float32)
            
            # Get genres and other metadata for additional matching
            anime_genres = str(df.iloc[i].get("Genres", "")).lower() if "Genres" in df.columns else ""
            anime_name = str(df.iloc[i].get("Name", "")).lower() if "Name" in df.columns else ""
            anime_synopsis = str(df.iloc[i].get("Synopsis", "")).lower() if "Synopsis" in df.columns else ""
            
            # Combine embedding and numeric features (same as during training)
            features = np.concatenate([anime_emb, num_features])
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Get predicted score from model (popularity prediction)
            model_score = model(features_tensor).item()
            
            # Calculate similarity importance
            sim = similarities[i]
            
            # Find matching terms for explanation
            matched_terms = []
            for term in search_terms:
                if term in anime_genres:
                    matched_terms.append(f"g�nero '{term}'")
                elif term in anime_name:
                    matched_terms.append(f"t�tulo '{term}'")
                elif term in anime_synopsis:
                    matched_terms.append(f"sinopsis '{term}'")
                    
            # Create explanation
            if matched_terms:
                explanation = f"Coincide con {', '.join(matched_terms)}"
            else:
                explanation = f"Tem�tica similar a '{keywords}'"
                
            # Add popularity factor if it's a highly rated anime
            if df.iloc[i].get("Score", 0) > 8:
                explanation += f" y es muy popular (Score: {df.iloc[i].get('Score', 0)})"
                
            explanations.append(explanation)
                
            # Keyword matching bonus - check if any keyword appears in the anime's genres or title
            keyword_bonus = 0.0
            for keyword in search_terms:
                if keyword in anime_genres:
                    keyword_bonus += 0.2  # Higher bonus for genre match
                if keyword in anime_name:
                    keyword_bonus += 0.15  # Good bonus for name match
                if keyword in anime_synopsis:
                    keyword_bonus += 0.1  # Small bonus for synopsis match
            
            # Final score: heavy weight on semantic similarity and keyword matching
            final_score = 0.15 * model_score + 0.55 * sim + 0.3 * keyword_bonus
            
            predictions.append(final_score)
    
    # Create a new column for explanations
    explanation_series = pd.Series(explanations, index=df.index)
    df["explanation"] = explanation_series
    
    # 3. Get indices of top N anime based on model predictions
    top_idx = np.argsort(predictions)[-min(len(df), top_n*3):][::-1]  # Get more candidates first
    
    # 4. Filter by relevance
    relevant_idx = []
    for idx in top_idx:
        if calculate_relevance(df.iloc[idx], keywords, anime_embeddings):
            relevant_idx.append(idx)
            if len(relevant_idx) >= top_n:
                break
    
    # If we couldn't find enough relevant animes, use the top scored ones
    if len(relevant_idx) < top_n:
        remaining = top_n - len(relevant_idx)
        for idx in top_idx:
            if idx not in relevant_idx:
                relevant_idx.append(idx)
                remaining -= 1
                if remaining <= 0:
                    break
    
    # 5. Return ALL columns for the top N relevant anime
    return df.iloc[relevant_idx][df.columns]

# Function to load the model and make predictions
def load_model_and_predict(keywords: str, top_n: int = 10) -> pd.DataFrame:
    """
    Load the pre-trained model and make predictions
    """
    model_path = "./model/anime_recommender.pt"
    
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run the script first to train and save the model.")
        return None
    
    # Load model
    model = AnimeRecommender(emb_dim, num_features)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load anime data
    if os.path.exists("./model/anime_data.pkl") and os.path.exists("./model/anime_embeddings.npy"):
        df = pd.read_pickle("./model/anime_data.pkl")
        anime_embeddings = np.load("./model/anime_embeddings.npy")
    else:
        print("Error: Model data not found. Please run the script first to process and save the data.")
        return None
    
    # Make recommendations
    return recommend_anime(keywords, top_n)

def print_usage():
    """Print usage instructions for the command line interface"""
    print("Uso: python modelFormer.py [comando] [keywords] [num_results]")
    print("\nComandos disponibles:")
    print("  train          - Entrena el modelo y lo guarda en la carpeta 'model'")
    print("  train-force    - Fuerza un reentrenamiento del modelo incluso si ya existe")
    print("  predict        - Utiliza el modelo guardado para hacer predicciones")
    print("\nArgumentos:")
    print("  keywords       - Palabras clave para buscar anime (por defecto: 'Comedy magic romance')")
    print("  num_results    - N�mero de resultados a mostrar (por defecto: 5)")
    print("\nEjemplos:")
    print("  python modelFormer.py train")
    print("  python modelFormer.py train-force")
    print("  python modelFormer.py predict \"Action adventure fantasy\" 10")
    print("  python modelFormer.py \"Action adventure fantasy\" 10  # Solo predicci�n (para compatibilidad)")

# Example usage
if __name__ == "__main__":
    import sys
    import io
    import locale
    import json
    
    # Check if help was requested
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["-h", "--help", "help"]:
        print_usage()
        sys.exit(0)
    
    # Set stdout encoding to utf-8 for Windows
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    
    try:
        # Check if first argument is a command
        mode = "predict"  # Default mode
        arg_offset = 0
        force_train = False
        
        if len(sys.argv) > 1:
            if sys.argv[1].lower() == "train":
                mode = "train"
                arg_offset = 1
            elif sys.argv[1].lower() == "train-force":
                mode = "train"
                force_train = True
                arg_offset = 1
            elif sys.argv[1].lower() == "predict":
                mode = "predict"
                arg_offset = 1
        
        # Get keywords and top_n based on mode and arguments
        keywords = sys.argv[1 + arg_offset] if len(sys.argv) > 1 + arg_offset else "Comedy magic romance"
        top_n = int(sys.argv[2 + arg_offset]) if len(sys.argv) > 2 + arg_offset else 5
        
        # Check if API call
        is_api_call = len(sys.argv) > 3 + arg_offset and sys.argv[3 + arg_offset] == "api_call"
        
        # Train mode: train and save the model
        if mode == "train":
            print("Modo: Entrenamiento del modelo")
            # Training is already handled in the model initialization part
            # Just ensure the model folder exists and model is saved
            os.makedirs("./model", exist_ok=True)
            
            # If model has already been trained above and not forcing retraining, print message
            if os.path.exists("./model/anime_recommender.pt") and not force_train:
                print("El modelo ya existe. Para reentrenar, use 'train-force'. Usando modelo existente...")
            else:
                if force_train:
                    print("Forzando reentrenamiento del modelo...")
                    
                # Train the model
                model = train_model(model, anime_embeddings, df, device=DEVICE)
                
                # Save model metadata
                model_info = {
                    "emb_dim": emb_dim,
                    "num_features": num_features,
                    "date_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(DEVICE),
                    "version": "1.0"
                }
                
                with open("./model/model_info.json", "w") as f:
                    json.dump(model_info, f)
                
                print("Modelo entrenado y guardado en la carpeta 'model'")
                
            # If no more arguments, exit
            if len(sys.argv) <= 1 + arg_offset:
                sys.exit(0)
        
        # Predict mode: use the model to make predictions
        if not is_api_call:
            print(f"Buscando animes con palabras clave: '{keywords}'")
            print(f"Mostrando los {top_n} mejores resultados...\n")
        
        # Use the recommend_anime function from above
        recs = recommend_anime(keywords, top_n=top_n)

        # Remove columns that are not needed in the frontend (like bert_emb)
        if "bert_emb" in recs.columns:
            recs = recs.drop(columns=["bert_emb"])
        
        # Si es llamado por la API, simplemente devuelve el JSON
        if is_api_call:
            print(recs.to_json(orient="records", force_ascii=False))
        else:
            # Versi�n m�s legible para pruebas manuales
            columns_to_display = ["Name", "Score", "Type", "Episodes", "Genres"]
            
            # Asegurarse de que todas las columnas solicitadas existen en el DataFrame
            columns_to_display = [col for col in columns_to_display if col in recs.columns]
            
            print("="*80)
            print(f"Recomendaciones usando modelo Deep Learning (carpeta 'model')")
            print(f"B�squeda por: '{keywords}'")
            print("="*80)
            
            for idx, anime in recs.iterrows():
                print(f"{idx+1}. {anime['Name']} - Score: {anime.get('Score', 'N/A')}")
                if "explanation" in anime:
                    print(f"   Por qu�: {anime['explanation']}")
                print(f"   Tipo: {anime.get('Type', 'N/A')} | Episodios: {anime.get('Episodes', 'N/A')}")
                print(f"   G�neros: {anime.get('Genres', 'N/A')}")
                if "Synopsis" in anime:
                    synopsis = anime["Synopsis"]
                    # Limitar sinopsis a 150 caracteres para la vista previa
                    if isinstance(synopsis, str) and len(synopsis) > 150:
                        synopsis = synopsis[:147] + "..."
                    print(f"   Sinopsis: {synopsis}")
                print("-"*80)
            
            # Muestra informaci�n sobre el modelo utilizado
            if os.path.exists("./model/model_info.json"):
                with open("./model/model_info.json", "r") as f:
                    model_info = json.load(f)
                print("\nInformaci�n del modelo:")
                print(f"Entrenado el: {model_info.get('date_trained', 'desconocido')}")
                print(f"Dispositivo: {model_info.get('device', 'desconocido')}")
                print(f"Versi�n: {model_info.get('version', '1.0')}")
                print(f"Dimensiones: {model_info.get('emb_dim', 768)} (embeddings) + {model_info.get('num_features', 4)} (caracter�sticas)")
                print("\nUso: python modelFormer.py [comando] [keywords] [num_results]")
                print("Para m�s informaci�n: python modelFormer.py --help")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print_usage()
        sys.exit(1)
