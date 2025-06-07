import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle

# Ruta a los modelos y datos
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'anime_embeddings.npy')
DATA_PATH = os.path.join(MODEL_DIR, 'anime_data.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'anime_recommender.pt')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Cargar datos y modelos
anime_embeddings = np.load(EMBEDDINGS_PATH)
with open(DATA_PATH, 'rb') as f:
    anime_data = pickle.load(f)
scaler = pickle.load(open(SCALER_PATH, 'rb'))
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Cargar vectorizador entrenado
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Prompt de ejemplo
prompt = "I like action and adventure anime with strong female protagonists"

# Extraer palabras clave (opcional)
keywords = vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else []

# Embedding del prompt usando el vectorizador entrenado
prompt_vec = vectorizer.transform([prompt]).toarray()
prompt_vec = scaler.transform(prompt_vec)
prompt_tensor = torch.tensor(prompt_vec, dtype=torch.float32)

# Obtener embedding del modelo (ajusta según tu modelo real)
with torch.no_grad():
    prompt_embedding = model(prompt_tensor).numpy().squeeze()

# Calcular similitud de coseno con todos los animes
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity([prompt_embedding], anime_embeddings)[0]
top5_idx = sims.argsort()[-5:][::-1]

# Mostrar top 5 recomendaciones
print("Top 5 recomendaciones:")
for idx in top5_idx:
    print(anime_data[idx])
