import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler

# Load dataset
DATASET_PATH = './dataset/small2.csv'
df = pd.read_csv(DATASET_PATH)

# Fill missing values
for col in ['Score', 'Rank', 'Popularity', 'Favorites']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

def preprocess_text(row):
    # Combine genres and synopsis for text input
    return f"{row['Genres']} {row['Synopsis']}"

df['text'] = df.apply(preprocess_text, axis=1)

# Normalize numeric features
scaler = MinMaxScaler()
df[['Score_norm', 'Rank_norm', 'Popularity_norm', 'Favorites_norm']] = scaler.fit_transform(df[['Score', 'Rank', 'Popularity', 'Favorites']])

# Load transformer model and tokenizer
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

# Detect device
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Function to get BERT embeddings for a batch of texts (batched for efficiency)
def get_bert_embeddings(texts, tokenizer, model, device=DEVICE, max_length=128, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Prepare anime embeddings (cache for efficiency)
if not hasattr(df, 'bert_emb'):
    anime_embeddings = get_bert_embeddings(df['text'].tolist(), tokenizer, bert_model, device=DEVICE)
    df['bert_emb'] = list(anime_embeddings)
else:
    anime_embeddings = np.stack(df['bert_emb'].values)

# Build the recommendation model
class AnimeRecommender(nn.Module):
    def __init__(self, emb_dim, num_features):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim + num_features, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate model
emb_dim = anime_embeddings.shape[1]
num_features = 4  # Score_norm, Rank_norm, Popularity_norm, Favorites_norm
model = AnimeRecommender(emb_dim, num_features)

# --- BLACKBOX RECOMMENDATION MODEL ---
def recommend_anime(keywords: str, top_n: int = 10) -> pd.DataFrame:
    """
    Blackbox anime recommendation model using BERT embeddings and a custom scoring formula.
    INPUT:  keywords (str) - e.g. 'Comedy magic romance'
    OUTPUT: DataFrame with ALL columns for the top N recommended anime (most famous and relevant first)
    """
    # 1. Get BERT embedding for the input keywords
    kw_emb = get_bert_embeddings([keywords], tokenizer, bert_model, device=DEVICE)[0]
    # 2. Compute cosine similarity between keywords and each anime
    cos_sim = np.sum(anime_embeddings * kw_emb, axis=1) / (np.linalg.norm(anime_embeddings, axis=1) * np.linalg.norm(kw_emb) + 1e-8)
    # 3. Invert Rank and Popularity so lower values mean more famous
    inv_rank = 1 - df['Rank_norm']
    inv_popularity = 1 - df['Popularity_norm']
    # 4. Weighted custom score: prioritize famous and relevant anime
    custom_score = 0.5 * cos_sim + 0.25 * inv_rank + 0.2 * inv_popularity + 0.05 * df['Score_norm']
    # 5. Get indices of top N anime
    top_idx = np.argsort(-custom_score)[:top_n]
    # 6. Return ALL columns for the top N anime
    return df.iloc[top_idx][df.columns]

# Example usage
if __name__ == '__main__':
    import sys
    import io
    import locale
    import json
    # Set stdout encoding to utf-8 for Windows
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

    keywords = sys.argv[1] if len(sys.argv) > 1 else "Comedy magic romance"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    recs = recommend_anime(keywords, top_n=top_n)

    # Remove columns that are not needed in the frontend (like bert_emb)
    if 'bert_emb' in recs.columns:
        recs = recs.drop(columns=['bert_emb'])
    # Convert DataFrame to JSON and print
    print(recs.to_json(orient='records', force_ascii=False))