# modelHandler.py
# ===============================
# Script de entrenamiento y gestión de modelo de recomendación de anime
# Algoritmo: Red neuronal profunda + embeddings BERT (DistilBERT) para similitud semántica
# El modelo prioriza animes recientes y populares, y responde a prompts completos del usuario
# ===============================

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
import pickle
import re
from datetime import datetime

# Ruta donde se guardará el modelo y los datos
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))
MODEL_PATH = os.path.join(MODEL_DIR, 'anime_recommender.pt')
DATA_PATH = os.path.join(MODEL_DIR, 'anime_data.pkl')
EMB_PATH = os.path.join(MODEL_DIR, 'anime_embeddings.npy')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# ===============================
# 1. Comprobar si existe el modelo
# ===============================
def modelo_existe():
    return os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH) and os.path.exists(EMB_PATH)

# ===============================
# 2. Cargar datos desde PostgreSQL y seleccionar campos relevantes
# ===============================
def cargar_datos_postgres(max_registros=5000):
    import sqlalchemy
    # Cambia los datos de conexión según tu entorno
    engine = sqlalchemy.create_engine('postgresql://anime_db:anime_db@localhost:5432/animes')
    # Selecciona solo las columnas útiles para el modelo
    cols = ['anime_id', 'name', 'genres', 'synopsis', 'score', 'popularity', 'aired']
    df = pd.read_sql(f'SELECT {", ".join(cols)} FROM anime LIMIT {max_registros}', engine)
    return df

# ===============================
# 3. Preprocesado de texto y generación de embeddings (optimizado)
# ===============================
def limpiar_texto(texto):
    # Limpieza básica: minúsculas, quitar símbolos, quitar stopwords en inglés
    stopwords = set([
        'the','a','an','and','or','in','on','at','of','for','to','with','by','from','is','it','this','that','as','but','be','are','was','were','so','if','into','out','about','up','down','over','under','again','further','then','once'
    ])
    texto = texto.lower()
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    tokens = [w for w in texto.split() if w not in stopwords and len(w) > 2]
    return ' '.join(tokens)

def preparar_datos(df, batch_size_emb=64):
    # Limpieza y combinación de campos relevantes para el texto
    df = df.head(1000)  # Solo los primeros 1000 registros
    df['genres'] = df['genres'].fillna('')
    df['synopsis'] = df['synopsis'].fillna('')
    df['name'] = df['name'].fillna('')
    df['text'] = (df['name'] + ' ' + df['genres'] + ' ' + df['synopsis']).apply(limpiar_texto)
    scaler = MinMaxScaler()
    num_cols = [c for c in ['score', 'popularity'] if c in df.columns]
    # Rellenar NaN en columnas numéricas antes de normalizar
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df[[c+'_norm' for c in num_cols]] = scaler.fit_transform(df[num_cols])
    df['year'] = df['aired'].str.extract(r'(\d{4})').fillna(0).astype(int)
    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1e-6)
    # Chequeo de NaN tras normalización
    for col in [c+'_norm' for c in num_cols] + ['year_norm']:
        if df[col].isnull().any():
            print(f"[ADVERTENCIA] Columna {col} contiene NaN tras normalizar. Se rellenan con 0.")
            df[col] = df[col].fillna(0)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[preparar_datos] Usando dispositivo: {device}")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    texts = df['text'].tolist()
    embs = []
    import time
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size_emb):
            batch = texts[i:i+batch_size_emb]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            start = time.time()
            # Mixed precision si está disponible
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state[:,0,:].detach().cpu().numpy().astype(np.float32)
            embs.append(batch_emb)
            print(f"Batch {i//batch_size_emb+1}: {len(batch)} registros, tiempo: {time.time()-start:.2f}s, memoria GPU: {torch.cuda.memory_allocated()/1e6 if device.type=='cuda' else 0:.1f}MB")
    embs = np.vstack(embs).astype(np.float32)
    # Chequeo de NaN en embeddings
    if np.isnan(embs).any():
        print("[ADVERTENCIA] Embeddings contienen NaN. Se reemplazan por 0.")
        embs = np.nan_to_num(embs)
    df.to_pickle(DATA_PATH)
    np.save(EMB_PATH, embs)
    return df, embs

# ===============================
# 4. Red neuronal profunda para recomendación
# ===============================
# Arquitectura: Capa densa -> ReLU -> Dropout -> Capa densa -> Salida
# Entrada: [embedding BERT + score_norm + popularity_norm + year_norm]
class RecommenderNN(nn.Module):
    def __init__(self, emb_dim, num_feat):
        super().__init__()
        # Ampliación de la arquitectura: más capas, más neuronas, regularización extra
        self.fc1 = nn.Linear(emb_dim + num_feat, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.drop5 = nn.Dropout(0.15)
        self.fc6 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.drop4(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.drop5(x)
        x = self.fc6(x)
        return x
# ===============================
# 5. Entrenamiento del modelo (optimizado para prompts semánticos)
# ===============================
def entrenar_modelo(df, embs, batch_size_train=512):
    import json
    from sklearn.metrics.pairwise import cosine_similarity
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    feat_cols = [c for c in df.columns if c.endswith('_norm')]
    # Asegura que embs y features sean float32 y sin NaN
    X = np.hstack([embs.astype(np.float32), df[feat_cols].values.astype(np.float32)])
    if np.isnan(X).any():
        raise ValueError('X contiene NaN')
    # 1. Prompts simulados (entrenamiento general)
    prompts_simulados_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/prompts_simulados.json'))
    with open(prompts_simulados_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    prompt_embs = []
    for p in prompts:
        clean_p = limpiar_texto(p)
        inp = tokenizer([clean_p], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = bert(**inp)
            prompt_embs.append(out.last_hidden_state[:,0,:].cpu().numpy()[0])
    prompt_embs = np.stack(prompt_embs).astype(np.float32)
    sim_matrix = cosine_similarity(embs, prompt_embs)
    y_general = 0.5 * sim_matrix.max(axis=1) + 0.3 * df['popularity_norm'].values + 0.2 * df['year_norm'].values
    y_general = y_general.reshape(-1,1).astype(np.float32)
    if np.isnan(y_general).any():
        raise ValueError('y_general contiene NaN')
    # 2. Ejemplos etiquetados (prompt_labels.json)
    prompt_labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/prompt_labels.json'))
    with open(prompt_labels_path, 'r', encoding='utf-8') as f:
        prompt_labels = json.load(f)
    X_sup, y_sup = [], []
    for entry in prompt_labels:
        prompt = limpiar_texto(entry['prompt'])
        inp = tokenizer([prompt], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            prompt_emb = bert(**inp).last_hidden_state[:,0,:].cpu().numpy()[0]
        for idx, row in df.iterrows():
            anime_id = row['anime_id']
            if anime_id in entry['anime_ids']:
                target = 1.0
            else:
                target = 0.15 + 0.15 * float(row['popularity_norm']) + 0.1 * float(row['year_norm'])
            X_sup.append(np.concatenate([prompt_emb, row[feat_cols].values]))
            y_sup.append([target])
    X_sup = np.array(X_sup, dtype=np.float32)
    y_sup = np.array(y_sup, dtype=np.float32)
    if np.isnan(X_sup).any() or np.isnan(y_sup).any():
        raise ValueError('X_sup o y_sup contiene NaN')
    # 3. Mezcla de datasets (más peso a ejemplos etiquetados)
    X_combined = np.vstack([X, X_sup, X_sup]).astype(np.float32)
    y_combined = np.vstack([y_general, y_sup, y_sup]).astype(np.float32)
    if np.isnan(X_combined).any() or np.isnan(y_combined).any():
        raise ValueError('X_combined o y_combined contiene NaN')
    ds = TensorDataset(torch.tensor(X_combined, dtype=torch.float32), torch.tensor(y_combined, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size_train, shuffle=True, pin_memory=(device.type=='cuda'))
    model = RecommenderNN(embs.shape[1], len(feat_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)  # learning rate más bajo
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(24):
        epoch_loss = 0
        start_epoch = time.time()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
            else:
                preds = model(xb)
                loss = loss_fn(preds, yb)
            if torch.isnan(loss):
                print('Loss es NaN, deteniendo entrenamiento')
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # clipping de gradientes
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/24 - Loss: {epoch_loss/len(dl.dataset):.6f} - Tiempo: {time.time()-start_epoch:.2f}s - Memoria GPU: {torch.cuda.memory_allocated()/1e6 if device.type=='cuda' else 0:.1f}MB")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'modelo entrenado y guardado en {MODEL_PATH} con integración de ejemplos etiquetados y priorización de animes relevantes, populares y recientes.')

# ===============================
# 6. Reentrenar modelo existente
# ===============================
def reentrenar():
    if not modelo_existe():
        print('No hay modelo para reentrenar. Usa "python modelHandler.py train" para crear uno.')
        return
    df = pd.read_pickle(DATA_PATH)
    embs = np.load(EMB_PATH)
    entrenar_modelo(df, embs)

# ===============================
# 7. Entrenar desde cero
# ===============================
def entrenar_desde_cero():
    df = cargar_datos_postgres(max_registros=1000)
    df, embs = preparar_datos(df, batch_size_emb=64)
    entrenar_modelo(df, embs, batch_size_train=128)

# ===============================
# 8. Main: gestión de argumentos y avisos
# ===============================
def main():
    if len(sys.argv) < 2:
        if modelo_existe():
            print('Ya existe un modelo entrenado en la carpeta model.')
            print('Puedes reentrenar con: python modelHandler.py retrain')
        else:
            print('No hay modelo entrenado. Usa: python modelHandler.py train')
        return
    cmd = sys.argv[1].lower()
    if cmd == 'train':
        entrenar_desde_cero()
    elif cmd == 'retrain':
        reentrenar()
    else:
        print('Comando no reconocido. Usa "train" o "retrain".')

if __name__ == '__main__':
    main()
# ===============================
# FIN DEL SCRIPT
# ===============================
