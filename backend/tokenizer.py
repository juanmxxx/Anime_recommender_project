# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# prompt = "I want to watch something funny and romantic."
# tokens = tokenizer.tokenize(prompt)
# print(tokens)



# Version 2

#
# # tokenizer.py
# from keybert import KeyBERT
#
# phrase = "I want something funny and isekai which the protagist is a girl with red head and if it can be with some romance."
#
# # prompt = input("Enter your prompt: ")
# kw_model = KeyBERT('all-mpnet-base-v2')
# keywords = kw_model.extract_keywords(
#     phrase,
#     keyphrase_ngram_range=(1, 2),
#     stop_words=None,
#     top_n=5
# )
# print([kw[0] for kw in keywords])


import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import torch


phrase = "I'm looking for a thrilling adventure set in space where the main character is a clever detective with blue eyes, and it would be great if there are some mystery elements."


kw_model = KeyBERT('all-mpnet-base-v2')
keywords = kw_model.extract_keywords(
    phrase,
    keyphrase_ngram_range=(1, 4),
    stop_words=None,
    top_n=10,
    use_maxsum=True,
    nr_candidates=20
)
print([kw[0] for kw in keywords])


custom_stopwords = {'want', 'something', 'which', 'the', 'be', 'with', 'and', 'if', 'can', 'is', 'a', 'of', 'to', 'it', 'some', 'which', 'and', 'be'}

# Filtrado de palabras clave para eliminar verbosidad

def clean_keyword(kw):
    words = [w for w in re.findall(r'\w+', kw.lower()) if w not in custom_stopwords]
    return ' '.join(words)

cleaned = []
for kw, _ in keywords:
    cleaned_kw = clean_keyword(kw)
    if cleaned_kw and cleaned_kw not in cleaned:
        cleaned.append(cleaned_kw)

print(cleaned)



# Use cleaned keywords for similarity filtering
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(cleaned, convert_to_tensor=True)

selected = []
selected_embeddings = []

for i, kw in enumerate(cleaned):
    if not selected:
        selected.append(kw)
        selected_embeddings.append(embeddings[i])
    else:
        sims = util.cos_sim(embeddings[i], torch.stack(selected_embeddings))
        if all(sim < 0.8 for sim in sims[0]):
            selected.append(kw)
            selected_embeddings.append(embeddings[i])

print(selected)
