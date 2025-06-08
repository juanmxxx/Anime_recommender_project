import spacy

# Carga el modelo de spaCy en inglés
nlp = spacy.load("en_core_web_sm")

def remove_prepositions(phrase):
    # Lista de preposiciones comunes en inglés
    prepositions = {
        'of', 'in', 'on', 'at', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'over', 'under',
        'again', 'further', 'then', 'once', 'by', 'as', 'like', 'off', 'out', 'around', 'among', 'and', 'the', 'a', 'an'
    }
    # Elimina preposiciones y artículos, mantiene el orden y la frase
    return ' '.join([w for w in phrase.split() if w.lower() not in prepositions])

def extract_keyphrases(prompt):
    doc = nlp(prompt)
    noun_chunks = list(doc.noun_chunks)
    keyphrases = []
    generic_words = {"anime", "series", "show", "shows", "movie", "movies"}
    i = 0
    while i < len(noun_chunks):
        chunk = noun_chunks[i]
        phrase = chunk.text.strip()
        # Completa frases como 'a slice' a 'a slice of life' si aparece en el prompt
        if phrase.lower() == "a slice":
            if "slice of life" in prompt.lower():
                phrase = "a slice of life"
        # Elimina palabras genéricas
        if any(word in generic_words for word in phrase.lower().split()):
            if all(word in generic_words for word in phrase.lower().split()):
                i += 1
                continue
            phrase = " ".join([w for w in phrase.split() if w.lower() not in generic_words])
            phrase = phrase.strip()
        # Intenta unir con el siguiente chunk si hay preposición entre ellos
        end = chunk.end
        merged = phrase
        while end < len(doc) and doc[end].pos_ == "ADP":  # ADP = preposition
            prep = doc[end].text
            # Busca el siguiente noun chunk después de la preposición
            next_chunks = [c for c in noun_chunks if c.start == end + 1]
            if next_chunks:
                next_chunk = next_chunks[0]
                merged += f" {prep} {next_chunk.text.strip()}"
                end = next_chunk.end
                i += 1  # Saltar el siguiente chunk porque ya fue unido
            else:
                break
        if len(merged.split()) > 1:
            keyphrases.append(merged)
        i += 1
    
    # Elimina duplicados y conserva el orden
    seen = set()
    keyphrases = [x for x in keyphrases if not (x.lower() in seen or seen.add(x.lower()))]
    
    # Limpia las keyphrases de preposiciones antes de devolverlas
    cleaned_keyphrases = [remove_prepositions(phrase) for phrase in keyphrases]
    # Filtra frases vacías o de una sola palabra después de limpiar
    cleaned_keyphrases = [phrase for phrase in cleaned_keyphrases if len(phrase.split()) > 1]
    
    return ", ".join(cleaned_keyphrases)