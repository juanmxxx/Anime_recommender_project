import spacy

# Carga el modelo de spaCy en inglés
nlp = spacy.load("en_core_web_sm")

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
    return ", ".join(keyphrases)

def remove_prepositions(phrase):
    # Lista de preposiciones comunes en inglés
    prepositions = {
        'of', 'in', 'on', 'at', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'over', 'under',
        'again', 'further', 'then', 'once', 'by', 'as', 'like', 'off', 'out', 'around', 'among', 'and', 'the', 'a', 'an'
    }
    # Elimina preposiciones y artículos, mantiene el orden y la frase
    return ' '.join([w for w in phrase.split() if w.lower() not in prepositions])

# Prompts largos y específicos de prueba
prompts = [
    "I'm interested in a coming-of-age story set in a futuristic city, with themes of friendship, betrayal, and advanced technology.",
    "Recommend me a dark fantasy with complex world-building, morally ambiguous characters, and epic battles between good and evil.",
    "Suggest a romantic comedy where the main character is a transfer student who joins a quirky school club and slowly falls in love.",
    "Find me a slice of life anime that explores the daily struggles of a single parent raising a gifted child in a small rural town.",
    "I want a psychological thriller involving time travel, memory manipulation, and a protagonist who questions their own reality.",
    "Give me a sports anime focused on teamwork and rivalry, set during the national high school basketball championship.",
    "Recommend a science fiction adventure with space pirates, intergalactic politics, and a search for a legendary lost planet.",
    "Suggest a supernatural mystery where the detective can see ghosts and must solve a series of crimes linked to ancient folklore.",
    "I'm looking for a historical drama centered on the rise and fall of a powerful samurai family during the Sengoku period.",
    "Find me a musical anime about a group of aspiring idols overcoming personal challenges to perform at the biggest concert of the year."
]
for p in prompts:
    print(f"Prompt: {p}")
    keyphrases = extract_keyphrases(p)
    # Segunda parte: estilo Google, elimina preposiciones y artículos
    if keyphrases:
        google_style = ', '.join(remove_prepositions(kp) for kp in keyphrases.split(', '))
        print("Final-keyphrase:", google_style)
    print()