"""
Procesador de prompts mejorado para el sistema de recomendación de anime

Este módulo implementa un procesador de prompts en lenguaje natural que extrae
palabras clave relevantes para recomendación de animes. Es una evolución del 
archivo tokenizer.py original, con capacidades mejoradas para entender
contexto y características específicas de anime.

"""

import re
import numpy as np
from keybert import KeyBERT                # Para extracción de palabras clave basada en BERT
from sentence_transformers import SentenceTransformer, util   # Para embeddings y similitud
import torch
import spacy                               # Para procesamiento de lenguaje natural avanzado
from collections import Counter
import os
import sys
import time                                # Para implementar timeouts

# Añadimos el directorio padre a la ruta para poder importar modelFormer
# Esto es necesario para integrar este procesador con el sistema de recomendación existente
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Importamos el módulo principal de recomendación para usar sus funciones
import modelFormer

# Constantes y configuración global
MAX_PROMPT_LENGTH = 500  # Máximo de caracteres para procesar sin truncar
PROCESSING_TIMEOUT = 30  # Tiempo máximo en segundos para procesamiento
MAX_KEYWORDS = 10        # Número máximo de palabras clave (aumentado para prompts largos)
FALLBACK_ENABLED = True  # Habilitar modo alternativo si falla el procesamiento completo

# Cargamos el modelo de spaCy para un mejor reconocimiento de entidades y extracción de palabras clave
# spaCy nos permite analizar la estructura gramatical y reconocer entidades nombradas
try:
    nlp = spacy.load("en_core_web_md")  # Modelo en inglés de tamaño medio con vectores de palabras
except OSError:
    print("Descargando modelo de spaCy...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

class PromptProcessor:
    """
    Clase principal para procesar prompts en lenguaje natural y extraer palabras clave relevantes
    para recomendaciones de anime.
    
    Esta clase se encarga de:
    1. Analizar el texto del prompt completo
    2. Extraer características importantes (género, personajes, atributos físicos)
    3. Filtrar palabras irrelevantes
    4. Conectar con el sistema de recomendación
    """
    
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        Inicializa el procesador de prompts con el modelo especificado.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers a utilizar para embeddings
        """
        # Inicializamos dos modelos principales:
        # 1. KeyBERT para extracción de palabras clave basada en BERT
        self.kw_model = KeyBERT(model_name)
        # 2. SentenceTransformer para generar embeddings semánticos
        self.st_model = SentenceTransformer(model_name)
        
        # Lista extendida de "stopwords" (palabras a ignorar) para consultas relacionadas con anime
        # Estas son palabras comunes que no aportan información relevante para la búsqueda
        self.custom_stopwords = {
            # Palabras de búsqueda generales
            'want', 'need', 'looking', 'something', 'which', 'the', 'be', 'with', 'and', 
            'if', 'can', 'is', 'a', 'of', 'to', 'it', 'some', 'what', 'where', 'when',
            'who', 'how', 'why', 'would', 'could', 'should', 'an', 'for', 'that', 'this',
            'these', 'those', 'my', 'our', 'your', 
            
            # Términos específicos de anime que son demasiado genéricos
            'anime', 'similar', 'like', 'recommend', 'recommendation', 'about', 
            
            # Verbos comunes que no ayudan a la búsqueda
            'has', 'have', 'had', 
            
            # Pronombres y expresiones conversacionales
            'me', 'i', 'show', 'please', 'thanks', 'thank', 'you', 'there', 'their', 'they', 
            
            # Términos relacionados con medios audiovisuales
            'series', 'film', 'movie', 'episode', 'season', 'character', 'characters', 
            'plot', 'story'
        }
        
        # Términos que deben recibir mayor importancia si aparecen en el prompt
        # Estos términos son específicos del dominio de anime y ayudan a identificar
        # características relevantes para la recomendación
        self.important_terms = {
            # Rasgos de personaje - importantes para identificar el tipo de personajes
            'protagonist', 'hero', 'heroine', 'villain', 'antagonist', 'main', 'character',
            'girl', 'boy', 'man', 'woman', 'female', 'male', 'kid', 'adult', 'teen', 'young',
            'tall', 'short', 'strong', 'weak', 'smart', 'intelligent', 'brave', 'cowardly',
            
            # Características visuales - cruciales para detectar atributos físicos mencionados
            # Estos son especialmente importantes en el contexto de anime donde los rasgos
            # visuales como color de pelo son distintivos
            'hair', 'eyes', 'blonde', 'brunette', 'redhead', 'bald', 'glasses', 'tall',
            'short', 'blue', 'red', 'green', 'black', 'white', 'pink', 'purple',
            
            # Géneros de anime - categorías estándar en la clasificación de anime
            'action', 'adventure', 'comedy', 'drama', 'fantasy', 'horror', 'magic',
            'mecha', 'mystery', 'psychological', 'romance', 'sci-fi', 'slice of life',
            'supernatural', 'thriller', 'cyberpunk', 'steampunk', 'historical', 'war',
            'sports', 'martial arts', 'music', 'school', 'space', 'vampire',
            
            # Ambientaciones - lugares o épocas donde se desarrollan las historias
            'school', 'academy', 'university', 'city', 'village', 'rural', 'urban',
            'future', 'past', 'medieval', 'modern', 'ancient', 'space', 'planet',
            'kingdom', 'empire', 'dystopia', 'utopia', 'forest', 'ocean', 'mountain',
            
            # Profesiones - ocupaciones comunes de personajes en animes
            # Muchos animes se definen por las profesiones de sus protagonistas
            'student', 'teacher', 'doctor', 'scientist', 'police', 'detective', 'samurai',
            'ninja', 'knight', 'wizard', 'witch', 'soldier', 'pilot', 'assassin', 'spy',
            'hunter', 'chef', 'musician', 'artist', 'athlete', 'gamer'
        }
    
    def clean_keyword(self, kw):
        """
        Limpia una palabra clave individual eliminando stopwords y caracteres innecesarios.
        
        Args:
            kw: La palabra clave a limpiar
            
        Returns:
            str: La palabra clave limpia o una cadena vacía si no queda nada relevante
        """
        # Extraemos solo caracteres alfanuméricos, convertimos a minúsculas y eliminamos stopwords
        words = [w for w in re.findall(r'\w+', kw.lower()) if w not in self.custom_stopwords]
        return ' '.join(words)
    
    def extract_attributes(self, prompt):
        """
        Extrae atributos de personajes y elementos de la historia del prompt.
        Esta función utiliza spaCy para análisis lingüístico avanzado y busca patrones
        específicos como colores de pelo, profesiones, etc.
        
        Args:
            prompt: El prompt completo en lenguaje natural
            
        Returns:
            dict: Diccionario con categorías de atributos extraídos
        """
        start_time = time.time()
        
        # Diccionario para organizar los atributos extraídos por categorías
        attributes = {
            'character_traits': [],  # Rasgos de carácter (valiente, inteligente, etc.)
            'visual_traits': [],     # Rasgos visuales (pelo rojo, ojos azules, etc.)
            'genres': [],            # Géneros de anime mencionados
            'setting': [],           # Ambientación (escuela, espacio, etc.)
            'profession': []         # Profesiones de personajes
        }
        
        # Limitamos la longitud del prompt para el análisis lingüístico profundo
        if len(prompt) > MAX_PROMPT_LENGTH:
            print(f"AVISO: Prompt muy largo ({len(prompt)} caracteres). Truncando para análisis lingüístico...")
            # Truncamos pero intentamos mantener frases completas
            prompt_truncated = prompt[:MAX_PROMPT_LENGTH].rsplit('.', 1)[0] + '.'
        else:
            prompt_truncated = prompt
            
        try:
            # Configuramos opciones de procesamiento para mejorar rendimiento en textos largos
            disabled_pipes = ['parser'] if len(prompt_truncated) > 300 else []
            doc = nlp(prompt_truncated, disable=disabled_pipes)
            
            # Verificar tiempo de procesamiento
            if time.time() - start_time > PROCESSING_TIMEOUT / 2:
                print("AVISO: El análisis está tardando demasiado, limitando procesamiento...")
                return attributes  # Devolvemos lo que tengamos hasta ahora
                
            # Lista de términos de color para buscar rasgos visuales
            # Estos son especialmente importantes para detectar características como "pelo rojo"
            color_terms = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'blonde', 
                           'brunette', 'pink', 'purple', 'orange', 'brown', 'gray', 'grey']
            
            # Términos relacionados con personajes para asociarlos con los colores
            character_terms = ['girl', 'boy', 'man', 'woman', 'character', 'protagonist',
                               'hair', 'eyes', 'person', 'hero', 'heroine']
            
            # Buscamos patrones de rasgos visuales como "ojos azules" o "pelo rojo"
            for token in doc:
                if check_timeout(start_time, PROCESSING_TIMEOUT, "extract_attributes"):
                    return attributes
                
                # Si encontramos una palabra de color, buscamos si está relacionada con características
                if token.text.lower() in color_terms:
                    # Buscamos términos de características en los "hijos" gramaticales del token
                    # Esto detecta relaciones como "red hair" donde "red" modifica a "hair"
                    for nearby in token.head.children:
                        if nearby.text.lower() in ['hair', 'eyes', 'haired']:
                            attributes['visual_traits'].append(f"{token.text} {nearby.text}")
                    
                    # También verificamos si la "cabeza" gramatical es pelo/ojos
                    # Esto detecta estructuras como "hair red" o construcciones similares
                    if token.head.text.lower() in ['hair', 'eyes', 'haired']:
                        attributes['visual_traits'].append(f"{token.text} {token.head.text}")
            
            # Extraemos entidades nombradas reconocidas por spaCy
            for ent in doc.ents:
                # Personas (nombres propios de personajes)
                if ent.label_ == 'PERSON':
                    attributes['character_traits'].append(ent.text)
                # Lugares (ciudades, países, instalaciones)
                elif ent.label_ in ['GPE', 'LOC', 'FAC']:
                    attributes['setting'].append(ent.text)
                # Nacionalidades, grupos religiosos o políticos
                elif ent.label_ == 'NORP':  
                    attributes['character_traits'].append(ent.text)
            
            # Lista de palabras relacionadas con profesiones y géneros
            profession_words = ['scientist', 'doctor', 'teacher', 'student', 'detective',
                                'police', 'warrior', 'knight', 'samurai', 'ninja', 'assassin',
                                'spy', 'agent', 'soldier', 'pilot', 'captain', 'chef', 'artist']
            
            genre_words = ['action', 'adventure', 'comedy', 'drama', 'fantasy', 'horror',
                          'magic', 'mecha', 'mystery', 'romance', 'sci-fi', 'thriller',
                          'supernatural', 'school', 'space', 'historical', 'war', 'sports']
            
            # Buscamos estas palabras en el texto y las clasificamos según corresponda
            for token in doc:
                # Verificamos si estamos tardando demasiado tiempo
                if check_timeout(start_time, PROCESSING_TIMEOUT, "extract_attributes"):
                    return attributes
                    
                # Identificamos profesiones mencionadas
                if token.text.lower() in profession_words:
                    attributes['profession'].append(token.text.lower())
                # Identificamos géneros mencionados
                if token.text.lower() in genre_words:
                    attributes['genres'].append(token.text.lower())
                    
        except Exception as e:
            print(f"Error durante análisis lingüístico: {e}")
        
        # Devolvemos el diccionario con todos los atributos extraídos
        return attributes
    def extract_keywords_simple(self, prompt):
        """
        Método alternativo simplificado para extraer palabras clave cuando el método completo falla.
        Este método es más rápido pero menos preciso.
        
        Args:
            prompt: El prompt completo en lenguaje natural
            
        Returns:
            list: Lista de palabras clave relevantes
        """
        keywords = []
        
        # Lista de palabras para buscar directamente en el texto
        # Importante: mantener ordenadas por prioridad
        important_categories = [
            # Colores (alta prioridad para anime)
            ['red', 'blue', 'green', 'black', 'white', 'blonde', 'pink', 'purple', 'blood', 'bloody'],
            
            # Características físicas y de personaje
            ['hair', 'eyes', 'tall', 'short', 'young', 'old', 'beautiful', 'handsome', 'strong', 'weak'],
            
            # Géneros principales
            ['action', 'adventure', 'comedy', 'romance', 'fantasy', 'horror', 'mystery', 'sci-fi', 'thriller'],
            
            # Profesiones y roles
            ['student', 'teacher', 'doctor', 'scientist', 'ninja', 'warrior', 'detective', 'pilot', 'hero', 'villain'],
            
            # Términos específicos de anime
            ['mecha', 'magical', 'school', 'supernatural', 'slice of life', 'isekai', 'monster', 'demon', 'robot'],
            
            # Términos de ambientación
            ['future', 'past', 'medieval', 'modern', 'ancient', 'space', 'planet', 'reality', 'parallel', 'alternate', 'world']
        ]
        
        # Términos "muy negativos" - buscaremos sus opuestos en el filtrado final
        negative_terms = ['ova', 'ovas', '1 episode', 'one episode', 'single episode', 'short']
        
        # Convertimos el prompt a minúsculas para búsqueda insensible a mayúsculas
        prompt_lower = prompt.lower()
        
        # Extraer frases clave específicas que podrían ser importantes
        key_phrases = [
            "parallel reality", "parallel world", "alternate reality", "alternate world",
            "blood", "bloody", "gore", "violent", "survival", "monster", "demons",
            "fighting monsters", "kill monsters", "delete monster", "destroy monster"
        ]
        
        for phrase in key_phrases:
            if phrase in prompt_lower and phrase not in keywords and len(keywords) < MAX_KEYWORDS:
                keywords.append(phrase)
        
        # Buscamos términos de cada categoría
        for category in important_categories:
            terms_from_category = 0
            for term in category:
                # Verificamos si el término aparece como palabra completa, no como subcadena
                term_pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(term_pattern, prompt_lower) and term not in keywords:
                    keywords.append(term)
                    terms_from_category += 1
                    
                # Limitamos a 2 términos por categoría como máximo
                if terms_from_category >= 2 or len(keywords) >= MAX_KEYWORDS:
                    break
        
        # Si no encontramos suficientes palabras clave, extraemos algunas palabras adicionales
        # usando una técnica simple basada en frecuencia
        if len(keywords) < 3:
            words = re.findall(r'\b\w{4,}\b', prompt_lower)  # Palabras de al menos 4 caracteres
            word_counts = Counter(words)
            
            # Filtramos stopwords y añadimos palabras frecuentes
            for word, count in word_counts.most_common(10):
                if (word not in self.custom_stopwords and 
                    word not in keywords and
                    len(keywords) < MAX_KEYWORDS):
                    keywords.append(word)
        
        # Verificamos si hay términos negativos en el prompt
        # Estos son términos que indican lo que NO quiere el usuario
        negative_found = False
        for term in negative_terms:
            if term in prompt_lower:
                negative_found = True
                break
                
        if negative_found:
            # Si se encontraron términos negativos, añadimos "series" y "multiple episodes"
            # para favorecer series completas
            if "series" not in keywords and len(keywords) < MAX_KEYWORDS:
                keywords.append("series")
            if "multiple episodes" not in keywords and len(keywords) < MAX_KEYWORDS:
                keywords.append("multiple episodes")
        
        return keywords
    
    def process_full_prompt(self, prompt, max_keywords=MAX_KEYWORDS, similarity_threshold=0.75):
        """
        Procesa un prompt completo en lenguaje natural y extrae las palabras clave más relevantes.
        Este es el método principal que combina todas las técnicas de extracción.
        
        Args:
            prompt: El prompt completo en lenguaje natural
            max_keywords: Número máximo de palabras clave a devolver
            similarity_threshold: Umbral para eliminar palabras clave similares
            
        Returns:
            list: Una lista de palabras clave relevantes para el motor de recomendación de anime
        """
        start_time = time.time()
        
        try:            # PASO 1: Extraer palabras clave usando KeyBERT
            # KeyBERT es un modelo que utiliza BERT para extraer las palabras más relevantes
            # Para prompts largos, limitamos el texto para evitar problemas de rendimiento
            prompt_for_keybert = prompt[:800] if len(prompt) > 800 else prompt  # Limitamos a 800 caracteres
            
            try:
                keywords = self.kw_model.extract_keywords(
                    prompt_for_keybert,
                    keyphrase_ngram_range=(1, 3),    # Reducimos a n-gramas de 1 a 3 palabras
                    stop_words=list(self.custom_stopwords),  # Excluye palabras irrelevantes
                    top_n=15,                         # Extrae las 15 mejores
                    use_maxsum=True,                  # Usa algoritmo MaxSum para diversidad
                    nr_candidates=20,                 # Reducimos candidatos para mayor velocidad
                    use_mmr=True,                     # Usamos MMR en lugar de MaxSum para prompts largos
                    diversity=0.7                     # Factor de diversidad para MMR
                )
            except Exception as e:
                print(f"Error al extraer palabras clave con KeyBERT: {e}")
                # Si falla KeyBERT, usamos el método alternativo directamente
                return self.extract_keywords_simple(prompt)
            
            # Verificamos si estamos tardando demasiado
            if check_timeout(start_time, PROCESSING_TIMEOUT, "process_full_prompt"):
                print("AVISO: Usando método alternativo de extracción de palabras clave...")
                return self.extract_keywords_simple(prompt)
              
            # PASO 2: Limpiar las palabras clave eliminando stopwords y duplicados
            cleaned = []
            for kw, score in keywords:
                cleaned_kw = self.clean_keyword(kw)
                if cleaned_kw and cleaned_kw not in cleaned:
                    cleaned.append(cleaned_kw)
            
            # PASO 3: Extraer atributos específicos usando análisis lingüístico
            # Esto captura características como color de pelo, profesiones, etc.
            attributes = self.extract_attributes(prompt)
            
            # Verificamos si estamos tardando demasiado
            if check_timeout(start_time, PROCESSING_TIMEOUT, "process_full_prompt"):
                if cleaned:
                    # Si tenemos palabras clave limpias, usamos esas (omitiendo análisis profundo)
                    return cleaned[:max_keywords]
                else:
                    # Si no tenemos palabras clave, usamos el método alternativo
                    return self.extract_keywords_simple(prompt)
            
            # PASO 4: Añadir atributos de alta prioridad a las palabras clave
            # Los rasgos visuales son especialmente importantes en anime
            # Por eso los añadimos al principio de la lista (mayor prioridad)
            for visual in attributes['visual_traits']:
                if visual not in cleaned:
                    cleaned.insert(0, visual)  # Añadir al principio para mayor prioridad
                    
            # Las profesiones son importantes para caracterizar personajes
            for profession in attributes['profession']:
                if profession not in cleaned:
                    cleaned.append(profession)
                    
            # Los géneros ayudan a clasificar el tipo de anime
            for genre in attributes['genres']:
                if genre not in cleaned:
                    cleaned.append(genre)
              
            # PASO 5: Calcular embeddings para filtrado por similitud
            # Si no encontramos palabras clave limpias, devolvemos lista vacía o usamos el fallback
            if not cleaned:
                return self.extract_keywords_simple(prompt) if FALLBACK_ENABLED else []
                
            # Verificamos si estamos tardando demasiado
            if check_timeout(start_time, PROCESSING_TIMEOUT, "process_full_prompt"):
                # Si hay muchas palabras clave, limitamos sin hacer análisis de similitud
                return cleaned[:max_keywords]
                
            # Convertimos las palabras clave a embeddings (vectores) para comparar similitud
            try:
                embeddings = self.st_model.encode(cleaned, convert_to_tensor=True)
            except Exception as e:
                print(f"Error al generar embeddings: {e}")
                # Si falla la generación de embeddings, devolvemos lo que tenemos hasta ahora
                return cleaned[:max_keywords]
            
            # PASO 6: Seleccionar palabras clave diversas usando filtrado por similitud
            # Esto evita redundancia (por ejemplo, no queremos "pelo rojo" y "cabello rojo")
            selected = []
            selected_embeddings = []
            
            for i, kw in enumerate(cleaned):
                # Verificamos si estamos tardando demasiado
                if check_timeout(start_time, PROCESSING_TIMEOUT, "process_full_prompt"):
                    # Si estamos tomando demasiado tiempo, añadimos lo que tenemos hasta ahora
                    if not selected:
                        return cleaned[:max_keywords]
                    return selected + cleaned[:max(0, max_keywords - len(selected))]
                
                if not selected:
                    # La primera palabra clave siempre se selecciona
                    selected.append(kw)
                    selected_embeddings.append(embeddings[i])
                else:
                    # Comparamos la similitud con palabras ya seleccionadas
                    sims = util.cos_sim(embeddings[i], torch.stack(selected_embeddings))
                    # Solo añadimos si no es demasiado similar a ninguna palabra ya seleccionada
                    if all(sim < similarity_threshold for sim in sims[0]):
                        selected.append(kw)
                        selected_embeddings.append(embeddings[i])
                        
                # Nos detenemos si ya tenemos suficientes palabras clave
                if len(selected) >= max_keywords:
                    break
              
            # PASO 7: Añadir términos de alta prioridad que aparecen en el prompt pero no fueron capturados
            # Solo hacemos esto si no estamos a punto de exceder el tiempo
            if not check_timeout(start_time, PROCESSING_TIMEOUT * 0.9, "process_full_prompt"):
                # Analizamos nuevamente el prompt para obtener los lemas (forma base) de las palabras
                # Limitamos a texto más corto para análisis rápido
                doc = nlp(prompt[:300].lower())
                tokens = [token.lemma_ for token in doc]
                
                # Verificamos si algún término importante aparece en el texto
                # pero no fue capturado por los métodos anteriores
                for term in self.important_terms:
                    if term in tokens and term not in selected and len(selected) < max_keywords:
                        selected.append(term)
                    
            # Devolvemos la lista final de palabras clave seleccionadas
            return selected
            
        except Exception as e:
            print(f"Error en process_full_prompt: {e}")
            # Si algo falla, usamos el método alternativo
            return self.extract_keywords_simple(prompt)
        
    def recommend_from_prompt(self, prompt, top_n=5):
        """
        Procesa un prompt y obtiene recomendaciones de anime directamente.
        Este método conecta el procesador de prompts con el sistema de recomendación.
        
        Args:
            prompt: Prompt completo en lenguaje natural
            top_n: Número de recomendaciones a devolver
            
        Returns:
            DataFrame: Recomendaciones de anime basadas en el prompt
        """
        start_time = time.time()
        try:
            # Verificamos si el prompt es demasiado largo y alertamos al usuario
            if len(prompt) > 1000:
                print(f"AVISO: El prompt es muy largo ({len(prompt)} caracteres). El procesamiento puede tardar más tiempo.")
            
            # PASO 1: Extraer palabras clave del prompt usando el método completo
            print("Procesando prompt y extrayendo palabras clave...")
            
            # Para prompts muy largos, vamos directo al método simple
            if len(prompt) > 1500:
                print("Prompt extremadamente largo, usando método simplificado directamente...")
                keywords = self.extract_keywords_simple(prompt)
            else:
                # Intentamos el método completo primero
                keywords = self.process_full_prompt(prompt)
                
                # Si no pudimos extraer palabras clave, intentamos con el método alternativo
                if not keywords and FALLBACK_ENABLED:
                    print("Usando método alternativo para extraer palabras clave...")
                    keywords = self.extract_keywords_simple(prompt)
                    
            # Si aún no tenemos palabras clave, informamos y devolvemos None
            if not keywords:
                print("No se pudieron extraer palabras clave significativas del prompt.")
                return None
                
            # PASO 2: Convertir las palabras clave a una cadena separada por espacios
            # Este es el formato que espera el modelo de recomendación existente
            keywords_str = " ".join(keywords)
            print(f"Palabras clave extraídas: {keywords_str}")
            
            # PASO 3: Utilizar el modelo de recomendación existente con manejo de errores
            print("Obteniendo recomendaciones del modelo...")
            
            # Establecemos un tiempo límite para la respuesta del modelo
            # Si ya hemos consumido mucho tiempo, nos aseguramos de dejar al menos 15 segundos
            # para la fase de recomendación
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 20:
                print("AVISO: El procesamiento ya ha tomado bastante tiempo. Optimizando búsqueda...")
                # Si se está demorando mucho, limitamos el top_n para acelerar
                if top_n > 5:
                    top_n = 5
                    print("Limitando a 5 recomendaciones para optimizar tiempo...")
            
            # Llamamos al modelo de recomendación
            recommendations = modelFormer.recommend_anime(keywords_str, top_n=top_n)
            
            return recommendations
            
        except Exception as e:
            print(f"Error al procesar el prompt: {e}")
            
            # En caso de error, intentamos un enfoque más simple
            if FALLBACK_ENABLED:
                print("Intentando enfoque alternativo...")
                simple_keywords = self.extract_keywords_simple(prompt)
                if simple_keywords:
                    # Usamos directamente el modelFormer con las palabras clave simples
                    return modelFormer.recommend_anime(" ".join(simple_keywords), top_n=top_n)
            
            return None

# Función auxiliar para verificar timeouts
def check_timeout(start_time, max_seconds, function_name=""):
    """Verifica si una función ha excedido el tiempo máximo de ejecución"""
    if time.time() - start_time > max_seconds:
        print(f"Advertencia: {function_name} excedió el límite de {max_seconds} segundos.")
        return True
    return False




# Ejemplo de uso como script independiente
# Este código se ejecutará si este archivo se ejecuta directamente (no al importarlo)
if __name__ == "__main__":
    # Procesamos argumentos de línea de comandos
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["-h", "--help", "help"]:
        print("Uso: python improved_prompt_processor.py \"tu prompt de recomendación de anime\" [num_resultados]")
        print("\nEjemplos:")
        print("  python improved_prompt_processor.py \"Recomiéndame un anime con un científico protagonista y una chica pelirroja\"")
        print("  python improved_prompt_processor.py \"Quiero un anime ambientado en el espacio con robots y romance\" 10")
        sys.exit(0)
        
    # Prompt predeterminado para pruebas si no se proporciona uno
    prompt = "Recommend an anime with a scientist protagonist and a red-haired girl"
    top_n = 5
    
    # Obtenemos el prompt de la línea de comandos si se proporciona
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
        
    # Obtenemos el número de recomendaciones si se proporciona
    if len(sys.argv) > 2:
        try:
            top_n = int(sys.argv[2])
        except ValueError:
            print(f"Advertencia: Número inválido '{sys.argv[2]}'. Usando el valor predeterminado de {top_n} recomendaciones.")
    
    # FLUJO PRINCIPAL DE EJECUCIÓN:
    
    # 1. Creamos el procesador de prompts (nuestro tokenizador principal)
    processor = PromptProcessor()
    
    print(f"Procesando prompt: \"{prompt}\"")
    
    # 2. Procesamos el prompt para extraer palabras clave
    # Este es el paso principal de tokenización
    keywords = processor.process_full_prompt(prompt)
    print(f"Palabras clave extraídas: {keywords}")
    
    # 3. Obtenemos recomendaciones basadas en las palabras clave
    # Aquí es donde nuestro tokenizador se conecta con modelFormer.py
    print(f"\nObteniendo recomendaciones basadas en: {' '.join(keywords)}")
    recommendations = processor.recommend_from_prompt(prompt, top_n=top_n)
      
    # 4. Si se encontraron recomendaciones, las mostramos
    # Esta parte muestra el resultado final al usuario
    if recommendations is not None:
        print("\n" + "="*80)
        print(f"TOP {len(recommendations)} RECOMENDACIONES DE ANIME")
        print("="*80)
        
        # Mostramos cada anime con su información relevante
        for idx, anime in recommendations.iterrows():
            print(f"{idx+1}. {anime['Name']} - Puntuación: {anime.get('Score', 'N/A')}")
            if "explanation" in anime:
                # Esta explicación viene de modelFormer.py y explica por qué se recomendó el anime
                print(f"   Por qué: {anime['explanation']}")
            print(f"   Tipo: {anime.get('Type', 'N/A')} | Episodios: {anime.get('Episodes', 'N/A')}")
            print(f"   Géneros: {anime.get('Genres', 'N/A')}")
            if "Synopsis" in anime:
                synopsis = anime["Synopsis"]
                # Limitamos la sinopsis para una vista previa limpia
                if isinstance(synopsis, str) and len(synopsis) > 150:
                    synopsis = synopsis[:147] + "..."
                print(f"   Sinopsis: {synopsis}")
            print("-"*80)
