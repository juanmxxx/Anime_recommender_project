"""
Interfaz de línea de comandos (CLI) para el sistema de recomendación de anime
que permite utilizar prompts completos en lenguaje natural.

Versión optimizada con mejor manejo de prompts largos.
"""

import argparse
import sys
import os
import time

# Importamos el procesador de prompts optimizado
from backend.anime_search_engine import PromptProcessor

def main():
    """
    Interfaz principal de línea de comandos para el recomendador de anime
    con soporte mejorado para prompts largos.
    """
    # Configuramos el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendación de Anime - Obtén recomendaciones desde prompts en lenguaje natural",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argumento para el prompt
    parser.add_argument(
        "prompt", 
        nargs="?",
        default=None, 
        help="Prompt en lenguaje natural describiendo el anime que estás buscando"
    )
    
    # Argumento para el número de resultados
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=5,
        help="Número de recomendaciones a mostrar (predeterminado: 5)"
    )
    
    # Opción para mostrar las palabras clave
    parser.add_argument(
        "-k", "--show-keywords",
        action="store_true",
        help="Mostrar las palabras clave extraídas del prompt"
    )
    
    # Procesamos los argumentos
    args = parser.parse_args()
    
    # Si no se proporciona un prompt, entramos en modo interactivo
    if args.prompt is None:
        print("\n" + "="*80)
        print("SISTEMA DE RECOMENDACIÓN DE ANIME - MODO INTERACTIVO")
        print("="*80)
        print("\nEscribe tu prompt a continuación, o 'exit' para salir:")
        args.prompt = input("> ")
        
        if args.prompt.lower() in ['exit', 'quit', 'q']:
            print("¡Hasta pronto!")
            sys.exit(0)
    
    # Medimos el tiempo de ejecución para diagnóstico
    start_time = time.time()
    
    # Creamos el procesador de prompts optimizado
    processor = PromptProcessor()
    
    # PASO 1: Extraer palabras clave del prompt
    print("\nProcesando tu prompt...")
    keywords = processor.process_prompt(args.prompt)
    
    # Mostramos las palabras clave si se solicita
    if args.show_keywords or True:  # Siempre mostramos las palabras clave en esta versión
        print("\nPalabras clave extraídas:")
        print(f"  {', '.join(keywords)}")
    
    # PASO 2: Obtener recomendaciones
    print("\nBuscando animes que coincidan con tu solicitud...")
    recommendations = processor.recommend_from_prompt(args.prompt, top_n=args.num_results)
    
    # PASO 3: Mostrar las recomendaciones
    if recommendations is not None and not recommendations.empty:
        print("\n" + "="*80)
        print(f"TOP {len(recommendations)} RECOMENDACIONES DE ANIME")
        print("="*80)
        
        for idx, anime in recommendations.iterrows():
            print(f"{idx+1}. {anime['Name']} - Puntuación: {anime.get('Score', 'N/A')}")
            if "explanation" in anime:
                print(f"   Por qué: {anime['explanation']}")
            print(f"   Tipo: {anime.get('Type', 'N/A')} | Episodios: {anime.get('Episodes', 'N/A')}")
            print(f"   Géneros: {anime.get('Genres', 'N/A')}")
            if "Synopsis" in anime:
                synopsis = anime["Synopsis"]
                if isinstance(synopsis, str) and len(synopsis) > 200:
                    synopsis = synopsis[:197] + "..."
                print(f"   Sinopsis: {synopsis}")
            print("-"*80)
    else:
        print("\nNo se encontraron recomendaciones. Por favor, intenta con un prompt diferente.")
    
    # Mostramos el tiempo total de ejecución
    elapsed_time = time.time() - start_time
    print(f"\nTiempo de procesamiento total: {elapsed_time:.2f} segundos")

# Punto de entrada principal
if __name__ == "__main__":
    main()
