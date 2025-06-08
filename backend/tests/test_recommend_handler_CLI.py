#!/usr/bin/env python3
"""
Script de prueba para el recomendador de anime CLI
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))


from recommendHandler import get_recommendations_text, get_recommendations_json

def test_recommendation():
    """
    Prueba el sistema de recomendaciones
    """

    print("Introduce un prompt para obtener recomendaciones de anime (por defecto se mostrara el top 10):")
    user_prompt = input("Prompt: ")
    
    # Prompt de prueba
    
    print("\n===== PRUEBA DE RECOMENDACIÓN =====")
    print(f"Prompt: '{user_prompt}'")
    
    print("\n--- FORMATO TEXTO ---")
    text_result = get_recommendations_text(user_prompt, top_n=10)
    print(text_result)

    formato_json = input("\n¿Quieres ver el resultado en formato JSON? (s/n): ").strip().lower()

    if formato_json == 's':
        print("\n--- FORMATO JSON ---")
        json_result = get_recommendations_json(user_prompt, top_n=10)
        print(json_result)
    

if __name__ == "__main__":
    test_recommendation()
