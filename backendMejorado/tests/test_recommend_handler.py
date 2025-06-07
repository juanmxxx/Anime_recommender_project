#!/usr/bin/env python3
"""
Script de prueba para el recomendador de anime
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))


from recommendHandler import get_recommendations_text, get_recommendations_json

def test_recommendation():
    """
    Prueba el sistema de recomendaciones
    """
    # Prompt de prueba
    test_prompt = "I want to watch an action anime with magic powers and strong characters"
    
    print("\n===== PRUEBA DE RECOMENDACIÓN =====")
    print(f"Prompt: '{test_prompt}'")
    
    print("\n--- FORMATO TEXTO ---")
    text_result = get_recommendations_text(test_prompt, top_n=5)
    print(text_result)
    
    print("\n--- FORMATO JSON ---")
    json_result = get_recommendations_json(test_prompt, top_n=5)
    print(json_result)

if __name__ == "__main__":
    test_recommendation()
