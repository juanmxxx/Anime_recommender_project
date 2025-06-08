#!/usr/bin/env python3
"""
Sistema de Recomendación Híbrido
================================

Combina el sistema content-based actual (que funciona bien) con el modelo de Deep Learning
para obtener los mejores resultados de ambos enfoques.

Estrategia híbrida:
1. Obtener recomendaciones del sistema content-based (más preciso para queries específicas)
2. Obtener recomendaciones del modelo ML (mejor para similitud semántica)
3. Combinar y re-rankear los resultados
4. Aplicar filtros de calidad

Uso:
    from hybridRecommendationSystem import get_hybrid_recommendations
    recommendations = get_hybrid_recommendations("aspiring idols", top_n=10)
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Agregar directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Importar sistemas existentes
    from recommendHandler import ImprovedAnimeRecommendationSystem
    from mlIntegrationImproved import get_recommendations_ml_improved, is_model_available
except ImportError as e:
    print(f"Error importando sistemas: {e}")

class HybridAnimeRecommendationSystem:
    """Sistema híbrido que combina content-based y deep learning"""
    
    def __init__(self):
        self.content_system = ImprovedAnimeRecommendationSystem()
        self.ml_available = is_model_available()
        
        # Pesos para la combinación híbrida
        self.content_weight = 0.7  # Mayor peso al sistema content-based (más preciso)
        self.ml_weight = 0.3       # Menor peso al ML (para similitud semántica)
        
        print(f"Sistema híbrido inicializado:")
        print(f"- Content-based: ✅ Disponible")
        print(f"- Deep Learning: {'✅ Disponible' if self.ml_available else '❌ No disponible'}")
    
    def normalize_scores(self, recommendations: List[Dict], score_key: str) -> List[Dict]:
        """Normaliza los scores de una lista de recomendaciones"""
        if not recommendations:
            return recommendations
        
        scores = [rec[score_key] for rec in recommendations]
        min_score = min(scores)
        max_score = max(scores)
        
        # Evitar división por cero
        if max_score == min_score:
            for rec in recommendations:
                rec[f'normalized_{score_key}'] = 1.0
        else:
            for rec in recommendations:
                original_score = rec[score_key]
                normalized = (original_score - min_score) / (max_score - min_score)
                rec[f'normalized_{score_key}'] = normalized
        
        return recommendations
    
    def merge_recommendations(self, content_recs: List[Dict], ml_recs: List[Dict]) -> List[Dict]:
        """Combina recomendaciones de ambos sistemas"""
        # Crear diccionario para fusionar por anime_id
        merged = {}
        
        # Normalizar scores de content-based
        content_recs = self.normalize_scores(content_recs, 'recommendation_score')
        
        # Procesar recomendaciones content-based
        for i, rec in enumerate(content_recs):
            anime_id = rec['anime_id']
            
            # Score basado en posición + score normalizado
            position_score = 1.0 - (i / len(content_recs))
            content_score = rec['normalized_recommendation_score']
            combined_content_score = (position_score + content_score) / 2
            
            merged[anime_id] = {
                **rec,
                'content_score': combined_content_score,
                'ml_score': 0.0,  # Default si no hay ML
                'hybrid_score': combined_content_score * self.content_weight,
                'source': 'content'
            }
        
        # Procesar recomendaciones ML si están disponibles
        if ml_recs and self.ml_available:
            ml_recs = self.normalize_scores(ml_recs, 'score')
            
            for i, rec in enumerate(ml_recs):
                anime_id = rec['anime_id']
                
                # Score basado en posición + score normalizado
                position_score = 1.0 - (i / len(ml_recs))
                ml_score = rec['normalized_score']
                combined_ml_score = (position_score + ml_score) / 2
                
                if anime_id in merged:
                    # Combinar con existente
                    merged[anime_id]['ml_score'] = combined_ml_score
                    merged[anime_id]['hybrid_score'] = (
                        merged[anime_id]['content_score'] * self.content_weight + 
                        combined_ml_score * self.ml_weight
                    )
                    merged[anime_id]['source'] = 'hybrid'
                else:
                    # Nuevo desde ML
                    merged[anime_id] = {
                        **rec,
                        'content_score': 0.0,
                        'ml_score': combined_ml_score,
                        'hybrid_score': combined_ml_score * self.ml_weight,
                        'source': 'ml'
                    }
        
        # Convertir a lista y ordenar por hybrid_score
        result = list(merged.values())
        result.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Re-enumerar ranks
        for i, rec in enumerate(result):
            rec['rank'] = i + 1
        
        return result
    
    def apply_quality_filters(self, recommendations: List[Dict], keyphrase: str) -> List[Dict]:
        """Aplica filtros de calidad y relevancia específicos"""
        filtered = []
        
        keyphrase_lower = keyphrase.lower()
        
        # Palabras clave para detectar consultas específicas
        idol_keywords = ['idol', 'aspiring', 'singer', 'performer', 'music', 'entertainment']
        action_keywords = ['action', 'fight', 'battle', 'combat', 'fighting']
        romance_keywords = ['romance', 'romantic', 'love', 'dating', 'couple']
        
        for rec in recommendations:
            synopsis = str(rec.get('synopsis', '')).lower()
            genres = str(rec.get('genres', '')).lower()
            name = str(rec.get('name', '')).lower()
            
            # Filtro de relevancia temática
            relevance_boost = 0.0
            
            # Boost para ídolos
            if any(keyword in keyphrase_lower for keyword in idol_keywords):
                if any(keyword in synopsis or keyword in genres or keyword in name 
                      for keyword in ['idol', 'singer', 'music', 'live', 'performance']):
                    relevance_boost += 0.3
                elif 'love live' in name or 'idolmaster' in name:
                    relevance_boost += 0.5
            
            # Boost para acción
            elif any(keyword in keyphrase_lower for keyword in action_keywords):
                if 'action' in genres or any(keyword in synopsis 
                                           for keyword in ['fight', 'battle', 'combat']):
                    relevance_boost += 0.2
            
            # Boost para romance
            elif any(keyword in keyphrase_lower for keyword in romance_keywords):
                if 'romance' in genres or any(keyword in synopsis 
                                            for keyword in ['love', 'romantic']):
                    relevance_boost += 0.2
            
            # Aplicar boost
            rec['hybrid_score'] += relevance_boost
            
            # Filtrar por score mínimo
            if rec['hybrid_score'] > 0.1:  # Threshold mínimo
                filtered.append(rec)
        
        # Re-ordenar después de aplicar boosts
        filtered.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Re-enumerar ranks
        for i, rec in enumerate(filtered):
            rec['rank'] = i + 1
        
        return filtered
    
    def get_recommendations(self, keyphrase: str, top_n: int = 10) -> List[Dict]:
        """Obtiene recomendaciones híbridas"""
        try:
            # Obtener recomendaciones del sistema content-based
            print("🔍 Obteniendo recomendaciones content-based...")
            content_recs = []
            
            # Usar el sistema content-based actual
            content_system_result = self.content_system.get_recommendations(keyphrase, top_n * 2)
            
            for rec in content_system_result:
                content_recs.append({
                    'anime_id': rec.get('anime_id', 0),
                    'name': rec.get('name', ''),
                    'english_name': rec.get('english_name', ''),
                    'genres': rec.get('genres', ''),
                    'synopsis': rec.get('synopsis', ''),
                    'favorites': rec.get('favorites', 0),
                    'aired': rec.get('aired', ''),
                    'type': rec.get('type', ''),
                    'episodes': rec.get('episodes', ''),
                    'image_url': rec.get('image_url', ''),
                    'recommendation_score': rec.get('recommendation_score', 0.0)
                })
            
            # Obtener recomendaciones ML si está disponible
            ml_recs = []
            if self.ml_available:
                try:
                    print("🤖 Obteniendo recomendaciones ML...")
                    ml_recs = get_recommendations_ml_improved(keyphrase, top_n * 2)
                except Exception as e:
                    print(f"⚠️ Error en ML: {e}")
                    ml_recs = []
            
            # Combinar sistemas
            print("🔀 Combinando recomendaciones...")
            merged_recs = self.merge_recommendations(content_recs, ml_recs)
            
            # Aplicar filtros de calidad
            print("✨ Aplicando filtros de calidad...")
            filtered_recs = self.apply_quality_filters(merged_recs, keyphrase)
            
            # Retornar top_n resultados
            final_recs = filtered_recs[:top_n]
            
            print(f"📊 Resultados finales: {len(final_recs)} recomendaciones")
            
            return final_recs
            
        except Exception as e:
            print(f"❌ Error en sistema híbrido: {e}")
            # Fallback al sistema content-based
            try:
                return self.content_system.get_recommendations(keyphrase, top_n)
            except Exception as e2:
                print(f"❌ Error en fallback: {e2}")
                return []

# Funciones de utilidad para compatibilidad
def get_hybrid_recommendations(keyphrase: str, top_n: int = 10) -> List[Dict]:
    """Función principal para obtener recomendaciones híbridas"""
    system = HybridAnimeRecommendationSystem()
    return system.get_recommendations(keyphrase, top_n)

def get_hybrid_recommendations_json(keyphrase: str, top_n: int = 10) -> str:
    """Obtiene recomendaciones híbridas en formato JSON"""
    try:
        recommendations = get_hybrid_recommendations(keyphrase, top_n)
        
        result = {
            "prompt": keyphrase,
            "model_type": "hybrid_content_ml",
            "total_results": len(recommendations),
            "recommendations": recommendations
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            "prompt": keyphrase,
            "model_type": "hybrid_content_ml", 
            "error": str(e),
            "recommendations": []
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)

def get_hybrid_recommendations_text(keyphrase: str, top_n: int = 10) -> str:
    """Obtiene recomendaciones híbridas en formato texto"""
    try:
        recommendations = get_hybrid_recommendations(keyphrase, top_n)
        
        if not recommendations:
            return f"❌ No se encontraron recomendaciones para: '{keyphrase}'"
        
        result = []
        result.append(f"🔍 Búsqueda: \"{keyphrase}\"")
        result.append(f"🔀 Modelo: Sistema Híbrido (Content-Based + Deep Learning)")
        result.append(f"✨ Top {len(recommendations)} Recomendaciones:")
        result.append("")
        
        for rec in recommendations:
            result.append(f"{rec['rank']}. {rec['name']}")
            
            if rec.get('english_name'):
                result.append(f"   Título en inglés: {rec['english_name']}")
            
            result.append(f"   Score híbrido: {rec['hybrid_score']:.4f}")
            
            # Mostrar scores individuales
            content_score = rec.get('content_score', 0.0)
            ml_score = rec.get('ml_score', 0.0)
            source = rec.get('source', 'unknown')
            
            result.append(f"   Fuente: {source} (Content: {content_score:.3f}, ML: {ml_score:.3f})")
            result.append(f"   Favoritos: {rec.get('favorites', 0):,}")
            result.append(f"   Géneros: {rec.get('genres', 'N/A')}")
            
            if rec.get('type') and rec.get('episodes'):
                result.append(f"   Tipo: {rec['type']} | Episodios: {rec['episodes']}")
            
            result.append(f"   Emitido: {rec.get('aired', 'N/A')}")
            
            # Sinopsis truncada
            synopsis = str(rec.get('synopsis', ''))
            if len(synopsis) > 300:
                synopsis = synopsis[:300] + "..."
            result.append(f"   Sinopsis: {synopsis}")
            result.append("   " + "-" * 50)
        
        return "\n".join(result)
        
    except Exception as e:
        return f"❌ Error en recomendaciones híbridas: {str(e)}"

if __name__ == "__main__":
    # Script de prueba
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Recomendación Híbrido')
    parser.add_argument('query', help='Consulta para probar')
    parser.add_argument('--top_n', type=int, default=5, help='Número de recomendaciones')
    parser.add_argument('--format', choices=['dict', 'json', 'text'], 
                       default='text', help='Formato de salida')
    
    args = parser.parse_args()
    
    if args.format == 'dict':
        result = get_hybrid_recommendations(args.query, args.top_n)
        for rec in result:
            print(f"{rec['rank']}. {rec['name']} (Score: {rec['hybrid_score']:.4f})")
    elif args.format == 'json':
        result = get_hybrid_recommendations_json(args.query, args.top_n)
        print(result)
    else:
        result = get_hybrid_recommendations_text(args.query, args.top_n)
        print(result)
