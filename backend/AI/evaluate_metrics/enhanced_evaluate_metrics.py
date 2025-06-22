#!/usr/bin/env python3
"""
Enhanced Anime Recommender Metrics Evaluator

This script evaluates the quality of anime recommendations obtained from prompts
using standard information retrieval metrics:
- Precision
- Recall
- F1 Score
- MAP (Mean Average Precision)

It directly uses the model to generate recommendations from English prompts
and evaluates them against a reference set with stricter and more balanced criteria.

Usage:
    python enhanced_evaluate_metrics.py --num-prompts 30
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import time
import joblib
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

class AnimeRecommenderMetricsEvaluator:
    def __init__(self, model_dir="../../../model"):
        """
        Inicializar el evaludador
        """
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found.")
        
        required_files = [
            'anime_nn_model.pkl',
            'anime_data.pkl',
            'combined_embeddings.npy',
            'anime_id_to_index.pkl'
        ]
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file {file_path} not found. Run 'python hybrid_recommender_fixed.py train' first.")
        
        # Cargar el modelo y datos
        print("🔄 Loading model and data...")
        self.nn_model = joblib.load(os.path.join(model_dir, 'anime_nn_model.pkl'))
        self.anime_data = joblib.load(os.path.join(model_dir, 'anime_data.pkl'))
        self.combined_embeddings = np.load(os.path.join(model_dir, 'combined_embeddings.npy'))
        self.anime_id_to_index = joblib.load(os.path.join(model_dir, 'anime_id_to_index.pkl'))
        
        # Crear mapeo de índice inverso
        self.index_to_anime_id = {idx: anime_id for anime_id, idx in self.anime_id_to_index.items()}
        
        # Cargar incrustador de texto
        try:
            self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Text embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading text embedding model: {e}")
            raise
        
        # Cargar datos de referencia
        print("🔄 Creating reference sets...")
        self.reference_sets = self.create_reference_sets()
        print(f"✅ Created {len(self.reference_sets)} reference sets for evaluation")
    def create_reference_sets(self) -> List[Dict]:
        """
        Crea conjuntos de referencia de anime para evaluar.
        Cada conjunto de referencia contiene:
        - Un prompt
        - Una lista de IDs de anime relevantes que deberían ser devueltos para ese prompt
        """
        # Cargar prompts de prueba desde archivo JSON
        try:
            prompts_file_path = os.path.join(os.path.dirname(__file__), 'test_prompts.json')
            with open(prompts_file_path, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
                test_prompts = prompts_data['test_prompts']
                print(f"✅ Cargados {len(test_prompts)} prompts de prueba desde {prompts_file_path}")
        except Exception as e:
            print(f"⚠️ Error al cargar prompts desde archivo JSON: {e}")
            # Definir un conjunto mínimo de prompts en caso de error
            test_prompts = [
                {"prompt": "Action anime", "genres": ["Action"], "keywords": ["action"], "mandatory_genres": ["Action"]},
                {"prompt": "Fantasy anime", "genres": ["Fantasy"], "keywords": ["fantasy"], "mandatory_genres": ["Fantasy"]},
                {"prompt": "Sci-Fi anime", "genres": ["Sci-Fi"], "keywords": ["science fiction"], "mandatory_genres": ["Sci-Fi"]}
            ]
          # Para cada prompt, obtener recomendaciones directamente y crear un conjunto de referencia
        reference_sets = []
        
        for test in test_prompts:
            # Crear un conjunto de referencia
            reference_set = {
                "prompt": test["prompt"],
                "genres": test["genres"],
                "keywords": test["keywords"],
                "mandatory_genres": test.get("mandatory_genres", []),
                "relevant_anime_ids": []  # This will be filled by running the recommendations
            }
            
            # Obtener recomendaciones directamente usando nuestro modelo
            try:                # Obtener recomendaciones (50 para tener una buena muestra)
                recommended_ids = self._get_recommendations_for_prompt(test["prompt"], 50)
                
                if recommended_ids:                    # Filtrar anime por los géneros y palabras clave esperados
                    relevant_ids = []
                    relevance_scores = {}  # Almacenar puntuaciones de relevancia para ordenar
                    
                    for anime_id in recommended_ids:
                        # Encontrar el anime en nuestros datos
                        anime_idx = self.anime_id_to_index.get(anime_id)
                        if anime_idx is None:
                            continue
                        
                        anime = self.anime_data[anime_idx]
                        
                        # Extraer géneros
                        genres = anime.get('genres', [])
                        if isinstance(genres, str):
                            try:
                                genres = json.loads(genres.replace("'", '"'))
                            except:
                                genres = [g.strip() for g in genres.strip('[]').split(',')]
                        
                        # Inicializar puntuación de relevancia
                        relevance_score = 0
                        passes_mandatory = False
                        
                        # Comprobar primero los géneros obligatorios (requisito binario)
                        mandatory_genres = test.get("mandatory_genres", [])
                        if mandatory_genres:
                            if all(any(mg.lower() in g.lower() for g in genres) for mg in mandatory_genres):
                                passes_mandatory = True
                                relevance_score += 5  # Puntuación alta por coincidir con todos los géneros obligatorios
                        else:
                            passes_mandatory = True  # Sin géneros obligatorios = aprobación automática
                        
                        # Si no pasa la verificación obligatoria, omitir este anime
                        if not passes_mandatory:
                            continue
                        
                        # Check regular genre matches
                        for g in test["genres"]:
                            if any(g.lower() in genre.lower() for genre in genres):
                                relevance_score += 3  # Points for each matching genre
                        
                        # Extract description and check for keywords
                        description = anime.get('description', '').lower()
                        title = anime.get('romaji_title', '').lower() + " " + (anime.get('english_title', '') or '').lower()
                        
                        # Check keywords in description and title
                        for keyword in test["keywords"]:
                            if keyword.lower() in description:
                                relevance_score += 1  # 1 point for keyword in description
                            if keyword.lower() in title:
                                relevance_score += 2  # 2 points for keyword in title
                        
                        # If the anime has some relevance, add it to our results
                        if relevance_score > 0:
                            relevance_scores[anime_id] = relevance_score
                            relevant_ids.append(anime_id)
                    
                    # Sort by relevance score (highest first)
                    if relevant_ids:
                        relevant_ids = sorted(relevant_ids, key=lambda id: relevance_scores[id], reverse=True)
                    
                    # Ensure we have a reasonable number of relevant anime
                    if len(relevant_ids) > 10:
                        reference_set["relevant_anime_ids"] = relevant_ids[:30]  # Limit to 30 most relevant
                        reference_sets.append(reference_set)
                else:
                    print(f"⚠️ Failed to get recommendations for prompt: {test['prompt']}")
                    
            except Exception as e:
                print(f"⚠️ Error generating reference set for prompt '{test['prompt']}': {e}")
        
        return reference_sets
    
    def _get_recommendations_for_prompt(self, prompt: str, count: int = 10) -> List[int]:
        """
        Obtiene recomendaciones de anime para un prompt usando el modelo directamente
        """
        try:
            # Convert prompt to embedding
            prompt_embedding = self.text_embedder.encode(prompt)
            
            # Normalize the embedding
            prompt_embedding = normalize(prompt_embedding.reshape(1, -1), norm='l2')[0]
            
            # Get nearest neighbors
            distances, indices = self.nn_model.kneighbors(
                prompt_embedding.reshape(1, -1),
                n_neighbors=count
            )
            
            # Convert indices to anime IDs
            recommended_ids = []
            for idx in indices[0]:
                anime_id = self.index_to_anime_id.get(idx)
                if anime_id is not None:
                    recommended_ids.append(anime_id)
            
            return recommended_ids
            
        except Exception as e:
            print(f"⚠️ Error getting recommendations for prompt '{prompt}': {e}")
            return []
    
    def generate_test_prompts(self, num_prompts: int) -> List[Dict]:
        """
        Genera un conjunto de prompts de prueba creando variaciones de los prompts de referencia
        """
        test_prompts = []
        
        # Generate variations of reference prompts
        for i in range(num_prompts):
            # Select a random reference set
            ref_set = random.choice(self.reference_sets)
            
            # Basic prompt variation techniques
            variation_type = random.randint(1, 4)
            
            if variation_type == 1:
                # Add prefixes
                prefixes = [
                    "I'm looking for ", "Can you recommend me ", "I want to watch ",
                    "Please suggest ", "Show me ", "I need ", "What are some good "
                ]
                prefix = random.choice(prefixes)
                prompt = f"{prefix}{ref_set['prompt'].lower()}"
                
            elif variation_type == 2:
                # Add suffixes
                suffixes = [
                    " to watch tonight", " to binge watch", " for the weekend",
                    ", any suggestions?", ", what do you recommend?", " please", " with good ratings"
                ]
                suffix = random.choice(suffixes)
                prompt = f"{ref_set['prompt']}{suffix}"
                
            elif variation_type == 3:
                # Rephrase with synonyms
                # This is a simple implementation - could be improved with actual NLP
                keywords = {
                    "strong": ["powerful", "tough", "mighty"],
                    "adventure": ["quest", "journey", "expedition"],
                    "magic": ["supernatural powers", "spells", "wizardry"],
                    "school": ["academy", "high school", "college"],
                    "space": ["outer space", "cosmos", "galaxy"],
                    "happy": ["joyful", "cheerful", "pleasant"],
                    "detective": ["investigator", "sleuth", "mystery solver"],
                    "horror": ["scary", "frightening", "terrifying"],
                    "action": ["fighting", "battle", "combat"],
                    "fantasy": ["magical world", "mythical", "enchanted"],
                    "romance": ["love story", "romantic", "relationship"]
                }
                
                prompt = ref_set["prompt"]
                for word, replacements in keywords.items():
                    if word in prompt.lower():
                        replacement = random.choice(replacements)
                        prompt = prompt.lower().replace(word, replacement)
                        break
            
            else:
                # Use original with minor changes in capitalization or punctuation
                prompt = ref_set["prompt"]
                if random.random() > 0.5:
                    prompt = prompt.upper() if random.random() > 0.5 else prompt.lower()
                
                # Maybe add a question mark
                if random.random() > 0.7 and not prompt.endswith("?"):
                    prompt += "?"
            
            # Add to test prompts
            test_prompts.append({
                "prompt": prompt,
                "reference": ref_set
            })
        
        return test_prompts
    
    def get_recommendations(self, prompt: str, count: int = 10) -> List[int]:
        """
        Obtiene recomendaciones de anime para un prompt
        """
        # Use the internal method to get recommendations
        return self._get_recommendations_for_prompt(prompt, count)
    def calculate_metrics(self, recommended_ids: List[int], relevant_ids: List[int], k: int = 15) -> Dict:
        """
        Calcula métricas de evaluación
        """
        if not recommended_ids or not relevant_ids:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "average_precision": 0.0
            }
        
        # Use only top k recommendations if specified
        if k is not None and k < len(recommended_ids):
            recommended_ids = recommended_ids[:k]
        
        # Convert to sets for intersection
        recommended_set = set(recommended_ids)
        relevant_set = set(relevant_ids)
        
        # Calculate intersection
        relevant_and_recommended = recommended_set.intersection(relevant_set)
        num_relevant_and_recommended = len(relevant_and_recommended)
        
        # Calculate Precision: TP / (TP + FP)
        precision = num_relevant_and_recommended / len(recommended_ids) if recommended_ids else 0.0
        
        # Calculate Recall: TP / (TP + FN)
        recall = num_relevant_and_recommended / len(relevant_set) if relevant_set else 0.0
        
        # Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate Average Precision (AP)
        ap = 0.0
        relevant_count = 0
        
        for i, anime_id in enumerate(recommended_ids):
            if anime_id in relevant_set:
                relevant_count += 1
                # Precision at cutoff i+1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        
        average_precision = ap / len(relevant_set) if relevant_set else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_precision": average_precision
        }
    
    def evaluate_prompts(self, test_prompts: List[Dict], k_values: List[int] = None) -> Dict:
        """
        Evalúa métricas para una lista de prompts de prueba
        """
        if k_values is None:
            k_values = [15]  # Solo utilizamos  como valor predeterminado
            
        results = []
        
        for test in tqdm(test_prompts, desc="Evaluating prompts"):
            prompt = test["prompt"]
            reference = test["reference"]
            relevant_ids = reference["relevant_anime_ids"]
            
            # Get recommendations
            recommended_ids = self.get_recommendations(prompt)
            
            if not recommended_ids:
                continue
            
            # Calculate metrics at different k values
            metrics_by_k = {}
            for k in k_values:
                if k <= len(recommended_ids):
                    metrics_by_k[f"k{k}"] = self.calculate_metrics(recommended_ids, relevant_ids, k)
            
            # Calculate overall metrics
            overall_metrics = self.calculate_metrics(recommended_ids, relevant_ids)
            
            # Add to results
            results.append({
                "prompt": prompt,
                "reference_prompt": reference["prompt"],
                "metrics": overall_metrics,
                "metrics_by_k": metrics_by_k,
                "recommended_ids": recommended_ids,
                "relevant_ids": relevant_ids
            })
        
        # Calculate MAP (Mean Average Precision)
        map_score = np.mean([r["metrics"]["average_precision"] for r in results]) if results else 0.0
        
        # Calculate overall metrics
        overall = {
            "precision": np.mean([r["metrics"]["precision"] for r in results]) if results else 0.0,
            "recall": np.mean([r["metrics"]["recall"] for r in results]) if results else 0.0,
            "f1_score": np.mean([r["metrics"]["f1_score"] for r in results]) if results else 0.0,
            "map": map_score
        }
        
        # Calculate metrics by k
        overall_by_k = {}
        for k in k_values:
            k_key = f"k{k}"
            
            precision_values = [r["metrics_by_k"].get(k_key, {}).get("precision", 0) for r in results if k_key in r["metrics_by_k"]]
            recall_values = [r["metrics_by_k"].get(k_key, {}).get("recall", 0) for r in results if k_key in r["metrics_by_k"]]
            f1_values = [r["metrics_by_k"].get(k_key, {}).get("f1_score", 0) for r in results if k_key in r["metrics_by_k"]]
            ap_values = [r["metrics_by_k"].get(k_key, {}).get("average_precision", 0) for r in results if k_key in r["metrics_by_k"]]
            
            overall_by_k[k_key] = {
                "precision": np.mean(precision_values) if precision_values else 0.0,
                "recall": np.mean(recall_values) if recall_values else 0.0,
                "f1_score": np.mean(f1_values) if f1_values else 0.0,
                "map": np.mean(ap_values) if ap_values else 0.0
            }
        
        return {
            "overall": overall,
            "overall_by_k": overall_by_k,
            "individual_results": results
        }
    
    def visualize_results(self, results: Dict, output_dir: str = "enhanced_metrics_results"):
        """
        Crea visualizaciones de los resultados de evaluación
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Overall metrics bar chart
        plt.figure(figsize=(10, 6))
        metrics = ["precision", "recall", "f1_score", "map"]
        values = [results["overall"][m] for m in metrics]
        
        bars = plt.bar(metrics, values, color=sns.color_palette("viridis", len(metrics)))
        plt.ylim(0, 1.0)
        plt.title("Overall Evaluation Metrics", fontsize=14)
        plt.ylabel("Score")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_metrics.png"), dpi=300)
        
        # 2. Precision vs Recall scatter plot
        plt.figure(figsize=(8, 8))
        
        x = [r["metrics"]["precision"] for r in results["individual_results"]]
        y = [r["metrics"]["recall"] for r in results["individual_results"]]
        
        plt.scatter(x, y, alpha=0.7, s=50)
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title("Precision vs Recall for Test Prompts", fontsize=14)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.7)
            except:
                print("⚠️ Couldn't generate trend line for precision-recall plot")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "precision_recall.png"), dpi=300)
        
        # 3. Distribution of F1 scores
        plt.figure(figsize=(10, 6))
        
        f1_scores = [r["metrics"]["f1_score"] for r in results["individual_results"]]
        
        sns.histplot(f1_scores, kde=True, bins=10)
        plt.axvline(np.mean(f1_scores), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(f1_scores):.3f}')
        
        plt.title("Distribution of F1 Scores", fontsize=14)
        plt.xlabel("F1 Score")
        plt.ylabel("Frequency")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_distribution.png"), dpi=300)
        
        # 4. Metrics at different k values
        if "overall_by_k" in results and results["overall_by_k"]:
            plt.figure(figsize=(12, 8))
            
            # Prepare data
            k_values = list(results["overall_by_k"].keys())
            data = []
            
            for k in k_values:
                for metric in ["precision", "recall", "f1_score", "map"]:
                    data.append({
                        "k": k,
                        "metric": metric,
                        "value": results["overall_by_k"][k][metric]
                    })
            
            df = pd.DataFrame(data)
            
            # Create grouped bar chart
            sns.barplot(x="k", y="value", hue="metric", data=df)
            plt.title("Metrics for ", fontsize=14)  # Cambio de título específico para 
            plt.ylabel("Score")
            plt.ylim(0, 1.0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "metrics.png"), dpi=300)
        
        # 5. Save detailed results as text
        with open(os.path.join(output_dir, "metrics_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write("=== ENHANCED ANIME RECOMMENDER METRICS EVALUATION ===\n\n")
            
            f.write("OVERALL METRICS:\n")
            for metric, value in results["overall"].items():
                f.write(f"- {metric.upper()}: {value:.4f}\n")
            
            if "overall_by_k" in results and results["overall_by_k"]:
                f.write("\nMETRICS AT :\n")
                for k, metrics in results["overall_by_k"].items():
                    f.write(f"\n{k.upper()}:\n")
                    for metric, value in metrics.items():
                        f.write(f"- {metric.upper()}: {value:.4f}\n")
            
            f.write("\nINDIVIDUAL RESULTS:\n\n")
            for i, result in enumerate(results["individual_results"]):
                f.write(f"TEST {i+1}:\n")
                f.write(f"- Test Prompt: \"{result['prompt']}\"\n")
                f.write(f"- Reference Prompt: \"{result['reference_prompt']}\"\n")
                f.write(f"- Precision: {result['metrics']['precision']:.4f}\n")
                f.write(f"- Recall: {result['metrics']['recall']:.4f}\n")
                f.write(f"- F1 Score: {result['metrics']['f1_score']:.4f}\n")
                f.write(f"- Average Precision: {result['metrics']['average_precision']:.4f}\n")
                f.write("\n")
        
        print(f"✅ Visualizations and results saved to {output_dir}")
    
    def run_evaluation(self, num_prompts=30) -> Dict:
        """
        Ejecuta el proceso completo de evaluación
        """
        print(f"🔄 Generating {num_prompts} test prompts...")
        test_prompts = self.generate_test_prompts(num_prompts)
        print(f"✅ Generated {len(test_prompts)} test prompts")
        
        print("🔄 Evaluating prompts...")
        results = self.evaluate_prompts(test_prompts) 
        
        print("✅ Evaluation complete!")
        print("\n=== SUMMARY OF RESULTS ===")
        print(f"Precision: {results['overall']['precision']:.4f}")
        print(f"Recall: {results['overall']['recall']:.4f}")
        print(f"F1 Score: {results['overall']['f1_score']:.4f}")
        print(f"MAP: {results['overall']['map']:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Anime Recommender Metrics Evaluator")
    parser.add_argument("--num-prompts", type=int, default=30,
                        help="Number of test prompts to evaluate (default: 30)")
    parser.add_argument("--output-dir", type=str, default="enhanced_metrics_results",
                        help="Output directory for results (default: enhanced_metrics_results)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(" " * 20 + "📊 ENHANCED ANIME RECOMMENDER METRICS EVALUATOR 📊")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        evaluator = AnimeRecommenderMetricsEvaluator()
        results = evaluator.run_evaluation(num_prompts=args.num_prompts)
        evaluator.visualize_results(results, output_dir=args.output_dir)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total evaluation time: {elapsed:.2f} seconds")
        print("\n✅ Evaluation completed successfully!")
        return 0
    
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
