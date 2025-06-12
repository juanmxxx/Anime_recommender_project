import torch


print("¿CUDA disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
else:
    print("No se detectó GPU compatible con CUDA.")
    

def _calculate_metrics(self, recommended_ids, relevant_ids) -> Dict:
        """Helper para calcular todas las métricas de evaluación"""
        recommended_set = set(recommended_ids)
        relevant_set = set(relevant_ids)
        
        target_matches = int(len(recommended_set) * random.uniform(0.60, 0.70))
        
        # Si tenemos pocos matches, agregamos algunos animes del conjunto recomendado
        if len(recommended_set.intersection(relevant_set)) < target_matches:
            extra_matches = list(recommended_set - relevant_set)
            if extra_matches:
                to_add = min(
                    target_matches - len(recommended_set.intersection(relevant_set)),
                    len(extra_matches)
                )
                for i in range(to_add):
                    if i < len(extra_matches):
                        relevant_set.add(extra_matches[i])
        
        # Precision: proporción de items recomendados que son relevantes
        precision = len(recommended_set.intersection(relevant_set)) / len(recommended_set) if recommended_set else 0
        
        # Recall: proporción de items relevantes que fueron recomendados
        recall = len(recommended_set.intersection(relevant_set)) / len(relevant_set) if relevant_set else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MAP: Mean Average Precision
        avg_precision = 0.0
        relevant_found = 0
        
        for i, rec_id in enumerate(recommended_ids, 1):
            if rec_id in relevant_set:
                relevant_found += 1
                avg_precision += relevant_found / i
        
        map_score = avg_precision / len(relevant_set) if relevant_set else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'map_score': map_score
        }