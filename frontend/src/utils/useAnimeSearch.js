import { useState } from 'react';
import { trackEvent, trackLoadTime } from './tracking';

/**
 * Hook personalizado useAnimeSearch
 * Gestiona el estado y la lógica para la búsqueda de animes
 * 
 * Proporciona:
 * - Estado de resultados de animes
 * - Gestión del texto de búsqueda (prompt)
 * - Estado de carga durante la búsqueda
 * - Funciones para buscar y limpiar resultados
 * 
 * @returns {Object} Estado y funciones para la búsqueda de animes
 */
export const useAnimeSearch = () => {
  // Estado para almacenar los resultados de búsqueda
  const [animes, setAnimes] = useState(() => {
    // Intentar recuperar resultados anteriores de localStorage
    const savedResults = localStorage.getItem("lastAnimeResults");
    return savedResults ? JSON.parse(savedResults) : [];
  });
  
  // Estado para el texto de búsqueda (prompt)
  const [prompt, setPrompt] = useState(() => {
    const savedPrompt = localStorage.getItem("lastAnimePrompt");
    return savedPrompt || "";
  });
  
  // Estado para indicar si hay una búsqueda en curso
  const [isLoading, setIsLoading] = useState(false);
  
  /**
   * Función para buscar animes según el prompt proporcionado
   * @param {string} searchPrompt - Texto de búsqueda para encontrar animes
   * @returns {Array|null} Arreglo de animes encontrados o null en caso de error
   */
  const searchAnime = async (searchPrompt) => {
    try {
      // Solo buscar si hay texto en el prompt
      if (!searchPrompt.trim()) {
        alert("Por favor ingresa una palabra clave para buscar animes");
        return;
      }
      
      setIsLoading(true);
      
      // Registrar evento de búsqueda
      await trackEvent('search', { prompt_text: searchPrompt });
      
      // Importar dinámicamente la configuración de API
      const { API_URLS } = await import('../config');
      
      // Medir tiempo de carga
      const startTime = performance.now();
      let loadTime;
      let data = null;
      
      try {
        // Siempre obtener las 100 mejores recomendaciones usando config.js
        const response = await fetch(API_URLS.recommend(searchPrompt, 100));
        data = await response.json();
          
        // Calcular tiempo de carga
        const endTime = performance.now();
        loadTime = Math.round(endTime - startTime);
        
        // Procesar datos recibidos
        if (data && data.recommendations && data.recommendations.length > 0) {
          // Guardar resultados en localStorage para persistencia
          localStorage.setItem("lastAnimeResults", JSON.stringify(data.recommendations));
          localStorage.setItem("lastAnimePrompt", searchPrompt);
          setAnimes(data.recommendations);
          
          if (data.keyphrases) {
            console.log("Keyphrases extraídas:", data.keyphrases);
          }
          
          setPrompt(searchPrompt);
          return data.recommendations;
        } else {
          alert("No se encontraron animes que coincidan con tu búsqueda");
          return [];
        }
      } catch (apiError) {
        console.error("Error al obtener recomendaciones:", apiError);
        alert("Ocurrió un error al buscar recomendaciones. Por favor intenta de nuevo.");
        loadTime = Math.round(performance.now() - startTime); // Seguir registrando tiempo incluso con error
        return [];
      } finally {
        // Registrar tiempo de carga fuera del try/catch de la API para que siempre se ejecute
        if (loadTime) {  // Solo si se pudo medir el tiempo
          console.log('Intentando registrar tiempo de carga:', loadTime, 'ms para prompt:', searchPrompt);
          try {
            await trackLoadTime(loadTime, searchPrompt);
            console.log('Tiempo de carga registrado exitosamente');
          } catch (logError) {
            console.error('Error al registrar tiempo de carga:', logError);
          }
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Limpia los resultados de búsqueda y el prompt
   * También elimina los datos guardados en localStorage
   */
  const clearSearch = () => {
    setAnimes([]);
    setPrompt("");
    localStorage.removeItem("lastAnimeResults");
    localStorage.removeItem("lastAnimePrompt");
  };

  // Retornar todas las variables y funciones necesarias para usar este hook
  return {
    animes,
    prompt,
    setPrompt,
    isLoading,
    searchAnime,
    clearSearch
  };
};

export default useAnimeSearch;
