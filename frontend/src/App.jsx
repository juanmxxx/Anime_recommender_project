import { useState } from 'react'

// Importar estilos
import './App.css'
import './styles/App.css'

// Importar componentes UI
import { AnimeCard, getAnimeField } from './components/AnimeCard'
import AnimeModal from './components/AnimeModal'
import MetricsPanel from './components/MetricsPanel'
import { BackgroundDecorator, LoadingIndicator, WelcomeScreen } from './components/UIElements'
import { SearchBar, DebugControls } from './components/Controls'

// Importar utilidades
import { trackEvent } from './utils/tracking'
import useAnimeSearch from './utils/useAnimeSearch'

/**
 * Componente principal de la aplicación
 * Smart Anime Recommender (S.A.R.)
 * 
 * Actúa como orquestador principal para todos los componentes de la aplicación
 */
function App() {  
  // Usar nuestro hook personalizado de búsqueda de anime
  const { animes, prompt, setPrompt, isLoading, searchAnime, clearSearch } = useAnimeSearch();
  
  // Estados para la interfaz de usuario
  const [top, setTop] = useState("top 5"); // Número de resultados a mostrar
  const [modalOpen, setModalOpen] = useState(false); // Control del modal
  const [selectedAnime, setSelectedAnime] = useState(null); // Anime seleccionado
  const [debugMode, setDebugMode] = useState(false); // Modo de depuración
  const [showMetrics, setShowMetrics] = useState(false); // Mostrar panel de métricas
  const [metrics, setMetrics] = useState(null); // Datos de métricas
  
  /**
   * Obtiene el valor numérico de la selección "top N"
   * @returns {number} Número de animes a mostrar
   */
  const getTopN = () => {
    return parseInt(top.split(' ')[1]);
  };
    
  /**
   * Manejador para clics en botones de sugerencia
   * @param {string} suggestion - Texto de sugerencia a buscar
   */
  const handleSuggestionClick = async (suggestion) => {
    if (suggestion && suggestion.trim()) {
      setPrompt(suggestion); // Actualizar el campo de entrada
      await searchAnime(suggestion); // Buscar con el prompt sugerido
    }
  };
  
  /**
   * Manejador para clics en tarjetas de anime
   * @param {number} idx - Índice del anime seleccionado
   */
  const handleCardClick = idx => {
    // Registrar evento de clic
    const anime = animes[idx];
    const animeName = getAnimeField(anime, 'name');
    const animeId = getAnimeField(anime, 'anime_id');
    
    trackEvent('click', { 
      anime_clicked: animeName,
      anime_id: animeId 
    });
    
    setSelectedAnime(idx);
    setModalOpen(true);
  };

  /**
   * Cierra el modal de confirmación
   */
  const handleModalClose = () => {
    setModalOpen(false);
    setSelectedAnime(null);
  };
  
  /**
   * Maneja la confirmación para ver un anime
   * Abre una nueva pestaña con la URL de AnimeFlv
   */
  const handleConfirm = async () => {
    if (selectedAnime !== null) {
      const anime = animes[selectedAnime];
      const name = getAnimeField(anime, 'name');
      const animeId = getAnimeField(anime, 'anime_id');
      
      // Registrar evento de clic
      await trackEvent('click', { 
        anime_clicked: name,
        anime_id: animeId 
      });
      
      // Obtener URL de AnimeFlv y abrir en nueva pestaña
      const animeFlvUrl = `https://www3.animeflv.net/browse?q=${encodeURIComponent(name)}`;
      window.open(animeFlvUrl, '_blank');
    }
    handleModalClose();
  };

  /**
   * Obtiene métricas de uso del backend
   */
  const fetchMetrics = async () => {
    try {
      const { API_URLS } = await import('./config');
      const response = await fetch(API_URLS.getMetrics(7)); // Últimos 7 días
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  /**
   * Maneja el cambio en el toggle de métricas
   */
  const handleMetricsToggle = (e) => {
    const newValue = e.target.checked;
    setShowMetrics(newValue);
    if (newValue) fetchMetrics();
  };  return (
    <div className="app-container">
      {/* Fondo decorativo */}
      <BackgroundDecorator />
      
      {/* Contenido principal */}
      <div className="main-content">
        <h1>Smart Anime Recommender</h1>
        
        {/* Barra de búsqueda y controles */}
        <SearchBar 
          prompt={prompt}
          setPrompt={setPrompt}
          isLoading={isLoading}
          onSearch={searchAnime}
          top={top}
          setTop={setTop}
          hasResults={animes.length > 0}
          onClear={clearSearch}
        />
        
        {/* Indicador de carga */}
        {isLoading && <LoadingIndicator />}
        
        {/* Pantalla de bienvenida (cuando no hay resultados ni está cargando) */}
        {!isLoading && animes.length === 0 && (
          <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
        )}
        
        {/* Lista de animes recomendados */}
        {!isLoading && animes.length > 0 && (
          <ul className="anime-list">
            {animes.slice(0, getTopN()).map((anime, idx) => (
              <AnimeCard
                key={idx}
                anime={anime}
                index={idx}
                onClick={handleCardClick}
                debugMode={debugMode}
              />
            ))}
          </ul>
        )}
        
        {/* Modal de confirmación de anime */}
        {modalOpen && selectedAnime !== null && (
          <AnimeModal
            anime={animes[selectedAnime]}
            onClose={handleModalClose}
            onConfirm={handleConfirm}
            debugMode={debugMode}
          />
        )}
        
        {/* Controles de depuración y métricas */}
        <DebugControls 
          debugMode={debugMode}
          setDebugMode={setDebugMode}
          showMetrics={showMetrics}
          onMetricsToggle={handleMetricsToggle}
        />

        {/* Panel de métricas */}
        {showMetrics && metrics && (
          <MetricsPanel 
            metrics={metrics}
            onClose={() => setShowMetrics(false)}
            onRefresh={fetchMetrics}
          />
        )}
      </div>
    </div>
  )
}

export default App


