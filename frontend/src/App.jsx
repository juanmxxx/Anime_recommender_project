import { useState } from 'react'
import './App.css'

// Importar componentes
import { AnimeCard, getAnimeField } from './components/AnimeCard'
import AnimeModal from './components/AnimeModal'
import MetricsPanel from './components/MetricsPanel'

// Importar utilidades
import { trackEvent } from './utils/tracking'
import useAnimeSearch from './utils/useAnimeSearch'

// Imágenes decorativas de fondo (chicas anime)
const backgroundGirls = [
  '/images/E2d2giGWQAMr6dx.jpg',
  '/images/Episodio_10_-_33.webp',
  '/images/Zerotwomain.webp',
];
const centralImage = '/images/Mayoi_Owari3.webp';

/**
 * Componente principal de la aplicación
 * Smart Anime Recommender (S.A.R.)
 */
function App() {  
  // Usar nuestro hook personalizado de búsqueda de anime
  const { animes, prompt, setPrompt, isLoading, searchAnime, clearSearch } = useAnimeSearch();
  
  // Otros estados
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

  return (
    <div style={{ position: 'relative', minHeight: '100vh', overflow: 'hidden', width: '1200px' }}>
      {/* Fondo decorativo chicas anime */}
      <div style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
      }}>
        {/* Imágenes laterales */}
        {backgroundGirls.map((url, idx) => (
          <img
            key={idx}
            src={url}
            alt="chica anime decorativa"
            style={{
              position: 'absolute',
              left: `${10 + idx * 40}%`,
              bottom: idx % 2 === 0 ? '0' : '10%',
              width: '320px',
              opacity: 0.13,
              filter: 'blur(2px) grayscale(0.2)',
              zIndex: 0,
              userSelect: 'none',
            }}
          />
        ))}
        {/* Imagen central superior */}
        <img
          src={centralImage}
          alt="chica anime central"
          style={{
            position: 'absolute',
            top: 0,
            left: '50%',
            transform: 'translateX(-50%)',
            width: '50vw',
            maxWidth: '700px',
            opacity: 0.13,
            filter: 'blur(2px) grayscale(0.2)',
            zIndex: 0,
            userSelect: 'none',
          }}
        />
      </div>      {/* Contenido principal */}
      <div style={{ position: 'relative', zIndex: 1, maxWidth: '1800px', margin: '0 auto', width: '90%', padding: '0 20px' }}>        <h1>Smart Anime Recommender</h1>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1rem' }}>          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Write your prompt or anime search keywords here..."
            style={{
              width: '80%',
              height: '80px',
              fontSize: '1.3rem',
              padding: '1rem',
              borderRadius: '10px',
              border: '2px solid #61dafb',
              resize: 'none',
              boxSizing: 'border-box',
              outline: 'none',
              fontWeight: 'bold',
              overflow: 'auto',
              display: 'block',
              maxHeight: '80px'
            }}
          />          <button
            onClick={() => searchAnime(prompt)}
            disabled={isLoading}
            style={{
              height: '80px',
              padding: '0 2rem',
              fontSize: '1.1rem',
              fontWeight: 'bold',
              background: '#61dafb',
              color: '#222',
              border: 'none',
              borderRadius: '10px',
              cursor: isLoading ? 'wait' : 'pointer',
              marginLeft: '0.5rem',
              whiteSpace: 'nowrap',
              opacity: isLoading ? 0.7 : 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              minWidth: '180px',
              transition: 'all 0.3s'
            }}
          >            {isLoading ? (
              <>
                <div className="loading-spinner"></div>
                Searching...
              </>            ) : 'Recommend'}
          </button>
        </div>        <div style={{ marginBottom: '2rem', textAlign: 'left', display: 'flex', alignItems: 'center' }}>
          <label htmlFor="top-select" style={{ fontWeight: 'bold', marginRight: '0.5rem' }}>Show:</label>
          <select
            id="top-select"
            value={top}
            onChange={e => setTop(e.target.value)}
            style={{
              fontSize: '1.1rem',
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              border: '1.5px solid #61dafb',
              outline: 'none',
              fontWeight: 'bold',
            }}
          >
            <option value="top 5">Top 5</option>
            <option value="top 10">Top 10</option>
            <option value="top 20">Top 20</option>
            <option value="top 50">Top 50</option>
            <option value="top 100">Top 100</option>
          </select>
          
          {animes.length > 0 && (
            <button              onClick={clearSearch}
              style={{
                marginLeft: 'auto',
                background: 'rgba(255, 100, 100, 0.2)',
                color: '#ff6b6b',
                border: '1px solid #ff6b6b',
                borderRadius: '8px',
                padding: '0.5rem 1rem',
                fontSize: '0.9rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '5px'
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
              New Search
            </button>
          )}
        </div>{isLoading && (
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            justifyContent: 'center',
            padding: '3rem'
          }}>
            <img 
              src="/images/inugami-korone-hololive.gif" 
              alt="Korone loading animation" 
              style={{ 
                width: '200px', 
                marginTop: '1.5rem',
                borderRadius: '12px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
              }} 
            />            <p style={{ marginTop: '1rem', fontSize: '1.2rem', color: '#61dafb' }}>
              Finding the best anime for you...
            </p>
          </div>
        )}        {!isLoading && animes.length === 0 && (
          <div style={{ 
            textAlign: 'center', 
            padding: '2rem',
            background: 'rgba(30, 30, 50, 0.5)',
            borderRadius: '12px',
            backdropFilter: 'blur(5px)',
            marginTop: '2rem'
          }}>
            <h2 style={{ color: '#61dafb', marginBottom: '1rem' }}>¡Welcome to S.A.R dear friend!</h2>
            <p style={{ fontSize: '1.1rem', lineHeight: '1.5', maxWidth: '600px', margin: '0 auto' }}>
              Enter keywords related to the type of anime you would like to watch.
              <br />For example, try genres such as “romance comedy”, “action adventure”, 
              or even specific themes such as “cyberpunk dystopia”..
            </p>
            <div style={{
              marginTop: '2rem',
              display: 'flex',
              justifyContent: 'center',
              gap: '1rem',
              flexWrap: 'wrap'
            }}>              {["romance comedy", "action adventure", "sports", "fantasy magic", "slice of life", "psychological drama"].map(suggestion => (
                <button 
                  key={suggestion}                  onClick={() => handleSuggestionClick(suggestion)}
                  style={{
                    padding: '0.5rem 1rem',
                    background: '#333',
                    color: '#fff',
                    border: '1px solid #61dafb',
                    borderRadius: '20px',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        
        {!isLoading && animes.length > 0 && (
          <ul style={{ listStyle: 'none', padding: 0 }}>            {animes.slice(0, getTopN()).map((anime, idx) => (
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
        
        {/* Anime confirmation modal */}
        {modalOpen && selectedAnime !== null && (
          <AnimeModal
            anime={animes[selectedAnime]}
            onClose={handleModalClose}
            onConfirm={handleConfirm}
            debugMode={debugMode}
          />
        )}
        
        {/* Debug Mode Toggle */}
        {process.env.NODE_ENV !== 'production' && (
          <div style={{
            position: 'fixed',
            right: 10,
            bottom: 10,
            background: 'rgba(0,0,0,0.6)',
            padding: '5px 10px',
            borderRadius: '4px',
            fontSize: '0.8rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '5px'
          }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={debugMode}
                onChange={() => setDebugMode(!debugMode)}
              />
              Debug Mode
            </label>
            
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={showMetrics}
                onChange={(e) => {
                  setShowMetrics(e.target.checked);
                  if (e.target.checked) fetchMetrics();
                }}
              />
              Show Metrics
            </label>
          </div>
        )}

        {/* Metrics Panel */}
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


