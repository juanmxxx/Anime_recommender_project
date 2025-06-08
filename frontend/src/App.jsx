// filepath: c:\proyectoIA\frontend\src\App.jsx
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

// Fondo decorativo estático con chicas anime (usando imágenes locales)
const backgroundGirls = [
  '/images/E2d2giGWQAMr6dx.jpg',
  '/images/Episodio_10_-_33.webp',
  '/images/Zerotwomain.webp',
];
const centralImage = '/images/Mayoi_Owari3.webp';

function App() {  
  // Initialize state with localStorage values if they exist, otherwise empty values
  const [prompt, setPrompt] = useState(() => {
    const savedPrompt = localStorage.getItem("lastAnimePrompt");
    return savedPrompt || "";
  });
  const [top, setTop] = useState("top 5");
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedAnime, setSelectedAnime] = useState(null);
  const [animes, setAnimes] = useState(() => {
    // Intentar recuperar resultados previos del localStorage
    const savedResults = localStorage.getItem("lastAnimeResults");
    return savedResults ? JSON.parse(savedResults) : [];
  });
  const [isLoading, setIsLoading] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  
  // Function to generate AnimeFlv URL based on anime name
  const getAnimeFlvUrl = (animeName) => {
    if (!animeName) return "#";
    // Format the name for a URL: lowercase, replace spaces with dashes, remove special chars
    const formattedName = animeName
      .toLowerCase()
      .replace(/[^\w\s-]/g, '') // Remove special characters
      .replace(/\s+/g, '-')     // Replace spaces with dashes
      .trim();
    return `https://www3.animeflv.net/browse?q=${encodeURIComponent(animeName)}`;
  };
  
  // Function to get numerical value from top selection
  const getTopN = () => {
    // Extract the number from strings like "top 5", "top 10", etc.
    return parseInt(top.split(' ')[1]);
  };
  
  // Función para normalizar el acceso a los campos del anime (independiente del case)
  const getAnimeField = (anime, fieldName) => {
    // Buscar el campo en diferentes formatos (camelCase, PascalCase)
    const possibleNames = [
      fieldName.toLowerCase(),  // genres
      fieldName,                // genres (como está)
      fieldName.charAt(0).toUpperCase() + fieldName.slice(1), // Genres
    ];
    
    // Devolver el primer campo que exista
    for (const name of possibleNames) {
      if (anime[name] !== undefined) {
        return anime[name];
      }
    }
    
    return null; // No se encontró el campo
  };
    // Debug function to log anime data structure
  const debugAnimeData = (anime) => {
    console.log("Estructura de datos del anime:", anime);
    return anime;
  };
  
  // Handler for suggestion button clicks
  const handleSuggestionClick = async (suggestion) => {
    if (suggestion && suggestion.trim()) {
      try {
        setIsLoading(true);
        
        // Importamos dinámicamente la configuración de API
        const { API_URLS } = await import('./config');
        
        // Use the suggestion directly without checking prompt state
        const response = await fetch(API_URLS.recommend(suggestion, 100));
        const data = await response.json();
        
        if (data && data.recommendations && data.recommendations.length > 0) {
          // Mostrar estructura en modo debug
          if (debugMode) {
            console.log("Estructura de datos recibida:", data.recommendations[0]);
          }
          
          // Save results to localStorage for persistence
          localStorage.setItem("lastAnimeResults", JSON.stringify(data.recommendations));
          localStorage.setItem("lastAnimePrompt", suggestion);
          
          // Guardar solo el array de recomendaciones
          setAnimes(data.recommendations);
        } else {
          // Si no hay resultados, mostrar un mensaje
          alert("No se encontraron animes que coincidan con tu búsqueda");
        }
      } catch (error) {
        console.error("Error al obtener recomendaciones:", error);
        alert("Ocurrió un error al buscar recomendaciones. Por favor intenta de nuevo.");
      } finally {
        setIsLoading(false);
      }
    }
  };
  
  // Fetch recommendations from backend  
  const fetchRecommendations = async () => {
    try {
      // Solo realizar búsqueda si hay texto en el prompt
      if (!prompt.trim()) {
        alert("Por favor ingresa una palabra clave para buscar animes");
        return;
      }
      
      setIsLoading(true);
      
      // Importamos dinámicamente la configuración de API
      const { API_URLS } = await import('./config');
      
      // Always fetch top 100 recommendations using config.js
      const response = await fetch(API_URLS.recommend(prompt, 100));
      const data = await response.json();
      
      if (data && data.recommendations && data.recommendations.length > 0) {
        // Mostrar estructura en modo debug
        if (debugMode) {
          console.log("Estructura de datos recibida:", data.recommendations[0]);
        }
        
        // Save results to localStorage for persistence
        localStorage.setItem("lastAnimeResults", JSON.stringify(data.recommendations));
        localStorage.setItem("lastAnimePrompt", prompt);
        
        // Guardar solo el array de recomendaciones
        setAnimes(data.recommendations);
        
        console.log("Keyphrases extraídas:", data.keyphrases);
      } else {
        // Si no hay resultados, mostrar un mensaje
        alert("No se encontraron animes que coincidan con tu búsqueda");
      }
    } catch (error) {
      console.error("Error al obtener recomendaciones:", error);
      alert("Ocurrió un error al buscar recomendaciones. Por favor intenta de nuevo.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCardClick = idx => {
    setSelectedAnime(idx);
    setModalOpen(true);
  };

  const handleModalClose = () => {
    setModalOpen(false);
    setSelectedAnime(null);
  };
  
  const handleConfirm = () => {
    if (selectedAnime !== null) {
      // Navigate to AnimeFlv search for this anime
      const anime = animes[selectedAnime];
      const name = getAnimeField(anime, 'name');
      window.open(getAnimeFlvUrl(name), '_blank');
    }
    handleModalClose();
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
            onClick={fetchRecommendations}
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
            <button
              onClick={() => {
                // Limpiar resultados y localStorage
                setAnimes([]);
                setPrompt("");
                localStorage.removeItem("lastAnimeResults");
                localStorage.removeItem("lastAnimePrompt");
              }}
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
                  key={suggestion}
                  onClick={() => {
                    setPrompt(suggestion);
                    // Ejecutamos la búsqueda inmediatamente con el valor de sugerencia
                    // en lugar de esperar a que se actualice el estado
                    handleSuggestionClick(suggestion);
                  }}
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
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {animes.slice(0, getTopN()).map((anime, idx) => (
              <li
                key={idx}
                onClick={() => handleCardClick(idx)}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1.5rem',
                  marginBottom: '2rem',
                  background: '#222',
                  borderRadius: '12px',
                  padding: '1.5rem',
                  boxShadow: '0 2px 8px #0002',
                  cursor: 'pointer',
                  transition: 'box-shadow 0.2s',
                  minHeight: '260px',
                  position: 'relative'
                }}
                title="Click to start this anime"
              >
                {/* Mostrar estructura en modo debug */}
                {debugMode && (
                  <div style={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    background: 'rgba(0,0,0,0.7)',
                    color: 'lime',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '0.7rem',
                    cursor: 'pointer',
                    zIndex: 10
                  }} onClick={(e) => {
                    e.stopPropagation();
                    console.log("Datos del anime:", anime);
                    alert("Datos mostrados en consola");
                  }}>
                    Ver datos
                  </div>
                )}
                
                {/* Imagen del anime */}
                <img 
                  src={getAnimeField(anime, "image_url")} 
                  alt={getAnimeField(anime, "name")} 
                  style={{ width: 120, borderRadius: 8 }} 
                  onError={(e) => { e.target.src = '/images/defaultImagePortrait.jpg'; }}
                />
                
                <div style={{ textAlign: 'left', width: '100%' }}>
                  <h2 style={{ margin: '0 0 0.5rem 0' }}>{getAnimeField(anime, "name")}</h2>
                  
                  {/* Sinopsis */}
                  <div style={{ 
                    margin: '0 0 0.5rem 0', 
                    color: '#bbb',
                    maxHeight: '180px', /* Increased height */
                    overflowY: 'auto',
                    paddingRight: '10px',
                    lineHeight: '1.5',
                    fontSize: '0.95rem'
                  }}>
                    {(() => {
                      const synopsis = getAnimeField(anime, "synopsis");
                      if (!synopsis) return <p>No hay sinopsis disponible.</p>;
                      
                      return synopsis.split('. ')
                        .filter(sentence => sentence.trim().length > 0)
                        .reduce((result, sentence, index, array) => {
                          // Group sentences in pairs (every 2 sentences)
                          if (index % 2 === 0) {
                            // If this is an even index and there's a next sentence, combine them
                            const nextSentence = array[index + 1];
                            const combinedText = nextSentence 
                              ? `${sentence.trim()}. ${nextSentence.trim()}${nextSentence.trim().endsWith('.') ? '' : '.'}`
                              : `${sentence.trim()}${sentence.trim().endsWith('.') ? '' : '.'}`;
                            
                            result.push(combinedText);
                          }
                          return result;
                        }, [])
                        .map((paragraph, i) => (
                          <p key={i} style={{ 
                            margin: '0 0 0.8rem 0',
                            textAlign: 'justify'
                          }}>
                            {paragraph}
                          </p>
                        ))
                    })()}
                  </div>
                  
                  {/* Basic info: score and ranking */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ fontWeight: 'bold', color: '#ffd700' }}>
                      Score: {getAnimeField(anime, "score") || "N/A"}
                    </div>
                    <div style={{ color: '#61dafb' }}>
                      Ranking #{getAnimeField(anime, "rank") || "?"}
                    </div>
                    {getAnimeField(anime, "recommendation_score") && (
                      <div style={{ fontWeight: 'bold', color: '#7df740' }}>
                        Match: {(getAnimeField(anime, "recommendation_score") * 100 * (-1)).toFixed(1)}%
                      </div>
                    )}
                  </div>
                  
                  {/* Línea separadora */}
                  <div style={{ height: '1px', background: '#444', margin: '8px 0' }}></div>
                  
                  {/* Info adicional */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
                    {/* Tipo (TV/OVA/Movie) */}
                    <div style={{ 
                      background: '#333', 
                      color: '#fff',
                      padding: '4px 8px', 
                      borderRadius: '4px',
                      fontSize: '0.9rem',
                      border: '1px solid #555'
                    }}>
                      {getAnimeField(anime, "type") || 'TV'}
                    </div>
                    
                    {/* Año de lanzamiento */}
                    {(() => {
                      const aired = getAnimeField(anime, "aired");
                      let year = null;
                      
                      if (aired) {
                        const match = aired.toString().match(/\d{4}/);
                        if (match) year = match[0];
                      }
                      
                      return year ? (
                        <div style={{ 
                          background: '#2d4838', 
                          color: '#a0e6b8',
                          padding: '4px 8px', 
                          borderRadius: '4px',
                          fontSize: '0.9rem',
                          border: '1px solid #365544'
                        }}>
                          {year}
                        </div>
                      ) : null;
                    })()}
                    
                    {/* Estado (Airing/Finished) */}
                    {(() => {
                      const status = getAnimeField(anime, "status");
                      const isAiring = status === 'Currently Airing' || status === 'Airing';
                      
                      return status ? (
                        <div style={{ 
                          background: isAiring ? '#9090c0' : '#2a2a3d', 
                          color: isAiring ? '#1e1e36' : '#ffffff',
                          padding: '4px 8px', 
                          borderRadius: '4px',
                          fontSize: '0.9rem',
                          fontWeight: isAiring ? 'bold' : 'normal'
                        }}>
                          {status}
                        </div>
                      ) : null;
                    })()}
                    
                    {/* Episodios */}
                    {(() => {
                      const episodes = getAnimeField(anime, "episodes");
                      return episodes ? (
                        <div style={{ 
                          background: '#3d3d2a', 
                          color: '#dfdfb0',
                          padding: '4px 8px', 
                          borderRadius: '4px',
                          fontSize: '0.9rem',
                        }}>
                          {episodes} eps
                        </div>
                      ) : null;
                    })()}
                  </div>
                  
                  {/* Géneros */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px' }}>
                    {(() => {
                      const genres = getAnimeField(anime, "genres");
                      
                      if (!genres) {
                        return <span style={{ color: '#888', fontSize: '0.9rem' }}>Sin información de géneros</span>;
                      }
                      
                      let genreList = genres;
                      if (typeof genres === 'string') {
                        genreList = genres.split(',').map(g => g.trim());
                      }
                      
                      if (!Array.isArray(genreList)) {
                        return <span style={{ color: '#888', fontSize: '0.9rem' }}>Formato de géneros desconocido</span>;
                      }
                      
                      return genreList.map((genre, i) => (
                        <span key={i} style={{ 
                          background: '#2a4555', 
                          color: '#7edeff',
                          padding: '3px 8px', 
                          borderRadius: '12px',
                          fontSize: '0.8rem',
                        }}>
                          {genre}
                        </span>
                      ));
                    })()}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
        
        {/* Modal personalizado */}
        {modalOpen && selectedAnime !== null && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            background: 'rgba(20, 24, 38, 0.75)',
            zIndex: 100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backdropFilter: 'blur(2px)'
          }}>
            <div style={{
              background: '#23243a',
              borderRadius: '18px',
              padding: '2.5rem 2rem 2rem 2rem',
              boxShadow: '0 8px 32px #0006',
              minWidth: 320,
              maxWidth: 380,
              textAlign: 'center',
              color: '#fff',
              position: 'relative',
            }}>
              {/* Mostrar estructura de datos en modo debug */}
              {debugMode && (
                <div 
                  style={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    background: 'rgba(0,0,0,0.7)',
                    color: 'lime',
                    fontSize: '0.7rem',
                    padding: '3px 6px',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    console.log("Datos del anime modal:", animes[selectedAnime]);
                    alert("Datos mostrados en consola");
                  }}
                >
                  Debug
                </div>
              )}
              
              <img 
                src={getAnimeField(animes[selectedAnime], "image_url")} 
                alt={getAnimeField(animes[selectedAnime], "name")} 
                style={{ width: 90, borderRadius: 12, marginBottom: 16, boxShadow: '0 2px 8px #0003' }} 
                onError={(e) => { e.target.src = '/images/defaultImagePortrait.jpg'; }}
              />
              <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.3rem', color: '#61dafb' }}>
                {getAnimeField(animes[selectedAnime], "name")}
              </h2>
              
              {(() => {
                const aired = getAnimeField(animes[selectedAnime], "aired");
                let year = null;
                
                if (aired) {
                  const match = aired.toString().match(/\d{4}/);
                  if (match) year = match[0];
                }
                
                return year ? (
                  <p style={{ 
                    color: '#a0e6b8',
                    fontSize: '0.9rem',
                    margin: '0 0 12px 0',
                    background: '#2d4838',
                    display: 'inline-block',
                    padding: '2px 8px',
                    borderRadius: '4px'
                  }}>
                    {year}
                  </p>
                ) : null;
              })()}
              
              <p style={{ color: '#bbb', marginBottom: 24 }}>¿Quieres ver este anime?</p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem' }}>
                <button onClick={handleConfirm} style={{
                  background: '#61dafb',
                  color: '#23243a',
                  border: 'none',
                  borderRadius: '8px',
                  fontWeight: 'bold',
                  fontSize: '1.1rem',
                  padding: '0.7rem 2.2rem',
                  cursor: 'pointer',
                  transition: 'background 0.2s'
                }}>
                  Sí
                </button>
                <button onClick={handleModalClose} style={{
                  background: '#444',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '1.1rem',
                  padding: '0.7rem 1rem',
                  cursor: 'pointer',
                  transition: 'background 0.2s'
                }}>
                  No
                </button>
              </div>
              
              {/* Botón de cierre (X) */}
              <button 
                onClick={handleModalClose}
                style={{
                  position: 'absolute',
                  top: 10,
                  right: 10,
                  background: 'none',
                  border: 'none',
                  color: '#999',
                  fontSize: '1.2rem',
                  cursor: 'pointer',
                  padding: '5px 10px'
                }}
              >
                ✕
              </button>
            </div>
          </div>
        )}
        
        {/* Toggle Debug Mode - Solo visible en desarrollo */}
        {process.env.NODE_ENV !== 'production' && (
          <div style={{
            position: 'fixed',
            right: 10,
            bottom: 10,
            background: 'rgba(0,0,0,0.6)',
            padding: '5px 10px',
            borderRadius: '4px',
            fontSize: '0.8rem'
          }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer' }}>
              <input 
                type="checkbox" 
                checked={debugMode}
                onChange={() => setDebugMode(!debugMode)}
              />
              Debug Mode
            </label>
          </div>
        )}
      </div>
    </div>
  )
}

export default App


