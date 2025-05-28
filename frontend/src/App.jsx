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
  // Initialize state from localStorage or use defaults
  const [prompt, setPrompt] = useState(() => {
    const saved = localStorage.getItem("lastAnimePrompt");
    return saved || "";
  });
  const [top, setTop] = useState("top 5");
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedAnime, setSelectedAnime] = useState(null);
  const [animes, setAnimes] = useState(() => {
    const saved = localStorage.getItem("lastAnimeResults");
    return saved ? JSON.parse(saved) : [];
  });
  const [isLoading, setIsLoading] = useState(false);
  
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
    // Fetch recommendations from backend
  const fetchRecommendations = async () => {
    try {
      setIsLoading(true);
      // Always fetch top 100 recommendations
      const response = await fetch(`http://localhost:8000/recommend?keywords=${encodeURIComponent(prompt)}&top_n=100`);
      const data = await response.json();
      
      // Save results to localStorage for persistence
      localStorage.setItem("lastAnimeResults", JSON.stringify(data));
      localStorage.setItem("lastAnimePrompt", prompt);
      
      setAnimes(data);
    } catch (error) {
      console.error("Error al obtener recomendaciones:", error);
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
      window.location.href = getAnimeFlvUrl(animes[selectedAnime].Name);
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
        </div>
        <div style={{ marginBottom: '2rem', textAlign: 'left' }}>
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
          </select>        </div>          {isLoading && (
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
        )}
        
        {!isLoading && (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {animes.slice(0, getTopN()).map((anime, idx) => (              <li
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
                }}
                title="Click to start this anime"
              >
                <img 
                  src={anime["Image URL"]} 
                  alt={anime.Name} 
                  style={{ width: 120, borderRadius: 8 }} 
                  onError={(e) => { e.target.src = '/images/defaultImagePortrait.jpg'; }}
                />                <div style={{ textAlign: 'left', width: '100%' }}>
                  <h2 style={{ margin: '0 0 0.5rem 0' }}>{anime.Name}</h2>
                  <div style={{ 
                    margin: '0 0 0.5rem 0', 
                    color: '#bbb',
                    maxHeight: '180px', /* Increased height */
                    overflowY: 'auto',
                    paddingRight: '10px',
                    lineHeight: '1.5',
                    fontSize: '0.95rem'
                  }}>
                    {anime.Synopsis?.split('. ')
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
                    }
                  </div>
                    {/* Basic info: score and ranking */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ fontWeight: 'bold', color: '#ffd700' }}>Score: {anime.Score}</div>
                    <div style={{ color: '#61dafb' }}>Ranking: #{anime.Rank}</div>
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
                      {anime.Type || 'TV'}
                    </div>                    {/* Estado (Airing/Finished) */}
                    <div style={{ 
                      background: anime.Status === 'Currently Airing' || anime.Status === 'Airing' ? '#9090c0' : '#2a2a3d', 
                      color: anime.Status === 'Currently Airing' || anime.Status === 'Airing' ? '#1e1e36' : '#ffffff',
                      padding: '4px 8px', 
                      borderRadius: '4px',
                      fontSize: '0.9rem',
                      fontWeight: anime.Status === 'Currently Airing' || anime.Status === 'Airing' ? 'bold' : 'normal'
                    }}>
                      {anime.Status || 'Finished'}
                    </div>
                      {/* Mostrar duración para películas o episodios para series */}
                    {anime.Type === 'Movie' ? (
                      <div style={{ 
                        background: '#3d1f52', 
                        color: '#e2b4ff',
                        padding: '4px 8px', 
                        borderRadius: '4px',
                        fontSize: '0.9rem',
                        border: '1px solid #5e3d7a'
                      }}>
                        Duration: {anime.Duration || '1:30:00'}
                      </div>
                    ) : (
                      anime.Status !== 'Currently Airing' && 
                      anime.Status !== 'Airing' && 
                      anime.Episodes && 
                      !isNaN(parseInt(anime.Episodes)) && 
                      parseInt(anime.Episodes) > 0 && (
                        <div style={{ 
                          background: '#3d3426', 
                          color: '#ffc670',
                          padding: '4px 8px', 
                          borderRadius: '4px',
                          fontSize: '0.9rem'
                        }}>
                          {`${parseInt(anime.Episodes)} eps`}
                        </div>
                      )
                    )}
                  </div>
                  
                  {/* Géneros */}
                  <div style={{ marginTop: '10px' }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                      {anime.Genres && anime.Genres.split(', ').map((genre, i) => (
                        <span key={i} style={{ 
                          background: '#2a4555', 
                          color: '#7edeff',
                          padding: '3px 8px', 
                          borderRadius: '12px',
                          fontSize: '0.8rem',
                        }}>
                          {genre}
                        </span>
                      ))}
                      {!anime.Genres && <span style={{ color: '#888', fontSize: '0.9rem' }}>Sin información de géneros</span>}
                    </div>
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
              <img 
                src={animes[selectedAnime]["Image URL"]} 
                alt={animes[selectedAnime].Name} 
                style={{ width: 90, borderRadius: 12, marginBottom: 16, boxShadow: '0 2px 8px #0003' }} 
                onError={(e) => { e.target.src = '/images/defaultImagePortrait.jpg'; }}
              />              <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.3rem', color: '#61dafb' }}>{animes[selectedAnime].Name}</h2>
              <p style={{ color: '#bbb', marginBottom: 24 }}>Do you want to watch this anime?</p>
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
                  transition: 'background 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <span>Yes</span>
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                    <polyline points="15 3 21 3 21 9"></polyline>
                    <line x1="10" y1="14" x2="21" y2="3"></line>
                  </svg>
                </button>
                <button onClick={handleModalClose} style={{
                  background: '#23243a',
                  color: '#fff',
                  border: '2px solid #61dafb',
                  borderRadius: '8px',
                  fontWeight: 'bold',
                  fontSize: '1.1rem',
                  padding: '0.7rem 2.2rem',
                  cursor: 'pointer',                  transition: 'background 0.2s',
                }}>No</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App
