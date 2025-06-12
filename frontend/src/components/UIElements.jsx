import React from 'react';

/**
 * Componente para el fondo decorativo con imágenes de anime
 * @param {Object} props - Propiedades del componente
 */
export const BackgroundDecorator = () => {
  // Imágenes decorativas de fondo (chicas anime)
  const backgroundGirls = [
    '/images/E2d2giGWQAMr6dx.jpg',
    '/images/Episodio_10_-_33.webp',
    '/images/Zerotwomain.webp',
  ];
  const centralImage = '/images/Mayoi_Owari3.webp';

  return (
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
    </div>
  );
};

/**
 * Componente para indicar que se está cargando
 */
export const LoadingIndicator = () => {
  return (
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
      />
      <p style={{ marginTop: '1rem', fontSize: '1.2rem', color: '#61dafb' }}>
        Finding the best anime for you...
      </p>
    </div>
  );
};

/**
 * Componente para la pantalla de bienvenida cuando no hay resultados
 * @param {Function} onSuggestionClick - Función para manejar clics en sugerencias
 */
export const WelcomeScreen = ({ onSuggestionClick }) => {
  // Sugerencias predefinidas para búsqueda rápida
  const suggestions = [
    "romance comedy", 
    "action adventure", 
    "sports", 
    "fantasy magic", 
    "slice of life", 
    "psychological drama"
  ];

  return (
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
        <br />For example, try genres such as "romance comedy", "action adventure", 
        or even specific themes such as "cyberpunk dystopia"..
      </p>
      <div style={{
        marginTop: '2rem',
        display: 'flex',
        justifyContent: 'center',
        gap: '1rem',
        flexWrap: 'wrap'
      }}>
        {suggestions.map(suggestion => (
          <button 
            key={suggestion}
            onClick={() => onSuggestionClick(suggestion)}
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
  );
};
