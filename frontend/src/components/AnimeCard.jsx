import React from 'react';

/**
 * Función de utilidad para acceder a campos de anime con diferentes formatos de casos
 * Busca el campo en diferentes formatos (camelCase, PascalCase) y devuelve el primero que encuentre
 */
const getAnimeField = (anime, fieldName) => {
  // Buscar campo en diferentes formatos (camelCase, PascalCase)
  const possibleNames = [
    fieldName.toLowerCase(),  // géneros
    fieldName,                // géneros (como está)
    fieldName.charAt(0).toUpperCase() + fieldName.slice(1), // Géneros
  ];
  
  // Devolver el primer campo que existe
  for (const name of possibleNames) {
    if (anime[name] !== undefined) {
      return anime[name];
    }
  }
  
  return null; // Campo no encontrado
};

/**
 * Función para generar URL de AnimeFlv basado en el nombre del anime
 * Formatea el nombre para una URL: minúsculas, reemplazo de espacios con guiones, etc.
 */
const getAnimeFlvUrl = (animeName) => {
  if (!animeName) return "#";
  // Formatear el nombre para una URL: minúsculas, reemplazar espacios con guiones, eliminar caracteres especiales
  const formattedName = animeName
    .toLowerCase()
    .replace(/[^\w\s-]/g, '') // Eliminar caracteres especiales
    .replace(/\s+/g, '-')     // Reemplazar espacios con guiones
    .trim();
  return `https://www3.animeflv.net/browse?q=${encodeURIComponent(animeName)}`;
};

/**
 * Componente AnimeCard
 * Muestra una tarjeta con la información detallada de un anime, incluyendo imagen, título,
 * sinopsis, puntuación, ranking, tipo, año, estado, episodios y géneros.
 */
const AnimeCard = ({ anime, index, onClick, debugMode }) => {
  return (
    <li
      onClick={() => onClick(index)}
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
      {/* Botón de depuración - visible solo en modo debug */}
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
        alt={getAnimeField(anime, "english_title")} 
        style={{ width: 120, borderRadius: 8 }} 
        onError={(e) => { e.target.src = '/images/defaultImagePortrait.jpg'; }}
      />
      
      <div style={{ textAlign: 'left', width: '100%' }}>
        <h2 style={{ margin: '0 0 0.5rem 0' }}>{getAnimeField(anime, "romaji_title")}</h2>
        
        {/* Sinopsis */}
        <div style={{ 
          margin: '0 0 0.5rem 0', 
          color: '#bbb',
          maxHeight: '180px',
          overflowY: 'auto',
          paddingRight: '10px',
          lineHeight: '1.5',
          fontSize: '0.95rem'
        }}>
          {/* Procesamiento y renderizado de la sinopsis */}
          {(() => {
            const synopsis = getAnimeField(anime, "description");
            if (!synopsis) return <p>No hay sinopsis disponible.</p>;
            
            // Dividir el texto en oraciones y agruparlas para mejor presentación
            return synopsis.split('. ')
              .filter(sentence => sentence.trim().length > 0)
              .reduce((result, sentence, index, array) => {
                // Agrupar oraciones en pares (cada 2 oraciones)
                if (index % 2 === 0) {
                  // Si es un índice par y hay una siguiente oración, combinarlas
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
        
        {/* Información básica: puntuación y ranking */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ fontWeight: 'bold', color: '#ffd700' }}>
            Score: {getAnimeField(anime, "average_score") || "N/A"}
          </div>
          <div style={{ color: '#61dafb' }}>
            Popularity #{getAnimeField(anime, "popularity") || "?"}
          </div>
          {getAnimeField(anime, "similarity") && (
            <div style={{ fontWeight: 'bold', color: '#7df740' }}>
              Similarity: {(getAnimeField(anime, "similarity") * 100).toFixed(1)}%
            </div>
          )}
        </div>
        
        {/* Línea separadora */}
        <div style={{ height: '1px', background: '#444', margin: '8px 0' }}></div>
        
        {/* Información adicional: tipo, año, estado, episodios */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
          {/* Tipo (TV/OVA/Película) */}
          <div style={{ 
            background: '#333', 
            color: '#fff',
            padding: '4px 8px', 
            borderRadius: '4px',
            fontSize: '0.9rem',
            border: '1px solid #555'
          }}>
            {getAnimeField(anime, "format") || 'TV'}
          </div>
          
          {/* Año de lanzamiento */}
          {(() => {
            const aired = getAnimeField(anime, "season_year");
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
          
          {/* Estado (En emisión/Finalizado) */}
          {/* {(() => {
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
          })()} */}
          
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
            
            // Mostrar cada género como una etiqueta coloreada
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
  );
};

export { AnimeCard, getAnimeField, getAnimeFlvUrl };
