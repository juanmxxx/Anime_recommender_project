import React from 'react';
import { getAnimeField } from './AnimeCard';

/**
 * Componente AnimeModal
 * Muestra un modal de confirmación cuando el usuario selecciona un anime
 * Permite al usuario confirmar si quiere ver el anime seleccionado
 * 
 * Props:
 * - anime: Objeto con los datos del anime seleccionado
 * - onClose: Función para cerrar el modal
 * - onConfirm: Función para confirmar la selección del anime
 * - debugMode: Booleano para mostrar información de depuración
 */
const AnimeModal = ({ anime, onClose, onConfirm, debugMode }) => {
  return (
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
        {/* Botón de depuración - solo visible en modo debug */}
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
              console.log("Datos del anime modal:", anime);
              alert("Datos mostrados en consola");
            }}
          >
            Debug
          </div>
        )}
        
        {/* Imagen del anime */}
        <img 
          src={getAnimeField(anime, "image_url")} 
          alt={getAnimeField(anime, "name")} 
          style={{ width: 90, borderRadius: 12, marginBottom: 16, boxShadow: '0 2px 8px #0003' }} 
          onError={(e) => { e.target.src = '/images/defaultImagePortrait.jpg'; }}
        />
        {/* Título del anime */}
        <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.3rem', color: '#61dafb' }}>
          {getAnimeField(anime, "name")}
        </h2>
        
        {/* Año de emisión (si está disponible) */}
        {(() => {
          const aired = getAnimeField(anime, "aired");
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
        
        {/* Pregunta al usuario */}
        <p style={{ color: '#bbb', marginBottom: 24 }}>¿Quieres ver este anime?</p>
        
        {/* Botones de acción */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem' }}>
          {/* Botón de confirmación */}
          <button onClick={onConfirm} style={{
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
          {/* Botón para cancelar */}
          <button onClick={onClose} style={{
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
          onClick={onClose}
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
  );
};

export default AnimeModal;
