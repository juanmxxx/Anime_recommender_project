import React from 'react';
import { topOptions } from '../config/uiConfig';

/**
 * Componente para la barra de búsqueda y controles relacionados
 * Incluye campo de texto, botón de búsqueda, selector de cantidad y botón de nueva búsqueda
 */
export const SearchBar = ({ 
  prompt, 
  setPrompt, 
  isLoading, 
  onSearch, 
  top, 
  setTop, 
  hasResults, 
  onClear 
}) => {
  return (
    <div>
      {/* Área de búsqueda */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'flex-start', 
        gap: '1rem', 
        marginBottom: '1rem' 
      }}>
        {/* Campo de texto para la búsqueda */}
        <textarea
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
        />
        
        {/* Botón de búsqueda */}
        <button
          onClick={() => onSearch(prompt)}
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
        >
          {isLoading ? (
            <>
              <div className="loading-spinner"></div>
              Searching...
            </>
          ) : 'Recommend'}
        </button>
      </div>
      
      {/* Selector de cantidad y botón de nueva búsqueda */}
      <div style={{ 
        marginBottom: '2rem', 
        textAlign: 'left', 
        display: 'flex', 
        alignItems: 'center' 
      }}>
        <label htmlFor="top-select" style={{ 
          fontWeight: 'bold', 
          marginRight: '0.5rem' 
        }}>Show:</label>
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
        >          {topOptions.map(option => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
        
        {/* Botón de nueva búsqueda (solo visible cuando hay resultados) */}
        {hasResults && (
          <button
            onClick={onClear}
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
      </div>
    </div>
  );
};

/**
 * Componente para controles de depuración y métricas
 * Solo visible en entorno de desarrollo
 */
export const DebugControls = ({ debugMode, setDebugMode, showMetrics, onMetricsToggle }) => {
  if (process.env.NODE_ENV === 'production') return null;
  
  return (
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
      {/* Toggle para modo depuración */}
      <label style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '5px', 
        cursor: 'pointer' 
      }}>
        <input 
          type="checkbox" 
          checked={debugMode}
          onChange={() => setDebugMode(!debugMode)}
        />
        Debug Mode
      </label>
      
      {/* Toggle para panel de métricas */}
      <label style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '5px', 
        cursor: 'pointer' 
      }}>
        <input 
          type="checkbox" 
          checked={showMetrics}
          onChange={onMetricsToggle}
        />
        Show Metrics
      </label>
    </div>
  );
};
