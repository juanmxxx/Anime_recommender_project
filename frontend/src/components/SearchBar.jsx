import React from 'react';
import { topOptions } from '../config/uiConfig';

/**
 * NOTA: Este componente ha sido consolidado en Controls.jsx
 * para mantener una estructura más organizada del proyecto.
 * 
 * Por favor use el componente SearchBar desde Controls.jsx en su lugar.
 * 
 * Componente para la barra de búsqueda y controles relacionados
 * 
 * Incluye el área de entrada de búsqueda, botón de búsqueda,
 * selector de cantidad de resultados y botón de nueva búsqueda
 * 
 * @param {Object} props - Propiedades del componente
 * @param {string} props.prompt - Texto actual de búsqueda
 * @param {Function} props.setPrompt - Función para actualizar el texto de búsqueda
 * @param {boolean} props.isLoading - Indica si hay una búsqueda en curso
 * @param {Function} props.onSearch - Función para realizar la búsqueda
 * @param {string} props.top - Valor actual del selector "top N"
 * @param {Function} props.setTop - Función para actualizar el valor del selector
 * @param {boolean} props.hasResults - Indica si hay resultados para mostrar el botón de limpieza
 * @param {Function} props.onClear - Función para limpiar los resultados
 * @deprecated Use el componente desde Controls.jsx
 */
const SearchBar = ({ 
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
    <>
      {/* Área de búsqueda */}
      <div className="search-area">
        {/* Campo de texto para la búsqueda */}
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Write your prompt or anime search keywords here..."
          className="search-textarea"
        />
        
        {/* Botón de búsqueda */}
        <button
          onClick={() => onSearch(prompt)}
          disabled={isLoading}
          className="search-button"
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
      <div className="filter-controls">
        <label htmlFor="top-select" style={{ fontWeight: 'bold', marginRight: '0.5rem' }}>Show:</label>
        <select
          id="top-select"
          value={top}
          onChange={e => setTop(e.target.value)}
          className="top-select"
        >
          {topOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        
        {/* Botón de nueva búsqueda (solo visible cuando hay resultados) */}
        {hasResults && (
          <button
            onClick={onClear}
            className="clear-button"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
            New Search
          </button>
        )}
      </div>
    </>
  );
};

export default SearchBar;
