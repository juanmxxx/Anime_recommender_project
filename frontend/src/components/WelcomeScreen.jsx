import React from 'react';
import { quickSuggestions } from '../config/uiConfig';

/**
 * NOTA: Este componente ha sido consolidado en UIElements.jsx
 * para mantener una estructura más organizada del proyecto.
 * 
 * Por favor use el componente WelcomeScreen desde UIElements.jsx en su lugar.
 * 
 * Componente de pantalla de bienvenida
 * 
 * Muestra un mensaje de bienvenida y sugerencias de búsqueda rápidas
 * cuando no hay resultados que mostrar
 * 
 * @param {Object} props - Propiedades del componente
 * @param {Function} props.onSuggestionClick - Función a llamar cuando se hace clic en una sugerencia
 * @deprecated Use el componente desde UIElements.jsx
 */
const WelcomeScreen = ({ onSuggestionClick }) => {
  return (
    <div className="welcome-container">
      <h2 className="welcome-title">¡Welcome to S.A.R dear friend!</h2>
      <p className="welcome-text">
        Enter keywords related to the type of anime you would like to watch.
        <br />For example, try genres such as "romance comedy", "action adventure", 
        or even specific themes such as "cyberpunk dystopia"..
      </p>
      
      {/* Botones de sugerencias rápidas */}
      <div className="suggestions-container">
        {quickSuggestions.map(suggestion => (
          <button 
            key={suggestion}
            onClick={() => onSuggestionClick(suggestion)}
            className="suggestion-button"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeScreen;
