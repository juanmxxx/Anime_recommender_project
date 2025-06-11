import React from 'react';
import { loadingImageUrl } from '../config/uiConfig';

/**
 * NOTA: Este componente ha sido consolidado en UIElements.jsx
 * para mantener una estructura más organizada del proyecto.
 * 
 * Por favor use el componente LoadingIndicator desde UIElements.jsx en su lugar.
 * 
 * Componente de indicador de carga
 * 
 * Muestra una animación de carga mientras se procesan las búsquedas
 * 
 * @deprecated Use el componente desde UIElements.jsx
 */
const LoadingIndicator = () => {
  return (
    <div className="loading-container">
      <img 
        src={loadingImageUrl} 
        alt="Korone loading animation" 
        className="loading-image" 
      />
      <p className="loading-text">
        Finding the best anime for you...
      </p>
    </div>
  );
};

export default LoadingIndicator;
