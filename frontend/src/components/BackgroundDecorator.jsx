import React from 'react';
import { backgroundGirls, centralImage } from '../config/uiConfig';

/**
 * NOTA: Este componente ha sido consolidado en UIElements.jsx
 * para mantener una estructura más organizada del proyecto.
 * 
 * Por favor use el componente BackgroundDecorator desde UIElements.jsx en su lugar.
 * 
 * Componente para el fondo decorativo con imágenes de anime
 * 
 * Renderiza las imágenes de fondo que dan ambientación a la aplicación,
 * incluyendo tanto imágenes laterales como una imagen central superior.
 * 
 * @deprecated Use el componente desde UIElements.jsx
 */
const BackgroundDecorator = () => {
  return (
    <div className="background-container">
      {/* Imágenes laterales */}
      {backgroundGirls.map((url, idx) => (
        <img
          key={idx}
          src={url}
          alt="chica anime decorativa"
          className="side-image"
          style={{
            left: `${10 + idx * 40}%`,
            bottom: idx % 2 === 0 ? '0' : '10%',
          }}
        />
      ))}
      
      {/* Imagen central superior */}
      <img
        src={centralImage}
        alt="chica anime central"
        className="central-image"
      />
    </div>
  );
};

export default BackgroundDecorator;
