import React from 'react';

/**
 * NOTA: Este componente ha sido consolidado en Controls.jsx
 * para mantener una estructura más organizada del proyecto.
 * 
 * Por favor use el componente DebugControls desde Controls.jsx en su lugar.
 * 
 * Componente para los controles de depuración y métricas
 * 
 * Muestra los controles para activar el modo de depuración y 
 * el panel de métricas en entornos de desarrollo
 * 
 * @param {Object} props - Propiedades del componente
 * @param {boolean} props.debugMode - Estado actual del modo de depuración
 * @param {Function} props.setDebugMode - Función para cambiar el modo de depuración
 * @param {boolean} props.showMetrics - Estado actual de visibilidad de métricas
 * @param {Function} props.onMetricsToggle - Función para cambiar la visibilidad de métricas
 * @deprecated Use el componente desde Controls.jsx
 */
const DebugControls = ({ debugMode, setDebugMode, showMetrics, onMetricsToggle }) => {
  // Solo mostrar en entorno de desarrollo
  if (process.env.NODE_ENV === 'production') {
    return null;
  }

  return (
    <div className="debug-panel">
      {/* Toggle para modo depuración */}
      <label className="debug-label">
        <input 
          type="checkbox" 
          checked={debugMode}
          onChange={() => setDebugMode(!debugMode)}
        />
        Debug Mode
      </label>
      
      {/* Toggle para panel de métricas */}
      <label className="debug-label">
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

export default DebugControls;
