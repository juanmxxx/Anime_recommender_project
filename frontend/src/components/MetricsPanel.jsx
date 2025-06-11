import React from 'react';

/**
 * Componente MetricsPanel
 * Muestra un panel con métricas de uso del sistema
 * Incluye estadísticas como búsquedas, clics, tasa de conversión y tiempo de carga
 * 
 * Props:
 * - metrics: Objeto con los datos de métricas a mostrar
 * - onClose: Función para cerrar el panel
 * - onRefresh: Función para actualizar las métricas
 */
const MetricsPanel = ({ metrics, onClose, onRefresh }) => {
  return (
    <div style={{
      position: 'fixed',
      top: 20,
      right: 20,
      background: 'rgba(0,0,0,0.9)',
      padding: '15px',
      borderRadius: '8px',
      color: '#fff',
      fontSize: '0.9rem',
      minWidth: '250px',
      zIndex: 50
    }}>
      {/* Cabecera del panel con título y botón de cierre */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
        <h3 style={{ margin: 0, color: '#61dafb' }}>Métricas (7 días)</h3>
        <button 
          onClick={onClose}
          style={{ 
            background: 'none', 
            border: 'none', 
            color: '#999', 
            cursor: 'pointer',
            fontSize: '1.2rem'
          }}
        >
          ✕
        </button>
      </div>
      
      {/* Contenido de métricas */}
      {metrics.metrics && metrics.metrics.length > 0 ? (
        <div>
          {/* Listado de métricas diarias (últimos 5 días) */}
          {metrics.metrics.slice(0, 5).map((day, idx) => (
            <div key={idx} style={{ 
              marginBottom: '8px', 
              padding: '8px', 
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '4px'
            }}>
              <div style={{ fontWeight: 'bold' }}>{day.date}</div>
              <div>Búsquedas: {day.searches}</div>
              <div>Clics: {day.clicks}</div>
              {/* Tasa de conversión con color condicional */}
              <div style={{ color: day.conversion_rate > 0 ? '#7df740' : '#ffd700' }}>
                Conversión: {day.conversion_rate}%
              </div>
              {/* Tiempo de carga con color condicional según velocidad */}
              <div style={{ 
                color: day.avg_load_time_sec < 1 ? '#7df740' : day.avg_load_time_sec < 3 ? '#ffd700' : '#ff6b6b',
                borderTop: '1px solid rgba(255,255,255,0.1)',
                marginTop: '5px',
                paddingTop: '5px'
              }}>
                Tiempo de carga: {day.avg_load_time_sec || 'N/A'} s
              </div>
            </div>
          ))}
          
          {/* Botón para actualizar las métricas */}
          <button 
            onClick={onRefresh}
            style={{
              background: '#61dafb',
              color: '#000',
              border: 'none',
              padding: '5px 10px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.8rem',
              marginTop: '10px'
            }}
          >
            Actualizar
          </button>
        </div>
      ) : (
        <div>No hay datos disponibles</div>
      )}
    </div>
  );
};

export default MetricsPanel;
