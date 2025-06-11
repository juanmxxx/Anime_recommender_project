/**
 * tracking.js - Utilidades para seguimiento de eventos y análisis de uso
 * Este archivo contiene funciones para gestionar sesiones de usuario y enviar
 * eventos de seguimiento al backend para análisis estadísticos.
 */

/**
 * Genera un ID de sesión único basado en timestamp y valor aleatorio
 * @returns {string} ID de sesión único
 */
const generateSessionId = () => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

/**
 * Obtiene el ID de sesión existente o crea uno nuevo si no existe
 * @returns {string} ID de sesión del usuario actual
 */
const getSessionId = () => {
  let sessionId = sessionStorage.getItem('sar_session_id');
  if (!sessionId) {
    sessionId = generateSessionId();
    sessionStorage.setItem('sar_session_id', sessionId);
  }
  return sessionId;
};

/**
 * Envía eventos de seguimiento al backend (búsquedas o clics)
 * @param {string} eventType - Tipo de evento ('search' o 'click')
 * @param {Object} data - Datos adicionales del evento
 */
const trackEvent = async (eventType, data = {}) => {
  try {
    // Importar dinámicamente las URLs de API desde config
    const { API_URLS } = await import('../config');
    const endpoint = eventType === 'search' ? API_URLS.logSearch : API_URLS.logClick;
    
    // Enviar datos al backend
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: getSessionId(),
        event_type: eventType,
        ...data
      })
    });
    
    if (!response.ok) {
      console.warn('Error enviando métrica:', response.status);
    }
  } catch (error) {
    console.warn('Error tracking event:', error);
  }
};

/**
 * Registra el tiempo de carga para análisis de rendimiento
 * @param {number} loadTimeMs - Tiempo de carga en milisegundos
 * @param {string} queryText - Texto de la consulta asociada
 */
const trackLoadTime = async (loadTimeMs, queryText) => {
  try {
    console.log('trackLoadTime llamado con:', loadTimeMs, 'ms,', queryText);
    const { API_URLS } = await import('../config');
    console.log('URL para logLoadTime:', API_URLS.logLoadTime);
    
    const sessionId = getSessionId();
    const requestBody = {
      session_id: sessionId,
      event_type: 'load_time',
      prompt_text: queryText,
      load_time_ms: loadTimeMs
    };
    console.log('Enviando datos:', JSON.stringify(requestBody));
    
    // Enviar datos de tiempo de carga al backend
    const response = await fetch(API_URLS.logLoadTime, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      console.warn('Error registrando tiempo de carga:', response.status);
      const errorText = await response.text();
      console.warn('Detalles del error:', errorText);
    } else {
      console.log(`Tiempo de carga registrado: ${loadTimeMs}ms (${(loadTimeMs/1000).toFixed(2)}s)`);
      const responseData = await response.json();
      console.log('Respuesta del servidor:', responseData);
    }
  } catch (error) {
    console.warn('Error tracking load time:', error);
  }
};

export { generateSessionId, getSessionId, trackEvent, trackLoadTime };
