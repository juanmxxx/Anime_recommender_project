// Archivo de configuración para URLs de API
// Cambiar la URL base según sea necesario para desarrollo, producción o contenedores Docker
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://api:8000'  // URL dentro de Docker si también el frontend está en Docker
  : 'http://localhost:8000'; // URL para desarrollo local

export const API_URLS = {
  recommend: (keywords, topN) => 
    `${API_BASE_URL}/recommend?keywords=${encodeURIComponent(keywords)}&top_n=${topN}`,
  health: `${API_BASE_URL}/health`
};

export default API_URLS;
