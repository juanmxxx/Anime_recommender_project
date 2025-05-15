import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // Cambia este valor al puerto que desees
    open: true // Abre el navegador automáticamente
  }
})
