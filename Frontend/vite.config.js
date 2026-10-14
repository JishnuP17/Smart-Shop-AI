import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // Use the port allowed by the local FastAPI CORS configuration.
    port: 5173,
    strictPort: true,
  },
});
