import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/speech-model/',
  plugins: [react()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
