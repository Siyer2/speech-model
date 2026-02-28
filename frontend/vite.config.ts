import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

// https://vite.dev/config/
export default defineConfig({
  base: '/speech-model/',
  plugins: [
    react(),
    {
      name: 'serve-model-files',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (!req.url?.startsWith('/models/')) return next()

          const fileName = path.basename(req.url)
          const filePath = path.resolve(
            __dirname,
            '..',
            'model',
            'checkpoints',
            fileName,
          )

          if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            return next()
          }

          const stat = fs.statSync(filePath)
          res.setHeader('Content-Type', 'application/octet-stream')
          res.setHeader('Content-Length', stat.size.toString())
          fs.createReadStream(filePath).pipe(res)
        })
      },
    },
  ],
  server: {
    allowedHosts: ['batholitic-unproclaimed-viki.ngrok-free.dev'],
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
