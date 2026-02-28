import { useState, useEffect, useRef } from 'react'
import * as ort from 'onnxruntime-web/webgpu'

const MODEL_URL = 'https://github.com/Siyer2/speech-model/releases/latest/download/model.onnx'
const CACHE_NAME = 'onnx-model-cache'

async function fetchModelWithProgress(
  onProgress: (percent: number) => void,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE_NAME)
  const cached = await cache.match(MODEL_URL)

  if (cached) {
    onProgress(100)
    return cached.arrayBuffer()
  }

  const response = await fetch(MODEL_URL, { signal })
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status}`)
  }
  if (!response.body) {
    throw new Error('ReadableStream not supported')
  }

  const contentLength = response.headers.get('Content-Length')
  const total = contentLength ? parseInt(contentLength, 10) : 0
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    chunks.push(value)
    received += value.length

    if (total > 0) {
      onProgress(Math.round((received / total) * 100))
    }
  }

  const modelBuffer = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    modelBuffer.set(chunk, offset)
    offset += chunk.length
  }

  onProgress(100)

  // Store in cache for next time
  await cache.put(MODEL_URL, new Response(modelBuffer, {
    headers: { 'Content-Length': String(received) },
  }))

  return modelBuffer.buffer
}

export function useModelSession() {
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [session, setSession] = useState<ort.InferenceSession | null>(null)
  const sessionRef = useRef<ort.InferenceSession | null>(null)

  useEffect(() => {
    const abortController = new AbortController()

    async function loadModel() {
      try {
        const buffer = await fetchModelWithProgress(
          setProgress,
          abortController.signal,
        )

        if (abortController.signal.aborted) return

        const inferenceSession = await ort.InferenceSession.create(
          buffer,
          { executionProviders: ['webgpu', 'wasm'] },
        )

        if (abortController.signal.aborted) {
          inferenceSession.release()
          return
        }

        sessionRef.current = inferenceSession
        setSession(inferenceSession)
        setLoading(false)
      } catch (err) {
        if (!abortController.signal.aborted) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      }
    }

    loadModel()

    return () => {
      abortController.abort()
      sessionRef.current?.release()
      sessionRef.current = null
    }
  }, [])

  return { session, loading, progress, error }
}
