import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const NUM_BARS = 30

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [barHeights, setBarHeights] = useState<number[]>(
    new Array(NUM_BARS).fill(0),
  )

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const animationFrameRef = useRef<number>(0)
  const streamRef = useRef<MediaStream | null>(null)

  const updateVisualization = useCallback(() => {
    if (!analyserRef.current) return

    const analyser = analyserRef.current
    const dataArray = new Uint8Array(analyser.frequencyBinCount)
    analyser.getByteFrequencyData(dataArray)

    const step = Math.max(1, Math.floor(dataArray.length / NUM_BARS))
    const heights = Array.from({ length: NUM_BARS }, (_, i) => {
      return dataArray[i * step] / 255
    })

    setBarHeights(heights)
    animationFrameRef.current = requestAnimationFrame(updateVisualization)
  }, [])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
        // TODO: preprocess and send to model
        console.log('Recording complete:', audioBlob.size, 'bytes')
      }

      mediaRecorder.start()
      setIsRecording(true)
      animationFrameRef.current = requestAnimationFrame(updateVisualization)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      console.error('Failed to start recording:', message)
      setError(message)
    }
  }, [updateVisualization])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state !== 'inactive') {
      mediaRecorderRef.current?.stop()
    }

    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null

    audioContextRef.current?.close()
    audioContextRef.current = null
    analyserRef.current = null

    cancelAnimationFrame(animationFrameRef.current)

    setIsRecording(false)
    setBarHeights(new Array(NUM_BARS).fill(0))
  }, [])

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationFrameRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      audioContextRef.current?.close()
    }
  }, [])

  return (
    <div className="app">
      <div className="word-image">
        <span className="emoji" role="img" aria-label="cup">&#x1F964;</span>
      </div>

      <h1 className="word-label">cup</h1>

      <div className="mic-section">
        <div className="mic-container">
          {isRecording && (
            <div className="visualizer">
              {barHeights.map((height, i) => (
                <div
                  key={i}
                  className="visualizer-bar"
                  style={{
                    transform: `rotate(${(i * 360) / NUM_BARS}deg) translateY(-52px)`,
                    height: `${6 + height * 14}px`,
                    opacity: 0.4 + height * 0.6,
                  }}
                />
              ))}
            </div>
          )}
          <button
            className={`mic-button${isRecording ? ' mic-button--recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
          >
            {isRecording ? (
              <div className="stop-icon" />
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="40"
                height="40"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" x2="12" y1="19" y2="22" />
              </svg>
            )}
          </button>
        </div>
        <p className="mic-hint">
          {isRecording ? 'Listening...' : 'Tap to speak'}
        </p>
        {error && <p className="mic-error">{error}</p>}
      </div>
    </div>
  )
}

export default App
