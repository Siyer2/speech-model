import { useState, useRef, useCallback, useEffect } from 'react'
import { useModelSession } from './useModelSession'
import { preprocessAudioBlob } from './audioPreprocessing'
import { runInference } from './inference'
import { normalizeForCer, detectSpeechErrors } from './speechRules'
import './App.css'

const NUM_BARS = 30

const ASSESSMENT_WORDS = [
  { word: 'cup', emoji: '\u{1F964}', ipa: '/kʌp/' },
  { word: 'duck', emoji: '\u{1F986}', ipa: '/dʌk/' },
  { word: 'green', emoji: '\u{1F7E2}', ipa: '/ɡɹin/' },
  { word: 'shovel', emoji: '\u26CF\uFE0F', ipa: '/ʃʌvɫ/' },
  { word: 'fish', emoji: '\u{1F41F}', ipa: '/fɪʃ/' },
  { word: 'soap', emoji: '\u{1F9FC}', ipa: '/sop/' },
  { word: 'zebra', emoji: '\u{1F993}', ipa: '/zibɹə/' },
  { word: 'red', emoji: '\u{1F534}', ipa: '/ɹɛd/' },
  { word: 'leaf', emoji: '\u{1F343}', ipa: '/lif/' },
  { word: 'spoon', emoji: '\u{1F944}', ipa: '/spun/' },
  { word: 'plate', emoji: '\u{1F37D}\uFE0F', ipa: '/plet/' },
  { word: 'chair', emoji: '\u{1FA91}', ipa: '/tʃɛɚ/' },
  { word: 'juice', emoji: '\u{1F9C3}', ipa: '/dʒus/' },
  { word: 'yellow', emoji: '\u{1F7E1}', ipa: '/jɛlo/' },
  { word: 'drum', emoji: '\u{1F941}', ipa: '/dɹʌm/' },
]

function App() {
  const { session, loading: modelLoading, progress: modelProgress, error: modelError } = useModelSession()
  const [isRecording, setIsRecording] = useState(false)
  const [hasRecorded, setHasRecorded] = useState(false)
  const [currentWordIndex, setCurrentWordIndex] = useState(0)
  const [isInferring, setIsInferring] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [barHeights, setBarHeights] = useState<number[]>(
    new Array(NUM_BARS).fill(0),
  )

  const currentWord = ASSESSMENT_WORDS[currentWordIndex]

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const animationFrameRef = useRef<number>(0)
  const streamRef = useRef<MediaStream | null>(null)
  const audioBlobRef = useRef<Blob | null>(null)
  const inferringRef = useRef(false)

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
        audioBlobRef.current = new Blob(chunksRef.current, { type: 'audio/webm' })
      }

      mediaRecorder.start()
      setIsRecording(true)
      setHasRecorded(false)
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
    setHasRecorded(true)
    setBarHeights(new Array(NUM_BARS).fill(0))
  }, [])

  const nextWord = useCallback(() => {
    const blob = audioBlobRef.current
    if (blob && session && !inferringRef.current) {
      const word = currentWord.word
      const targetIpa = currentWord.ipa.replace(/\//g, '')
      audioBlobRef.current = null
      inferringRef.current = true
      setIsInferring(true)

      console.log(`Inference triggered for: ${word}`)
      preprocessAudioBlob(blob)
        .then(({ pcmFloat32 }) => runInference(session, pcmFloat32))
        .then((predicted) => {
          const normTarget = normalizeForCer(targetIpa)
          const normPredicted = normalizeForCer(predicted)
          console.log(`Target: ${normTarget} | Predicted: ${normPredicted}`)

          const errors = detectSpeechErrors(word, targetIpa, predicted)
          for (const err of errors) {
            console.log(`suspected ${err.pattern} (${err.detail})`)
          }
        })
        .catch((err) => {
          console.error(`Inference failed for ${word}:`, err)
        })
        .finally(() => {
          inferringRef.current = false
          setIsInferring(false)
        })
    }

    if (currentWordIndex < ASSESSMENT_WORDS.length - 1) {
      setCurrentWordIndex((prev) => prev + 1)
    }
    setHasRecorded(false)
  }, [session, currentWord])

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationFrameRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      audioContextRef.current?.close()
    }
  }, [])

  if (modelLoading) {
    return (
      <div className="loading-screen">
        <p className="loading-text">Downloading model...</p>
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${modelProgress}%` }}
          />
        </div>
        <p className="loading-progress">{modelProgress}%</p>
      </div>
    )
  }

  if (modelError) {
    return (
      <div className="loading-screen">
        <p className="loading-error">Failed to load model: {modelError}</p>
      </div>
    )
  }

  return (
    <div className="app">
      <div className="word-image">
        <span className="emoji" role="img" aria-label={currentWord.word}>{currentWord.emoji}</span>
      </div>

      <h1 className="word-label">{currentWord.word}</h1>

      <div className="mic-section">
        <div className="button-row">
          <div className="mic-column">
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
              {isRecording
                ? 'Listening...'
                : hasRecorded
                  ? 'Tap to redo'
                  : 'Tap to speak'}
            </p>
            {error && <p className="mic-error">{error}</p>}
          </div>

          {hasRecorded && !isRecording && (
            <button
              className="next-word-button"
              onClick={nextWord}
              disabled={isInferring}
            >
              {isInferring ? (
                <span className="spinner" />
              ) : currentWordIndex >= ASSESSMENT_WORDS.length - 1 ? (
                'Finish'
              ) : (
                <>
                  Next word
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M5 12h14" />
                    <path d="m12 5 7 7-7 7" />
                  </svg>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
