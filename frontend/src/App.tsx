import { useState, useRef, useCallback, useEffect } from 'react'
import { useModelSession } from './useModelSession'
import { preprocessAudioBlob } from './audioPreprocessing'
import { runInference } from './inference'
import { normalizeForCer, detectSpeechErrors } from './speechRules'
import {
  buildConfirmationTasks,
  formatPatternName,
  type WordResult,
  type ConfirmationTask,
  type ConfirmationResult,
} from './confirmationLogic'
import './App.css'
import TARGET_WORDS from './target_words.json'

const NUM_BARS = 30

const WORD_DETAILS: Record<string, { emoji: string; ipa: string }> = {
  cup: { emoji: '\u{1F964}', ipa: '/kʌp/' },
  duck: { emoji: '\u{1F986}', ipa: '/dʌk/' },
  green: { emoji: '\u{1F7E2}', ipa: '/ɡɹin/' },
  shovel: { emoji: '\u26CF\uFE0F', ipa: '/ʃʌvɫ/' },
  fish: { emoji: '\u{1F41F}', ipa: '/fɪʃ/' },
  soap: { emoji: '\u{1F9FC}', ipa: '/sop/' },
  zebra: { emoji: '\u{1F993}', ipa: '/zibɹə/' },
  red: { emoji: '\u{1F534}', ipa: '/ɹɛd/' },
  leaf: { emoji: '\u{1F343}', ipa: '/lif/' },
  spoon: { emoji: '\u{1F944}', ipa: '/spun/' },
  plate: { emoji: '\u{1F37D}\uFE0F', ipa: '/plet/' },
  chair: { emoji: '\u{1FA91}', ipa: '/tʃɛɚ/' },
  juice: { emoji: '\u{1F9C3}', ipa: '/dʒus/' },
  yellow: { emoji: '\u{1F7E1}', ipa: '/jɛlo/' },
  drum: { emoji: '\u{1F941}', ipa: '/dɹʌm/' },
}

const ASSESSMENT_WORDS = TARGET_WORDS.map((word) => ({
  word,
  ...WORD_DETAILS[word],
}))

type AppPhase = 'assessment' | 'confirmation' | 'results'

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

  const [phase, setPhase] = useState<AppPhase>('assessment')
  const [assessmentResults, setAssessmentResults] = useState<WordResult[]>([])
  const [confirmationTasks, setConfirmationTasks] = useState<ConfirmationTask[]>([])
  const [confirmationIndex, setConfirmationIndex] = useState(0)
  const [confirmationResults, setConfirmationResults] = useState<ConfirmationResult[]>([])
  const [previousPrediction, setPreviousPrediction] = useState<string | null>(null)

  const activeWord =
    phase === 'confirmation'
      ? ASSESSMENT_WORDS[confirmationTasks[confirmationIndex].confirmWordIndex]
      : ASSESSMENT_WORDS[currentWordIndex]

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const animationFrameRef = useRef<number>(0)
  const streamRef = useRef<MediaStream | null>(null)
  const audioBlobRef = useRef<Blob | null>(null)
  const inferringRef = useRef(false)

  // Transition from assessment to confirmation (or results) when all words are done
  useEffect(() => {
    if (phase !== 'assessment' || assessmentResults.length !== ASSESSMENT_WORDS.length) return

    const tasks = buildConfirmationTasks(assessmentResults, ASSESSMENT_WORDS)
    if (tasks.length === 0) {
      setPhase('results')
    } else {
      setConfirmationTasks(tasks)
      setConfirmationIndex(0)
      setHasRecorded(false)
      setPhase('confirmation')
    }
  }, [assessmentResults, phase])

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

  const handleNextAssessment = useCallback(async () => {
    const blob = audioBlobRef.current
    if (!blob || !session || inferringRef.current) return

    const word = activeWord.word
    const targetIpa = activeWord.ipa.replace(/\//g, '')
    audioBlobRef.current = null
    inferringRef.current = true
    setIsInferring(true)

    try {
      const { pcmFloat32 } = await preprocessAudioBlob(blob)
      const predicted = await runInference(session, pcmFloat32)
      const normTarget = normalizeForCer(targetIpa)
      const normPredicted = normalizeForCer(predicted)
      console.log(`Target: ${normTarget} | Predicted: ${normPredicted}`)

      const errors = detectSpeechErrors(word, targetIpa, predicted)
      for (const err of errors) {
        console.log(`suspected ${err.pattern} (${err.detail})`)
      }

      setAssessmentResults((prev) => [...prev, { word, errors }])
      setPreviousPrediction(predicted)
    } catch (err) {
      console.error(`Inference failed for ${word}:`, err)
      setAssessmentResults((prev) => [...prev, { word, errors: [] }])
    } finally {
      inferringRef.current = false
      setIsInferring(false)
    }

    // Advance to next word (phase transition handled by useEffect)
    if (currentWordIndex < ASSESSMENT_WORDS.length - 1) {
      setCurrentWordIndex((prev) => prev + 1)
    }
    setHasRecorded(false)
  }, [session, activeWord, currentWordIndex])

  const handleNextConfirmation = useCallback(async () => {
    const blob = audioBlobRef.current
    if (!blob || !session || inferringRef.current) return

    const task = confirmationTasks[confirmationIndex]
    const wordData = ASSESSMENT_WORDS[task.confirmWordIndex]
    const targetIpa = wordData.ipa.replace(/\//g, '')
    audioBlobRef.current = null
    inferringRef.current = true
    setIsInferring(true)

    try {
      const { pcmFloat32 } = await preprocessAudioBlob(blob)
      const predicted = await runInference(session, pcmFloat32)
      const errors = detectSpeechErrors(wordData.word, targetIpa, predicted)
      const confirmed = errors.some((e) => e.pattern === task.pattern)

      setPreviousPrediction(predicted)
      setConfirmationResults((prev) => [
        ...prev,
        {
          pattern: task.pattern,
          detail: task.originalDetail,
          originalWord: task.originalWord,
          confirmWord: task.confirmWord,
          confirmed,
        },
      ])
    } catch (err) {
      console.error(`Confirmation inference failed:`, err)
    } finally {
      inferringRef.current = false
      setIsInferring(false)
    }

    if (confirmationIndex < confirmationTasks.length - 1) {
      setConfirmationIndex((prev) => prev + 1)
    } else {
      setPhase('results')
    }
    setHasRecorded(false)
  }, [session, confirmationTasks, confirmationIndex])

  const handleNext = phase === 'confirmation' ? handleNextConfirmation : handleNextAssessment

  const handleRestart = useCallback(() => {
    setPhase('assessment')
    setCurrentWordIndex(0)
    setHasRecorded(false)
    setAssessmentResults([])
    setConfirmationTasks([])
    setConfirmationIndex(0)
    setConfirmationResults([])
    setPreviousPrediction(null)
    setError(null)
  }, [])

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
        <p className="loading-tip">Tip: This model is large and may not work on a phone. Use a laptop.</p>
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

  if (phase === 'results') {
    const confirmed = confirmationResults.filter((r) => r.confirmed)

    return (
      <div className="app results-screen">
        {confirmed.length === 0 ? (
          <div className="results-clear">
            <h1 className="results-heading">You are all clear!</h1>
          </div>
        ) : (
          <div className="results-findings">
            <h1 className="results-heading">Assessment Results</h1>
            <p className="results-recommendation">
              Consider contacting a{' '}
              <a
                href="https://www.google.com/maps/search/speech+pathologist"
                target="_blank"
                rel="noopener noreferrer"
                className="results-link"
              >
                Speech Pathologist
              </a>{' '}
              for:
            </p>
            <ul className="results-list">
              {confirmed.map((r, i) => (
                <li key={i} className="results-item">
                  <p className="results-condition">
                    <strong>{formatPatternName(r.pattern)}</strong> as there
                    could be <strong>{r.detail}</strong> on{' '}
                    <strong>{r.originalWord}</strong>
                  </p>
                </li>
              ))}
            </ul>
          </div>
        )}
        <button className="restart-button" onClick={handleRestart}>
          Start Over
        </button>
      </div>
    )
  }

  const isLastStep =
    phase === 'assessment'
      ? currentWordIndex >= ASSESSMENT_WORDS.length - 1
      : confirmationIndex >= confirmationTasks.length - 1

  const buttonLabel =
    phase === 'confirmation'
      ? isLastStep
        ? 'See results'
        : 'Next'
      : isLastStep
        ? 'Finish'
        : 'Next word'

  return (
    <div className="app">
      {phase === 'assessment' && (
        <p className="phase-progress">
          Word {currentWordIndex + 1} of {ASSESSMENT_WORDS.length}
        </p>
      )}
      {phase === 'confirmation' && (
        <p className="phase-progress">
          Let's try again! ({confirmationIndex + 1} of {confirmationTasks.length})
        </p>
      )}

      <div className="word-image">
        <span className="emoji" role="img" aria-label={activeWord.word}>{activeWord.emoji}</span>
      </div>

      <h1 className="word-label">{activeWord.word}</h1>

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
              onClick={handleNext}
              disabled={isInferring}
            >
              {isInferring ? (
                <span className="spinner" />
              ) : isLastStep ? (
                buttonLabel
              ) : (
                <>
                  {buttonLabel}
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

      {previousPrediction !== null && (
        <div className="previous-prediction">
          <p className="previous-prediction-label">Previous prediction</p>
          <p className="previous-prediction-value">/{previousPrediction}/</p>
        </div>
      )}
    </div>
  )
}

export default App
