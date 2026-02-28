const TARGET_SAMPLE_RATE = 16000

/**
 * Decode a recorded audio Blob (e.g. WebM from MediaRecorder) into a
 * mono Float32Array at 16 kHz — the format the speech model expects.
 */
export async function preprocessAudioBlob(
  blob: Blob,
): Promise<{ wavBytes: ArrayBuffer; pcmFloat32: Float32Array }> {
  const arrayBuffer = await blob.arrayBuffer()

  const offlineCtx = new OfflineAudioContext(1, 1, TARGET_SAMPLE_RATE)
  const decoded = await offlineCtx.decodeAudioData(arrayBuffer)

  const mono = mixToMono(decoded)
  const resampled = await resample(mono, decoded.sampleRate, TARGET_SAMPLE_RATE)

  const wavBytes = encodeWav(resampled, TARGET_SAMPLE_RATE)
  return { wavBytes, pcmFloat32: resampled }
}

/**
 * Down-mix an AudioBuffer to a single-channel Float32Array by averaging
 * all channels.
 */
function mixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0)
  }

  const length = buffer.length
  const mono = new Float32Array(length)
  const numChannels = buffer.numberOfChannels

  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = buffer.getChannelData(ch)
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i]
    }
  }

  for (let i = 0; i < length; i++) {
    mono[i] /= numChannels
  }

  return mono
}

/**
 * Resample a mono Float32Array from `srcRate` to `dstRate` using
 * OfflineAudioContext (lets the browser do the heavy lifting).
 */
async function resample(
  samples: Float32Array,
  srcRate: number,
  dstRate: number,
): Promise<Float32Array> {
  if (srcRate === dstRate) {
    return samples
  }

  const dstLength = Math.round((samples.length * dstRate) / srcRate)
  const offlineCtx = new OfflineAudioContext(1, dstLength, dstRate)
  const srcBuffer = offlineCtx.createBuffer(1, samples.length, srcRate)
  srcBuffer.getChannelData(0).set(samples)

  const source = offlineCtx.createBufferSource()
  source.buffer = srcBuffer
  source.connect(offlineCtx.destination)
  source.start()

  const rendered = await offlineCtx.startRendering()
  return rendered.getChannelData(0)
}

/**
 * Encode a mono Float32Array as a 16-bit PCM WAV file (ArrayBuffer).
 * The model loads audio via soundfile which reads WAV natively.
 */
function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numSamples = samples.length
  const bytesPerSample = 2 // 16-bit
  const dataSize = numSamples * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeString(view, 8, 'WAVE')

  // fmt chunk
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * bytesPerSample, true) // byte rate
  view.setUint16(32, bytesPerSample, true) // block align
  view.setUint16(34, 16, true) // bits per sample

  // data chunk
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  // Convert float32 [-1, 1] to int16
  let offset = 44
  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
    view.setInt16(offset, int16, true)
    offset += 2
  }

  return buffer
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}
