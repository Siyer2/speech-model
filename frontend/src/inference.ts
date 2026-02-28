import * as ort from 'onnxruntime-web/webgpu'

/**
 * Index-to-phone mapping matching the Python Vocab (sorted phones, 0=blank, 1=UNK, 2+=phones).
 * Generated from: Vocab.from_phones() in model/src/speech_model/dataset.py
 */
const IDX_TO_PHONE: Record<number, string> = {
  2: ' ',
  3: '*',
  4: 'a',
  5: 'b',
  6: 'd',
  7: 'e',
  8: 'f',
  9: 'g',
  10: 'h',
  11: 'i',
  12: 'j',
  13: 'k',
  14: 'l',
  15: 'm',
  16: 'n',
  17: 'o',
  18: 'p',
  19: 'r',
  20: 's',
  21: 't',
  22: 'u',
  23: 'v',
  24: 'w',
  25: 'z',
  26: 'æ',
  27: 'ð',
  28: 'ŋ',
  29: 'ɑ',
  30: 'ɔ',
  31: 'ə',
  32: 'ɚ',
  33: 'ɛ',
  34: 'ɝ',
  35: 'ɡ',
  36: 'ɪ',
  37: 'ɫ',
  38: 'ɹ',
  39: 'ɾ',
  40: 'ʃ',
  41: 'ʊ',
  42: 'ʌ',
  43: 'ʒ',
  44: 'ʔ',
  45: 'ʤ',
  46: 'ʧ',
  47: 'ʰ',
  48: 'ˈ',
  49: 'ˌ',
  50: 'ː',
  51: '\u0329',
  52: '\u032F',
  53: '\u0361',
  54: 'θ',
}

const BLANK_IDX = 0
const UNK_IDX = 1

/**
 * Greedy CTC decode: argmax each timestep, collapse repeats, remove blanks.
 */
function greedyCtcDecode(logits: Float32Array, numFrames: number, vocabSize: number): string {
  let prev = -1
  const phones: string[] = []

  for (let t = 0; t < numFrames; t++) {
    const offset = t * vocabSize
    let maxIdx = 0
    let maxVal = logits[offset]
    for (let v = 1; v < vocabSize; v++) {
      if (logits[offset + v] > maxVal) {
        maxVal = logits[offset + v]
        maxIdx = v
      }
    }

    if (maxIdx !== prev) {
      if (maxIdx !== BLANK_IDX && maxIdx !== UNK_IDX) {
        const phone = IDX_TO_PHONE[maxIdx]
        if (phone !== undefined) {
          phones.push(phone)
        }
      }
      prev = maxIdx
    }
  }

  return phones.join('')
}

/**
 * Run ONNX inference on preprocessed PCM audio and return predicted phonemes.
 */
export async function runInference(
  session: ort.InferenceSession,
  pcmFloat32: Float32Array,
): Promise<string> {
  const length = pcmFloat32.length
  const inputTensor = new ort.Tensor('float32', pcmFloat32, [1, length])
  const attentionMask = new ort.Tensor('int64', new BigInt64Array(length).fill(1n), [1, length])
  const results = await session.run({ input_values: inputTensor, attention_mask: attentionMask })

  const outputKey = session.outputNames[0]
  const output = results[outputKey]
  const [, numFrames, vocabSize] = output.dims as [number, number, number]
  const logits = output.data as Float32Array

  return greedyCtcDecode(logits, numFrames, vocabSize)
}
