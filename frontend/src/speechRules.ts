/**
 * Post-processing normalization and speech error detection rules.
 *
 * normalizeForCer mirrors the Python normalize_for_cer from dataset.py.
 * detectSpeechErrors compares target vs predicted phonemes and returns
 * suspected error patterns per model/docs/assessment_words.md.
 */

// ---------------------------------------------------------------------------
// Normalization (mirrors Python normalize_for_cer)
// ---------------------------------------------------------------------------

/** Strip combining diacritics (NFD decompose then drop combining chars). */
function normalizePhonetic(text: string): string {
  return text.normalize('NFD').replace(/\p{M}/gu, '')
}

const CER_REPLACEMENTS: [string, string][] = [
  ['ʤ', 'dʒ'],
  ['ʧ', 'tʃ'],
  ['ʦ', 'ts'],
  ['ʨ', 'tɕ'],
  ['g', 'ɡ'], // ASCII g → IPA ɡ (U+0261)
  ['ɝ', 'ɚ'],
]

const STRIP_CHARS = new Set(['ˈ', 'ˌ', 'ː', 'ʰ', '*', ' '])

export function normalizeForCer(text: string): string {
  let out = normalizePhonetic(text)
  for (const [old, rep] of CER_REPLACEMENTS) {
    out = out.replaceAll(old, rep)
  }
  return [...out].filter((c) => !STRIP_CHARS.has(c)).join('')
}

// ---------------------------------------------------------------------------
// Assessment word targets (IPA without slashes, pre-normalized)
// ---------------------------------------------------------------------------

interface WordTarget {
  ipa: string // normalized target phones
  clusters: string[][] // consonant clusters in the word, e.g. [['ɡ','ɹ']]
}

/** Pre-normalized target IPA for every assessment word. */
const WORD_TARGETS: Record<string, WordTarget> = {
  cup: { ipa: 'kʌp', clusters: [] },
  duck: { ipa: 'dʌk', clusters: [] },
  green: { ipa: 'ɡɹin', clusters: [['ɡ', 'ɹ']] },
  shovel: { ipa: 'ʃʌvɫ', clusters: [] },
  fish: { ipa: 'fɪʃ', clusters: [] },
  soap: { ipa: 'sop', clusters: [] },
  zebra: { ipa: 'zibɹə', clusters: [['b', 'ɹ']] },
  red: { ipa: 'ɹɛd', clusters: [] },
  leaf: { ipa: 'lif', clusters: [] },
  spoon: { ipa: 'spun', clusters: [['s', 'p']] },
  plate: { ipa: 'plet', clusters: [['p', 'l']] },
  chair: { ipa: 'tʃɛɚ', clusters: [] },
  juice: { ipa: 'dʒus', clusters: [] },
  yellow: { ipa: 'jɛlo', clusters: [] },
  drum: { ipa: 'dɹʌm', clusters: [['d', 'ɹ']] },
}

// ---------------------------------------------------------------------------
// Phoneme-level helpers
// ---------------------------------------------------------------------------

/** Split a phoneme string into an array of logical phoneme units. */
function splitPhones(s: string): string[] {
  const digraphs = ['dʒ', 'tʃ']
  const phones: string[] = []
  let i = 0
  while (i < s.length) {
    let matched = false
    for (const dg of digraphs) {
      if (s.startsWith(dg, i)) {
        phones.push(dg)
        i += dg.length
        matched = true
        break
      }
    }
    if (!matched) {
      phones.push(s[i])
      i++
    }
  }
  return phones
}

// ---------------------------------------------------------------------------
// Detection rules
// ---------------------------------------------------------------------------

export interface SpeechError {
  pattern: string
  detail: string
}

/**
 * Compare target and predicted phones for a given assessment word.
 * Returns an array of suspected speech error patterns.
 */
export function detectSpeechErrors(
  word: string,
  rawTarget: string,
  rawPredicted: string,
): SpeechError[] {
  const target = normalizeForCer(rawTarget)
  const predicted = normalizeForCer(rawPredicted)

  if (target === predicted) return []

  const tPhones = splitPhones(target)
  const pPhones = splitPhones(predicted)
  const errors: SpeechError[] = []
  const seen = new Set<string>()

  const add = (pattern: string, detail: string) => {
    const key = `${pattern}:${detail}`
    if (!seen.has(key)) {
      seen.add(key)
      errors.push({ pattern, detail })
    }
  }

  const wt = WORD_TARGETS[word.toLowerCase()]

  // --- Initial consonant deletion ---
  if (tPhones.length > 0 && pPhones.length > 0) {
    // If the first target phoneme is entirely absent at the start
    if (pPhones.length < tPhones.length && pPhones[0] !== tPhones[0]) {
      // Check if rest aligns with target[1:]
      const restTarget = tPhones.slice(1).join('')
      if (predicted.startsWith(restTarget.slice(0, 2))) {
        add('initial_consonant_deletion', `${tPhones[0]} missing`)
      }
    }
  }

  // --- Final consonant deletion ---
  if (tPhones.length > 0 && pPhones.length < tPhones.length) {
    const lastTarget = tPhones[tPhones.length - 1]
    const lastPredicted = pPhones.length > 0 ? pPhones[pPhones.length - 1] : ''
    if (lastPredicted !== lastTarget) {
      // Check that vowels preceding the final consonant are still present
      const withoutFinal = tPhones.slice(0, -1).join('')
      if (predicted === withoutFinal || predicted.endsWith(withoutFinal.slice(-2))) {
        add('final_consonant_deletion', `final ${lastTarget} missing`)
      }
    }
  }

  // --- Fronting (velar/postalveolar → alveolar) ---
  const frontingMap: Record<string, string> = {
    k: 't',
    ɡ: 'd',
    ʃ: 's',
    ŋ: 'n',
  }
  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    const t = tPhones[i]
    const p = pPhones[i]
    if (frontingMap[t] && p === frontingMap[t]) {
      const pos = i === 0 ? 'onset' : i === tPhones.length - 1 ? 'final' : 'medial'
      add('fronting', `${pos} ${t}→${p}`)
    }
  }

  // --- Backing (alveolar → velar) ---
  const backingMap: Record<string, string> = { t: 'k', d: 'ɡ' }
  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    const t = tPhones[i]
    const p = pPhones[i]
    if (backingMap[t] && p === backingMap[t]) {
      const pos = i === 0 ? 'onset' : i === tPhones.length - 1 ? 'final' : 'medial'
      add('backing', `${pos} ${t}→${p}`)
    }
  }

  // --- Stopping of fricatives ---
  // f→p, s→t
  const stoppingFs: Record<string, string> = { f: 'p', s: 't' }
  // v→b, z→d
  const stoppingVz: Record<string, string> = { v: 'b', z: 'd' }
  // ʃ→t, tʃ→t, dʒ→d
  const stoppingShChJ: Record<string, string> = { ʃ: 't', tʃ: 't', dʒ: 'd' }

  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    const t = tPhones[i]
    const p = pPhones[i]
    const pos = i === 0 ? 'onset' : i === tPhones.length - 1 ? 'final' : 'medial'

    if (stoppingFs[t] && p === stoppingFs[t]) {
      add('stopping_f_s', `${pos} ${t}→${p}`)
    }
    if (stoppingVz[t] && p === stoppingVz[t]) {
      add('stopping_v_z', `${pos} ${t}→${p}`)
    }
    if (stoppingShChJ[t] && p === stoppingShChJ[t]) {
      add('stopping_sh_ch_j', `${pos} ${t}→${p}`)
    }
  }

  // --- Deaffrication (affricate → fricative) ---
  const deaffMap: Record<string, string> = { tʃ: 'ʃ', dʒ: 'ʒ' }
  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    const t = tPhones[i]
    const p = pPhones[i]
    if (deaffMap[t] && p === deaffMap[t]) {
      add('deaffrication', `${t}→${p}`)
    }
  }

  // --- Voicing (voiceless stop → voiced) ---
  const voicingMap: Record<string, string> = { p: 'b', t: 'd', k: 'ɡ' }
  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    const t = tPhones[i]
    const p = pPhones[i]
    if (voicingMap[t] && p === voicingMap[t]) {
      const pos = i === 0 ? 'onset' : i === tPhones.length - 1 ? 'final' : 'medial'
      add('voicing', `${pos} ${t}→${p}`)
    }
  }

  // --- Gliding /ɹ/ → /w/ ---
  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    if (tPhones[i] === 'ɹ' && pPhones[i] === 'w') {
      add('gliding_r', 'ɹ→w')
    }
  }

  // --- Gliding /l/ → /w/ or /j/ ---
  for (let i = 0; i < tPhones.length && i < pPhones.length; i++) {
    if (
      (tPhones[i] === 'l' || tPhones[i] === 'ɫ') &&
      (pPhones[i] === 'w' || pPhones[i] === 'j')
    ) {
      add('gliding_l', `${tPhones[i]}→${pPhones[i]}`)
    }
  }

  // --- Cluster reduction ---
  if (wt) {
    for (const cluster of wt.clusters) {
      const clusterStr = cluster.join('')
      const clusterIdx = target.indexOf(clusterStr)
      if (clusterIdx === -1) continue

      // Check if the cluster is reduced to just one of its members
      const predAtCluster = predicted.slice(clusterIdx)
      if (predAtCluster.length > 0) {
        const first = cluster[0]
        const second = cluster[1]
        // Cluster present as single phoneme (one member kept, other dropped)
        if (
          (predAtCluster.startsWith(first) && !predAtCluster.startsWith(clusterStr)) ||
          (predAtCluster.startsWith(second) && !predAtCluster.startsWith(clusterStr))
        ) {
          const kept = predAtCluster.startsWith(first) ? first : second
          add('cluster_reduction', `${clusterStr}→${kept}`)
        }
      }
    }
  }

  // --- Coalescence (sp→f) ---
  if (word.toLowerCase() === 'spoon') {
    if (predicted.startsWith('f') && !predicted.startsWith('sp')) {
      add('coalescence', 'sp→f')
    }
  }

  // --- Assimilation (yellow: j→l creating lɛlo) ---
  if (word.toLowerCase() === 'yellow') {
    if (predicted.startsWith('l') && tPhones[0] === 'j') {
      add('assimilation', 'j→l')
    }
  }

  return errors
}
