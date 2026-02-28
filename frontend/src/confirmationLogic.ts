import type { SpeechError } from './speechRules'

export interface WordResult {
  word: string
  errors: SpeechError[]
}

export interface ConfirmationTask {
  pattern: string
  originalWord: string
  originalDetail: string
  confirmWord: string
  confirmWordIndex: number
}

export interface ConfirmationResult {
  pattern: string
  detail: string
  originalWord: string
  confirmWord: string
  confirmed: boolean
}

/** Which assessment words test each error pattern (from assessment_words.md). */
const PATTERN_WORDS: Record<string, string[]> = {
  fronting: ['cup', 'duck', 'green', 'shovel', 'fish'],
  final_consonant_deletion: ['cup', 'duck', 'soap', 'red', 'leaf'],
  cluster_reduction: ['green', 'spoon', 'plate', 'drum', 'zebra'],
  gliding_r: ['red', 'green', 'zebra', 'drum'],
  gliding_l: ['leaf', 'plate', 'yellow'],
  stopping_f_s: ['fish', 'soap', 'leaf'],
  stopping_v_z: ['shovel', 'zebra'],
  stopping_sh_ch_j: ['shovel', 'fish', 'chair', 'juice'],
  deaffrication: ['chair', 'juice'],
  voicing: ['cup', 'plate'],
  backing: ['duck', 'drum'],
  coalescence: ['spoon'],
  assimilation: ['yellow'],
  initial_consonant_deletion: [
    'cup', 'duck', 'green', 'shovel', 'fish',
    'soap', 'zebra', 'red', 'leaf', 'spoon',
    'plate', 'chair', 'juice', 'yellow', 'drum',
  ],
}

/**
 * Build confirmation tasks from assessment results.
 * For each unique suspected pattern, pick a different word to re-test.
 * If no alternate word exists, re-use the original.
 */
export function buildConfirmationTasks(
  assessmentResults: WordResult[],
  assessmentWords: { word: string }[],
): ConfirmationTask[] {
  const patternTriggers = new Map<string, { words: Set<string>; detail: string }>()

  for (const result of assessmentResults) {
    for (const err of result.errors) {
      if (!patternTriggers.has(err.pattern)) {
        patternTriggers.set(err.pattern, { words: new Set(), detail: err.detail })
      }
      patternTriggers.get(err.pattern)!.words.add(result.word)
    }
  }

  const tasks: ConfirmationTask[] = []

  for (const [pattern, { words: triggerWords, detail }] of patternTriggers) {
    const candidates = PATTERN_WORDS[pattern] ?? []
    const alternate = candidates.find((w) => !triggerWords.has(w))
    const originalWord = [...triggerWords][0]
    const chosenWord = alternate ?? originalWord
    const wordIndex = assessmentWords.findIndex((aw) => aw.word === chosenWord)

    tasks.push({
      pattern,
      originalWord,
      originalDetail: detail,
      confirmWord: chosenWord,
      confirmWordIndex: wordIndex,
    })
  }

  return tasks
}

/** Convert snake_case pattern names to display format. */
export function formatPatternName(pattern: string): string {
  return pattern
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
