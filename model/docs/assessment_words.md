# Assessment Words

15 words selected for a downstream rules engine that compares target phonetic transcription
against the model's predicted phonetic transcription to deduce speech error patterns.

Selection criteria:
1. The model performs well on the word (high accuracy on correct speech in beam decode)
2. The model has seen the word produced with real speech errors in training data
3. Sufficient coverage across datasets and participants for confidence
4. 3-5 phonemes (model's optimal range)
5. Each word enables clear, unambiguous rules for detecting ontology error patterns

## Word List

| # | Word | IPA | Total | Child | Beam N | Correct Perfect | Patterns Tested | Detection Rules |
|---|------|-----|------:|------:|-------:|----------------:|-----------------|-----------------|
| 1 | cup | /kʌp/ | 353 | 287 | 76 | 92% | fronting, FCD, voicing | onset k→t: fronting | onset k→g: voicing | final p missing: FCD |
| 2 | duck | /dʌk/ | 342 | 283 | 72 | 91% | backing, fronting, FCD | onset d→g: backing | final k→t: fronting | final k missing: FCD |
| 3 | green | /ɡɹin/ | 442 | 332 | 89 | 98% | fronting, cluster_reduction, gliding_r | onset g→d: fronting | ɹ→w: gliding_r | ɡɹ→ɡ or ɡɹ→ɹ: cluster_reduction |
| 4 | shovel | /ʃʌvɫ/ | 430 | 326 | 87 | 89% | fronting, stopping_sh_ch_j, stopping_v_z | onset ʃ→s: fronting | onset ʃ→t: stopping_sh_ch_j | v→b: stopping_v_z |
| 5 | fish | /fɪʃ/ | 248 | 160 | 49 | 95% | stopping_f_s, fronting | onset f→p: stopping_f_s | final ʃ→s: fronting | final ʃ→t: stopping_sh_ch_j |
| 6 | soap | /sop/ | 223 | 166 | 46 | 100% | stopping_f_s, FCD | onset s→t: stopping_f_s | final p missing: FCD |
| 7 | zebra | /zibɹə/ | 293 | 201 | 56 | 81% | stopping_v_z, gliding_r | onset z→d: stopping_v_z | ɹ→w: gliding_r | bɹ→b or bɹ→ɹ: cluster_reduction |
| 8 | red | /ɹɛd/ | 166 | 166 | 34 | 94% | gliding_r, FCD | onset ɹ→w: gliding_r | final d missing: FCD |
| 9 | leaf | /lif/ | 283 | 197 | 56 | 87% | gliding_l, FCD, stopping_f_s | onset l→w or l→j: gliding_l | final f missing: FCD | final f→p: stopping_f_s |
| 10 | spoon | /spun/ | 192 | 118 | 44 | 96% | cluster_reduction, coalescence | sp→p or sp→s: cluster_reduction | sp→f: coalescence |
| 11 | plate | /plet/ | 273 | 208 | 51 | 94% | cluster_reduction, gliding_l, voicing | pl→p or pl→l: cluster_reduction | l→w in cluster: gliding_l | onset p→b: voicing |
| 12 | chair | /tʃɛɚ/ | 389 | 283 | 82 | 93% | deaffrication, stopping_sh_ch_j | onset tʃ→ʃ: deaffrication | onset tʃ→t: stopping_sh_ch_j |
| 13 | juice | /dʒus/ | 262 | 163 | 50 | 98% | deaffrication, stopping_sh_ch_j | onset dʒ→ʒ: deaffrication | onset dʒ→d: stopping_sh_ch_j |
| 14 | yellow | /jɛlo/ | 401 | 333 | 81 | 97% | assimilation, gliding_l | j→l (lɛlo): assimilation | l→w or l→j: gliding_l |
| 15 | drum | /dɹʌm/ | 412 | 310 | 86 | 95% | cluster_reduction, gliding_r, backing | dɹ→d or dɹ→ɹ: cluster_reduction | ɹ→w: gliding_r | onset d→g: backing |

## Pattern Coverage Matrix

| Ontology Pattern | Tested By | Rule Summary |
|------------------|-----------|--------------|
| **fronting** (k→t, g→d, sh→s, ng→n) | cup (k→t), duck (k→t final), green (g→d), shovel (sh→s), fish (sh→s final) | Target velar/postalveolar predicted as alveolar |
| **final_consonant_deletion** | cup, duck, soap, red, leaf | Final phoneme in target absent from prediction |
| **cluster_reduction** | green (gr), spoon (sp), plate (pl), drum (dr), zebra (br) | Two-consonant cluster reduced to one in prediction |
| **gliding_r** | red, green, zebra, drum | Target /ɹ/ predicted as /w/ |
| **gliding_l** | leaf, plate, yellow | Target /l/ predicted as /w/ or /j/ |
| **stopping_f_s** (f→p, s→t) | fish (f→p), soap (s→t), leaf (f→p final) | Target fricative /f/ or /s/ predicted as stop /p/ or /t/ |
| **stopping_v_z** (v→b, z→d) | shovel (v→b), zebra (z→d) | Target fricative /v/ or /z/ predicted as stop /b/ or /d/ |
| **stopping_sh_ch_j** (sh→t, ch→t, j→d) | shovel (sh→t), fish (sh→t final), chair (ch→t), juice (j→d) | Target /ʃ/, /tʃ/, or /dʒ/ predicted as stop |
| **deaffrication** (ch→sh, j→zh) | chair (ch→sh), juice (j→zh) | Target affricate predicted as corresponding fricative |
| **voicing** (t→d, p→b, k→g) | cup (k→g), plate (p→b) | Target voiceless stop predicted as voiced counterpart |
| **backing** (t→k, d→g) | duck (d→g), drum (d→g) | Target alveolar stop predicted as velar |
| **coalescence** | spoon (sp→f) | Cluster replaced by a single phoneme sharing features of both |
| **assimilation** | yellow (j→l, creating lɛlo) | One sound takes on characteristics of another in the word |
| **initial_consonant_deletion** | Any word | First phoneme in target absent from prediction |

## Validated Against Real Speech Errors

The model has seen these words produced with real speech errors in training. Key examples
from beam decode where the model correctly captured disordered speech:

| Word | Child Said | Model Predicted | Error Detected |
|------|-----------|-----------------|----------------|
| green | /din/ (fronting + cluster reduction) | /din/ | fronting (g→d), cluster_reduction (gr→d) |
| green | /ɡwin/ (gliding_r in cluster) | /ɡwin/ | gliding_r (ɹ→w) |
| leaf | /wif/ (gliding_l) | /wif/ | gliding_l (l→w) |
| juice | /dus/ (stopping) | /dus/ | stopping_sh_ch_j (dʒ→d) |
| spoon | /fun/ (coalescence) | /fun/ | coalescence (sp→f) |
| spoon | /bun/ (cluster reduction) | /bun/ | cluster_reduction (sp→b) |
| duck | /dʌt/ (fronting of final k) | /dʌt/ | fronting (k→t) |
| plate | /pwet/ (gliding_l in cluster) | /pwet/ | gliding_l (l→w) |
| plate | /bet/ (cluster reduction + voicing) | /bet/ | cluster_reduction (pl→b), voicing (p→b) |
| drum | /dwʌm/ (gliding_r in cluster) | /dwʌm/ | gliding_r (ɹ→w) |
| yellow | /lɛlo/ (assimilation) | /lɛlo/ | assimilation (j→l) |
| yellow | /jɛwo/ (gliding_l) | /jɛwo/ | gliding_l (l→w) |
| soap | /dʒoʊp/ (stopping) | /dʒoʊp/ | correctly captured unexpected substitution |
| chair | /dɛr/ (stopping) | captured as stop onset | stopping_sh_ch_j (tʃ→d) |
| fish | /fɪs/ (fronting) | /fɪs/ | fronting (ʃ→s) |

## Confidence Notes

- **zebra** (81% correct-speech accuracy) is the weakest word. Errors concentrate in the
  medial /bɹə/ segment, not the /z/ onset, so z→d stopping detection on the initial phoneme
  remains reliable. Monitor in production.
- **red** has excellent 94% accuracy but spans only 2 datasets (TDChildrenandAdults,
  SuspectedSSD). It is the only high-frequency gliding_r test word available in isolation.
  Green, zebra, and drum provide additional gliding_r coverage within clusters.
- **shovel** (89% correct-speech accuracy) is reliable for onset /ʃ/ detection (fronting,
  stopping). The medial /v/→/b/ stopping_v_z test has reduced reliability due to occasional
  model v/b confusion in medial position.
- **voicing** (t→d) has no direct single-word test because no /t/-initial word has sufficient
  coverage. Cup tests k→g and plate tests p→b, covering 2 of 3 voicing pairs.

## Dropped Patterns

The following ontology patterns are excluded from assessment word coverage:

1. **weak_syllable_deletion** — Requires multi-syllable words (6+ phonemes) where model
   accuracy drops significantly. Best candidates: elephant (49% beam perfect), banana
   (163 samples from only 3 datasets). Neither is reliable enough.
2. **fricative_simplification** (th→f) — Model accuracy on /θ/ is only 71.5% with a 7%
   θ→f confusion rate, meaning the model itself produces the exact substitution we'd be
   trying to detect. Unacceptable false positive rate.
3. **lateral_lisp** — Requires detecting articulatory quality of /s/ and /z/ (air escaping
   over tongue sides). A phoneme-level model outputs the same symbol /s/ regardless of
   articulation quality. Not detectable at this level.
4. **interdental_lisp / interdental_lisp_extended** — Same limitation. The model cannot
   distinguish correctly-placed /s/ from interdentalized /s/. Both produce the same phoneme
   symbol.


