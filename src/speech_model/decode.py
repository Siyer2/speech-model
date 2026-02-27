"""Constrained CTC decoding via Viterbi forced alignment against ontology-derived variants."""

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import Vocab

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLANK_IDX = 0
UNK_IDX = 1

# Map ontology English-like notation → IPA characters in the vocab
_ENGLISH_TO_IPA: dict[str, str] = {
    "t": "t",
    "d": "d",
    "p": "p",
    "b": "b",
    "k": "k",
    "g": "ɡ",
    "sh": "ʃ",
    "ng": "ŋ",
    "s": "s",
    "z": "z",
    "f": "f",
    "v": "v",
    "ch": "ʧ",
    "j": "ʤ",
    "zh": "ʒ",
    "th": "θ",
    "l": "l",
    "r": "ɹ",
    "w": "w",
    "y": "j",
    "n": "n",
    "m": "m",
}

# IPA consonants for detecting clusters and word-edge consonants
_CONSONANTS = frozenset("bdɡfhʤklmnŋpɹrsʃtθvwzðɫɾʒʔʧj")

# Vowel substitution pairs for vowel distortion variants.
# Each (A, B) pair generates both A→B and B→A substitutions.
_VOWEL_PAIRS: list[tuple[str, str]] = [
    ("i", "ɪ"),  # tense-lax
    ("u", "ʊ"),  # tense-lax
    ("e", "ɛ"),  # tense-lax
    ("o", "ɔ"),  # tense-lax
    ("æ", "ɛ"),  # front vowel height
    ("ɑ", "æ"),  # low vowel front-back
    ("ɑ", "ɔ"),  # low vowel rounding
    ("ʌ", "ə"),  # centralization
    ("ɚ", "ə"),  # de-rhoticization
    ("ɝ", "ɛ"),  # r-colored → front mid
    ("a", "ɑ"),  # open vowel variant
]


# ---------------------------------------------------------------------------
# Ontology loading
# ---------------------------------------------------------------------------


def load_ontology(path: str | Path) -> dict:
    """Load ontology.yaml and return the raw error_patterns dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["error_patterns"]


def _parse_substitution_rule(rule_str: str) -> tuple[str, str] | None:
    """Parse 'source → target' into (ipa_source, ipa_target).
    Returns None for non-substitution entries."""
    if "→" not in rule_str:
        return None
    parts = rule_str.split("→")
    if len(parts) != 2:
        return None
    src = parts[0].strip()
    tgt = parts[1].strip()
    ipa_src = _ENGLISH_TO_IPA.get(src, src)
    ipa_tgt = _ENGLISH_TO_IPA.get(tgt, tgt)
    return (ipa_src, ipa_tgt)


def _parse_lisp_rule(rule_str: str) -> str | None:
    """Parse lisp-style 's*' into the affected IPA consonant. Returns None if not a lisp entry."""
    rule_str = rule_str.strip()
    if rule_str.endswith("*") and len(rule_str) >= 2:
        eng = rule_str[:-1]
        return _ENGLISH_TO_IPA.get(eng, eng)
    return None


def parse_ontology_rules(
    ontology: dict,
) -> dict[str, list]:
    """Parse ontology into structured rules per pattern.

    Returns dict mapping pattern_name → list of rule tuples:
      - ("sub", source_ipa, target_ipa)      for substitution rules
      - ("del_final",)                        for final consonant deletion
      - ("del_initial",)                      for initial consonant deletion
      - ("cluster_red",)                      for cluster reduction
      - ("lisp", consonant_ipa)              for lisp patterns
    """
    rules: dict[str, list] = {}

    for pattern_name, pattern_data in ontology.items():
        sounds = pattern_data.get("sounds_affected", []) or []
        pattern_rules: list = []

        if pattern_name == "final_consonant_deletion":
            pattern_rules.append(("del_final",))
        elif pattern_name == "initial_consonant_deletion":
            pattern_rules.append(("del_initial",))
        elif pattern_name == "cluster_reduction":
            pattern_rules.append(("cluster_red",))
            # Also parse specific examples as substitution rules
            for rule_str in sounds:
                parsed = _parse_substitution_rule(rule_str)
                if parsed:
                    pattern_rules.append(("sub", parsed[0], parsed[1]))
        elif pattern_name in (
            "lateral_lisp",
            "interdental_lisp",
            "interdental_lisp_extended",
        ):
            for rule_str in sounds:
                consonant = _parse_lisp_rule(rule_str)
                if consonant:
                    pattern_rules.append(("lisp", consonant))
        elif pattern_name in (
            "weak_syllable_deletion",
            "assimilation",
            "coalescence",
            "substitution_error",
        ):
            # Skip patterns without automatable rules
            continue
        else:
            # Standard substitution patterns
            for rule_str in sounds:
                parsed = _parse_substitution_rule(rule_str)
                if parsed:
                    pattern_rules.append(("sub", parsed[0], parsed[1]))

        if pattern_rules:
            rules[pattern_name] = pattern_rules

    return rules


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


def _find_consonant_clusters(text: str) -> list[tuple[int, int]]:
    """Find (start, end) index spans of consecutive consonant runs of length >= 2."""
    clusters = []
    i = 0
    while i < len(text):
        if text[i] in _CONSONANTS:
            j = i
            while j < len(text) and text[j] in _CONSONANTS:
                j += 1
            if j - i >= 2:
                clusters.append((i, j))
            i = j
        else:
            i += 1
    return clusters


def _apply_substitutions(text: str, sub_rules: list[tuple[str, str]]) -> list[str]:
    """Apply substitution rules to generate variants.

    For each rule, applies it at each matching position (single substitution).
    Also applies all rules from the same set simultaneously (full pattern).
    """
    variants: set[str] = set()

    # Single substitutions: apply one rule at one position
    for src, tgt in sub_rules:
        idx = 0
        while idx < len(text):
            pos = text.find(src, idx)
            if pos == -1:
                break
            variant = text[:pos] + tgt + text[pos + len(src) :]
            variants.add(variant)
            idx = pos + 1

    # Full pattern: apply ALL rules simultaneously (left to right, non-overlapping)
    if len(sub_rules) > 1:
        full = list(text)
        applied = [False] * len(text)
        for src, tgt in sub_rules:
            idx = 0
            while idx <= len(full) - len(src):
                # Check if this position matches the source in the original text
                original_pos = idx
                if (
                    text[original_pos : original_pos + len(src)] == src
                    and not applied[original_pos]
                ):
                    # Replace in the working copy
                    full[original_pos : original_pos + len(src)] = list(tgt)
                    for k in range(original_pos, min(original_pos + len(src), len(applied))):
                        applied[k] = True
                    idx = original_pos + len(tgt)
                else:
                    idx += 1
        full_str = "".join(full)
        if full_str != text:
            variants.add(full_str)

    return list(variants)


def generate_variants(
    target_phonetic: str,
    parsed_rules: dict[str, list],
) -> list[str]:
    """Generate all allowed phoneme sequences from a target + error pattern rules.

    Args:
        target_phonetic: The target pronunciation (after normalize_phonetic)
        parsed_rules: Output of parse_ontology_rules()

    Returns:
        List of unique allowed sequences including the target itself.
    """
    target = target_phonetic
    variants: set[str] = {target}

    # Per-pattern variants (for pairwise combinations later)
    per_pattern_variants: dict[str, set[str]] = {}

    for pattern_name, rules in parsed_rules.items():
        pattern_variants: set[str] = set()

        sub_rules = [(r[1], r[2]) for r in rules if r[0] == "sub"]
        if sub_rules:
            pattern_variants.update(_apply_substitutions(target, sub_rules))

        for rule in rules:
            if rule[0] == "del_final" and target:
                # Remove last consonant
                for i in range(len(target) - 1, -1, -1):
                    if target[i] in _CONSONANTS:
                        pattern_variants.add(target[:i] + target[i + 1 :])
                        break

            elif rule[0] == "del_initial" and target:
                # Remove first consonant
                for i in range(len(target)):
                    if target[i] in _CONSONANTS:
                        pattern_variants.add(target[:i] + target[i + 1 :])
                        break

            elif rule[0] == "cluster_red":
                # For each consonant cluster, remove each consonant in turn
                clusters = _find_consonant_clusters(target)
                for start, end in clusters:
                    for k in range(start, end):
                        v = target[:k] + target[k + 1 :]
                        pattern_variants.add(v)

            elif rule[0] == "lisp":
                consonant = rule[1]
                # Insert * after each occurrence of the consonant
                idx = 0
                result = list(target)
                offset = 0
                while idx < len(target):
                    if target[idx] == consonant:
                        insert_pos = idx + 1 + offset
                        result.insert(insert_pos, "*")
                        offset += 1
                    idx += 1
                lisp_variant = "".join(result)
                if lisp_variant != target:
                    pattern_variants.add(lisp_variant)

        # Filter out empty strings
        pattern_variants.discard("")

        if pattern_variants:
            per_pattern_variants[pattern_name] = pattern_variants
            variants.update(pattern_variants)

    # Vowel distortion variants: single-vowel substitutions from _VOWEL_PAIRS
    vowel_variants: set[str] = set()
    for a, b in _VOWEL_PAIRS:
        # A → B
        idx = 0
        while idx < len(target):
            pos = target.find(a, idx)
            if pos == -1:
                break
            vowel_variants.add(target[:pos] + b + target[pos + len(a) :])
            idx = pos + 1
        # B → A
        idx = 0
        while idx < len(target):
            pos = target.find(b, idx)
            if pos == -1:
                break
            vowel_variants.add(target[:pos] + a + target[pos + len(b) :])
            idx = pos + 1
    vowel_variants.discard("")
    vowel_variants.discard(target)
    if vowel_variants:
        per_pattern_variants["vowel_distortions"] = vowel_variants
        variants.update(vowel_variants)

    # Build combined rules dict (parsed_rules + synthetic vowel rules) for pairwise combos
    all_rules = dict(parsed_rules)
    if "vowel_distortions" in per_pattern_variants:
        all_rules["vowel_distortions"] = [("sub", a, b) for a, b in _VOWEL_PAIRS] + [
            ("sub", b, a) for a, b in _VOWEL_PAIRS
        ]

    # Pairwise combinations: apply pattern B to each variant from pattern A
    pattern_names = list(per_pattern_variants.keys())
    for name_a, name_b in combinations(pattern_names, 2):
        rules_b = all_rules[name_b]
        sub_rules_b = [(r[1], r[2]) for r in rules_b if r[0] == "sub"]

        for variant_a in per_pattern_variants[name_a]:
            # Apply substitution rules from pattern B to variant A
            if sub_rules_b:
                combo_variants = _apply_substitutions(variant_a, sub_rules_b)
                variants.update(v for v in combo_variants if v)

            # Apply deletion/cluster/lisp rules from B to variant A
            for rule in rules_b:
                if rule[0] == "del_final" and variant_a:
                    for i in range(len(variant_a) - 1, -1, -1):
                        if variant_a[i] in _CONSONANTS:
                            v = variant_a[:i] + variant_a[i + 1 :]
                            if v:
                                variants.add(v)
                            break
                elif rule[0] == "del_initial" and variant_a:
                    for i in range(len(variant_a)):
                        if variant_a[i] in _CONSONANTS:
                            v = variant_a[:i] + variant_a[i + 1 :]
                            if v:
                                variants.add(v)
                            break
                elif rule[0] == "cluster_red":
                    clusters = _find_consonant_clusters(variant_a)
                    for start, end in clusters:
                        for k in range(start, end):
                            v = variant_a[:k] + variant_a[k + 1 :]
                            if v:
                                variants.add(v)

    return list(variants)


# ---------------------------------------------------------------------------
# Viterbi Forced Alignment
# ---------------------------------------------------------------------------


def _encode_variant(variant: str, vocab: Vocab) -> list[int]:
    """Encode variant to token indices, filtering UNK tokens."""
    return [idx for idx in vocab.encode(variant) if idx != UNK_IDX]


def _ctc_collapse(ids: list[int], vocab: Vocab) -> str:
    """CTC greedy decode: collapse repeats, remove blanks, decode to text."""
    collapsed = []
    prev = -1
    for i in ids:
        if i != prev:
            if i != BLANK_IDX:
                collapsed.append(i)
            prev = i
    return vocab.decode(collapsed)


def viterbi_align(log_probs_np: "np.ndarray", token_ids: list[int]) -> float:
    """Score a token sequence against CTC log-probabilities via Viterbi forced alignment.

    Builds the standard CTC state topology (interleaved blanks) and finds the
    best monotonic alignment path through the log-prob matrix.

    Args:
        log_probs_np: (T, V) numpy array of log-softmax output.
        token_ids: List of vocab indices for the target sequence (no blanks, no UNK).

    Returns:
        Log-probability of the best alignment path. Returns -inf for empty sequences.
    """
    n = len(token_ids)
    if n == 0:
        return -math.inf

    t = log_probs_np.shape[0]
    s = 2 * n + 1  # CTC states: blank, tok0, blank, tok1, ..., tokN-1, blank
    neg_inf = -math.inf

    # State-to-vocab-index mapping
    state_idx = [BLANK_IDX] * s
    for j in range(n):
        state_idx[2 * j + 1] = token_ids[j]

    # DP: dp_prev[j] = best log-prob ending in state j at previous timestep
    dp_prev = [neg_inf] * s
    dp_prev[0] = float(log_probs_np[0, state_idx[0]])
    dp_prev[1] = float(log_probs_np[0, state_idx[1]])

    for time in range(1, t):
        dp_curr = [neg_inf] * s
        for j in range(s):
            best = dp_prev[j]  # self-loop

            if j >= 1 and dp_prev[j - 1] > best:
                best = dp_prev[j - 1]  # advance from previous state

            # Skip-blank: jump from token k-1 directly to token k (odd states only)
            if j >= 2 and j % 2 == 1:
                k = j // 2
                if token_ids[k] != token_ids[k - 1] and dp_prev[j - 2] > best:
                    best = dp_prev[j - 2]

            dp_curr[j] = best + float(log_probs_np[time, state_idx[j]])

        dp_prev = dp_curr

    return max(dp_prev[s - 1], dp_prev[s - 2])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def constrained_decode(
    log_probs: torch.Tensor,
    target_phonetic: str,
    parsed_rules: dict[str, list],
    vocab: Vocab,
    beam_width: int = 20,
) -> str:
    """Decode a single utterance using Viterbi forced alignment against ontology variants.

    Generates all phonemically plausible variants from the target + error patterns,
    adds the greedy CTC decode as a fallback candidate, scores each via CTC Viterbi
    forced alignment, and returns the highest-scoring one.

    Args:
        log_probs: (T, V) log-softmax probabilities for the utterance.
        target_phonetic: The target pronunciation (after normalize_phonetic).
        parsed_rules: Pre-parsed ontology rules from parse_ontology_rules().
        vocab: The phoneme vocabulary.
        beam_width: Unused, kept for API compatibility.

    Returns:
        Decoded phoneme string.
    """
    if not target_phonetic:
        return ""

    variants = generate_variants(target_phonetic, parsed_rules)

    # Add greedy decode as a candidate so we never regress vs greedy
    greedy_ids = log_probs.argmax(dim=-1).tolist()
    greedy_text = _ctc_collapse(greedy_ids, vocab)
    if greedy_text:
        variants.append(greedy_text)

    # Convert to numpy once for all variants
    lp = log_probs.detach().cpu().numpy()

    best_score = -math.inf
    best_variant = target_phonetic

    for variant in variants:
        token_ids = _encode_variant(variant, vocab)
        if not token_ids:
            continue
        score = viterbi_align(lp, token_ids)
        if score > best_score:
            best_score = score
            best_variant = variant

    return best_variant
