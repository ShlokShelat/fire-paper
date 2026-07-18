"""
FiRE Pipeline — Worker 1, Stage 2 + Stage 3  (single-model, axis-verified)
============================================================================
Stage 2: Single LLM Event Extractor
  Model: GPT-5.1   (the same model FiRE's paper uses as its extraction
  model, so that any accuracy gap between FiRE and a "GPT-5.1 direct
  scoring" baseline is attributable to the symbolic/verification
  scaffolding around it, not to a different or stronger model doing the
  extraction. This is the paper's "same-model ablation" claim, and it
  only holds if extraction genuinely runs through one model.)

Stage 3: Four-Axis Verification Engine (pure code, deterministic)
    1. CONSTRUCT   — claimed experience_type matches the text, and the
                      synthesized `event` is entailed by its own
                      source_sentence (source -> event, directional).
    2. ACTOR/ROLE   — the claimed agent is the one the source assigns the
                      action to.
    3. AGE WINDOW   — claimed life_stage is consistent with age evidence.
    4. GROUNDEDNESS — every person/action in the event is anchored in the
                      source text or immediate context.

Output: validated_events.json — Worker 1 final output → Worker 2 input

Usage:
  python3 fire_stage2_3_single_model.py stage1_output.json \
    --patient-id patient03 \
    --api-key YOUR_KEY \
    [--output validated_events.json] [--concurrency 5] [--dry-run]
  python3 fire_stage2_3_single_model.py --selftest      # offline regression guard

Key also via env: OPENAI_API_KEY
"""

import json
import re
import os
import sys
import asyncio
import aiohttp
import argparse
import time
import logging
import hashlib
import tempfile
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1.  CONSTANTS AND CONFIG
# ─────────────────────────────────────────────────────────────

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Single source of truth for which model performs extraction.
MODEL_NAME = "gpt-5.1"

# Reasoning-family detection (GPT-5.x, o1/o3/o4). These models reject
# `temperature` and `max_tokens` on Chat Completions.
_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)

# Default reasoning effort for gpt-5.1. Its own default is "none"; setting it
# explicitly keeps behaviour stable and is the closest achievable analogue to
# the paper's temperature-0 determinism intent for a reasoning model.
DEFAULT_REASONING_EFFORT = "none"

# ─────────────────────────────────────────────────────────────
# 1A.  AXIS-BASED CONFIDENCE CALIBRATION
# ─────────────────────────────────────────────────────────────
AXIS_CONFIDENCE_MATRIX = {
    ("HIGH",   4): ("CONFIRMED",       0.95),
    ("HIGH",   3): ("REVIEW_REQUIRED", 0.70),
    ("HIGH",   2): ("DISCARD",         0.40),
    ("HIGH",   1): ("DISCARD",         0.20),
    ("HIGH",   0): ("DISCARD",         0.05),

    ("MEDIUM", 4): ("TENTATIVE",       0.75),
    ("MEDIUM", 3): ("REVIEW_REQUIRED", 0.55),
    ("MEDIUM", 2): ("DISCARD",         0.30),
    ("MEDIUM", 1): ("DISCARD",         0.15),
    ("MEDIUM", 0): ("DISCARD",         0.05),

    ("LOW",    4): ("TENTATIVE",       0.50),
    ("LOW",    3): ("DISCARD",         0.30),
    ("LOW",    2): ("DISCARD",         0.15),
    ("LOW",    1): ("DISCARD",         0.05),
    ("LOW",    0): ("DISCARD",         0.00),
}
AXIS_CONFIDENCE_DEFAULT = ("REVIEW_REQUIRED", 0.55)

# Semantic similarity thresholds — used inside the CONSTRUCT axis.
COSINE_THRESHOLD  = 0.82
JACCARD_THRESHOLD = 0.30

FRAGMENT_CONNECTORS = [
    "connected to", "related to", "following",
    "as a result", "due to", "because of",
    "impact:", "contributed to", "this led to",
    "which resulted in", "likely reinforced",
    "possibly related", "noted as", "described as",
    "represents", "reflects", "mirrors", "noted clinically",
    "identified as", "noted by therapist",
]

INDIAN_CULTURAL_GLOSSARY = {
    "kaka":                       "paternal uncle (father's brother)",
    "mama":                       "maternal uncle (mother's brother)",
    "chacha":                     "paternal uncle",
    "maasi":                      "maternal aunt (mother's sister)",
    "bua":                        "paternal aunt (father's sister)",
    "nani":                       "maternal grandmother",
    "dadi":                       "paternal grandmother",
    "nana":                       "maternal grandfather",
    "dada":                       "paternal grandfather",
    "kanyadaan":                  "ritual of father giving daughter at wedding — paternal absence at kanyadaan is clinically significant",
    "joint family":               "multi-generational household with shared authority, limited individual autonomy, collective decision-making",
    "came home in a bad state":   "in Indian clinical notes, frequently implies intoxication or aggressive behaviour",
    "boarding school":            "in Indian family context, often experienced as abandonment or ejection from family",
    "mil":                        "mother-in-law",
    "fil":                        "father-in-law",
    "sil":                        "sister-in-law or brother-in-law",
    "pind":                       "ancestral village",
    "izzat":                      "family honour/reputation — threat to izzat is a major stressor in Indian families",
}

PERSON_REFERENCES = [
    "father", "mother", "brother", "sister",
    "grandfather", "grandmother",
    "husband", "wife", "son", "daughter",
    "uncle", "aunt", "cousin",
    "teacher", "classmate", "colleague",
    "manager", "boss", "friend",
    "partner", "spouse", "therapist",
    "kaka", "mama", "chacha", "nani", "dadi",
    "nana", "dada", "mil", "fil",
    "in-laws", "in-law", "sibling", "twin", "batchmate",
]

PERSON_REFERENCE_NORMALISE = {
    "in-laws": "in-law",
    "in-law":  "in-law",
    "mil":     "mil",
    "fil":     "fil",
}

PERSON_ANCHOR_EXPANSIONS = {
    "mil":          ["mother", "mother-in-law", "in-law", "in-laws"],
    "fil":          ["father", "father-in-law", "in-law", "in-laws"],
    "sil":          ["sister", "brother", "sibling",
                     "sister-in-law", "brother-in-law", "in-law", "in-laws"],
    "paternal":     ["father", "grandfather"],
    "maternal":     ["mother", "grandmother"],
    "mother-in-law": ["mother", "in-law", "in-laws"],
    "father-in-law": ["father", "in-law", "in-laws"],
    "in-law":       ["in-laws", "in-law"],
    "in-laws":      ["in-law", "in-laws"],
}

ACTION_SYNONYMS = {
    "hit":        ["beat", "struck", "abused", "slapped", "punched", "hurt",
                   "physical", "violence"],
    "died":       ["passed away", "deceased", "death", "lost", "gone", "died"],
    "sent":       ["relocated", "moved", "transferred", "placed", "sent away"],
    "rejected":   ["refused", "turned down", "declined", "dismissed",
                   "mocked", "excluded", "bullied", "humiliated", "teased",
                   "rejected", "rejection"],
    "discovered": ["found out", "learned", "realised", "uncovered", "revealed",
                   "disclosed", "discovered", "found"],
    "married":    ["wed", "got married", "marriage", "married", "wedding"],
    "divorced":   ["separated", "split", "broke up", "divorce", "separation"],
    "diagnosed":  ["identified", "detected", "found to have", "diagnosed",
                   "diagnosis", "condition"],
    "arrested":   ["taken into custody", "detained", "police", "custody",
                   "arrested", "legal"],
    "bullied":    ["teased", "mocked", "humiliated", "excluded", "bullied",
                   "bullying", "harassed", "taunted"],
}
# Passive-voice participle so "was beaten by..." matches the "hit" family.
ACTION_SYNONYMS["hit"].append("beaten")

SKIP_PREFIXES = {
    "therapy goals", "seeking", "previous therapy",
    "current living", "referred by", "note on communication",
}

SKIP_EXCLUSION_REASONS = {
    "standalone_age_marker",
    "administrative_metadata",
    "pure_clinical_observation",
}

CLINICAL_LABELS = [
    "ptsd", "post-traumatic", "depression", "depressive",
    "anxiety disorder", "panic", "dissociation", "dissociative",
    "borderline", "bipolar", "psychosis", "psychotic",
    "ocd", "obsessive", "narcissistic", "borderline personality",
    "attachment disorder", "complex trauma",
]

LIFE_STAGE_AGE_RANGES = {
    "childhood": (0, 12),
    "teenage":   (12, 19),
    "adult":     (18, 59),
    "mid-life":  (35, 65),
    "unknown":   None,
}

VALID_LIFE_STAGES = {"childhood", "teenage", "adult", "mid-life", "unknown"}
VALID_FREQUENCIES = {"single", "recurring", "repeated", "chronic", "unknown"}
VALID_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}


# ─────────────────────────────────────────────────────────────
# 2.  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class ProcessingUnit:
    unit_id:           str
    line_number:       int
    subsection:        str
    age_context:       Optional[str]
    source_type:       str
    stage1_confidence: str
    sentences:         list
    sentence_objects:  list
    context_before:    list
    context_after:     list


@dataclass
class LLMExtraction:
    source_sentence:        str
    event_present:          bool
    event:                  Optional[str]
    experience_type:        str
    persons_involved:       Optional[list]
    life_stage:              str
    frequency:               str
    confidence:              str
    null_reason:             Optional[str]
    requires_new_state:      bool = False
    new_state_description:   Optional[str] = None
    possible_hallucination:  bool = False


@dataclass
class UnitLLMResult:
    unit_id:          str
    model_name:       str
    extractions:      list
    status:           str
    raw_response:     str = ""
    tokens_used:      int = 0
    latency_ms:       int = 0
    reasoning_tokens: int = 0   # hidden reasoning tokens (GPT-5 family)


@dataclass
class ValidatedEvent:
    unit_id:               str
    source_sentence:       str
    event:                 str
    experience_type:       str
    persons_involved:      list
    life_stage:            str
    frequency:             str
    confidence:            float
    status:                str
    verified:              bool
    human_review_required: bool
    cultural_note:         Optional[str]
    stage1_confidence:     str
    line_number:           int
    age_context:            Optional[str]
    subsection:              str
    construct_check_passed:    bool
    actor_role_check_passed:   bool
    age_window_check_passed:   bool
    groundedness_check_passed: bool
    axes_passed:               int
    verification_flags:        list
    self_groundedness_score:   Optional[float]
    requires_new_state:            bool = False
    new_state_description:         Optional[str] = None
    stage4_passed:                 bool = True
    stage4_flags:                  list = None
    verification_flags_structured: list = None   # auto-derived below

    def __post_init__(self):
        if self.stage4_flags is None:
            self.stage4_flags = []
        if self.verification_flags_structured is None:
            self.verification_flags_structured = structure_flags(self.verification_flags)


# ─────────────────────────────────────────────────────────────
# 3.  STEP 1 — GROUPER   (model-independent)
# ─────────────────────────────────────────────────────────────

def extract_subsection(sentence: str) -> str:
    if sentence.startswith("["):
        return ""
    if ": " not in sentence:
        return ""
    prefix = sentence.split(": ", 1)[0]
    return prefix.strip() if len(prefix.strip()) <= 80 else ""


def strip_prefix(sentence: str) -> str:
    if ": " not in sentence:
        return sentence
    prefix = sentence.split(": ", 1)[0]
    if len(prefix.strip()) <= 80:
        return sentence.split(": ", 1)[1]
    return sentence


def is_fragment(text_after_prefix: str) -> bool:
    s = text_after_prefix.strip().lower()
    if not s:
        return True
    if len(s) < 55:
        return True
    if any(s.startswith(conn) for conn in FRAGMENT_CONNECTORS):
        return True
    event_verbs_re = re.compile(
        r'\b(is|was|were|had|has|have|did|does|do|said|told|went|came|left|'
        r'died|hit|beat|married|diagnosed|sent|found|discovered|felt|became|'
        r'started|stopped|moved|developed|experienced|reported|described|noted|'
        r'arrested|divorced|separated|rejected|disclosed|witnessed|saw)\b'
    )
    if not event_verbs_re.search(s):
        return True
    return False


def group_sentences(stage1_data: list, patient_id: str) -> list:
    filtered = []
    for r in stage1_data:
        excl = r.get("exclusion_reason", "")
        if excl in SKIP_EXCLUSION_REASONS:
            continue
        if r.get("confidence_tag") == "EXCLUDE" and excl in SKIP_EXCLUSION_REASONS:
            continue
        prefix_lower = extract_subsection(r["sentence"]).lower()
        if any(prefix_lower.startswith(sp) for sp in SKIP_PREFIXES):
            continue
        filtered.append(r)

    filtered.sort(key=lambda x: x["line_number"])

    line_groups: dict = defaultdict(list)
    for r in filtered:
        line_groups[r["line_number"]].append(r)
    sorted_lines = sorted(line_groups.keys())

    def _group_key(sentence_dict: dict) -> str:
        sentence = sentence_dict.get("sentence", "")
        prefix   = extract_subsection(sentence)
        if prefix:
            return ("prefix", prefix)
        age_ctx = sentence_dict.get("age_context") or "unknown"
        return ("age", age_ctx)

    merged_groups = []
    i = 0
    while i < len(sorted_lines):
        base_ln   = sorted_lines[i]
        ln        = base_ln
        cur_group = list(line_groups[ln])
        cur_key   = _group_key(cur_group[0])

        while i + 1 < len(sorted_lines):
            next_ln = sorted_lines[i + 1]
            if next_ln != ln + 1:
                break
            next_group = line_groups[next_ln]
            next_key   = _group_key(next_group[0])
            if next_key != cur_key:
                break
            cur_group = cur_group + list(next_group)
            ln = next_ln
            i += 1

        merged_groups.append((base_ln, cur_group))
        i += 1

    units = []
    for idx, (base_ln, group) in enumerate(merged_groups):

        ctx_before = []
        for j in range(max(0, idx - 2), idx):
            ctx_before.extend(s["sentence"] for s in merged_groups[j][1])

        ctx_after = []
        for j in range(idx + 1, min(len(merged_groups), idx + 3)):
            ctx_after.extend(s["sentence"] for s in merged_groups[j][1])

        subsection = extract_subsection(group[0]["sentence"])
        if not subsection:
            subsection = group[0].get("age_context") or ""
        age_ctx  = group[0].get("age_context")
        src_type = group[0].get("source_type", "NARRATIVE")

        s1_confs = [s.get("confidence_tag", "MEDIUM") for s in group]
        s1_conf  = ("HIGH"   if "HIGH"   in s1_confs else
                    "MEDIUM" if "MEDIUM" in s1_confs else "LOW")

        unit_id = f"{patient_id}_L{base_ln}_{idx:04d}"

        all_sentences = [s["sentence"] for s in group]
        units.append(ProcessingUnit(
            unit_id=unit_id,
            line_number=base_ln,
            subsection=subsection,
            age_context=age_ctx,
            source_type=src_type,
            stage1_confidence=s1_conf,
            sentences=all_sentences,
            sentence_objects=group,
            context_before=ctx_before,
            context_after=ctx_after,
        ))

    print(f"[GROUPER] {len(filtered)} sentences → {len(units)} section-chunk units")
    return units


# ─────────────────────────────────────────────────────────────
# 4.  PROMPT  (model-independent)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical experience extractor for the FiRE system (Finite automata inspired Reasoning Engine), a neurosymbolic trauma assessment pipeline built on Indian clinical consultation notes from [REDACTED: clinical partner site].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL STRUCTURAL MANDATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You must return EXACTLY ONE comprehensive event object per clinical section heading paragraph.
- Do NOT break down or split separate sentences into separate JSON objects.
- You MUST synthesize all sequential actions, symptoms, patterns, and consequences within the paragraph text into a single, cohesive clinical compound summary string.
- Failure to merge text details into one single compound statement per section violates pipeline constraints.

Your job is to extract comprehensive clinical content from each narrative section paragraph — including concrete events, psychological states, trauma responses, relational patterns, ongoing symptoms, and explicit belief formations. You synthesize what is explicitly stated in the text into unified compound experiences. You do not infer, interpret, or add context not present in the source.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDIAN CLINICAL CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Notes come from [REDACTED: clinical partner site], India.
- kaka/chacha = paternal uncle | mama = maternal uncle
- nani/dadi = maternal/paternal grandmother | nana/dada = maternal/paternal grandfather
- MIL/FIL = mother-in-law / father-in-law
- joint family = multi-generational household with shared authority and limited personal autonomy
- kanyadaan = ritual of father giving daughter at wedding (paternal presence is highly significant)
- "came home in a bad state" → often implies intoxication in Indian clinical notes
- "sent to boarding school" → in Indian context, frequently experienced as parental rejection
- "main hi strong hoon" → "I am the only strong one" — common Indian expression of over-responsibility

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO EXTRACT — extract ALL of these
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE: event
Concrete single-occurrence or time-bounded incidents. Things that happened.
→ "Father beat mother during childhood"
→ "Client was sent to boarding school at age 13"
→ "Grandmother died of brain cancer"
→ "Cousin inappropriately touched client on 2-3 occasions"
→ "Father was taken into police custody"
→ "Client disclosed the incident to parents at age 20"
→ "Client got married", "Client gave birth to first child"

TYPE: pattern
Ongoing relational dynamics, chronic behaviours, repeated situations explicitly stated.
→ "Husband repeatedly fails to take a stand for client in conflicts"
→ "Parents had frequent fights throughout childhood"
→ "Client was consistently compared unfavourably to siblings"
→ "Grandfather shouted at children and parents regularly"
→ "Client was never made to feel emotionally special at key moments"

TYPE: symptom
Concrete psychological or physical symptom presentations explicitly described.
→ "Client is triggered by sounds linked to the postpartum period"
→ "Client experiences anxiety and emotional restlessness"
→ "Client has persistent low mood"
→ "Client feels stuck, unsettled, lacking contentment"
→ "Client suppressed emotions and carried the secret for years"
→ "Client feels emotionally lonely and unseen"
→ "Client experiences hypervigilance about safety of loved ones"

TYPE: belief
Explicit belief statements, core wounds, or identity formations present in the notes.
→ "Client holds core belief: I am not enough"
→ "Client formed core wound: nobody stood for me when I needed support"
→ "Client believes she must always be the strong one"
→ "Client feels she is not allowed to have a weak moment"
→ "Client believes: I must earn love through achievement"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL — SHORT FRAGMENTS ARE VALID EXTRACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Short 2-3 word sentences ARE clinically significant. Extract them:
✓ "Emotions suppressed."            → symptom: "Client suppressed emotions"
✓ "Abandonment feelings."           → symptom: "Client experienced feelings of abandonment"
✓ "Feels fearful."                  → symptom: "Client experienced fear"
✓ "Deep hurt and resentment begin." → symptom: "Client began experiencing deep hurt and resentment"
✓ "Career stagnation."              → symptom: "Client experiences career stagnation"
✓ "Growing resentment."             → symptom: "Client experiences growing resentment"

These are NOT vague — they are compressed clinical notes. Extract them as symptoms.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL — PSYCHOLOGICAL STATES ARE VALID EXTRACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Do NOT apply a "concrete physical event" filter. These are all valid:
✓ "Ongoing resentment — linked to postpartum period and in-law dynamics"
  → symptom: "Client experiences ongoing resentment linked to postpartum period and in-law dynamics"
✓ "Feels she missed out on fully living her college years"
  → symptom: "Client feels she missed out on fully living her college years"
✓ "Career feels stagnant — wants growth, movement, travel"
  → symptom: "Client feels career is stagnant and desires growth, movement, and travel"
✓ "Main complaint: husband does not take a stand for me"
  → pattern: "Husband does not take a stand for client"
✓ "Trust in husband as protector significantly damaged"
  → belief: "Client's trust in husband as protector was significantly damaged"
✓ "First child born."
  → event: "Client gave birth to first child"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT NOT TO EXTRACT — return null for these
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✗ Pure therapist formulations with no patient signal: "suggests parentification pattern", "identified as a likely contributor to"
✗ Admin content: "therapy goals: improve confidence", "seeking career coaching"
✗ Vague demographic facts with no clinical signal: "younger brother is 4 years younger"
✗ General positive descriptors: "mother was supportive", "father was present"
✗ Pure structural labels: "Domain: Self", "Section 3", "ACE 3 confirmed"
✗ Pattern descriptions that are entirely therapist interpretation with no patient content

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRES_NEW_STATE FLAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you find something clinically significant that does not fit standard ACE/trauma categories
— set requires_new_state: true and describe it in new_state_description.
This expands the state space for future patients.
Example: "Client experienced chronic invalidation of professional identity by family" might not
fit any existing cluster and should be flagged.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DO NOT INFER. Only describe what is explicitly in the target text.
2. DO NOT ADD persons, places, or actions not present in source + context.
   For persons_involved: always list the AGENT (perpetrator / active subject) first.
   Active voice: "Father beat mother" → ["Father", "Mother"] — Father acted, Mother received.
   Passive voice: "Client was abused by her father" → ["Father", "Client"] — Father still acted,
   even though Client appears first in the sentence. Order by agency, not by text position.
3. DO NOT ADD temporal information not in the text.
4. PREFER null only for genuinely empty paragraphs — demographic labels, admin, structural markers.
5. For short fragments ("Proposes.", "Rejected."): use context to understand who/what, anchor to target only.
6. One consolidated summary object per narrative heading. Synthesize paragraph content into compound events; do not split sentence lines."""


def build_prompt_user(unit: ProcessingUnit) -> str:
    """Build the structured user message for one processing unit."""

    def _strip_section_prefix(sentence: str, subsection: str) -> str:
        if subsection and sentence.startswith(subsection + ": "):
            return sentence[len(subsection) + 2:].strip()
        if ": " in sentence:
            prefix = sentence.split(": ", 1)[0].strip()
            if len(prefix) <= 80:
                return sentence.split(": ", 1)[1].strip()
        return sentence

    def _to_paragraph(sentences: list, subsection: str, is_timeline: bool) -> str:
        if is_timeline:
            stripped = sentences
        else:
            stripped = [_strip_section_prefix(s, subsection) for s in sentences]

        collapsed = []
        i = 0
        while i < len(stripped):
            s = stripped[i]
            if ": " in s and len(s.split(": ", 1)[0]) <= 60:
                sub   = s.split(": ", 1)[0].strip()
                items = [s.split(": ", 1)[1].strip()]
                j = i + 1
                while j < len(stripped):
                    ns = stripped[j]
                    if ": " in ns and ns.split(": ", 1)[0].strip() == sub:
                        items.append(ns.split(": ", 1)[1].strip())
                        j += 1
                    else:
                        break
                if len(items) > 1:
                    collapsed.append(f"{sub}: {', '.join(items)}")
                else:
                    collapsed.append(s)
                i = j
            else:
                collapsed.append(s)
                i += 1

        return ". ".join(c.rstrip(". ") for c in collapsed if c)

    parts = []
    parts.append(f"AGE CONTEXT: {unit.age_context or 'Not specified'}")
    parts.append(f"STAGE 1 SOURCE TYPE: {unit.source_type}")
    parts.append("")

    if unit.context_before:
        parts.append("CONTEXT BEFORE — reference only, do NOT extract from here:")
        for s in unit.context_before:
            parts.append(f"  • {s}")
        parts.append("")

    if unit.context_after:
        parts.append("CONTEXT AFTER — reference only, do NOT extract from here:")
        for s in unit.context_after:
            parts.append(f"  • {s}")
        parts.append("")

    heading_label = unit.subsection or ""
    is_timeline   = (unit.source_type == "TIMELINE"
                     or all(s.get("source_type") == "TIMELINE"
                            for s in unit.sentence_objects))

    if heading_label:
        if is_timeline:
            parts.append(f"AGE PERIOD: {heading_label}")
        else:
            parts.append(f"SECTION: {heading_label}")
        parts.append("")

    paragraph = _to_paragraph(unit.sentences, heading_label, is_timeline)
    parts.append("CLINICAL NOTES:")
    parts.append(paragraph)
    parts.append("")

    parts.append("Extract all distinct clinical events from the notes above.")
    parts.append("")
    parts.append("RULES:")
    parts.append("- Return EXACTLY ONE comprehensive object per clinical section heading.")
    parts.append("  Do NOT break down a narrative paragraph into individual sentence lines.")
    parts.append("")
    parts.append("- SYNTHESIZE COMPOUND EVENTS: Merge all sequential actions, symptoms, and")
    parts.append("  consequences within the section into a single, cohesive clinical summary string.")
    parts.append("  WRONG (Sentence Splitting):")
    parts.append("    Object 1: 'Father lost his job.'")
    parts.append("    Object 2: 'Financial insecurity entered the family environment.'")
    parts.append("    Object 3: 'The house they lived in was disputed.'")
    parts.append("  RIGHT (Consolidated Compound Event):")
    parts.append("    'Family experienced severe financial insecurity and housing instability following the father losing his job and a dispute over their home.'")
    parts.append("")
    parts.append("- FOLD IN: Context, consequences, or supporting details fold into the main event.")
    parts.append("  WRONG: separate object for 'Employer showed leniency'")
    parts.append("  RIGHT: fold into — 'Client stole from employer; employer showed leniency'")
    parts.append("")
    parts.append("- EXCEPTION: Only extract multiple objects if a section explicitly contains")
    parts.append("  completely disconnected incidents separated by years or completely different perpetrators.")
    parts.append("")
    parts.append("- OMIT clinical labels and significance markers — these are therapist annotations,")
    parts.append("  not patient events. Do NOT extract phrases like:")
    parts.append("  'Identified as one of the earliest emotionally significant experiences'")
    parts.append("  'This is considered a core wound'  'Noted as a major turning point'")
    parts.append("  If the note is purely a label or severity marker with no patient content, skip it.")
    parts.append("")
    parts.append("- OMIT positive/protective facts with no clinical bearing.")
    parts.append("  Do NOT extract: 'Client is close to mother and brother'")
    parts.append("                  'Client has a supportive relationship with X'")
    parts.append("  These are not adverse experiences and carry no trauma signal.")
    parts.append("  Extract them ONLY if the context makes them clinically relevant")
    parts.append("  (e.g. 'only support system is X — all other relationships broken').")
    parts.append("")
    parts.append("- OMIT notes with no clinical content entirely. No null objects.")
    parts.append("")
    parts.append('Return {"events": [...]} where each element has this schema:')
    parts.append("{")
    parts.append('  "source_sentence": "the relevant text from CLINICAL NOTES above",')
    parts.append('  "experience_type": "event" | "pattern" | "symptom" | "belief",')
    parts.append('  "event": "one plain English sentence describing the clinical experience",')
    parts.append('  "persons_involved": ["AGENT first, then others"] or null,')
    parts.append('  "life_stage": "childhood" | "teenage" | "adult" | "mid-life" | "unknown",')
    parts.append('  "frequency": "single" | "recurring" | "repeated" | "chronic" | "unknown",')
    parts.append('  "confidence": "HIGH" | "MEDIUM" | "LOW",')
    parts.append('  "requires_new_state": false,')
    parts.append('  "new_state_description": null')
    parts.append("}")
    parts.append("")
    parts.append("- experience_type: event=concrete incident, pattern=ongoing dynamic,")
    parts.append("  symptom=psychological/physical state, belief=core belief statement")
    parts.append("- requires_new_state: true only if fits no standard ACE/trauma category")
    parts.append("Return ONLY valid JSON. No prose, no markdown, no explanation.")
    return "\n".join(parts)


RETRY_NUDGE = """

REMINDER — re-read before answering:
- Short 2-3 word sentences ARE extractable: "Emotions suppressed." → symptom, "Abandonment feelings." → symptom
- Psychological states, feelings, and relational patterns ARE clinically significant — extract them
- DO NOT return null just because a sentence is short, emotional, or lacks a physical action
- Only return null for pure structural labels, admin, or positive-only descriptors
"""


# ─────────────────────────────────────────────────────────────
# 5.  LLM CLIENT  — single model only
# ─────────────────────────────────────────────────────────────

def is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))


def build_payload(model_name, system_msg, user_msg,
                  max_output_tokens=2000, reasoning_effort=DEFAULT_REASONING_EFFORT):
    """
    Build a Chat Completions payload valid for both classic chat models
    and GPT-5-family reasoning models (which reject temperature/max_tokens).
    """
    payload = {
        "model": model_name,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
    }
    if is_reasoning_model(model_name):
        payload["max_completion_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = max_output_tokens
    return payload


def extract_usage(data: dict) -> dict:
    """Surface hidden reasoning tokens so accounting is honest."""
    usage   = data.get("usage", {}) or {}
    details = usage.get("completion_tokens_details", {}) or {}
    return {
        "total_tokens":      usage.get("total_tokens", 0),
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens":  details.get("reasoning_tokens", 0),
    }


async def call_extraction_model(
    session:     aiohttp.ClientSession,
    unit:        ProcessingUnit,
    api_key:     str,
    max_retries: int = 5,
    is_retry:    bool = False,
) -> UnitLLMResult:
    """Call the single extraction model (MODEL_NAME)."""
    start      = time.time()
    headers    = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_msg = (SYSTEM_PROMPT + RETRY_NUDGE) if is_retry else SYSTEM_PROMPT
    payload    = build_payload(MODEL_NAME, system_msg, build_prompt_user(unit))

    for attempt in range(max_retries):
        try:
            async with session.post(
                OPENAI_API_URL, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 429:
                    wait = min(2 ** (attempt + 2), 60)
                    logging.warning(f"{MODEL_NAME} rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    text = await resp.text()
                    logging.error(f"{MODEL_NAME} HTTP {resp.status}: {text[:300]}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return UnitLLMResult(unit.unit_id, MODEL_NAME, [], "API_ERROR",
                                        f"HTTP {resp.status}: {text[:300]}")
                data    = await resp.json()
                raw     = data["choices"][0]["message"]["content"]
                usage   = extract_usage(data)
                latency = int((time.time() - start) * 1000)
                extractions, status = parse_llm_response(raw, unit)
                return UnitLLMResult(unit.unit_id, MODEL_NAME,
                                     extractions, status, raw,
                                     usage["total_tokens"], latency,
                                     usage["reasoning_tokens"])

        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                return UnitLLMResult(unit.unit_id, MODEL_NAME, [], "API_ERROR", "TIMEOUT")
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            logging.error(f"{MODEL_NAME} connection error: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                return UnitLLMResult(unit.unit_id, MODEL_NAME, [], "API_ERROR", str(e))
            await asyncio.sleep(2 ** attempt)

    return UnitLLMResult(unit.unit_id, MODEL_NAME, [], "API_ERROR", "MAX_RETRIES")


# ─────────────────────────────────────────────────────────────
# 6.  RESPONSE PARSER  (operates on one response)
# ─────────────────────────────────────────────────────────────

def parse_llm_response(raw: str, unit: ProcessingUnit) -> tuple:
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$',       '', text, flags=re.MULTILINE)
    text = text.strip()

    parsed_list = None
    try:
        outer = json.loads(text)
        if isinstance(outer, list):
            parsed_list = outer
        elif isinstance(outer, dict):
            for key in ("events", "extractions", "results", "items", "data"):
                if key in outer and isinstance(outer[key], list):
                    parsed_list = outer[key]
                    break
            if parsed_list is None:
                for v in outer.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        parsed_list = v
                        break
            if parsed_list is None:
                if "event_present" in outer:
                    parsed_list = [outer]
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                parsed_list = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if parsed_list is None:
        logging.debug(f"No parseable JSON found for {unit.unit_id}: {text[:150]}")
        return [], "PARSE_ERROR"

    if not isinstance(parsed_list, list):
        return [], "INVALID_SCHEMA"

    extractions = []
    context_text = " ".join(unit.context_before + unit.context_after).lower()

    for idx, item in enumerate(parsed_list):
        if not isinstance(item, dict):
            continue
        if item.get("event_present") is False:
            continue

        event = item.get("event")
        if not event or not str(event).strip():
            continue
        confidence         = str(item.get("confidence", "MEDIUM")).upper()
        life_stage         = str(item.get("life_stage", "unknown")).lower().replace("–", "-")
        frequency          = str(item.get("frequency", "unknown")).lower()
        experience_type    = str(item.get("experience_type") or "unknown").lower()
        requires_new_state = bool(item.get("requires_new_state", False))
        new_state_desc     = item.get("new_state_description")

        raw_indices = item.get("sentence_indices")
        if not raw_indices:
            si = item.get("sentence_index")
            raw_indices = [si] if si is not None else []
        indices_0 = [
            int(x) - 1 for x in raw_indices
            if x is not None and 0 < int(x) <= len(unit.sentences)
        ]

        valid_exp_types = {"event", "pattern", "symptom", "belief", "unknown"}
        if experience_type not in valid_exp_types:
            experience_type = "unknown"

        if confidence not in VALID_CONFIDENCES:
            confidence = "MEDIUM"
        if life_stage not in VALID_LIFE_STAGES:
            if "child"  in life_stage: life_stage = "childhood"
            elif "teen" in life_stage: life_stage = "teenage"
            elif "adult"in life_stage: life_stage = "adult"
            elif "mid"  in life_stage: life_stage = "mid-life"
            else:                      life_stage = "unknown"
        if frequency not in VALID_FREQUENCIES:
            if "chron"   in frequency: frequency = "chronic"
            elif "repea" in frequency: frequency = "repeated"
            elif "recur" in frequency: frequency = "recurring"
            elif "singl" in frequency: frequency = "single"
            else:                      frequency = "unknown"

        if indices_0:
            source = " | ".join(
                unit.sentences[i] for i in indices_0 if i < len(unit.sentences)
            ).strip()
        else:
            source = str(item.get("source_sentence", "")).strip()
        if not source and idx < len(unit.sentences):
            source = unit.sentences[idx]

        possible_hallucination = False
        if event:
            possible_hallucination = _light_hallucination_check(
                event, source, context_text, experience_type
            )

        extractions.append(LLMExtraction(
            source_sentence=source,
            event_present=True,
            event=event,
            experience_type=experience_type,
            persons_involved=item.get("persons_involved") or [],
            life_stage=life_stage,
            frequency=frequency,
            confidence=confidence,
            null_reason=None,
            requires_new_state=requires_new_state,
            new_state_description=new_state_desc if requires_new_state else None,
            possible_hallucination=possible_hallucination,
        ))

    return extractions, "SUCCESS"


def _normalise_person(p: str) -> str:
    return PERSON_REFERENCE_NORMALISE.get(p, p)


def _person_in_text(person: str, text: str) -> bool:
    norm = _normalise_person(person)
    if "-" in norm:
        return norm in text
    return bool(re.search(r'\b' + re.escape(person) + r'\b', text))


def _source_anchors_person(person: str, source: str, context: str) -> bool:
    person_lower = person.lower()
    all_text     = (source + " " + context).lower()
    for trigger, anchored_persons in PERSON_ANCHOR_EXPANSIONS.items():
        if trigger in all_text:
            if person_lower in anchored_persons:
                return True
    return False


def _light_hallucination_check(
    event:           str,
    source:          str,
    context_lower:   str,
    experience_type: str = "event",
) -> bool:
    if (experience_type or "event").lower() == "belief":
        return False
    event_lower  = event.lower()
    source_lower = source.lower()
    all_text     = source_lower + " " + context_lower
    for person in PERSON_REFERENCES:
        if _person_in_text(person, event_lower):
            if not _person_in_text(person, all_text):
                if not _source_anchors_person(person, source.lower(), context_lower):
                    return True
    return False


# ─────────────────────────────────────────────────────────────
# 7.  SEMANTIC SIMILARITY  (event-vs-source self-groundedness)
# ─────────────────────────────────────────────────────────────

_similarity_backend = None
_similarity_model   = None
_nli_model          = None
_nli_available      = None

NLI_COSINE_LOWER  = 0.50
NLI_COSINE_UPPER  = 0.82
NLI_JACCARD_LOWER = 0.10
NLI_JACCARD_UPPER = 0.30
NLI_ENTAILMENT_MIN = 0.55


def _load_nli_model() -> bool:
    global _nli_model, _nli_available
    if _nli_available is not None:
        return _nli_available
    try:
        from sentence_transformers import CrossEncoder
        _nli_model     = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        _nli_available = True
        logging.info("NLI Layer 2: cross-encoder/nli-MiniLM2-L6-H768 loaded")
    except Exception as e:
        _nli_available = False
        logging.info(f"NLI Layer 2 unavailable ({e}) — using cosine/Jaccard only")
    return _nli_available


def _nli_entailment_score(premise: str, hypothesis: str) -> float:
    """
    Entailment probability that `premise` entails `hypothesis`.
    (Callers pass premise=source, hypothesis=event.)
    """
    if not _nli_available or _nli_model is None:
        return 0.0
    try:
        import numpy as np
        scores     = _nli_model.predict([(premise, hypothesis)])
        exp_scores = np.exp(scores[0] - scores[0].max())
        probs      = exp_scores / exp_scores.sum()
        return float(probs[1])   # index 1 = entailment
    except Exception as e:
        logging.debug(f"NLI inference error: {e}")
        return 0.0


def compute_semantic_similarity(text_a: str, text_b: str) -> tuple:
    """
    Two-layer similarity between text_a and text_b.
    Layer 1 — cosine (sentence-transformers) or Jaccard fallback (symmetric).
    Layer 2 — NLI entailment (directional): text_a treated as premise,
              text_b as hypothesis. Callers that need directionality must
              pass (premise, hypothesis) in that order.
    Returns (score, backend).
    """
    global _similarity_backend, _similarity_model

    cosine_score  = None
    using_jaccard = False

    if _similarity_backend is None:
        try:
            from sentence_transformers import SentenceTransformer
            _similarity_model   = SentenceTransformer("all-MiniLM-L6-v2")
            _similarity_backend = "transformers"
            logging.info("Semantic similarity: sentence-transformers loaded (cosine + NLI)")
        except Exception:
            _similarity_backend = "jaccard"
            logging.info("Semantic similarity: sentence-transformers unavailable, Jaccard + NLI")

    if _similarity_backend == "transformers":
        try:
            from sentence_transformers import util as st_util
            ea = _similarity_model.encode(text_a, convert_to_tensor=True)
            eb = _similarity_model.encode(text_b, convert_to_tensor=True)
            cosine_score = float(st_util.cos_sim(ea, eb))
        except Exception:
            _similarity_backend = "jaccard"

    if cosine_score is None:
        using_jaccard = True
        CLINICAL_FAMILIES = {
            "beat":       {"hit","struck","abused","slapped","punched","hurt","beat","beating"},
            "hit":        {"hit","beat","struck","slapped","punched","hurt","abuse"},
            "died":       {"died","death","passed","deceased","lost","gone","passing"},
            "rejected":   {"rejected","rejection","dismissed","excluded","refused","mocked","teased"},
            "bullied":    {"bullied","bullying","mocked","teased","humiliated","excluded","harassed"},
            "relocated":  {"relocated","moved","transferred","migration","move","sent"},
            "discovered": {"discovered","found","learned","revealed","uncovered","disclosed"},
            "married":    {"married","marriage","wedding","wed"},
            "arrested":   {"arrested","custody","detained","police","legal"},
            "diagnosed":  {"diagnosed","diagnosis","identified","detected","condition"},
        }
        def expand(words):
            expanded = set(words)
            for w in list(words):
                for fam in CLINICAL_FAMILIES.values():
                    if w in fam:
                        expanded |= fam
            return expanded
        stopwords = {"the","a","an","is","was","were","had","has","have","of","in",
                     "to","and","or","for","with","at","by","from","on","that","this",
                     "it","as","be","been","being","do","did","does","not","but","so",
                     "client","her","his","their","him","she","he","they","who","which"}
        wa = expand(set(re.findall(r'\b\w+\b', text_a.lower())) - stopwords)
        wb = expand(set(re.findall(r'\b\w+\b', text_b.lower())) - stopwords)
        cosine_score = (len(wa & wb) / len(wa | wb)) if (wa and wb) else 0.0

    nli_lower = NLI_JACCARD_LOWER if using_jaccard else NLI_COSINE_LOWER
    nli_upper = NLI_JACCARD_UPPER if using_jaccard else NLI_COSINE_UPPER

    if nli_lower <= cosine_score < nli_upper:
        _load_nli_model()
        if _nli_available:
            entailment = _nli_entailment_score(text_a, text_b)  # premise=text_a
            if entailment >= NLI_ENTAILMENT_MIN:
                upgraded_score = max(cosine_score, nli_upper + 0.01)
                backend = "jaccard_nli_upg" if using_jaccard else "nli_upgrade"
                return upgraded_score, backend
            else:
                return cosine_score, "jaccard" if using_jaccard else "nli_no_upgrade"

    return cosine_score, "jaccard" if using_jaccard else "transformers"


def is_agreed(score: float, backend: str) -> bool:
    if backend in ("nli_upgrade", "jaccard_nli_upg"):
        return True
    if backend == "nli_no_upgrade":
        return score >= COSINE_THRESHOLD
    if backend == "transformers":
        return score >= COSINE_THRESHOLD
    return score >= JACCARD_THRESHOLD


# ─────────────────────────────────────────────────────────────
# 8.  STAGE 3 — FOUR-AXIS VERIFICATION ENGINE
# ─────────────────────────────────────────────────────────────

def detect_cultural_note(event: str, source: str) -> Optional[str]:
    WORD_BOUNDARY_TERMS = {
        "mil", "fil", "sil", "pind", "izzat",
        "kaka", "mama", "bua", "nani", "dadi", "nana", "dada",
    }
    combined = (event + " " + source).lower()
    notes = []
    for term, explanation in INDIAN_CULTURAL_GLOSSARY.items():
        if term in WORD_BOUNDARY_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', combined):
                notes.append(f"'{term}': {explanation}")
        else:
            if term in combined:
                notes.append(f"'{term}': {explanation}")
    return "; ".join(notes) if notes else None


def full_hallucination_check(
    event:              str,
    source:             str,
    context_sentences:  list,
    experience_type:    str = "event",
) -> list:
    """
    AXIS 4 — GROUNDEDNESS. Returns list of flag strings (empty = clean).
    The action anchor fires on ANY asserted variant, not only the
    canonical key word.
    """
    flags        = []
    event_lower  = event.lower()
    source_lower = source.lower()
    ctx_lower    = " ".join(context_sentences).lower()
    all_text     = source_lower + " " + ctx_lower

    exp = experience_type.lower() if experience_type else "event"

    if exp == "belief":
        return []

    for person in PERSON_REFERENCES:
        if _person_in_text(person, event_lower):
            if not _person_in_text(person, all_text):
                if not _source_anchors_person(person, source_lower, ctx_lower):
                    flags.append(f"PERSON_NOT_IN_SOURCE: '{person}'")

    if exp in ("event", "unknown"):
        for canonical, synonyms in ACTION_SYNONYMS.items():
            variants = [canonical] + synonyms
            asserted = [v for v in variants
                        if re.search(r'\b' + re.escape(v) + r'\b', event_lower)]
            if not asserted:
                continue
            if not any(re.search(r'\b' + re.escape(v) + r'\b', all_text)
                       for v in variants):
                flags.append(
                    f"ACTION_NOT_ANCHORED: '{canonical}' (event says '{asserted[0]}')")

    if exp == "symptom":
        for label in CLINICAL_LABELS:
            if re.search(r'\b' + re.escape(label) + r'\b', event_lower):
                if not re.search(r'\b' + re.escape(label) + r'\b', all_text):
                    flags.append(f"CLINICAL_LABEL_NOT_IN_SOURCE: '{label}'")

    return flags


_EXPERIENCE_TYPE_CUES = {
    "event":   re.compile(
        r'\b(was born|died|beat|hit|struck|married|divorced|arrested|'
        r'diagnosed|sent to|moved to|discovered|disclosed|gave birth|'
        r'took custody|admitted to|hospitalised|hospitalized)\b'),
    "pattern": re.compile(
        r'\b(repeatedly|regularly|consistently|always|never|chronically|'
        r'throughout|ongoing|habitually|routinely|every time|ever since|'
        r'ever\b|frequently|often)\b'),
    "symptom": re.compile(
        r'\b(feels?|felt|experiences?|experienced|suffers?|suffered|'
        r'anxiety|anxious|depress|hypervigilan|restless|insomnia|'
        r'triggered|suppress|lonely|unseen|numb|stuck)\b'),
    "belief":  re.compile(
        r'\b(believes?|belief|core wound|not enough|i am\b|must always|'
        r"i must|convinced|feels she must|feels he must)\b"),
}


def check_construct_correctness(extraction) -> tuple:
    """
    AXIS 1 — CONSTRUCT. Returns (passed: bool, flags: list[str], self_score).
    Self-entailment is directional (source premise -> event hypothesis).
    """
    flags = []
    exp = (extraction.experience_type or "unknown").lower()
    combined = (extraction.event + " " + extraction.source_sentence).lower()

    cue_re = _EXPERIENCE_TYPE_CUES.get(exp)
    if cue_re is not None and not cue_re.search(combined):
        flags.append(
            f"CONSTRUCT_CUE_WEAK: no '{exp}'-type marker found in event/source text")

    self_score = None
    if extraction.source_sentence and extraction.event:
        # source is the premise, event is the hypothesis.
        self_score, backend = compute_semantic_similarity(
            extraction.source_sentence, extraction.event)
        if not is_agreed(self_score, backend):
            flags.append(
                f"CONSTRUCT_NOT_ENTAILED: event (similarity={self_score:.2f}, "
                f"backend={backend}) not entailed by its cited source_sentence")

    hard_fail = any(f.startswith("CONSTRUCT_NOT_ENTAILED") for f in flags)
    return (not hard_fail), flags, self_score


def _regex_actor_check(event: str, source: str, persons_involved: list) -> tuple:
    """
    AXIS 2 — ACTOR/ROLE (heuristic). Only a recognised person reference
    counts as the parsed agent, so adverbs/qualifiers before the verb
    don't false-flag. Fails open on ambiguity. Checks source_sentence only.
    """
    if not persons_involved or len(persons_involved) < 2:
        return True, []

    claimed_agent = str(persons_involved[0]).lower().strip()
    event_lower   = event.lower()
    source_lower  = source.lower()
    flags = []
    person_set = set(PERSON_REFERENCES)

    for canonical, synonyms in ACTION_SYNONYMS.items():
        variants = [canonical] + synonyms
        if not any(re.search(r'\b' + re.escape(v) + r'\b', event_lower) for v in variants):
            continue

        for v in variants:
            v_re = re.escape(v)
            passive = re.search(
                rf'\bwas\s+{v_re}\s+by\s+(?:her|his|the\s+)?([a-z\-]+)', source_lower)
            if passive:
                found_agent = passive.group(1).strip()
                if found_agent in person_set \
                        and found_agent not in claimed_agent \
                        and claimed_agent not in found_agent:
                    flags.append(
                        f"ACTOR_ROLE_MISMATCH: source attributes '{v}' (passive) to "
                        f"'{found_agent}', but persons_involved lists '{claimed_agent}' first")
                break

            active = re.search(rf'\b([a-z\-]+)\s+{v_re}\b', source_lower)
            if active:
                found_agent = active.group(1).strip()
                if found_agent in person_set \
                        and found_agent not in claimed_agent \
                        and claimed_agent not in found_agent:
                    flags.append(
                        f"ACTOR_ROLE_MISMATCH: source attributes '{v}' (active) to "
                        f"'{found_agent}', but persons_involved lists '{claimed_agent}' first")
                break

    return (len(flags) == 0), flags


def check_actor_role(extraction) -> tuple:
    """AXIS 2 wrapper — see _regex_actor_check for the heuristic limitations."""
    return _regex_actor_check(
        extraction.event, extraction.source_sentence,
        extraction.persons_involved or []
    )


_EXPLICIT_AGE_RE = re.compile(
    r'\bage[d]?\s*(\d{1,2})\b|\bat\s+(?:age\s+)?(\d{1,2})\b|\b(\d{1,2})\s*years?\s*old\b',
    re.IGNORECASE,
)
_AGE_RANGE_RE = re.compile(r'(\d{1,2})\s*[-–to]+\s*(\d{1,2})', re.IGNORECASE)


def _parse_ages_from_text(text: str) -> list:
    ages = []
    for m in _AGE_RANGE_RE.finditer(text):
        try:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 <= a <= 99 and 0 <= b <= 99:
                ages.extend([a, b])
        except ValueError:
            continue
    for m in _EXPLICIT_AGE_RE.finditer(text):
        for g in m.groups():
            if g:
                try:
                    n = int(g)
                    if 0 <= n <= 99:
                        ages.append(n)
                except ValueError:
                    continue
    return sorted(set(ages))   # dedup boundary ages


def check_age_window(extraction, unit: ProcessingUnit) -> tuple:
    """
    AXIS 3 — AGE WINDOW. Contradiction-detector between claimed life_stage
    and available age evidence. Fails open when no numeric evidence exists.
    Instrument-specific 0-18 ACE windowing belongs to stage (iv), not here.
    """
    flags = []
    claimed = (extraction.life_stage or "unknown").lower()
    claimed_range = LIFE_STAGE_AGE_RANGES.get(claimed)

    if claimed_range is None:
        return True, []

    candidate_ages = []
    if unit.age_context:
        candidate_ages.extend(_parse_ages_from_text(unit.age_context))
    candidate_ages.extend(_parse_ages_from_text(extraction.source_sentence))

    if not candidate_ages:
        return True, []

    lo, hi = claimed_range
    contradicting = [a for a in candidate_ages if a < lo - 2 or a > hi + 2]

    if contradicting and len(contradicting) == len(candidate_ages):
        flags.append(
            f"AGE_WINDOW_MISMATCH: life_stage='{claimed}' (expected ~{lo}-{hi}) "
            f"but source/age_context implies age(s) {sorted(set(contradicting))}")
        return False, flags

    return True, []


# ── machine-readable flags → E1..E6 taxonomy ──────────────────────
_FLAG_TO_MODE = {
    "CONSTRUCT_NOT_ENTAILED":       ("construct",    "E1"),
    "CONSTRUCT_CUE_WEAK":           ("construct",    "E1"),
    "CLINICAL_LABEL_NOT_IN_SOURCE": ("groundedness", "E1"),
    "ACTOR_ROLE_MISMATCH":          ("actor_role",   "E3"),
    "PERSON_NOT_IN_SOURCE":         ("groundedness", "E3"),
    "ACTION_NOT_ANCHORED":          ("groundedness", "E4"),
    "AGE_WINDOW_MISMATCH":          ("age_window",   "E5"),
    "MODEL_STATUS":                 ("extraction",   "NA"),
}


def structure_flags(flags) -> list:
    out = []
    for f in flags or []:
        code   = f.split(":", 1)[0].strip()
        detail = f.split(":", 1)[1].strip() if ":" in f else ""
        axis, mode = _FLAG_TO_MODE.get(code, ("unknown", "NA"))
        out.append({"axis": axis, "code": code, "error_mode": mode, "detail": detail})
    return out


def verify_extraction(unit: ProcessingUnit, extraction) -> ValidatedEvent:
    """Run all four axis checks on a single extraction and assemble a ValidatedEvent."""
    subsection_ctx = [unit.subsection] if unit.subsection else []
    ctx_all_list   = subsection_ctx + unit.context_before + unit.context_after

    construct_passed, construct_flags, self_score = check_construct_correctness(extraction)
    actor_passed, actor_flags = check_actor_role(extraction)
    age_passed, age_flags = check_age_window(extraction, unit)
    grounded_flags = full_hallucination_check(
        extraction.event, extraction.source_sentence, ctx_all_list,
        extraction.experience_type)
    grounded_passed = len(grounded_flags) == 0

    all_flags   = construct_flags + actor_flags + age_flags + grounded_flags
    axes_passed = sum([construct_passed, actor_passed, age_passed, grounded_passed])

    status, base_conf = AXIS_CONFIDENCE_MATRIX.get(
        (unit.stage1_confidence, axes_passed), AXIS_CONFIDENCE_DEFAULT)

    if self_score is not None:
        base_conf = min(1.0, base_conf * (0.85 + 0.15 * min(self_score, 1.0)))

    human_review = (
        status in ("REVIEW_REQUIRED",)
        or axes_passed < 4
        or bool(all_flags)
    )
    verified = (axes_passed == 4 and status == "CONFIRMED" and not all_flags)

    cultural_note = detect_cultural_note(extraction.event, extraction.source_sentence)

    return ValidatedEvent(
        unit_id=unit.unit_id,
        source_sentence=extraction.source_sentence,
        event=extraction.event,
        experience_type=extraction.experience_type,
        persons_involved=extraction.persons_involved or [],
        life_stage=extraction.life_stage,
        frequency=extraction.frequency,
        confidence=round(base_conf, 3),
        status=status,
        verified=verified,
        human_review_required=human_review,
        cultural_note=cultural_note,
        stage1_confidence=unit.stage1_confidence,
        line_number=unit.line_number,
        age_context=unit.age_context,
        subsection=unit.subsection,
        construct_check_passed=construct_passed,
        actor_role_check_passed=actor_passed,
        age_window_check_passed=age_passed,
        groundedness_check_passed=grounded_passed,
        axes_passed=axes_passed,
        verification_flags=all_flags,
        self_groundedness_score=(round(self_score, 3) if self_score is not None else None),
        requires_new_state=extraction.requires_new_state,
        new_state_description=extraction.new_state_description,
    )


def verify_unit(unit: ProcessingUnit, result: UnitLLMResult) -> list:
    validated = []

    if result.status != "SUCCESS":
        validated.append(ValidatedEvent(
            unit_id=unit.unit_id,
            source_sentence=unit.sentences[0] if unit.sentences else "",
            event="", experience_type="unknown",
            persons_involved=[], life_stage="unknown",
            frequency="unknown", confidence=0.0,
            status="EXTRACTION_FAILED", verified=False,
            human_review_required=True, cultural_note=None,
            stage1_confidence=unit.stage1_confidence,
            line_number=unit.line_number, age_context=unit.age_context,
            subsection=unit.subsection,
            construct_check_passed=False, actor_role_check_passed=False,
            age_window_check_passed=False, groundedness_check_passed=False,
            axes_passed=0, verification_flags=[f"MODEL_STATUS: {result.status}"],
            self_groundedness_score=None,
        ))
        return validated

    if not result.extractions:
        if unit.stage1_confidence == "HIGH":
            validated.append(ValidatedEvent(
                unit_id=unit.unit_id,
                source_sentence=" | ".join(unit.sentences[:3]),
                event="", experience_type="unknown",
                persons_involved=[], life_stage="unknown",
                frequency="unknown", confidence=0.0,
                status="NO_EVENT", verified=True,
                human_review_required=True, cultural_note=None,
                stage1_confidence=unit.stage1_confidence,
                line_number=unit.line_number, age_context=unit.age_context,
                subsection=unit.subsection,
                construct_check_passed=True, actor_role_check_passed=True,
                age_window_check_passed=True, groundedness_check_passed=True,
                axes_passed=4, verification_flags=[],
                self_groundedness_score=None,
            ))
        return validated

    for extraction in result.extractions:
        validated.append(verify_extraction(unit, extraction))

    return validated


# ─────────────────────────────────────────────────────────────
# 8A.  REPRODUCIBILITY MANIFEST + ATOMIC WRITE
# ─────────────────────────────────────────────────────────────

def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_run_manifest() -> dict:
    return {
        "model":                MODEL_NAME,
        "verification_method":  "four_axis_deterministic",
        "system_prompt_sha256": _sha(SYSTEM_PROMPT),
        "retry_nudge_sha256":   _sha(RETRY_NUDGE),
        "thresholds": {
            "COSINE_THRESHOLD":   COSINE_THRESHOLD,
            "JACCARD_THRESHOLD":  JACCARD_THRESHOLD,
            "NLI_ENTAILMENT_MIN": NLI_ENTAILMENT_MIN,
            "NLI_COSINE_LOWER":   NLI_COSINE_LOWER,
            "NLI_COSINE_UPPER":   NLI_COSINE_UPPER,
        },
        "similarity_backend":   _similarity_backend,
        "nli_available":        _nli_available,
        "generated_utc":        datetime.now(timezone.utc).isoformat(),
    }


def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ─────────────────────────────────────────────────────────────
# 9.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

async def run_pipeline(
    units:       list,
    api_key:     str,
    concurrency: int = 5,
    resume_set:  set = None,
) -> tuple:
    if resume_set is None:
        resume_set = set()

    pending = [u for u in units if u.unit_id not in resume_set]
    skipped = len(units) - len(pending)
    if skipped:
        print(f"[RESUME] Skipping {skipped} already-processed units")

    # NOTE: estimate only — a single per-token rate ignores the input/output
    # split and reasoning-token billing. Do NOT quote this figure anywhere.
    MODEL_COST_PER_TOKEN = 0.0000030

    all_results: dict = {}
    total_tokens     = 0
    total_reasoning  = 0
    total_cost       = 0.0

    semaphore = asyncio.Semaphore(concurrency)

    async def process_unit(session: aiohttp.ClientSession, unit: ProcessingUnit):
        async with semaphore:
            res = await call_extraction_model(session, unit, api_key)
            if unit.stage1_confidence == "HIGH":
                all_null  = res.status == "SUCCESS" and len(res.extractions) == 0
                api_error = res.status == "API_ERROR"
                if all_null or api_error:
                    reason = "all-null" if all_null else "API_ERROR"
                    logging.info(f"Retry ({reason} on HIGH unit): {unit.unit_id}")
                    if api_error:
                        await asyncio.sleep(5)
                    res = await call_extraction_model(
                        session, unit, api_key,
                        max_retries=5, is_retry=all_null,
                    )
            return unit.unit_id, res

    print(f"\n[STAGE 2] {len(pending)} units × 1 model ({MODEL_NAME})  concurrency={concurrency}\n")

    async with aiohttp.ClientSession() as session:
        tasks     = [process_unit(session, u) for u in pending]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            uid, res = await coro
            all_results[uid] = res
            completed       += 1
            total_tokens    += res.tokens_used
            total_reasoning += res.reasoning_tokens
            total_cost      += res.tokens_used * MODEL_COST_PER_TOKEN
            s = "✓" if res.status == "SUCCESS" else f"✗({res.status})"
            print(f"  [{completed:3d}/{len(pending)}] {uid}  {MODEL_NAME}:{s}({res.latency_ms}ms)")

    print(f"\n[STAGE 2] Done. tokens={total_tokens:,} "
          f"(reasoning={total_reasoning:,})  est_cost=${total_cost:.4f}")

    print("\n[STAGE 3] Four-axis verification...")
    all_validated = []
    audit_log     = []
    unit_map      = {u.unit_id: u for u in units}

    for uid, unit in unit_map.items():
        res = all_results.get(uid)
        if res is None:
            continue
        events = verify_unit(unit, res)
        all_validated.extend(events)
        audit_log.append({
            "unit_id":         uid,
            "model_status":    res.status,
            "tokens":          res.tokens_used,
            "reasoning_tokens": res.reasoning_tokens,
            "latency_ms":      res.latency_ms,
            "raw_response":    res.raw_response[:500] if res.raw_response else "",
            "n_events":        len(events),
            "statuses":        [e.status for e in events],
            "axes_passed":     [e.axes_passed for e in events],
        })

    from collections import Counter
    status_counts = Counter(v.status for v in all_validated)
    print(f"\n[STAGE 3] Done. {len(all_validated)} events processed:")
    for s, n in sorted(status_counts.items()):
        print(f"  {s}: {n}")

    final_counts = Counter(v.status for v in all_validated)
    n_passing    = sum(final_counts.get(k, 0) for k in ["CONFIRMED", "TENTATIVE", "REVIEW_REQUIRED"])
    n_review     = sum(1 for v in all_validated if v.human_review_required)

    # error-mode tally straight from structured flags
    mode_counts = Counter()
    for v in all_validated:
        for sf in v.verification_flags_structured:
            if sf["error_mode"] != "NA":
                mode_counts[sf["error_mode"]] += 1

    print(f"\n[SUMMARY] Passing to Worker 2: {n_passing}")
    print(f"[SUMMARY] Human review queue: {n_review}")
    if mode_counts:
        print(f"[SUMMARY] Flagged error modes: {dict(sorted(mode_counts.items()))}")

    stats = {
        "total_units":        len(units),
        "pending_units":      len(pending),
        "total_events":       len(all_validated),
        "status_counts":      dict(final_counts),
        "error_mode_counts":  dict(mode_counts),
        "tokens_used":        total_tokens,
        "reasoning_tokens":   total_reasoning,
        "cost_usd_estimate":  round(total_cost, 5),
        "human_review_count": n_review,
        "model":              MODEL_NAME,
    }

    return all_validated, stats, audit_log


# ─────────────────────────────────────────────────────────────
# 10.  DRY RUN
# ─────────────────────────────────────────────────────────────

def dry_run(units: list, n: int = 5) -> None:
    print(f"\n{'='*65}")
    print(f"DRY RUN — {len(units)} total units, showing first {n}")
    print(f"{'='*65}\n")
    for unit in units[:n]:
        print(f"UNIT: {unit.unit_id}")
        print(f"  subsection:  {repr(unit.subsection)}")
        print(f"  age_context: {unit.age_context}")
        print(f"  source_type: {unit.source_type}")
        print(f"  s1_conf:     {unit.stage1_confidence}")
        print(f"  sentences ({len(unit.sentences)}):  ctx_before={len(unit.context_before)}  ctx_after={len(unit.context_after)}")
        for s in unit.sentences:
            print(f"    [{unit.sentences.index(s)+1}] {s[:90]}")
        print()
        print("  ── PROMPT (first 25 lines) ──")
        for line in build_prompt_user(unit).split("\n")[:25]:
            print(f"  {line}")
        print("  ...")
        print()


# ─────────────────────────────────────────────────────────────
# 10A.  OFFLINE SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    p, _ = _regex_actor_check("Father emotionally beat mother",
                              "Father emotionally beat mother", ["Father", "Mother"])
    check(p, "actor: adverb before verb does not false-flag")
    p, _ = _regex_actor_check("Father repeatedly beat mother",
                              "Father repeatedly beat mother", ["Father", "Mother"])
    check(p, "actor: 'repeatedly' before verb does not false-flag")
    p, f = _regex_actor_check("Father beat mother",
                              "Mother beat the children", ["Father", "Mother"])
    check((not p) and f, "actor: genuine agent swap still flagged")

    check(full_hallucination_check("Father beat mother", "Father and mother were present.", []),
          "grounded: unanchored synonym 'beat' now flags")
    check(full_hallucination_check("Father slapped mother", "Father and mother were present.", []),
          "grounded: unanchored synonym 'slapped' now flags")
    check(not full_hallucination_check("Father beat mother", "Father struck mother.", []),
          "grounded: synonym anchored in source passes")

    rp = build_payload("gpt-5.1", "sys", "usr")
    check("temperature" not in rp and "max_completion_tokens" in rp,
          "payload: gpt-5.1 drops temperature, uses max_completion_tokens")
    cp = build_payload("gpt-4.1", "sys", "usr")
    check("temperature" in cp and "max_tokens" in cp,
          "payload: classic model keeps temperature + max_tokens")

    check(structure_flags(["ACTOR_ROLE_MISMATCH: x"])[0]["error_mode"] == "E3",
          "flags: structured flag maps to error mode")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# 11.  CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FiRE Worker 1 Stage 2+3 — Single-Model Extractor + Four-Axis Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage1_json", nargs="?",
                        help="Path to stage1_output.json")
    parser.add_argument("--patient-id",   default="patient",
                        help="Patient identifier prefix for unit IDs")
    parser.add_argument("--api-key",      default=os.getenv("OPENAI_API_KEY", ""),
                        help=f"API key for {MODEL_NAME} (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output",       default="validated_events.json",
                        help="Output file for validated events (default: validated_events.json)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print processing units and prompts without calling the API")
    parser.add_argument("--selftest",     action="store_true",
                        help="Run offline regression checks and exit")
    parser.add_argument("--concurrency",  type=int, default=5,
                        help="Parallel units per batch (default: 5)")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip units already in --output file (resume partial run)")
    parser.add_argument("--show-n",       type=int, default=5,
                        help="Number of units to show in --dry-run (default: 5)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.stage1_json:
        parser.error("stage1_json is required (unless using --selftest)")

    with open(args.stage1_json, encoding="utf-8") as f:
        stage1_data = json.load(f)
    print(f"[LOAD] {len(stage1_data)} sentences from {args.stage1_json}")

    units = group_sentences(stage1_data, args.patient_id)

    if args.dry_run:
        dry_run(units, args.show_n)
        return

    if not args.api_key:
        print(f"\nERROR: --api-key required (model: {MODEL_NAME}).")
        print("  export OPENAI_API_KEY=your_key")
        sys.exit(1)

    resume_path = args.output.replace(".json", ".resume")
    resume_set: set = set()
    if args.resume:
        if Path(resume_path).exists():
            try:
                with open(resume_path, encoding="utf-8") as f:
                    resume_set = set(json.load(f))
                print(f"[RESUME] Found {len(resume_set)} previously processed unit IDs")
                print(f"[RESUME] Reading from: {resume_path}")
            except Exception as e:
                print(f"[RESUME] Could not load resume file: {e} — starting fresh")
        else:
            print(f"[RESUME] No resume file found at {resume_path} — starting fresh")

    validated, stats, audit_log = asyncio.run(run_pipeline(
        units, args.api_key, args.concurrency, resume_set,
    ))

    new_state_candidates = [
        {
            "unit_id":         v.unit_id,
            "source_sentence": v.source_sentence,
            "event":           v.event,
            "experience_type": v.experience_type,
            "description":     v.new_state_description,
            "life_stage":      v.life_stage,
            "age_context":     v.age_context,
        }
        for v in validated
        if v.requires_new_state and v.new_state_description
    ]

    output_data = {
        "patient_id":           args.patient_id,
        "pipeline_stage":       "worker1_stage2_3_single_model",
        "model":                MODEL_NAME,
        "verification_method":  "four_axis_deterministic",
        "manifest":             build_run_manifest(),
        "statistics":           stats,
        "events":               [asdict(v) for v in validated],
        "new_state_candidates": new_state_candidates,
    }

    try:
        processed_ids = list({e["unit_id"] for e in output_data["events"]})
        atomic_write_json(resume_path, processed_ids)
    except Exception as e:
        logging.warning(f"Could not save resume file: {e}")

    atomic_write_json(args.output, output_data)
    print(f"\n[SAVE] Validated events → {args.output}")

    audit_path = args.output.replace(".json", "_audit.json")
    atomic_write_json(audit_path, audit_log)
    print(f"[SAVE] Audit log        → {audit_path}")

    if new_state_candidates:
        print(f"\n[NEW STATES] {len(new_state_candidates)} candidates flagged for state space expansion:")
        for c in new_state_candidates[:5]:
            print(f"  → [{c['experience_type']}] {c['description'][:70] if c['description'] else c['event'][:70]}")
        if len(new_state_candidates) > 5:
            print(f"  ... and {len(new_state_candidates) - 5} more (see new_state_candidates in output)")

    print(f"\n[DONE] Patient: {args.patient_id}  "
          f"Units: {stats['total_units']}  "
          f"Events: {stats['total_events']}  "
          f"Cost(est): ~${stats['cost_usd_estimate']}")


if __name__ == "__main__":
    main()
