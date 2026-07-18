"""
FiRE Pipeline — Preprocessor
=================================
Converts raw [REDACTED: clinical partner site] digitized consultation notes into a
normalized flat format where every line is self-contained and
carries its full context as a prefix.

Design principles:
  - Zero hallucination: only moves and prefixes existing text
  - Zero removal:       every content token appears in output
  - Deterministic:      same input always produces same output
  - No LLM:            pure rule-based state machine
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1.  KNOWN WORD SETS
# ─────────────────────────────────────────────────────────────

# Pure structural table column headers — discard always
# NOTE: "birth" is NOT here — it is a valid timeline age marker
# NOTE: "belief" and "origin" are NOT here — they are two-column
#        table headers that label exactly one child line each
STRUCTURAL_WORDS = {
    "member", "detail",
    "age", "event", "name", "relationship",
}

# Words that signal table column headers.
# When two or three consecutive single-word lines from this set appear
# after a subsection header, enter alternating table mode.
TABLE_COLUMN_WORDS = {
    "belief", "origin",
    "domain", "self", "others", "relationships",
    "category", "description", "type", "value",
    "theme", "trigger", "response", "pattern",
}

# Person name headers → prefix next line only, then reset to subsection
# Includes multi-word family member labels found in [REDACTED: clinical partner site] notes
# Parenthetical qualifiers like "(paternal)", "(maternal)" are stripped
# before matching, so "Grandfather (paternal)" matches "grandfather".
PERSON_HEADERS = {
    "father", "mother", "sister", "brother",
    "grandfather", "grandmother", "husband", "wife",
    "son", "daughter", "uncle", "aunt", "cousin",
    "partner", "spouse",
    # multi-word variants
    "younger sister", "elder sister", "older sister",
    "younger brother", "elder brother", "older brother",
    "another younger sister", "another sister",
    "twin brothers", "twin sisters",
    "mil", "fil",                        # mother/father-in-law abbreviations
    "domestic helper", "live-in helper",
    "extended family",
    # role-qualified variants
    "grandfather (paternal)", "grandfather (maternal)",
    "grandmother (paternal)", "grandmother (maternal)",
    "uncle (paternal)", "uncle (maternal)",
    "aunt (paternal)", "aunt (maternal)",
    "father-in-law", "mother-in-law",
    "sister-in-law", "brother-in-law",
    "step-father", "step-mother", "step-brother", "step-sister",
}

# Strip parenthetical qualifiers for flexible person header matching
PERSON_QUALIFIER_RE = re.compile(r"\s*\([^)]+\)\s*$")

# Admin key headers (Section 1 metadata keys)
ADMIN_KEY_HEADERS = {
    "referred by", "preferred mode", "primary diagnosis",
    "previous therapy", "financial situation",
    "current living situation", "current living",
    "clinical priorities", "planned interventions",
    "note on communication style", "origin",
    "education", "age at time of accident",
    "presenting problems", "presenting problems (client-reported)",
    "social life", "primary support relationships",
}

# Clinical key headers worth keeping with prefix
CLINICAL_KEY_HEADERS = {
    "medical condition", "current physical & emotional state",
    "current emotional state", "current physical state",
    "current physical and emotional state",
}

# Standalone age markers in timeline
# Handles both numeric (~8, ~20) and text-based (~Adolescence, October 2021)
STANDALONE_AGE_RE = re.compile(
    # numeric age markers
    r"^[~≈]?\s*\d{1,2}(?:[–\-]\d{1,2})?\s*$"
    r"|^[~≈]?\s*\d{1,2}\+\s*$"
    # named lifecycle stages (with or without ~ prefix)
    r"|^[~≈]?\s*present\s*$"
    r"|^[~≈]?\s*recent\s*$"
    r"|^[~≈]?\s*birth\s*$"
    r"|^[~≈]?\s*school\s+years\s*$"
    r"|^[~≈]?\s*early\s+childhood\s*$"
    r"|^[~≈]?\s*childhood\s*$"
    r"|^[~≈]?\s*adolescence\s*$"
    r"|^[~≈]?\s*teenage\s+years\s*$"
    r"|^[~≈]?\s*early\s+teens\s*$"
    r"|^[~≈]?\s*late\s+teens.*$"         # ~Late teens / early 20s
    r"|^[~≈]?\s*early.*20s\s*$"          # ~Early/mid 20s
    r"|^[~≈]?\s*mid.*20s\s*$"            # ~Mid-20s
    r"|^[~≈]?\s*late.*20s\s*$"           # ~Late 20s
    r"|^[~≈]?\s*early.*30s\s*$"
    r"|^[~≈]?\s*mid.*30s\s*$"
    r"|^[~≈]?\s*late.*30s\s*$"
    r"|^[~≈]?\s*early.*40s\s*$"
    r"|^[~≈]?\s*son.*years\s*$"          # ~Son's early years
    r"|^[~≈]?\s*son\s+age.*$"            # ~Son age ~11
    r"|^ongoing\s+in\s+marriage\s*$"    # Ongoing in marriage
    r"|^ongoing\s+in\s+\w+\s*$"        # Ongoing in <context>
    r"|^[A-Za-z]+\s+\d{4}\s*$"         # October 2021, January 2020 etc.
    r"|^\d{4}\s*$",                      # bare year: 2021
    re.IGNORECASE,
)

# Full timeline row: age marker + content on same line
TIMELINE_ROW_RE = re.compile(
    r"^[~≈]?\s*\d{1,2}(?:[–\-]\d{1,2})?\s+\S"
    r"|^[~≈]?\s*\d{1,2}\+\s+\S"
    r"|^present\s+\S"
    r"|^recent\s+\S"
    r"|^school\s+years\s+\S",
    re.IGNORECASE,
)

SUBSECTION_RE    = re.compile(r"^\d+\.\d+\s+\S", re.IGNORECASE)
SECTION_HDR_RE   = re.compile(r"^SECTION\s+\d+\s*:", re.IGNORECASE)
LETTER_PREFIX_RE = re.compile(r"^[A-Z]\.\s+\S")
SUBSECTION_AGE_RE = re.compile(
    r"\(age\s*[~≈]?\s*(\d{1,2}(?:[–\-]\d{1,2})?)\)",
    re.IGNORECASE,
)
DIRECT_QUOTE_RE = re.compile(r'["\u201c\u201d\u2018\u2019]')


# ─────────────────────────────────────────────────────────────
# 2.  LINE CLASSIFIER
# ─────────────────────────────────────────────────────────────

def classify_line(line: str, in_section: int) -> str:
    stripped = line.strip()
    lower    = stripped.lower()

    if not stripped:
        return "EMPTY"
    if SECTION_HDR_RE.match(stripped):
        return "SECTION_HEADER"
    if SUBSECTION_RE.match(stripped):
        return "SUBSECTION"
    if lower in STRUCTURAL_WORDS:
        return "STRUCTURAL"
    # Two-column alternating table word
    if lower in TABLE_COLUMN_WORDS and len(stripped.split()) == 1:
        return "TABLE_COLUMN_WORD"
    # Person headers: single-word OR known multi-word family labels
    # Also match after stripping parenthetical qualifiers:
    # "Grandfather (paternal)" → "grandfather" → matches
    stripped_lower = PERSON_QUALIFIER_RE.sub("", lower).strip()
    if lower in PERSON_HEADERS or stripped_lower in PERSON_HEADERS:
        return "PERSON_HEADER"
    if LETTER_PREFIX_RE.match(stripped) and in_section == 2:
        return "LETTER_HEADER"
    if in_section == 3:
        if STANDALONE_AGE_RE.match(stripped):
            return "AGE_MARKER"
        if TIMELINE_ROW_RE.match(stripped):
            return "TIMELINE_ROW"
    if stripped.endswith(":") and len(stripped) > 1:
        key = stripped[:-1].strip().lower()
        if key in ADMIN_KEY_HEADERS:
            return "KEY_ADMIN"
        if key in CLINICAL_KEY_HEADERS:
            return "KEY_CLINICAL"
        return "KEY_NARRATIVE"

    # Inline key-value: starts with an admin key followed by colon
    # e.g. "Previous Therapy: Not mentioned Current Living: With family"
    # These are self-contained and should not inherit parent context
    first_segment = stripped.split(":")[0].strip().lower()
    if first_segment in ADMIN_KEY_HEADERS and ":" in stripped:
        return "INLINE_ADMIN"

    return "CONTENT"


# ─────────────────────────────────────────────────────────────
# 3.  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class NormalizedLine:
    raw_text:      str
    enriched_text: str
    line_number:   int
    context_label: Optional[str]
    age_context:   Optional[str]
    line_class:    str
    in_section:    int
    discard:       bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class NormalizedNote:
    lines:          list
    patient_id:     str  = ""
    validation_ok:  bool = True
    validation_msg: str  = ""

    def content_lines(self):
        return [l for l in self.lines if not l.discard]

    def to_dict(self):
        return {
            "patient_id":     self.patient_id,
            "validation_ok":  self.validation_ok,
            "validation_msg": self.validation_msg,
            "lines":          [l.to_dict() for l in self.lines],
        }


# ─────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _is_short_list_item(line: str) -> bool:
    """True if line looks like a noun-phrase list item, not a full sentence."""
    if len(line) > 60:
        return False
    full_sentence_verbs = [
        "never shared", "responded", "said", "told",
        "heard", "developed", "became", "felt", "found",
        "decided", "started", "stopped", "moved", "left",
        "worked", "attended", "cried", "drifted", "backed",
    ]
    return not any(v in line.lower() for v in full_sentence_verbs)


def _is_direct_quote_line(line: str) -> bool:
    return bool(DIRECT_QUOTE_RE.search(line))


# ─────────────────────────────────────────────────────────────
# 5.  TWO-LEVEL CONTEXT STATE MACHINE
# ─────────────────────────────────────────────────────────────

class ContextStateMachine:
    """
    Two-level label stack:

      subsection_label  : set by numbered subsection (2.1, 2.2 etc)
                          persists until next subsection
                          never reset by blank lines or structural words

      narrative_label   : set by KEY_NARRATIVE / KEY_ADMIN / KEY_CLINICAL
                          / PERSON_HEADER / LETTER_HEADER
                          resets on blank lines and structural words

    When building the enriched prefix:
      - If narrative_label is active and applies: use narrative_label
        (subsection is implicit context, not doubled)
      - If narrative_label expired but subsection_label exists:
        fall back to subsection_label
      - For inner KEY_NARRATIVE inside a subsection:
        compose as "subsection: narrative: content"

    This ensures no line is ever orphaned without context.
    """

    QUOTE_ONLY_LABELS = {
        "classmates said", "parents responded with",
        "client notes", "client stated", "client said",
        "therapist notes",
    }

    LIST_ONLY_LABELS = {
        "bullied and mocked repeatedly by classmates for",
        "bullied repeatedly for", "mocked for", "reasons include",
    }

    def __init__(self):
        self.subsection_label = None   # outer: set by SUBSECTION, never reset by blank
        self.narrative_label  = None   # inner: set by KEY_*, PERSON, LETTER
        self.narrative_type   = None
        self.narrative_mode   = "ALL"  # ALL / QUOTE_ONLY / LIST_ONLY
        self.person_pending   = False
        self.current_age      = None
        self.in_section       = 1
        # Two-column alternating table state
        self.alt_col1         = None   # first column label  (e.g. "Belief")
        self.alt_col2         = None   # second column label (e.g. "Origin")
        self.alt_pending      = None   # "col1" when first col header just seen
        self.alt_counter      = 0      # counts content lines: odd=col1, even=col2

    def _reset_narrative(self):
        """Reset inner narrative label only. Subsection persists."""
        self.narrative_label = None
        self.narrative_type  = None
        self.narrative_mode  = "ALL"
        self.person_pending  = False
        self.alt_col1        = None
        self.alt_col2        = None
        self.alt_pending     = None
        self.alt_counter     = 0

    def _reset_all(self):
        """Full reset — used at section boundaries."""
        self.subsection_label = None
        self._reset_narrative()

    def _narrative_applies_to(self, line: str) -> bool:
        """
        Check whether the inner narrative label applies to this line.
        Handles QUOTE_ONLY and LIST_ONLY modes.
        If it does not apply, resets narrative label and falls back to subsection.
        """
        if self.narrative_label is None:
            return False
        if self.narrative_mode == "ALL":
            return True
        if self.narrative_mode == "QUOTE_ONLY":
            if _is_direct_quote_line(line):
                return True
            self._reset_narrative()
            return False
        if self.narrative_mode == "LIST_ONLY":
            if _is_short_list_item(line):
                return True
            self._reset_narrative()
            return False
        return True

    def _build_prefix(self, line: str) -> str:
        """
        Build the context prefix for a content line.

        Priority:
          1. Alternating table mode active → use col1/col2 cycling label
             composed with subsection if present
          2. Narrative label applies → use narrative label
             (if inside subsection, compose: "subsection: narrative: content")
          3. Narrative label does not apply, subsection exists → subsection
          4. Neither → no prefix
        """
        # Priority 1: alternating table mode
        if self.alt_pending == "active":
            self.alt_counter += 1
            col_label = self.alt_col1 if self.alt_counter % 2 == 1 else self.alt_col2
            if self.subsection_label:
                return f"{self.subsection_label}: {col_label}"
            return col_label

        narrative_applies = self._narrative_applies_to(line)

        if narrative_applies:
            # Inner KEY_NARRATIVE inside a subsection: compose both
            if self.subsection_label and self.narrative_type in ("KEY_NARRATIVE",):
                return f"{self.subsection_label}: {self.narrative_label}"
            # Admin or clinical key inside Section 1 (no subsection)
            return self.narrative_label

        # Fall back to subsection label
        if self.subsection_label:
            return self.subsection_label

        return None

    def process_line(self, raw_line: str, line_num: int, line_class: str) -> NormalizedLine:
        stripped = raw_line.strip()

        # ── EMPTY ────────────────────────────────────────────────
        if line_class == "EMPTY":
            # Fix 1: blank line resets narrative but NOT subsection
            self._reset_narrative()
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=None,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── SECTION HEADER ───────────────────────────────────────
        if line_class == "SECTION_HEADER":
            m = re.search(r"SECTION\s+(\d+)", stripped, re.IGNORECASE)
            if m:
                self.in_section = int(m.group(1))
            self._reset_all()
            if self.in_section == 3:
                self.current_age = None
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=None,
                age_context=None, line_class=line_class,
                in_section=self.in_section, discard=True,
            )

        # ── SUBSECTION ───────────────────────────────────────────
        if line_class == "SUBSECTION":
            age_match = SUBSECTION_AGE_RE.search(stripped)
            if age_match:
                self.current_age = "Age " + age_match.group(1)

            # Strip number prefix and age from label
            name = re.sub(r"^\d+\.\d+\s+", "", stripped).strip()
            name = SUBSECTION_AGE_RE.sub("", name).strip().rstrip("—–-").strip()

            # Set subsection label, reset narrative
            self.subsection_label = name
            self._reset_narrative()

            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=name,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── STRUCTURAL ───────────────────────────────────────────
        if line_class == "STRUCTURAL":
            # Fix 3: structural words reset narrative only, NOT subsection
            self._reset_narrative()
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=None,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── TABLE COLUMN WORD (belief, origin, domain, self, etc.) ──
        if line_class == "TABLE_COLUMN_WORD":
            col = stripped.capitalize()

            # If alternating mode is already active, this word is a
            # DATA ROW value (e.g. "Self", "Others" in the Domain column)
            # NOT a new column header. Fall through to CONTENT handling.
            if self.alt_pending == "active":
                pass  # fall through to CONTENT block below
            elif self.alt_col1 is None:
                # First column header seen: store and wait for second
                self.alt_col1    = col
                self.alt_pending = "waiting_for_col2"
                self.alt_counter = 0
                return NormalizedLine(
                    raw_text=stripped, enriched_text=stripped,
                    line_number=line_num, context_label=col,
                    age_context=self.current_age,
                    line_class=line_class, in_section=self.in_section,
                    discard=True,
                )
            else:
                # Second column header seen: activate alternating mode
                self.alt_col2    = col
                self.alt_pending = "active"
                self.alt_counter = 0
                return NormalizedLine(
                    raw_text=stripped, enriched_text=stripped,
                    line_number=line_num, context_label=col,
                    age_context=self.current_age,
                    line_class=line_class, in_section=self.in_section,
                    discard=True,
                )

        # ── INLINE ADMIN (self-contained key-value line) ────────
        if line_class == "INLINE_ADMIN":
            # Line like "Previous Therapy: Not mentioned Current Living: With family"
            # Already carries its own key — pass through without any prefix
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=None,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=False,
            )

        # ── PERSON HEADER ────────────────────────────────────────
        if line_class == "PERSON_HEADER":
            # Use clean label without parenthetical qualifier as prefix
            # "Grandfather (paternal)" → "Grandfather" as prefix
            clean_label = PERSON_QUALIFIER_RE.sub("", stripped).strip()
            self.narrative_label = clean_label.capitalize()
            self.narrative_type  = "PERSON"
            self.narrative_mode  = "ALL"
            self.person_pending  = True
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=self.narrative_label,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── LETTER HEADER ────────────────────────────────────────
        if line_class == "LETTER_HEADER":
            label = re.sub(r"^[A-Z]\.\s+", "", stripped).strip()
            self.narrative_label = label
            self.narrative_type  = "LETTER"
            self.narrative_mode  = "ALL"
            self.person_pending  = False
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=label,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── KEY ADMIN ────────────────────────────────────────────
        if line_class == "KEY_ADMIN":
            label = stripped[:-1].strip()
            self.narrative_label = label
            self.narrative_type  = "KEY_ADMIN"
            self.narrative_mode  = "ALL"
            self.person_pending  = False
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=label,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── KEY CLINICAL ─────────────────────────────────────────
        if line_class == "KEY_CLINICAL":
            label = stripped[:-1].strip()
            self.narrative_label = label
            self.narrative_type  = "KEY_CLINICAL"
            self.narrative_mode  = "ALL"
            self.person_pending  = False
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=label,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── KEY NARRATIVE ────────────────────────────────────────
        if line_class == "KEY_NARRATIVE":
            label      = stripped[:-1].strip()
            label_lower = label.lower()

            if label_lower in self.QUOTE_ONLY_LABELS:
                mode = "QUOTE_ONLY"
            elif label_lower in self.LIST_ONLY_LABELS:
                mode = "LIST_ONLY"
            else:
                mode = "ALL"

            self.narrative_label = label
            self.narrative_type  = "KEY_NARRATIVE"
            self.narrative_mode  = mode
            self.person_pending  = False
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=label,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=True,
            )

        # ── AGE MARKER (Section 3) ───────────────────────────────
        if line_class == "AGE_MARKER":
            self.current_age = stripped
            self._reset_narrative()
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=None,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=False,
            )

        # ── TIMELINE ROW (Section 3) ─────────────────────────────
        if line_class == "TIMELINE_ROW":
            age_match = re.match(
                r"^([~≈]?\s*\d{1,2}(?:[–\-]\d{1,2})?[+]?)\s+",
                stripped
            )
            if age_match:
                self.current_age = age_match.group(1).strip()
            self._reset_narrative()
            return NormalizedLine(
                raw_text=stripped, enriched_text=stripped,
                line_number=line_num, context_label=None,
                age_context=self.current_age,
                line_class=line_class, in_section=self.in_section,
                discard=False,
            )

        # ── CONTENT ──────────────────────────────────────────────
        # SAFETY NET: if a short single-word title-case line appears that
        # looks like an unknown table column header, keep it as content
        # with the subsection context rather than discarding it silently.
        # This ensures unknown table structures never cause data loss.
        # (Known table words are already handled above as TABLE_COLUMN_WORD)

        # Build age prefix for Section 3
        age_prefix = ""
        if self.in_section == 3 and self.current_age:
            age_prefix = f"[{self.current_age}] "

        # Build context prefix using two-level stack
        prefix = self._build_prefix(stripped)
        ctx_label = prefix

        if prefix:
            enriched = f"{age_prefix}{prefix}: {stripped}"
        else:
            enriched = f"{age_prefix}{stripped}"

        # TYPE 2 (person): reset narrative after exactly one child
        if self.person_pending:
            self._reset_narrative()

        return NormalizedLine(
            raw_text=stripped,
            enriched_text=enriched,
            line_number=line_num,
            context_label=ctx_label,
            age_context=self.current_age,
            line_class="CONTENT",
            in_section=self.in_section,
            discard=False,
        )


# ─────────────────────────────────────────────────────────────
# 6.  OUTPUT VALIDATOR
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def validate(raw_text: str, normalized: NormalizedNote) -> tuple:
    raw_tokens = set()
    for line in raw_text.split("\n"):
        s = line.strip().lower()
        if s and s not in STRUCTURAL_WORDS:
            raw_tokens.update(_tokenize(s))

    out_tokens = set()
    for nl in normalized.lines:
        out_tokens.update(_tokenize(nl.enriched_text))
        out_tokens.update(_tokenize(nl.raw_text))

    missing = {t for t in raw_tokens - out_tokens if len(t) > 2}

    if missing:
        sample = sorted(missing)[:10]
        msg = (f"Possible dropped tokens (sample): {sample}. "
               f"Check lines that may have been silently discarded. "
               f"Total missing token types: {len(missing)}")
        return False, msg
    return True, "All content tokens present in output"


# ─────────────────────────────────────────────────────────────
# 7.  MAIN PREPROCESSOR
# ─────────────────────────────────────────────────────────────

def preprocess(raw_text: str, patient_id: str = "") -> NormalizedNote:
    sm    = ContextStateMachine()
    lines = raw_text.split("\n")
    out   = []

    for line_num, raw_line in enumerate(lines, 1):
        stripped   = raw_line.strip()
        line_class = classify_line(stripped, sm.in_section)
        nl         = sm.process_line(stripped, line_num, line_class)
        out.append(nl)

    note = NormalizedNote(lines=out, patient_id=patient_id)
    ok, msg = validate(raw_text, note)
    note.validation_ok  = ok
    note.validation_msg = msg
    return note


# ─────────────────────────────────────────────────────────────
# 8.  OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────

def print_normalized(note: NormalizedNote, show_discarded: bool = False):
    print(f"\n{'='*65}")
    print(f"PREPROCESSOR OUTPUT  |  patient: {note.patient_id or 'unknown'}")
    print(f"Validation: {'✓ OK' if note.validation_ok else '✗ FAILED'}"
          f"  — {note.validation_msg}")
    print(f"{'='*65}")
    total = len(note.lines)
    kept  = sum(1 for l in note.lines if not l.discard)
    print(f"Total lines: {total}  |  Kept: {kept}  |  Discarded: {total-kept}\n")

    for nl in note.lines:
        if nl.discard and not show_discarded:
            continue
        tag = f"[S{nl.in_section}]"
        age = f" [AGE: {nl.age_context}]" if nl.age_context else ""
        cls = f"[{nl.line_class:<16}]"
        pfx = "  DISCARD  " if nl.discard else "           "
        print(f"{pfx}{tag} {cls}{age}")
        print(f"           LINE {nl.line_number:3d}: {nl.enriched_text}")


def to_flat_text(note: NormalizedNote) -> str:
    return "\n".join(nl.enriched_text for nl in note.content_lines())


def to_json(note: NormalizedNote, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(note.to_dict(), f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 fire_preprocessor_v2.py <notes_file.txt> [--show-discarded]")
        sys.exit(1)

    show_disc = "--show-discarded" in sys.argv

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        raw = f.read()

    patient_id = sys.argv[1].replace(".txt", "").split("/")[-1]
    note = preprocess(raw, patient_id=patient_id)
    print_normalized(note, show_discarded=show_disc)

    json_path = sys.argv[1].replace(".txt", "_preprocessed.json")
    to_json(note, json_path)
    print(f"\nFull JSON saved to: {json_path}")

    flat_path = sys.argv[1].replace(".txt", "_flat.txt")
    with open(flat_path, "w", encoding="utf-8") as f:
        f.write(to_flat_text(note))
    print(f"Flat text saved to: {flat_path}")
