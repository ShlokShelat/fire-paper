"""
FiRE Pipeline — Worker 1, Stage 1: Sentence Tagger
=====================================================
Input  : Flat normalized text produced by fire_preprocessor.py
         Every line is already self-contained and context-prefixed.
Output : List of tagged sentence objects ready for Stage 2

What this stage does:
  1. Detects line type: TIMELINE / DIRECT_QUOTE / NARRATIVE
  2. Extracts temporal markers, hedging flags, clinical flags
  3. Assigns confidence: HIGH / MEDIUM / LOW / EXCLUDE
  4. Optionally splits long lines into sub-sentences

What this stage does NOT do (preprocessor already handled it):
  - Section detection
  - Header-child context tracking
  - Age context inheritance
  - Structural word filtering

Each output object:
{
    "sentence"        : str,   # full enriched sentence text
    "source_type"     : str,   # TIMELINE / DIRECT_QUOTE / NARRATIVE
    "confidence_tag"  : str,   # HIGH / MEDIUM / LOW / EXCLUDE
    "temporal_markers": list,  # age/period references found
    "age_context"     : str,   # age bracket if line starts with [~9-12]
    "hedging_flags"   : list,  # hedging words found
    "clinical_flags"  : list,  # therapist observation markers found
    "exclusion_reason": str,   # reason if EXCLUDE, else None
    "is_direct_quote" : bool,
    "line_number"     : int
}
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import spacy


# ─────────────────────────────────────────────
# 1.  PATTERNS
# ─────────────────────────────────────────────

DIRECT_QUOTE_RE = re.compile(r'["\u201c\u201d\u2018\u2019]')

# Timeline lines from Section 3: start with ANY [bracket] prefix
# This matches ALL age markers produced by the preprocessor:
# [~8], [~6-8], [Present], [Birth], [~Adolescence],
# [~Teenage years], [October 2021], [Ongoing in marriage] etc.
TIMELINE_PREFIX_RE = re.compile(
    r"^\[.+?\]\s+\S",  # [anything] followed by content
    re.IGNORECASE
)

# Standalone age markers from Section 3
STANDALONE_AGE_RE = re.compile(
    r"^[~≈]?\s*\d{1,2}(?:[–\-]\d{1,2})?\s*$"
    r"|^[~≈]?\s*\d{1,2}\+\s*$"
    r"|^[~≈]?\s*present\s*$|^[~≈]?\s*recent\s*$"
    r"|^[~≈]?\s*birth\s*$|^[~≈]?\s*school\s+years\s*$"
    r"|^[~≈]?\s*early\s+childhood\s*$"
    r"|^[~≈]?\s*childhood\s*$"
    r"|^[~≈]?\s*adolescence\s*$"
    r"|^[~≈]?\s*teenage\s+years\s*$"
    r"|^[~≈]?\s*early\s+teens\s*$"
    r"|^[~≈]?\s*late\s+teens.*$"
    r"|^[~≈]?\s*early.*20s\s*$"
    r"|^[~≈]?\s*mid.*20s\s*$"
    r"|^[~≈]?\s*late.*20s\s*$"
    r"|^[~≈]?\s*early.*30s\s*$"
    r"|^[~≈]?\s*mid.*30s\s*$"
    r"|^[~≈]?\s*late.*30s\s*$"
    r"|^[~≈]?\s*early.*40s\s*$"
    r"|^[~≈]?\s*son.*years\s*$"
    r"|^[~≈]?\s*son\s+age.*$"
    r"|^ongoing\s+in\s+\w+\s*$"
    r"|^[A-Za-z]+\s+\d{4}\s*$"
    r"|^\d{4}\s*$",
    re.IGNORECASE
)

# Extract age bracket from ANY [bracket] prefix
# Handles: [~8], [~6-8], [Present], [~Early childhood],
# [~Late teens / early 20s], [October 2021], [Ongoing in marriage]
AGE_BRACKET_RE = re.compile(
    r"^\[([^\]]+)\]\s*",
    re.IGNORECASE
)

HEDGING_WORDS = [
    "possible", "possibly", "suggests", "suggested",
    "appears to", "appeared to", "seems", "seemed",
    "likely", "unlikely", "probable", "may have",
    "needs clarification", "not confirmed",
    "reportedly", "described as", "client believes",
    "client feels", "indicates", "could indicate",
    "might suggest", "possibly related", "unclear",
    "not fully clear", "reported to", "noted as possible",
]

CLINICAL_OBSERVATION_MARKERS = [
    # Strong therapist interpretation signals — reliably appear in
    # SENTENCE BODIES as commentary, not in section headings.
    "marked as", "noted as", "flagged as",
    "identified as", "presents as", "consistent with",
    "this suggests", "this indicates",
    "no further elaboration", "marked repeatedly",
    "therapist notes", "therapist describes",
    "therapist marks", "therapist specifically marked",
    "associated emotion recorded",
    "core trigger identified",
    "impact:", "characteristics:",
    # REMOVED: "major trauma", "turning point", "formative wound",
    # "major turning point", "ace confirmed"
    # Reason: these appear in SECTION HEADINGS as labels
    # (e.g. "Grandfather's Suicide — Witnessed — Major Trauma:")
    # and are too ambiguous to reliably identify commentary sentences.
    # Real commentary sentences use "identified as", "this suggests" etc.
]

EVENT_VERBS = [
    "died", "death", "beat", "hit", "abused", "left",
    "moved", "fired", "married", "divorced", "separated",
    "discovered", "found", "looted", "stole", "burned",
    "crashed", "accident", "diagnosed", "passed away",
    "forced", "placed", "packed", "shamed", "manipulated",
    "targeted", "touched", "suffered", "arrested", "threatened",
    "killed", "lost", "began", "started", "migrated",
    "witnessed", "saw", "cried", "wept", "backed out",
    "drifted", "broke", "compared", "criticized",
    "humiliated", "excluded", "rejected", "bullied", "teased",
    "mocked", "failed", "collapsed", "attempted", "disclosed",
    "told", "informed", "withdrew",
    # Additional verbs covering deception, abandonment, coercion
    "deceived", "betrayed", "cheated", "lied", "hidden",
    "abandoned", "neglected", "assaulted", "coerced", "pressured",
    "controlled", "engaged", "proposed", "relocated", "hospitalized",
    "expelled", "suspended", "fired", "evicted", "separated",
    "experienced", "placed", "exposed", "subjected",
]
# Note: "reported", "born", "ended", "stopped" removed to prevent
# false positives on admin/description text like "reported as good",
# "born in Uttarakhand", "attended" (contains ended), etc.
# Word-boundary safe alternatives added in assign_confidence below.
WORD_BOUNDARY_EVENT_VERBS = [
    "stopped", "ended", "born with", "sent",
]

DESCRIPTIVE_QUALITY_MARKERS = [
    "strict", "abusive", "controlling", "intimidating",
    "unavailable", "demanding", "orthodox", "critical",
    "supportive", "frightening", "loud", "aggressive",
    "passive", "neglectful", "absent", "emotionally distant",
    "cold", "protective", "rule-oriented", "punitive",
    "violent", "volatile", "unpredictable", "manipulative",
    "dismissive", "scary", "strained",
]

EXCLUSION_BELIEF_PATTERNS = [
    # NOTE: "core belief" removed intentionally.
    # Lines containing "core belief" have rich subsection context
    # and should pass to Stage 2 for LLM judgment, not be excluded here.
    r"\bself.worth\b.*\blinked\b",
    r"\bidentity\b.*\bshift\b",
    r"^achievement became.*self.worth",
    r"^achievement became.*coping strategy",
]

EXCLUSION_PURE_CLINICAL_PATTERNS = [
    r"core emotions during this period",
    r"^associated emotion",
    r"^possible (themes|schema|core belief|dynamic|pattern)",
]

TEMPORAL_PATTERNS = [
    (r"\bage[d]?\s*[~≈]?\s*(\d{1,2}(?:[\-–]\d{1,2})?)\b", "age"),
    (r"\bat\s+[~≈]?\s*(\d{1,2})\b",                        "age"),
    (r"\bwhen\s+(?:she|he|client)\s+was\s+[~≈]?\s*(\d{1,2})\b", "age"),
    (r"\b(\d{1,2})\s+years?\s+old\b",                       "age"),
    (r"\bapproximately\s+(\d{1,2})\b",                      "age_approx"),
    (r"\baround\s+age\s+[~≈]?\s*(\d{1,2})\b",              "age_approx"),
    (r"\b(\d{1,2})[–\-](\d{1,2})\b",                       "age_range"),
    (r"\b(childhood|teenage|adolescence|college|school years|"
     r"post.marriage|post.divorce|adulthood|early adulthood)\b", "lifecycle"),
    (r"\bafter (mother|father|grandfather|grandmother)[''s]* death\b",
     "relative_time"),
    (r"\bduring (college|school|marriage|teenage years)\b", "relative_time"),
    (r"\bpost.(?:accident|divorce|marriage|covid)\b",       "relative_time"),
]


# ─────────────────────────────────────────────
# 2.  DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class TaggedSentence:
    sentence:         str
    source_type:      str
    confidence_tag:   str
    temporal_markers: list          = field(default_factory=list)
    age_context:      Optional[str] = None
    hedging_flags:    list          = field(default_factory=list)
    clinical_flags:   list          = field(default_factory=list)
    exclusion_reason: Optional[str] = None
    is_direct_quote:  bool          = False
    line_number:      int           = 0

    def to_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────
# 3.  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_age_context(line: str) -> Optional[str]:
    """Extract age bracket from [~9-12] prefix if present."""
    m = AGE_BRACKET_RE.match(line)
    if m:
        return m.group(1).strip()
    return None


def strip_age_bracket(line: str) -> str:
    """Remove [~9-12] prefix from line text."""
    return AGE_BRACKET_RE.sub("", line).strip()


def extract_temporal_markers(text: str) -> list:
    found = []
    text_lower = text.lower()
    seen = set()
    for pattern, label in TEMPORAL_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            val = match.group(0).strip()
            if val not in seen:
                found.append({"type": label, "value": val})
                seen.add(val)
    return found


def extract_hedging_flags(text: str) -> list:
    text_lower = text.lower()
    return [w for w in HEDGING_WORDS if w in text_lower]


def extract_clinical_flags(text: str) -> list:
    text_lower = text.lower()
    return [m for m in CLINICAL_OBSERVATION_MARKERS if m in text_lower]


def is_exclusion_candidate(text: str) -> tuple:
    text_lower = text.lower()
    for pattern in EXCLUSION_PURE_CLINICAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True, "pure_clinical_observation"
    for pattern in EXCLUSION_BELIEF_PATTERNS:
        if re.search(pattern, text_lower):
            return True, "belief_or_schema_language"
    return False, None


def assign_confidence(
    source_type:     str,
    hedging_flags:   list,
    clinical_flags:  list,
    is_direct_quote: bool,
    text:            str,
) -> str:
    """
    Confidence rules in priority order:

    1. Direct quote                         → HIGH
    2. Timeline line                        → HIGH
    3. Has clinical observation markers     → LOW
    4. Has event verb IN CONTENT ONLY       → HIGH
    5. Has descriptive quality marker       → MEDIUM
    6. Has hedging language                 → MEDIUM
    7. Default                              → MEDIUM

    CRITICAL: Event verb checks run only on the CONTENT portion
    of the line, not the context prefix. This prevents prefixes
    like "Death of Younger Sister:" from causing false HIGH tags
    on every child line that inherits that prefix.
    """
    if is_direct_quote:
        return "HIGH"
    if source_type == "TIMELINE":
        return "HIGH"

    # Strip ONLY the outermost section heading prefix for clinical flag check.
    # The section heading is the FIRST segment before ": " that is <= 80 chars.
    # Everything after that first heading IS the clinical content.
    # Sub-clauses like "Long-term impact:" remain in the content string
    # so their markers ("impact:") can be detected correctly.
    #
    # For event verb detection, we iterate further (stripping sub-headings
    # like "Turning Point — Birthday Incident:" to reach the actual sentence).
    # This distinction is intentional and correct:
    #   Clinical flag: "Long-term impact: fear..." → content → LOW ✓
    #   Event verb:    "Turning Point — Birthday: Made a decision" → stripped → MEDIUM ✓
    if ": " in text:
        first_prefix = text.split(": ", 1)[0].strip()
        if len(first_prefix) <= 80:
            clinical_content = text.split(": ", 1)[1]
        else:
            clinical_content = text
    else:
        clinical_content = text

    if extract_clinical_flags(clinical_content):
        return "LOW"

    # Extract content for event verb and quality checks
    # (iterative stripping removes all prefix segments)
    content = text
    if ": " in text:
        parts = text.split(": ")
        for i in range(len(parts) - 1):
            if len(parts[i]) < 80:
                content = ": ".join(parts[i+1:])
            else:
                break

    content_lower = content.lower()
    full_lower    = text.lower()

    if any(v in content_lower for v in EVENT_VERBS):
        return "HIGH"

    # Word-boundary checked verbs — content only
    import re as _re
    if any(_re.search(r"\b" + _re.escape(v) + r"\b", content_lower)
           for v in WORD_BOUNDARY_EVENT_VERBS):
        return "HIGH"

    if any(q in full_lower for q in DESCRIPTIVE_QUALITY_MARKERS):
        return "MEDIUM"
    if hedging_flags:
        return "MEDIUM"
    return "MEDIUM"


def detect_line_type(line: str) -> str:
    """
    Classify each flat file line into:

    STANDALONE_AGE : bare age marker (~6-8, Present, Birth)
                     → kept for traceability, not passed to Stage 2
    TIMELINE       : [~6-8] content line from Section 3
    DIRECT_QUOTE   : line containing quotation marks
    NARRATIVE      : everything else (already context-prefixed)
    """
    if STANDALONE_AGE_RE.match(line):
        return "STANDALONE_AGE"
    if TIMELINE_PREFIX_RE.match(line):
        return "TIMELINE"
    if DIRECT_QUOTE_RE.search(line):
        return "DIRECT_QUOTE"
    return "NARRATIVE"


# ─────────────────────────────────────────────
# 5.  SENTENCE SPLITTER
# ─────────────────────────────────────────────

def split_if_needed(text: str, nlp) -> list:
    """
    Split a line into sub-sentences if it is long and does NOT
    already have a context prefix (contains ': ').

    Lines with a context prefix are kept whole because splitting
    would orphan sub-sentences from their context.

    For TIMELINE lines (starting with [age bracket]), the bracket
    prefix is stripped before splitting so sentences are clean,
    but the age_context is extracted separately and propagated
    by the caller to every sub-sentence.
    """
    # Already has a context prefix → keep whole (never split)
    if ": " in text:
        return [text]
    # Short enough → keep whole
    if len(text) <= 120:
        return [text]
    # Long plain sentence → split
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def split_timeline_line(text: str, nlp) -> tuple:
    """
    Special splitter for TIMELINE lines like:
      [October 2021] Relocates to Hong Kong. Cultural adjustment...

    Returns (age_context: str, sub_sentences: list[str])
    where age_context is the extracted bracket and sub_sentences
    have the bracket stripped but content preserved.

    This lets the caller attach age_context to every sub-sentence.
    """
    age_ctx = extract_age_context(text)
    # Strip the [bracket] prefix to get clean content
    content_only = AGE_BRACKET_RE.sub("", text).strip()

    if not content_only:
        return age_ctx, [text]

    # Short enough → keep as single sentence
    if len(content_only) <= 120:
        return age_ctx, [content_only]

    # Long → split by sentence boundaries
    doc = nlp(content_only)
    subs = [s.text.strip() for s in doc.sents if s.text.strip()]
    return age_ctx, subs if subs else [content_only]


# ─────────────────────────────────────────────
# 6.  MAIN PROCESSOR
# ─────────────────────────────────────────────

def process_stage1(flat_text: str) -> list:
    """
    Main Stage 1 processor.

    Reads flat file line by line. Each line is already enriched
    with context from the preprocessor. Stage 1 only needs to:
      1. Detect line type
      2. Extract metadata
      3. Assign confidence
      4. Optionally split long plain lines
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    results = []
    lines   = flat_text.split("\n")

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue

        line_type = detect_line_type(line)

        # ── STANDALONE AGE MARKER ────────────────────────────────
        # Keep for traceability but do not pass to Stage 2
        if line_type == "STANDALONE_AGE":
            results.append(TaggedSentence(
                sentence=line,
                source_type="TIMELINE",
                confidence_tag="EXCLUDE",
                age_context=line,
                exclusion_reason="standalone_age_marker",
                line_number=line_num,
            ).to_dict())
            continue

        # ── TIMELINE LINE ────────────────────────────────────────
        # [~6-8] content — extract age bracket, split if long,
        # propagate age_context to ALL sub-sentences (Fix 2)
        if line_type == "TIMELINE":
            age_ctx, subs = split_timeline_line(line, nlp)

            for sub in subs:
                if not sub.strip():
                    continue
                results.append(TaggedSentence(
                    sentence=sub,
                    source_type="TIMELINE",
                    confidence_tag="HIGH",
                    temporal_markers=extract_temporal_markers(sub),
                    age_context=age_ctx,
                    hedging_flags=extract_hedging_flags(sub),
                    clinical_flags=extract_clinical_flags(sub),
                    is_direct_quote=bool(DIRECT_QUOTE_RE.search(sub)),
                    line_number=line_num,
                ).to_dict())
            continue

        # ── NARRATIVE / DIRECT_QUOTE ─────────────────────────────
        # Split only if no context prefix and line is long
        sub_sentences = split_if_needed(line, nlp)

        for sub in sub_sentences:
            if not sub.strip():
                continue

            temporal_markers = extract_temporal_markers(sub)
            hedging_flags    = extract_hedging_flags(sub)
            clinical_flags   = extract_clinical_flags(sub)
            is_direct_quote  = bool(DIRECT_QUOTE_RE.search(sub))

            should_exclude, excl_reason = is_exclusion_candidate(sub)

            if should_exclude:
                results.append(TaggedSentence(
                    sentence=sub,
                    source_type="DIRECT_QUOTE" if is_direct_quote else "NARRATIVE",
                    confidence_tag="EXCLUDE",
                    temporal_markers=temporal_markers,
                    hedging_flags=hedging_flags,
                    clinical_flags=clinical_flags,
                    exclusion_reason=excl_reason,
                    is_direct_quote=is_direct_quote,
                    line_number=line_num,
                ).to_dict())
                continue

            source_type = "DIRECT_QUOTE" if is_direct_quote else "NARRATIVE"
            confidence  = assign_confidence(
                source_type=source_type,
                hedging_flags=hedging_flags,
                clinical_flags=clinical_flags,
                is_direct_quote=is_direct_quote,
                text=sub,
            )

            results.append(TaggedSentence(
                sentence=sub,
                source_type=source_type,
                confidence_tag=confidence,
                temporal_markers=temporal_markers,
                hedging_flags=hedging_flags,
                clinical_flags=clinical_flags,
                exclusion_reason=None,
                is_direct_quote=is_direct_quote,
                line_number=line_num,
            ).to_dict())

    return results


# ─────────────────────────────────────────────
# 7.  OUTPUT HELPERS
# ─────────────────────────────────────────────

def filter_for_stage2(results: list) -> list:
    """Return only HIGH / MEDIUM / LOW sentences for Stage 2."""
    return [r for r in results if r["confidence_tag"] in ("HIGH", "MEDIUM", "LOW")]


def summary_stats(results: list) -> dict:
    from collections import Counter
    tags  = Counter(r["confidence_tag"]  for r in results)
    types = Counter(r["source_type"]     for r in results)
    excl  = Counter(
        r["exclusion_reason"]
        for r in results if r["exclusion_reason"]
    )
    return {
        "total_sentences":    len(results),
        "confidence_counts":  dict(tags),
        "source_type_counts": dict(types),
        "exclusion_reasons":  dict(excl),
        "passed_to_stage2":   len(filter_for_stage2(results)),
    }


# ─────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 fire_stage1.py <flat_file.txt>")
        print("       (flat_file.txt is output from fire_preprocessor.py)")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        flat_text = f.read()

    results = process_stage1(flat_text)
    stats   = summary_stats(results)

    print("\n" + "="*60)
    print("STAGE 1 SUMMARY")
    print("="*60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("SENTENCES PASSED TO STAGE 2")
    print("="*60)
    for r in filter_for_stage2(results):
        age = r.get("age_context")
        print(f"\n[Line {r['line_number']:3d}] "
              f"[{r['confidence_tag']:6s}] "
              f"[{r['source_type']:12s}]"
              f"{' [AGE: '+age+']' if age else ''}")
        print(f"  TEXT: {r['sentence']}")
        if r["temporal_markers"]:
            print(f"  TIME:  {r['temporal_markers']}")
        if r["hedging_flags"]:
            print(f"  HEDGE: {r['hedging_flags']}")
        if r["clinical_flags"]:
            print(f"  CLIN:  {r['clinical_flags']}")

    output_path = "stage1_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull output saved to: {output_path}")
