"""
Clinical State Space Matcher — General Domain Pipeline

Maps validated clinical events to a shared state space taxonomy via Oracle
(event decomposition) → Global Vector Search → Judge (state matching) → 
Genesis (new state creation) loop. Implements provisional state scoping and
clinician review gate per paper Section 3.2.

Key features:
  - Flat global semantic search across all states (no cluster pre-filtering)
  - Unified judge with no cross-type veto (same mechanism = same state)
  - Provisional new states scoped to triggering patient until clinician approval
  - Deterministic dedup sweep for same-cluster near-duplicates post-run
  - Grounding proof: all sub-events must be grounded in source clinical text
  - Cross-process safe writes with merge-on-disk conflict resolution
"""

import json
import os
import sys
import asyncio
import aiohttp
import argparse
import logging
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl   # POSIX advisory file locking for safe parallel runs
except ImportError:
    fcntl = None   # Windows: lock is a no-op (single-process use still safe)
from typing import Optional
from sentence_transformers import SentenceTransformer, util


# ─────────────────────────────────────────────────────────────
# 1.  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

ORACLE_PROMPT = """You are the Oracle for the FiRE clinical trauma database.
Your job is to read a complex compound clinical event and decompose it into
atomic sub-events, each mapping to exactly one clinical state.

DECOMPOSITION RULES — follow all of them:

1. ATOMIC DOMAIN SPLIT
   Split across clinical domains — never merge different domains in one sub-event.
   Separate: family trauma | peer trauma | financial hardship | medical | internal states
   Example: "bullied by classmates and father lost job" → 2 sub-events (peer domain + economic domain)

2. TRIGGER vs RESPONSE SPLIT
   Separate external events (what happened) from internal responses (how the patient reacted).
   WRONG:  "Witnessed domestic violence and developed protective feelings" [one sub-event]
   CORRECT: Sub-event 1: "Client [Childhood] witnessed domestic violence between parents." [event]
            Sub-event 2: "Client [Childhood] developed parentification and strong protective feelings toward mother." [symptom]

3. BELIEF ISOLATION
   Each distinct core belief is its own sub-event of experience_type "belief".
   Never merge multiple beliefs into one sub-event.

4. GRANULARITY
   One sub-event = one clinical mechanism = one state in the database.
   If a paragraph has 5 distinct clinical mechanisms, return 5 sub-events.

   COMPOUND STATES ARE FORBIDDEN:
   NEVER combine multiple symptoms into one state description.
   WRONG ideal_state_name: "Depression, anxiety, and insomnia"
   If you are listing multiple conditions separated by commas, split them immediately.

   SYMPTOM ISOLATION — mandatory:
   Each symptom domain is its own sub-event. NEVER group them.
   "Client experienced depression, anxiety, and insomnia" → THREE sub-events:
     Sub-event 1: depressive symptoms (experience_type: symptom)
     Sub-event 2: anxiety symptoms   (experience_type: symptom)
     Sub-event 3: sleep disturbance  (experience_type: symptom)
   Apply this to ALL symptom lists, no matter how short the source sentence.

5. DYNAMIC ANCHORING
   Prefix every sub-event text with the life stage in brackets, e.g. "Client [Childhood]..."

6. NEUTRALITY — strict definition
   NEUTRAL = a demographically routine fact with ZERO trauma signal.
   Examples of NEUTRAL: "Client completed a university degree", "Client moved to a new city".
   ADVERSE = everything else, including:
     - Maladaptive coping strategies (achievement as defence, people-pleasing)
     - Trauma-born beliefs ("I must prove myself", "My future is insecure")
     - Any behavioural or emotional response born from an adverse experience
     - Relational patterns developed as self-protection

7. CLINICAL ABSTRACTION & PERMANENT GENERALISATION
   Replace ALL patient-specific details with clinical abstractions:
      - Names:   "father" → "caregiver/parent",  "classmates" → "peers"
      - Details: "hearing loss" → "visible physical difference"
   Keep specific names and details ONLY in the sub-event text field.

8. CHRONOLOGY & AGE ANCHORING (CRITICAL)
   You MUST extract the exact integer onset_age and end_age for EVERY sub-event.
   - If the text provides an age range (e.g., "~23-25" or "Age 8-10"), set onset_age to the
     lower bound (23) and end_age to the upper bound (25). DO NOT hallucinate intermediate ages.
   - If a single age is given (e.g., "at age 12"), set both onset_age and end_age to 12.
   - If an event is tied to a life stage (e.g. "Childhood"), infer the standard age range
     (Childhood: 0-12, Teenage: 13-19, Adult: 20+).
   - Set is_ongoing to true if the text says "current", "ongoing", "until recently", or "present".
   - A DEATH is a POINT EVENT, not a range: if the sub-event is that someone died, set
     onset_age and end_age to the SAME value (the age at which the death occurred), even if the
     surrounding text spans a wider age range for that person's presence in the client's life.
   - Output integers only. If completely unknown, output null.

9. GROUNDING PROOF — MANDATORY
   Every sub-event MUST be grounded ONLY in the ORIGINAL CLINICAL TEXT, never the
   BACKGROUND CONTEXT. To prove this, copy the EXACT span, word-for-word, from the
   ORIGINAL CLINICAL TEXT that supports the sub-event into "supporting_quote". If you
   cannot find real words in the ORIGINAL CLINICAL TEXT to quote, the sub-event is not
   grounded there — do not emit it (a different line likely carries that fact instead).

OUTPUT EXACTLY THIS JSON — no other text:
{
  "decomposition_reasoning": "Brief explanation of how and why you split.",
  "sub_events": [
    {
      "status": "ADVERSE or NEUTRAL",
      "experience_type": "event or symptom or pattern or belief",
      "neutral_reason": "Only fill if NEUTRAL — why this is routine",
      "text": "Client [LifeStage] anchored sub-event text",
      "supporting_quote": "exact verbatim span from the ORIGINAL CLINICAL TEXT",
      "onset_age": 8,
      "end_age": 12,
      "is_ongoing": false,
      "ideal_cluster_name": "Generalised cluster name (only if ADVERSE)",
      "ideal_cluster_definition": "Broad textbook definition (only if ADVERSE)",
      "ideal_state_name": "Generalised state name (only if ADVERSE)",
      "ideal_state_description": "Generalised clinical mechanism (only if ADVERSE)"
    }
  ]
}"""


MATCH_JUDGE_PROMPT = """You are the Supreme Matching Judge of the FiRE clinical state space.
Your single job: decide whether a PROPOSED clinical state already exists in the
database, so the database never stores the same mechanism twice.

PROPOSED STATE:
  Name:        {proposed_name}
  Type:        {proposed_type}
  Description: {proposed_desc}

EXISTING CANDIDATES — the true nearest neighbours across the WHOLE database,
sorted by semantic similarity (highest first). This list is complete; if a real
match existed it is in here:
{candidates_json}

DECISION RULES — apply in order:

RULE 0 — EMPTY: If the candidate list is empty, output MAP=false (create new).

RULE 1 — SAME NAME = SAME STATE:
  If any candidate has the same or near-identical name, MAP to it.
  "Insomnia" vs "Sleep Disturbance" vs "Sleep deprivation" -> MAP (same mechanism).

RULE 2 — SAME MECHANISM = SAME STATE (ignore surface wording AND type):
  Map whenever the underlying clinical mechanism is the same, even if the
  experience_type label differs (symptom vs pattern vs event). The Oracle is
  inconsistent about type; do NOT let a type difference block a real match.

  BIAS TOWARD MAPPING. If two states describe the same core clinical phenomenon,
  MAP — even when one uses a clinical term and the other uses lay wording, and
  even when one adds a secondary detail. A new state is justified ONLY when the
  CORE mechanism is genuinely different, not merely differently phrased.

  Clinical-term vs lay-wording pairs that ALWAYS map (same phenomenon):
    Alexithymia = Emotional disconnection = Emotional numbing = Can't identify emotions
    Insomnia = Sleep disturbance = Sleep deprivation = Difficulty sleeping
    Hypervigilance = Startle response = Hyperarousal = Fear response (post-trauma)
    Anhedonia = Loss of interest = Inability to feel pleasure
    Emotional neglect = Emotionally unavailable caregiver
    Parentification = Emotional role reversal = Caretaking of parent
    Substance use = Substance abuse = Addiction (same substance family)
    Social anxiety = Fear of judgment = Social-evaluative fear = Social avoidance
    Medication discontinuation = Medication non-adherence

  SUPERSET RULE: if the proposed state is the existing state PLUS an extra
  behaviour ("Impulsive stealing" vs "Impulsive lying AND stealing"), and both
  sit in the same impulse/behaviour family, MAP to the existing one rather than
  minting a near-identical compound. Do not split a mechanism just because the
  new wording bundles one more symptom.

  SAME-CLUSTER IS NOT SAMENESS: candidates in the same cluster are RELATED BY
  DESIGN — the cluster groups distinct sub-mechanisms of one family. Do NOT map
  just because two states share a cluster. Map ONLY when they describe the SAME
  underlying mechanism. Examples of DISTINCT same-cluster states that must stay
  SEPARATE (never merge these):
    Parentification (child meets parent's emotional needs)
      != Restrictive/controlling parenting (parent controls child's social world)
      != Emotional neglect (parent provides no emotional support)
    Fear of death != Generalized anxiety != Rumination
    Impulsive stealing != Post-impulsive regret
  These share a cluster but are different experiences. CREATE a new state.

RULE 3 — GENUINELY DIFFERENT MECHANISM = CREATE:
  Only create when the proposed state is clinically distinct from ALL candidates.
  Physical abuse != Emotional neglect. PTSD hyperarousal != Impulse-control deficit.
  Trauma symptom != Medication side-effect (different etiology).

RULE 4 — NO CONTRADICTION SELF-CHECK:
  Before answering, re-read your reasoning. If it contains "no match",
  "does not match", "different mechanism", or "create a new state", then your
  output MUST be map=false. Never output map=true while concluding no match.

OUTPUT EXACTLY THIS JSON — no other text:
{{
  "map": true or false,
  "code": "existing state code if map=true, else null",
  "reasoning": "One sentence: which rule and why."
}}"""


# ─────────────────────────────────────────────────────────────
# 2.  TOKEN / COST TRACKER
# ─────────────────────────────────────────────────────────────

class TokenCounter:
    """Thread-safe (asyncio-safe) token accumulator."""
    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.calls             = 0

    def add(self, usage: dict) -> None:
        self.prompt_tokens     += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.calls             += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * 0.14 + self.completion_tokens * 0.28) / 1_000_000

    def summary(self) -> str:
        return (
            f"LLM calls: {self.calls} | "
            f"tokens: {self.prompt_tokens:,} in + {self.completion_tokens:,} out "
            f"= {self.total_tokens:,} total | "
            f"est. cost: ${self.cost_usd:.4f}"
        )


# ─────────────────────────────────────────────────────────────
# 3.  RESUME MANAGER
# ─────────────────────────────────────────────────────────────

class ResumeManager:
    def __init__(self, resume_path: Path, enabled: bool):
        self.path    = resume_path
        self.enabled = enabled
        self._data   = {}
        self._lock   = asyncio.Lock()

        if enabled and resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                self._data = json.load(f)
            logging.info("[RESUME] Loaded %d already-completed events from %s",
                         len(self._data), resume_path)

    def is_done(self, unit_id: str) -> bool:
        return unit_id in self._data

    def get(self, unit_id: str) -> dict:
        return self._data[unit_id]

    async def mark_done(self, unit_id: str, event: dict) -> None:
        if not self.enabled:
            return
        async with self._lock:
            self._data[unit_id] = event
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# 4.  CONTEXT HELPERS & MARKERS
# ─────────────────────────────────────────────────────────────

def build_section_map(stage1_data: list) -> tuple[dict, dict]:
    if not stage1_data:
        return {}, {}

    _BULLET_LIST_HEADERS = [
        "presenting problem", "chief complaint",
        "current concern",    "presenting issue",
    ]

    def _prefix(text: str) -> str:
        if ": " in text:
            return re.sub(r"\s+", " ", text.split(": ", 1)[0].strip())
        return ""

    sections: dict[str, list] = {}
    for s in stage1_data:
        sentence = str(s.get("sentence") or "")
        pfx = _prefix(sentence)
        if not pfx:
            continue
        content = sentence.split(": ", 1)[1].strip()
        if any(x in pfx.lower() for x in _BULLET_LIST_HEADERS):
            continue
        sections.setdefault(pfx, []).append(content)

    section_map = {p: " ".join(c) for p, c in sections.items()}

    context_map: dict[int, str] = {}
    for i, sent in enumerate(stage1_data):
        ln  = sent.get("line_number", i)
        pfx = _prefix(str(sent.get("sentence") or ""))
        before = [str(s.get("sentence") or "") for s in stage1_data[max(0, i-2):i]
                  if _prefix(str(s.get("sentence") or "")) == pfx]
        after  = [str(s.get("sentence") or "") for s in stage1_data[i+1:i+3]
                  if _prefix(str(s.get("sentence") or "")) == pfx]
        context_map[ln] = " ".join(before + after)

    return section_map, context_map


def lookup_section(heading: str, section_map: dict) -> str:
    if not heading:
        return ""
    h = re.sub(r"\s+", " ", heading.strip())
    if h in section_map:
        return section_map[h]
    best_key, best_score = None, 0
    for key in section_map:
        if key.startswith(h) or h.startswith(key):
            score = len(key)
            if score > best_score:
                best_key, best_score = key, score
            continue
        cp = 0
        while cp < len(h) and cp < len(key) and h[cp] == key[cp]:
            cp += 1
        if cp >= 12 and cp > best_score:
            best_key, best_score = key, cp
    return section_map[best_key] if best_key else ""


def build_enriched_context(event_obj: dict, section_map: dict, context_map: dict) -> str:
    subsection = str(event_obj.get("subsection") or "")
    line_no    = event_obj.get("line_number")
    if subsection:
        para = lookup_section(subsection, section_map)
        if para:
            return para[:600].rstrip()
    if line_no is not None:
        ctx = str(context_map.get(line_no) or "")
        if ctx:
            return ctx[:300].rstrip()
    return ""


def _safe_int(val) -> Optional[int]:
    """Safely converts Oracle age output to int (handles None, floats, strings like '~24')."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    try:
        nums = re.findall(r'\d+', str(val))
        if nums:
            return int(nums[0])
    except Exception:
        pass
    return None


# ── Age extraction — ported from v4.7, generalised (no ACE wording) ───────
_AGE_TAG_RE        = re.compile(r"\[\s*age\b([^\]]*)\]", re.IGNORECASE)
_DURATION_UNIT_RE  = re.compile(
    r"^\s*(?:hour|hr|minute|min|second|sec|day|week|month|year|yr)s?\b", re.IGNORECASE)
_YEAR_RE           = re.compile(r"\b(19|20)\d{2}\b")

_DEATH_MARKERS = ("died", " death", "passed away", "passed on", "death of", "demise")


def _parse_age_tag_content(content: str):
    """Parse the inside of an [Age ...] tag into (onset, end) or (None, None)."""
    c = str(content or "").lower()
    c = _YEAR_RE.sub(" ", c)
    mm = re.search(r"[~\u2248]?\s*(\d{1,3})\s*month", c)
    if mm:
        yrs = int(mm.group(1)) // 12
        return (yrs, yrs)
    m = re.search(r"[~\u2248]?\s*(\d{1,3})\s*(?:-|\u2013|\u2014|to)\s*(\d{1,3})", c)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a <= 120 and 0 <= b <= 120:
            return (min(a, b), max(a, b))
    m = re.search(r"[~\u2248]?\s*(\d{1,3})", c)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 120:
            return (n, n)
    return (None, None)


def _segment_tagged_source(source: str):
    """Split a source sentence into (clause_text, (onset,end)) pairs, one per
    trailing [Age ...] tag. Returns [] when there are no such tags."""
    if not source:
        return []
    segs, last = [], 0
    for m in _AGE_TAG_RE.finditer(source):
        segs.append((source[last:m.start()], _parse_age_tag_content(m.group(1))))
        last = m.end()
    return segs


def _age_from_tagged_source(source: str, quote: str):
    """Find the [Age ...] tag attached to the clause containing `quote`, so a
    multi-age sentence binds each sub-event to its OWN clause's age rather than
    the first age in the whole sentence."""
    segs = _segment_tagged_source(source)
    if not segs or not quote:
        return (None, None)

    def _norm(s):
        return re.sub(r"\s+", " ", str(s or "").lower()).strip(" .,\"'\u201c\u201d")

    qn = _norm(quote)
    if not qn:
        return (None, None)
    for seg_text, age in segs:
        if qn in _norm(seg_text):
            return age
    words = qn.split()
    if len(words) >= 4:
        chunk = " ".join(words[:6])
        for seg_text, age in segs:
            if chunk in _norm(seg_text):
                return age
    return (None, None)


def extract_temporal_anchor(sub_event_text: str, parent_context: dict,
                            supporting_quote: str = None):
    """
    Regex age safety net. Returns (onset, end, provenance).
    provenance in {'quote_tag','subtext_tag','sub_text','source','parent',None}
    tells the caller how strong the evidence is.
    """
    def _is_duration(t, end_idx):
        return bool(_DURATION_UNIT_RE.match(t[end_idx:]))

    def _search(text):
        if not text:
            return (None, None)
        t = str(text).lower()
        t = _YEAR_RE.sub("    ", t)
        m = re.search(r"[~\u2248]?\s*(\d{1,3})\s*months?\b", t)
        if m:
            yrs = int(m.group(1)) // 12
            return yrs, yrs
        m = re.search(r"(?:age|at)\s*[~\u2248]?\s*(\d+)\s*(?:-|to|\u2013|\u2014)\s*(\d+)", t)
        if m and not _is_duration(t, m.end()):
            return int(m.group(1)), int(m.group(2))
        for m in re.finditer(r"[~\u2248]?\s*(\d{1,2})\s*(?:-|\u2013|\u2014|to)\s*(\d{1,2})\b", t):
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b <= 120 and not _is_duration(t, m.end()):
                return a, b
        m = re.search(r"(?:age|at)\s*[~\u2248]?\s*(\d+)", t)
        if m and not _is_duration(t, m.end()):
            return int(m.group(1)), int(m.group(1))
        m = re.match(r"\s*[~\u2248]?\s*(\d{1,2})\s*(?:\u2014|\u2013|-|\s)(?!\s*\d)", t)
        if m and not _is_duration(t, m.end()):
            n = int(m.group(1))
            if n <= 120:
                return n, n
        return (None, None)

    source = parent_context.get("source_sentence")

    if supporting_quote:
        a, b = _age_from_tagged_source(source, supporting_quote)
        if a is not None:
            return a, b, "quote_tag"

    sub_segs = _segment_tagged_source(sub_event_text)
    if sub_segs:
        for _txt, age in sub_segs:
            if age[0] is not None:
                return age[0], age[1], "subtext_tag"

    r = _search(sub_event_text)
    if r != (None, None):
        return r[0], r[1], "sub_text"

    if not _segment_tagged_source(source):
        r = _search(source)
        if r != (None, None):
            return r[0], r[1], "source"

    parent_age = parent_context.get("age_context") or parent_context.get("subsection")
    if parent_age:
        nums = re.findall(r"\d+", str(parent_age))
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1]), "parent"
        if len(nums) == 1:
            return int(nums[0]), int(nums[0]), "parent"
    return None, None, None


_RECOLLECTION_MARKERS = [
    r"post[-\s](\w[\w\s]*)",
    r"triggered by (\w[\w\s]*)",
    r"reminded (?:him|her|them) of",
    r"reminds (?:him|her|them) of",
    r"triggered memories of",
    r"brought back memories",
    r"since the (\w[\w\s]*) (?:war|attack|disaster|event|trauma|incident)",
    r"after the (\w[\w\s]*) (?:war|attack|disaster|event|trauma|incident)",
    r"related to (?:the )?(\w[\w\s]*) trauma",
    r"felt like childhood again",
    r"same feeling as when",
    r"echoes of",
    r"resurfaced",
    r"reactivated",
]

def detect_recollection(text: str) -> dict:
    t = text.lower()
    for pattern in _RECOLLECTION_MARKERS:
        m = re.search(pattern, t)
        if m:
            ref = m.group(1).strip() if m.lastindex else "past trauma"
            return {"is_recollection": True, "recollection_reference": ref}
    return {"is_recollection": False, "recollection_reference": None}


def _quote_grounded(source_sentence: str, supporting_quote: str) -> bool:
    """Is `supporting_quote` actually present in `source_sentence`?"""
    def _norm(t):
        return re.sub(r"\s+", " ", str(t or "").lower()).strip(" .,\"'\u201c\u201d")
    src, q = _norm(source_sentence), _norm(supporting_quote)
    if not q:
        return False
    if q in src:
        return True
    words = set(q.split())
    if len(q.split()) >= 3 and words:
        return sum(1 for w in words if w in src) / len(words) >= 0.7
    return False


# ── Cluster-name canonicalisation ─────────────
_CLUSTER_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "with", "by"}

def canonical_cluster_name(name: str) -> str:
    """Normalises a cluster name so trivial variants collapse to one identity."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    words = []
    for w in s.split():
        if w in _CLUSTER_STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        words.append(w)
    words.sort()
    return " ".join(words) if words else s.strip()


def cluster_key_from_name(name: str) -> str:
    """Collision-free cluster key: canonical form + short hash."""
    canon  = canonical_cluster_name(name)
    digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:8]
    slug   = re.sub(r"\s+", "_", canon)[:40]
    return f"{slug}__{digest}"


# ── Retrieval / matching thresholds ─────────────────────
TOP_K_CANDIDATES     = 8
CANDIDATE_MIN_SCORE  = 0.30
SANITY_FLOOR         = 0.30
CLUSTER_MATCH_SCORE  = 0.80
SAME_CLUSTER_AUTO_MAP = 0.82


# ── Provisional state / clinician gate ────────────────
def _state_visible_to(state_data: dict, patient_id: Optional[str]) -> bool:
    """A permanent state is visible to everyone. A provisional state is visible only to
    the patient(s) whose events triggered its creation, until a clinician promotes it."""
    status = state_data.get("state_status", "permanent")
    if status != "provisional":
        return True
    owners = state_data.get("provisional_for_patients") or []
    if not owners:
        return True
    return patient_id is not None and patient_id in owners


def _new_provisional_state_fields(patient_id: Optional[str], source: dict) -> dict:
    """Metadata for a freshly created provisional state."""
    return {
        "state_status":              "provisional",
        "provisional_for_patients":  [patient_id] if patient_id else [],
        "provisional_created_at":    datetime.now(timezone.utc).isoformat(),
        "provisional_source":        source,
        "review": {
            "decision":    None,
            "reviewer":    None,
            "reviewed_at": None,
            "note":        None,
        },
    }


def _merge_domains_into(target_db: dict, on_disk: dict, exclude_codes: set = None) -> None:
    """Union on-disk clusters/states into target_db. Never drops target_db's own additions."""
    exclude_codes = exclude_codes or set()
    my_domains = target_db.setdefault("domains", {})
    for dk, dom in on_disk.get("domains", {}).items():
        my_dom = my_domains.setdefault(dk, {
            "domain_name": dom.get("domain_name", dk),
            "domain_definition": dom.get("domain_definition", ""),
            "clusters": {},
        })
        my_clusters = my_dom.setdefault("clusters", {})
        for ck, clu in dom.get("clusters", {}).items():
            if ck not in my_clusters:
                kept = [s for s in clu.get("states", [])
                       if s.get("code") not in exclude_codes]
                if kept:
                    clu = dict(clu, states=kept)
                    my_clusters[ck] = clu
                continue
            existing_codes = {s.get("code") for s in my_clusters[ck].get("states", [])}
            for s in clu.get("states", []):
                code = s.get("code")
                if code in exclude_codes:
                    continue
                if code not in existing_codes:
                    my_clusters[ck].setdefault("states", []).append(s)


def save_state_space_dict(filepath: str, db: dict, exclude_codes: set = None) -> None:
    """Cross-process-safe write of state-space dict to disk."""
    lock_path = filepath + ".lock"
    lock_fh = None
    try:
        if fcntl is not None:
            lock_fh = open(lock_path, "w")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

        if os.path.exists(filepath):
            try:
                with open(filepath, encoding="utf-8") as f:
                    on_disk = json.load(f)
                _merge_domains_into(db, on_disk, exclude_codes=exclude_codes)
            except (json.JSONDecodeError, OSError):
                pass

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logging.error("State space disk write failed: %s", e)
            raise
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                lock_fh.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 5.  STATE SPACE TREE — GLOBAL VECTOR ENGINE
# ─────────────────────────────────────────────────────────────

class StateSpaceTree:
    """Flat global semantic search over ALL states (no cluster gating)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        logging.info("Loading SentenceTransformer: all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.state_index:   dict = {}
        self.cluster_index: dict = {}

        if os.path.exists(filepath):
            with open(filepath, encoding="utf-8") as f:
                self.db = json.load(f)
            logging.info("State space loaded: %s", filepath)
        else:
            logging.warning("State space file not found: %s — starting with empty DB.", filepath)
            self.db = {"domains": {}}

        self._build_indexes()

        self.index_lock    = asyncio.Lock()
        self.creation_lock = asyncio.Lock()

    def _build_indexes(self) -> None:
        new_state_idx:   dict = {}
        new_cluster_idx: dict = {}

        for dom_data in self.db.get("domains", {}).values():
            for clu_key, clu_data in dom_data.get("clusters", {}).items():
                cname = clu_data.get("cluster_name", "")
                if clu_key not in new_cluster_idx:
                    ctext = cname + " " + str(clu_data.get("cluster_definition", ""))
                    new_cluster_idx[clu_key] = {
                        "name":   cname,
                        "canon":  canonical_cluster_name(cname),
                        "vector": self.embedder.encode(ctext, convert_to_tensor=True),
                    }

                for state in clu_data.get("states", []):
                    code = state.get("code")
                    if not code:
                        continue
                    _state_text = (
                        str(state.get("name", "")).strip()
                        + ". "
                        + str(state.get("description", "")).strip()
                    ).strip(". ")
                    new_state_idx[code] = {
                        "cluster_key":  clu_key,
                        "cluster_name": cname,
                        "data":         state,
                        "vector":       self.embedder.encode(_state_text, convert_to_tensor=True),
                    }

        self.state_index   = new_state_idx
        self.cluster_index = new_cluster_idx
        logging.info("Index built: %d clusters | %d states",
                     len(self.cluster_index), len(self.state_index))

    async def global_search(self, text: str, top_k: int = TOP_K_CANDIDATES,
                            patient_id: Optional[str] = None) -> list[dict]:
        """Flat search across every VISIBLE state — no cluster gating."""
        async with self.index_lock:
            idx = self.state_index
            if not idx or not text:
                return []
            query_vec = await asyncio.to_thread(
                self.embedder.encode, text, convert_to_tensor=True
            )

        scored = [
            {
                "code":            code,
                "score":           round(util.cos_sim(query_vec, info["vector"]).item(), 4),
                "name":            info["data"].get("name", ""),
                "description":     info["data"].get("description", ""),
                "cluster_name":    info["cluster_name"],
                "experience_type": info["data"].get("experience_type", ""),
                "state_status":    info["data"].get("state_status", "permanent"),
            }
            for code, info in idx.items()
            if _state_visible_to(info["data"], patient_id)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [c for c in scored if c["score"] >= CANDIDATE_MIN_SCORE][:top_k]

    async def _match_existing_cluster(self, proposed_name: str) -> Optional[str]:
        """Cluster-level dedup: returns existing cluster_key when the proposed name
        is a trivial/semantic variant of one already there."""
        if not self.cluster_index:
            return None
        canon = canonical_cluster_name(proposed_name)
        for ckey, info in self.cluster_index.items():
            if info["canon"] == canon:
                return ckey
        async with self.index_lock:
            qv = await asyncio.to_thread(
                self.embedder.encode, proposed_name, convert_to_tensor=True
            )
        best_key, best = None, 0.0
        for ckey, info in self.cluster_index.items():
            sc = util.cos_sim(qv, info["vector"]).item()
            if sc > best:
                best_key, best = ckey, sc
        return best_key if best >= CLUSTER_MATCH_SCORE else None

    def _save_db_sync(self, exclude_codes: set = None) -> None:
        """Rebuild indexes, then save with concurrent-safe merge+write."""
        self._build_indexes()
        save_state_space_dict(self.filepath, self.db, exclude_codes=exclude_codes)

    async def dedup_sweep(self, threshold: float) -> dict:
        """Deterministic post-run cleanup: merge same-cluster near-duplicates."""
        async with self.index_lock:
            entries = [
                {"code": code, "vector": info["vector"],
                 "cluster_key": info["cluster_key"],
                 "name": info["data"].get("name", ""),
                 "status": info["data"].get("state_status", "permanent")}
                for code, info in self.state_index.items()
            ]

        def _codenum(c: str) -> int:
            m = re.search(r"\d+", c)
            return int(m.group()) if m else 0

        remap: dict = {}
        n = len(entries)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = entries[i], entries[j]
                if a["cluster_key"] != b["cluster_key"]:
                    continue
                score = util.cos_sim(a["vector"], b["vector"]).item()
                if score < threshold:
                    continue

                a_perm = a["status"] != "provisional"
                b_perm = b["status"] != "provisional"
                if a_perm and not b_perm:
                    lo, hi = a["code"], b["code"]
                elif b_perm and not a_perm:
                    lo, hi = b["code"], a["code"]
                else:
                    lo, hi = sorted([a["code"], b["code"]], key=_codenum)

                winner = lo
                while winner in remap:
                    winner = remap[winner]
                if hi != winner:
                    remap[hi] = winner
                    logging.info("DEDUP SWEEP: merging %s -> %s (score=%.3f)",
                                 hi, winner, score)

        if not remap:
            return {}

        losers = set(remap.keys())

        by_code = {s.get("code"): s
                  for dom in self.db.get("domains", {}).values()
                  for clu in dom.get("clusters", {}).values()
                  for s in clu.get("states", [])}
        for loser_code, winner_code in remap.items():
            loser_state  = by_code.get(loser_code)
            winner_state = by_code.get(winner_code)
            if not loser_state or not winner_state:
                continue
            if winner_state.get("state_status") == "provisional":
                owners = set(winner_state.get("provisional_for_patients") or [])
                owners |= set(loser_state.get("provisional_for_patients") or [])
                winner_state["provisional_for_patients"] = sorted(owners)

        for dom in self.db.get("domains", {}).values():
            for clu in dom.get("clusters", {}).values():
                clu["states"] = [s for s in clu.get("states", [])
                                 if s.get("code") not in losers]
            empty = [ck for ck, clu in dom.get("clusters", {}).items()
                     if not clu.get("states")]
            for ck in empty:
                del dom["clusters"][ck]

        async with self.index_lock:
            await asyncio.to_thread(self._save_db_sync, losers)

        logging.info("DEDUP SWEEP: %d state(s) merged away.", len(remap))
        return remap

    async def add_new_state(
        self,
        cluster_name: str,
        cluster_def: str,
        state_name: str,
        state_desc: str,
        lifecycle_tier: str = "Lifespan",
        experience_type: str = "unknown",
        patient_id: Optional[str] = None,
        source: Optional[dict] = None,
    ) -> dict:
        """Creates a new PROVISIONAL state, scoped to patient_id."""
        domain_key  = "general_trauma"

        if domain_key not in self.db["domains"]:
            self.db["domains"][domain_key] = {
                "domain_name":       "General Trauma (Dynamic)",
                "domain_definition": "Dynamically generated states for novel clinical mechanisms.",
                "clusters":          {},
            }
        clusters = self.db["domains"][domain_key]["clusters"]

        existing_ck = await self._match_existing_cluster(cluster_name)
        if existing_ck and existing_ck in clusters:
            cluster_key = existing_ck
        else:
            cluster_key = cluster_key_from_name(cluster_name)
            if cluster_key not in clusters:
                clusters[cluster_key] = {
                    "cluster_name":       cluster_name,
                    "cluster_definition": cluster_def or "Dynamically generated cluster.",
                    "states":             [],
                }

        total_states = sum(
            len(c.get("states", []))
            for d in self.db.get("domains", {}).values()
            for c in d.get("clusters", {}).values()
        )
        new_code = f"dyn-{total_states + 1:03d}"

        new_state = {
            "code":            new_code,
            "name":            state_name,
            "description":     state_desc,
            "lifecycle_tier":  lifecycle_tier,
            "experience_type": experience_type,
            **_new_provisional_state_fields(patient_id, source or {}),
        }
        clusters[cluster_key]["states"].append(new_state)

        async with self.index_lock:
            await asyncio.to_thread(self._save_db_sync)

        logging.info("NEW PROVISIONAL STATE: %s | cluster=%s | %s | pending clinician review",
                     new_code, clusters[cluster_key]["cluster_name"], state_name)
        return {"code": new_code,
                "cluster_name": clusters[cluster_key]["cluster_name"],
                "description": state_desc,
                "state_status": "provisional"}


# ─────────────────────────────────────────────────────────────
# 5A.  CLINICIAN REVIEW WORKFLOW
# ─────────────────────────────────────────────────────────────

def list_provisional_states(db: dict, patient_id: Optional[str] = None) -> list[dict]:
    """All provisional states not yet reviewed, optionally filtered by patient."""
    out = []
    for dom_key, dom in db.get("domains", {}).items():
        for clu_key, clu in dom.get("clusters", {}).items():
            for s in clu.get("states", []):
                if s.get("state_status") != "provisional":
                    continue
                review = s.get("review") or {}
                if review.get("decision") is not None:
                    continue
                owners = s.get("provisional_for_patients") or []
                if patient_id is not None and patient_id not in owners:
                    continue
                out.append({
                    "code":                      s.get("code"),
                    "name":                      s.get("name"),
                    "description":               s.get("description"),
                    "cluster_name":              clu.get("cluster_name"),
                    "experience_type":           s.get("experience_type"),
                    "provisional_for_patients":  owners,
                    "provisional_created_at":    s.get("provisional_created_at"),
                    "provisional_source":        s.get("provisional_source"),
                    "decision":                  None,
                    "reviewer":                  None,
                    "note":                      None,
                })
    return out


def export_provisional_review(db: dict, out_path: str,
                              patient_id: Optional[str] = None) -> int:
    """Write the pending-review queue to a JSON file for clinician editing."""
    queue = list_provisional_states(db, patient_id=patient_id)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pending_review": queue}, f, indent=2, ensure_ascii=False)
    logging.info("Exported %d provisional state(s) for review -> %s", len(queue), out_path)
    return len(queue)


def apply_provisional_review(state_space_path: str, review_path: str) -> dict:
    """Apply clinician decisions to the state-space file."""
    with open(state_space_path, encoding="utf-8") as f:
        db = json.load(f)
    with open(review_path, encoding="utf-8") as f:
        review_data = json.load(f)
    entries = review_data.get("pending_review", review_data if isinstance(review_data, list) else [])

    approved, rejected, skipped = [], [], []
    by_code = {s.get("code"): s
              for dom in db.get("domains", {}).values()
              for clu in dom.get("clusters", {}).values()
              for s in clu.get("states", [])}

    for entry in entries:
        code     = entry.get("code")
        decision = entry.get("decision")
        state    = by_code.get(code)
        if not state or state.get("state_status") != "provisional":
            continue
        if decision == "approve":
            state["state_status"] = "permanent"
            state["review"] = {
                "decision":    "approve",
                "reviewer":    entry.get("reviewer"),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "note":        entry.get("note"),
            }
            approved.append(code)
        elif decision == "reject":
            state["review"] = {
                "decision":    "reject",
                "reviewer":    entry.get("reviewer"),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "note":        entry.get("note"),
            }
            rejected.append(code)
        else:
            skipped.append(code)

    if rejected:
        rejected_set = set(rejected)
        for dom in db.get("domains", {}).values():
            for clu in dom.get("clusters", {}).values():
                clu["states"] = [s for s in clu.get("states", [])
                                 if s.get("code") not in rejected_set]
            empty = [ck for ck, clu in dom.get("clusters", {}).items()
                     if not clu.get("states")]
            for ck in empty:
                del dom["clusters"][ck]

    save_state_space_dict(state_space_path, db, exclude_codes=set(rejected))

    logging.info("Review applied: %d approved, %d rejected, %d still pending.",
                 len(approved), len(rejected), len(skipped))
    return {"approved": approved, "rejected": rejected, "skipped": skipped}


def flag_rejected_states_in_events(events: list, rejected_codes: list) -> int:
    """Flag sub-events that mapped to rejected states for re-processing."""
    if not rejected_codes:
        return 0
    rejected_set = set(rejected_codes)
    changed = 0
    for ev in events:
        if isinstance(ev.get("state_codes"), list):
            if any(c in rejected_set for c in ev["state_codes"]):
                ev["state_codes"] = [c for c in ev["state_codes"] if c not in rejected_set]
        for sub in ev.get("sub_events", []):
            if sub.get("state_code") in rejected_set:
                sub["status"] = "REJECTED_STATE_NEEDS_REMAP"
                sub["rejected_state_code"] = sub.pop("state_code")
                sub["mapping_reason"] = (
                    "Previously mapped state was rejected by clinician review; "
                    "needs re-processing.")
                changed += 1
    return changed


# ─────────────────────────────────────────────────────────────
# 6.  LLM CALLER
# ─────────────────────────────────────────────────────────────

MODEL_NAME  = "gpt-5.1"
API_URL     = "https://api.openai.com/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"

_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)


def _is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))


def _build_llm_payload(model_name: str, system_prompt: str, user_prompt: str,
                       max_output_tokens: int) -> dict:
    payload = {
        "model": model_name,
        "response_format": {"type": "json_object"},
        "messages": (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": user_prompt}],
    }
    if _is_reasoning_model(model_name):
        payload["max_completion_tokens"] = max_output_tokens
        payload["reasoning_effort"] = "none"
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = max_output_tokens
    return payload


async def call_llm(
    session:       aiohttp.ClientSession,
    api_key:       str,
    system_prompt: str,
    user_prompt:   str,
    counter:       TokenCounter,
    max_retries:   int = 3,
    max_output_tokens: int = 4000,
) -> dict | None:
    payload = _build_llm_payload(MODEL_NAME, system_prompt, user_prompt, max_output_tokens)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                API_URL,
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    counter.add(data.get("usage", {}))
                    try:
                        raw = data["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError) as e:
                        logging.warning("Malformed API response attempt %d: %s", attempt+1, e)
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if raw.startswith("```"):
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw).strip()
                    return json.loads(raw)
                if resp.status == 429:
                    wait = 2 ** (attempt + 1)
                    logging.warning("Rate limited — waiting %ds", wait)
                    await asyncio.sleep(wait)
                else:
                    logging.warning("LLM HTTP %d on attempt %d", resp.status, attempt+1)
                    await asyncio.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            logging.warning("JSON parse error on attempt %d: %s — retrying", attempt+1, e)
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logging.debug("LLM attempt %d failed: %s", attempt+1, e)
            await asyncio.sleep(2 ** attempt)

    return None


# ─────────────────────────────────────────────────────────────
# 7.  SINGLE EVENT PROCESSOR
# ─────────────────────────────────────────────────────────────

async def process_event(
    session:       aiohttp.ClientSession,
    api_key:       str,
    db_tree:       StateSpaceTree,
    event_obj:     dict,
    idx:           int,
    total:         int,
    section_map:   dict,
    context_map:   dict,
    counter:       TokenCounter,
    dry_run:       bool,
    patient_id:    Optional[str] = None,
) -> dict:
    raw_event   = str(event_obj.get("event") or "")
    life_stage  = str(event_obj.get("life_stage") or "Unknown").capitalize()
    unit_id     = str(event_obj.get("unit_id") or f"event_{idx}")
    orig_status = str(event_obj.get("status") or "")

    enriched_ctx    = build_enriched_context(event_obj, section_map, context_map)
    source_sentence = str(event_obj.get("source_sentence") or "")
    llm_agreement   = str(event_obj.get("llm_agreement") or "AGREED")
    is_partial      = llm_agreement in ("A_ONLY", "B_ONLY")

    if is_partial and source_sentence and source_sentence.strip() != raw_event.strip():
        partial_note = (
            "\n⚠️ PARTIAL EXTRACTION — only one model extracted this event "
            "(the other returned null). The pipeline extraction below may be INCOMPLETE.\n"
            "Decompose from the ORIGINAL CLINICAL TEXT."
        )
    else:
        partial_note = ""

    _clinical_text      = source_sentence or raw_event
    _show_extraction    = raw_event.strip() and raw_event.strip() != _clinical_text.strip()
    _extraction_section = (
        f"\nPIPELINE EXTRACTION (reference — may be incomplete for A_ONLY/B_ONLY):\n"
        f"{raw_event}"
    ) if _show_extraction else ""

    oracle_req = (
        f"PATIENT LIFE STAGE: [{life_stage}]\n\n"
        f"BACKGROUND CONTEXT (read-only — DO NOT extract sub-events from this):\n"
        f"{enriched_ctx or '(none provided)'}\n\n"
        f"ORIGINAL CLINICAL TEXT (the ONLY valid source for sub-events):\n"
        f"{_clinical_text}"
        f"{partial_note}"
        f"{_extraction_section}"
    )

    logging.info("[%d/%d] %s", idx, total, unit_id)

    if dry_run:
        print(f"\n{'='*60}\nDRY RUN — {unit_id}\n{'='*60}")
        print("ORACLE PROMPT (system):")
        print(ORACLE_PROMPT[:300] + "…")
        print("\nORACLE REQUEST (user):")
        print(oracle_req)
        out = event_obj.copy()
        out["status"] = "DRY_RUN"
        return out

    # ── STEP 1: ORACLE ───────────────────────────────────────
    oracle_res = await call_llm(session, api_key, ORACLE_PROMPT, oracle_req, counter)
    if not oracle_res:
        logging.error("Oracle failed for %s — returning original event.", unit_id)
        return event_obj

    sub_events_raw = oracle_res.get("sub_events", [])
    if not isinstance(sub_events_raw, list) or not sub_events_raw:
        logging.warning("Oracle returned no sub_events for %s", unit_id)
        return event_obj

    logging.debug("Oracle reasoning: %s", oracle_res.get("decomposition_reasoning", "")[:120])

    final_sub_events: list = []
    mapped_codes:     list = []
    mapped_clusters:  list = []

    for sub_idx, sub in enumerate(sub_events_raw):
        sub_text      = str(sub.get("text") or "").strip()
        _raw_exp_type = str(sub.get("experience_type") or "unknown").lower().strip()
        exp_type      = _raw_exp_type if _raw_exp_type in (
            "event", "symptom", "pattern", "belief"
        ) else "unknown"
        raw_status    = str(sub.get("status") or "").upper()
        recoll        = detect_recollection(sub_text)

        supporting_quote = str(sub.get("supporting_quote") or "").strip()

        # ── AGE EXTRACTION ───────
        onset = _safe_int(sub.get("onset_age"))
        end   = _safe_int(sub.get("end_age"))
        ex_onset, ex_end, ex_src = extract_temporal_anchor(
            sub_text, event_obj, supporting_quote=supporting_quote)
        ex_explicit = ex_src in ("quote_tag", "subtext_tag", "sub_text")
        if onset is None:
            if ex_onset is not None:
                onset = ex_onset
                if end is None:
                    end = ex_end
        else:
            if ex_explicit and ex_onset is not None and ex_onset != onset:
                onset = ex_onset
                end = ex_end
            elif end is None and ex_onset == onset and ex_end is not None:
                end = ex_end
        if end is None and onset is not None:
            end = onset

        _death_text = (sub_text + " " + supporting_quote).lower()
        if any(w in _death_text for w in _DEATH_MARKERS) and onset is not None \
                and end is not None and onset != end:
            onset = end

        quote_grounded = _quote_grounded(source_sentence or raw_event, supporting_quote)

        sub_record = {
            "sub_id":                 f"{unit_id}_{chr(97+sub_idx) if sub_idx<26 else f's{sub_idx}'}",
            "event":                  sub_text,
            "experience_type":        exp_type,
            "status":                 "UNKNOWN",
            "onset_age":              onset,
            "end_age":                end,
            "is_ongoing":             bool(sub.get("is_ongoing", False)),
            "is_recollection":        recoll["is_recollection"],
            "recollection_reference": recoll["recollection_reference"],
            "supporting_quote":       supporting_quote or None,
            "quote_grounded":         quote_grounded,
        }

        if sub_text and supporting_quote and not quote_grounded:
            logging.warning(
                "[%s] sub-event may be UNGROUNDED in source (quote not found): "
                "event=%r quote=%r", unit_id, sub_text[:60], supporting_quote[:60])

        _proposed_name = str(sub.get("ideal_state_name") or "")
        if _proposed_name.count(",") >= 2:
            logging.warning("Rejected compound state name for %s: '%s'",
                            unit_id, _proposed_name[:60])
            sub_record["status"]            = "UNRESOLVED"
            sub_record["unresolved_reason"] = (
                "Compound state name rejected. List multiple conditions as separate sub-events."
            )
            final_sub_events.append(sub_record)
            continue

        if not sub_text:
            continue

        if raw_status == "NEUTRAL":
            sub_record["status"]         = "NEUTRAL"
            sub_record["neutral_reason"] = str(sub.get("neutral_reason") or "")
            final_sub_events.append(sub_record)
            continue

        # ── GROUNDING GATE ─────────────────────────────────────
        if supporting_quote and not quote_grounded:
            sub_record["status"] = "UNGROUNDED"
            sub_record["ungrounded_reason"] = (
                "supporting_quote not found in this event's source sentence — "
                "likely from background context; recorded but excluded from state mapping.")
            final_sub_events.append(sub_record)
            continue

        # ── STEP 2+3: GLOBAL SEARCH -> JUDGE -> MAP/CREATE ──
        async with db_tree.creation_lock:
            fresh_name  = str(sub.get("ideal_state_name") or "").strip()
            fresh_desc  = str(sub.get("ideal_state_description") or sub_text).strip()
            fresh_query = (
                f"{fresh_name}. {fresh_desc}" if fresh_name else fresh_desc
            ).strip(". ")

            candidates = await db_tree.global_search(fresh_query, top_k=TOP_K_CANDIDATES,
                                                     patient_id=patient_id)

            matched   = None
            judge_res = None
            judge_called = False

            if candidates:
                judge_req = MATCH_JUDGE_PROMPT.format(
                    proposed_name   = fresh_name or "(unnamed)",
                    proposed_type   = exp_type,
                    proposed_desc   = fresh_desc,
                    candidates_json = json.dumps(candidates, indent=2, ensure_ascii=False),
                )
                judge_called = True
                judge_res = await call_llm(session, api_key, "", judge_req, counter,
                                           max_output_tokens=300)

                if judge_res and judge_res.get("map") is True and judge_res.get("code"):
                    code = str(judge_res["code"])
                    cand = next((c for c in candidates if c["code"] == code), None)
                    if cand and cand["score"] >= SANITY_FLOOR:
                        matched = cand

                if matched is None and judge_res is not None and candidates:
                    proposed_cluster = str(sub.get("ideal_cluster_name") or "")
                    top = candidates[0]
                    same_cluster = (
                        canonical_cluster_name(top["cluster_name"])
                        == canonical_cluster_name(proposed_cluster)
                    )
                    if same_cluster and top["score"] >= SAME_CLUSTER_AUTO_MAP:
                        matched = top
                        logging.info(
                            "SAME-CLUSTER GUARD: auto-mapped to %s (%s, score=%.3f)",
                            top["code"], top["name"], top["score"],
                        )

            if matched:
                sub_record.update({
                    "status":              "MAPPED",
                    "state_code":          matched["code"],
                    "cluster_name":        matched["cluster_name"],
                    "similarity":          matched["score"],
                    "mapping_reason":      str((judge_res or {}).get("reasoning") or "Matched existing state."),
                    "mapped_state_status": matched.get("state_status", "permanent"),
                })
                mapped_codes.append(matched["code"])
                mapped_clusters.append(matched["cluster_name"])

            elif judge_called and judge_res is None:
                sub_record["status"] = "JUDGE_FAILED"
                logging.warning("Judge call failed for %s — not creating a state.",
                                sub_record["sub_id"])

            else:
                new_state = await db_tree.add_new_state(
                    cluster_name    = str(sub.get("ideal_cluster_name") or "General Trauma"),
                    cluster_def     = str(sub.get("ideal_cluster_definition") or ""),
                    state_name      = fresh_name or "New State",
                    state_desc      = fresh_desc,
                    lifecycle_tier  = str(sub.get("lifecycle_tier") or "Lifespan"),
                    experience_type = exp_type,
                    patient_id      = patient_id,
                    source = {
                        "unit_id":          unit_id,
                        "sub_id":           sub_record["sub_id"],
                        "event_text":       sub_text,
                        "supporting_quote": supporting_quote or None,
                    },
                )
                sub_record.update({
                    "status":              "NEW_STATE_CREATED",
                    "state_code":          new_state["code"],
                    "cluster_name":        new_state["cluster_name"],
                    "state_description":   new_state["description"],
                    "mapping_reason":      "No semantic match in global search — created "
                                           "as PROVISIONAL, pending clinician review.",
                    "mapped_state_status": "provisional",
                })
                mapped_codes.append(new_state["code"])
                mapped_clusters.append(new_state["cluster_name"])

        final_sub_events.append(sub_record)

    out = event_obj.copy()
    out.update({
        "status":                  "COMPOUND",
        "was_review_required":     (orig_status == "REVIEW_REQUIRED"),
        "decomposition_reasoning": str(oracle_res.get("decomposition_reasoning") or ""),
        "sub_event_count":         len(final_sub_events),
        "sub_events":              final_sub_events,
        "state_codes":             list(dict.fromkeys(mapped_codes)),
        "cluster_symbols":         list(dict.fromkeys(mapped_clusters)),
    })
    return out


# ─────────────────────────────────────────────────────────────
# 8.  MAIN LOOP
# ─────────────────────────────────────────────────────────────

def _remap_event_codes(events: list, remap: dict) -> int:
    """Rewrite merged state codes inside patient events after dedup sweep."""
    if not remap:
        return 0
    changes = 0
    for ev in events:
        if isinstance(ev.get("state_codes"), list):
            new = []
            for c in ev["state_codes"]:
                if c in remap:
                    changes += 1
                    new.append(remap[c])
                else:
                    new.append(c)
            ev["state_codes"] = list(dict.fromkeys(new))
        for sub in ev.get("sub_events", []):
            if sub.get("state_code") in remap:
                sub["state_code"] = remap[sub["state_code"]]
                changes += 1
    return changes


async def main_loop(args: argparse.Namespace) -> None:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key and not args.dry_run:
        logging.error("%s environment variable is required.", API_KEY_ENV)
        sys.exit(1)

    if args.fresh_start:
        sp = Path(args.state_space)
        if sp.exists():
            sp.unlink()
        with open(args.state_space, "w", encoding="utf-8") as _f:
            json.dump({"domains": {}}, _f)
        logging.info("[FRESH START] Empty state space written: %s", args.state_space)

    logging.info("Initialising Vector Engine: %s", args.state_space)
    db_tree = StateSpaceTree(args.state_space)

    try:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error("Input file not found: %s", args.input)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON: %s — %s", args.input, e)
        sys.exit(1)

    if not isinstance(data, dict) or "events" not in data:
        logging.error("Input is not a valid Worker 1 output (needs 'events' key).")
        sys.exit(1)

    all_events = data.get("events") or []
    logging.info("Input events loaded: %d total", len(all_events))

    patient_id = str(data.get("patient_id") or Path(args.input).stem)
    logging.info("Patient scope for provisional states: %s", patient_id)

    section_map, context_map = {}, {}
    if args.stage1 and Path(args.stage1).exists():
        try:
            with open(args.stage1, encoding="utf-8") as f:
                stage1_data = json.load(f)
            if isinstance(stage1_data, list):
                section_map, context_map = build_section_map(stage1_data)
                logging.info("Stage 1 context loaded: %d sections", len(section_map))
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("Could not load stage1 file: %s — context skipped.", e)

    resume_path = Path(args.output).with_suffix(".resume.json")
    resumer     = ResumeManager(resume_path, enabled=args.resume)
    counter     = TokenCounter()

    eligible_indices = [
        i for i, e in enumerate(all_events)
        if str(e.get("status") or "") in ("CONFIRMED", "TENTATIVE", "REVIEW_REQUIRED")
        and str(e.get("event") or "").strip()
    ]
    logging.info("Eligible: %d / %d", len(eligible_indices), len(all_events))

    if args.concurrency < 1:
        args.concurrency = 1
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bound_worker(rank: int, ev: dict) -> dict:
        unit_id = str(ev.get("unit_id") or f"event_{rank}")
        if resumer.is_done(unit_id):
            logging.info("[SKIP] %s (already done)", unit_id)
            return resumer.get(unit_id)
        try:
            async with semaphore:
                result = await process_event(
                    session=session, api_key=api_key or "",
                    db_tree=db_tree, event_obj=ev,
                    idx=rank, total=len(eligible_indices),
                    section_map=section_map, context_map=context_map,
                    counter=counter, dry_run=args.dry_run,
                    patient_id=patient_id,
                )
        except Exception as exc:
            logging.error("Unhandled error for %s: %s", unit_id, exc, exc_info=True)
            result = ev
        if result.get("status") == "COMPOUND":
            await resumer.mark_done(unit_id, result)
        return result

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[bound_worker(rank+1, all_events[i]) for rank, i in enumerate(eligible_indices)],
            return_exceptions=True,
        )

    for orig_idx, result in zip(eligible_indices, results):
        if isinstance(result, Exception):
            logging.error("Task failed for event %d: %s", orig_idx, result)
        else:
            all_events[orig_idx] = result

    # ── FINAL DEDUP SWEEP ────────────────────────────────────────────────
    if not args.dry_run and not args.no_dedup_sweep:
        logging.info("Running final dedup sweep (threshold=%.2f)...", args.dedup_threshold)
        remap = await db_tree.dedup_sweep(args.dedup_threshold)
        if remap:
            changed = _remap_event_codes(all_events, remap)
            logging.info("Dedup sweep merged %d state(s); remapped %d code reference(s).",
                         len(remap), changed)
            data["worker2_dedup_merges"] = remap
        else:
            logging.info("Dedup sweep: no same-cluster duplicates found.")

    pending_for_patient = len(list_provisional_states(db_tree.db, patient_id=patient_id))

    data["pipeline_stage"] = "worker2_v48_general"
    data["events"]         = all_events
    data["worker2_stats"]  = {
        "eligible_processed":               len(eligible_indices),
        "llm_calls":                        counter.calls,
        "tokens_total":                     counter.total_tokens,
        "cost_usd_estimate":                round(counter.cost_usd, 4),
        "patient_id":                       patient_id,
        "provisional_states_pending_review": pending_for_patient,
    }
    if pending_for_patient:
        logging.info(
            "%d provisional state(s) created for this patient are pending "
            "clinician review — run with --export-review <path> to list them.",
            pending_for_patient)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logging.error("Failed to write output: %s", e)
        sys.exit(1)

    if not args.dry_run and resume_path.exists() and not args.keep_resume:
        resume_path.unlink()
        logging.info("Resume file deleted.")

    logging.info("Output written to: %s", args.output)
    logging.info("State space updated: %s", args.state_space)
    logging.info(counter.summary())


# ─────────────────────────────────────────────────────────────
# 9.  SELF-TEST — offline regression guard
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    check(canonical_cluster_name("Anxiety Symptoms") == canonical_cluster_name("Anxiety Symptom"),
          "canonical_cluster_name: plural/singular variants collapse")
    check(cluster_key_from_name("Anxiety Symptoms") != cluster_key_from_name("Anger Symptoms"),
          "cluster_key_from_name: distinct names never collide")

    onset, end, prov = extract_temporal_anchor("Client [Childhood] was hit, age 8-10", {})
    check((onset, end) == (8, 10), "age: explicit 'age 8-10' range parsed")

    onset, end, prov = extract_temporal_anchor("Client cried for 10-11 hours after the incident", {})
    check((onset, end) != (10, 11), "age: '10-11 hours' duration is not misread as age range")

    onset, end, prov = extract_temporal_anchor("Client was born prematurely at 6 months", {})
    check(onset == 0, "age: '6 months' converts to age 0 (floor)")

    onset, end, prov = extract_temporal_anchor("Client felt unloved", {"age_context": "12-16"})
    check((onset, end) == (12, 16) and prov == "parent",
          "age: falls back to parent age_context when nothing else is present")

    check(_quote_grounded("Father beat mother during childhood.", "Father beat mother"),
          "quote_grounded: exact substring passes")
    check(not _quote_grounded("Client felt unloved and abandoned.", "grandmother was abusive"),
          "quote_grounded: unrelated quote correctly fails")

    rp = _build_llm_payload("gpt-5.1", "sys", "usr", 500)
    check("temperature" not in rp and rp.get("max_completion_tokens") == 500
          and rp.get("reasoning_effort") == "none",
          "payload: gpt-5.1 drops temperature, sets max_completion_tokens + reasoning_effort=none")
    cp = _build_llm_payload("gpt-4.1", "sys", "usr", 500)
    check(cp.get("temperature") == 0.0 and cp.get("max_tokens") == 500,
          "payload: non-reasoning model keeps temperature + max_tokens")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical State Space Matcher")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--state-space", default="fire_state_space_tree.json")
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--stage1", default=None)
    parser.add_argument("--output", default="final_mapped_events.json")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="Run offline regression checks and exit")
    parser.add_argument("--no-dedup-sweep", action="store_true",
                        help="Disable the automatic final same-cluster dedup sweep")
    parser.add_argument("--dedup-threshold", type=float, default=0.82,
                        help="Cosine threshold for final dedup sweep (default 0.82)")

    parser.add_argument("--export-review", metavar="PATH", default=None,
                        help="Write pending provisional-state review queue to PATH and exit.")
    parser.add_argument("--review-patient", default=None,
                        help="With --export-review, filter queue to this patient only.")
    parser.add_argument("--apply-review", metavar="PATH", default=None,
                        help="Apply clinician decisions from PATH to state-space and exit.")
    parser.add_argument("--patch-events", metavar="INPUT_OUTPUT_PAIR", nargs=2,
                        default=None,
                        help="With --apply-review: patch events file if states were rejected.")

    args = parser.parse_args()
    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt = "%H:%M:%S",
    )

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.export_review:
        with open(args.state_space, encoding="utf-8") as f:
            _db = json.load(f)
        export_provisional_review(_db, args.export_review, patient_id=args.review_patient)
        sys.exit(0)

    if args.apply_review:
        result = apply_provisional_review(args.state_space, args.apply_review)
        print(f"Approved: {len(result['approved'])}  "
              f"Rejected: {len(result['rejected'])}  "
              f"Still pending: {len(result['skipped'])}")
        if args.patch_events and result["rejected"]:
            events_in, events_out = args.patch_events
            with open(events_in, encoding="utf-8") as f:
                _data = json.load(f)
            changed = flag_rejected_states_in_events(_data.get("events", []), result["rejected"])
            with open(events_out, "w", encoding="utf-8") as f:
                json.dump(_data, f, indent=2, ensure_ascii=False)
            print(f"Flagged {changed} sub-event(s) referencing rejected state -> {events_out}")
        sys.exit(0)

    if not args.input:
        parser.error("input is required (unless using --selftest / --export-review / --apply-review)")

    asyncio.run(main_loop(args))
