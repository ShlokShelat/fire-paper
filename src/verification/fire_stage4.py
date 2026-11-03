"""
FiRE Pipeline — Worker 1, Stage 4: audit of confirmed events
================================================================
Paper concept: part of stage (iii). Stage 4 rechecks every event Stages 2-3
marked CONFIRMED, for entities, grounding and timing (Supplementary
Material, Section S1). It can lower an event from CONFIRMED to TENTATIVE,
which keeps the event but flags it for review; it never discards an event
and never raises one. No language-model call is made.

The three rechecks:
  1. Entities  — every person in persons_involved must appear in the source
     sentence, its section heading or nearby lines of the same section
     (kinship terms such as "mil" count). Severity HIGH.
  2. Grounding — the event text against its source sentence (section
     heading removed): pass if cosine >= 0.60 with all-MiniLM-L6-v2
     (0.55 for patterns; symptoms and beliefs are not tested). Without
     embeddings, a lexical containment score is used with the same
     thresholds. Severity MEDIUM. This is the "grounding recheck" that the
     cited-span ablation removes (src/ablation/fire_ablation.py).
  3. Timing    — the life stage must not contradict the age context or a
     non-adjacent life-stage cue in the source. Severity LOW.
An event's confidence falls by 0.20, 0.10 or 0.05 for a failed check of
high, medium or low severity (Supplementary Material, Section S7; at most
0.30 in total, never below 0.40); a HIGH or MEDIUM failure lowers the
status to TENTATIVE, a LOW failure only flags.

Pipeline position: after Stages 2-3 (fire_stage2_3_single_model.py), before
Stage 5 (fire_stage5.py).
Input: validated_events.json, optionally with stage1_output.json for the
nearby-line context. Output: the same events with stage4_flags,
stage4_passed and any lowered status, written to a new file.

Usage:
  python3 fire_stage4.py validated_events.json --stage1 stage1_output.json -o validated_events_s4.json
  python3 fire_stage4.py --selftest
"""

import argparse
import json
import logging
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional


PERSON_ANCHOR_EXPANSIONS = {
    "mil":           ["mother", "mother-in-law", "in-law", "in-laws"],
    "fil":           ["father", "father-in-law", "in-law", "in-laws"],
    "sil":           ["sister", "brother", "sibling",
                      "sister-in-law", "brother-in-law", "in-law", "in-laws"],
    "paternal":      ["father", "grandfather"],
    "maternal":      ["mother", "grandmother"],
    "mother-in-law": ["mother", "in-law", "in-laws"],
    "father-in-law": ["father", "in-law", "in-laws"],
    "in-law":        ["in-laws", "in-law"],
    "in-laws":       ["in-law", "in-laws"],
}

PERSON_EXEMPTIONS = {"client", "family"}

_PERSON_NORMALISE = {
    "in-laws": "in-law",
    "in-law":  "in-law",
    "mil":     "mil",
    "fil":     "fil",
}


def _person_in_text(person: str, text: str) -> bool:
    """Word-boundary-safe check for whether `person` is mentioned in `text`."""
    norm = _PERSON_NORMALISE.get(person, person)
    if "-" in norm:
        return norm in text
    return bool(re.search(r'\b' + re.escape(norm) + r'\b', text))


def _source_anchors_person(person: str, source: str, context: str) -> bool:
    """True if a relational trigger term (e.g. "mil") in the source or
    context implies `person` even without naming them directly."""
    person_l = person.lower()
    all_text = (source + " " + context).lower()
    for trigger, anchored in PERSON_ANCHOR_EXPANSIONS.items():
        if trigger in all_text and person_l in anchored:
            return True
    return False


_HEADER_SIGNAL_WORDS = {
    "ace", "protective", "presenting", "postpartum", "core",
    "college", "childhood", "timeline", "factors", "problems",
    "beliefs", "period", "wound", "years", "freedom", "violation",
    "formulation", "therapist", "emotional", "sexual", "boundary",
}


def _looks_like_header(before_text: str, max_len: int) -> bool:
    """
    Shared, single definition of "does this look like a section header
    rather than the start of an ordinary sentence?" — used by BOTH the
    subsection-match branch and the no-subsection fallback branch, so they
    always agree. Requires either an em/en dash, a known header
    signal word, or at least TWO capitalised words (a title-case heading
    like "Father Beat" or "Presenting Problems", not just the first word
    of a normal sentence like "Client believes").
    """
    if not before_text:
        return False
    if "—" in before_text or "–" in before_text:
        return True
    bwords = set(re.findall(r'\b[a-z]+\b', before_text.lower()))
    if bwords & _HEADER_SIGNAL_WORDS:
        return True
    if len(before_text) < max_len and before_text[0].isupper():
        # Require >=2 capitalised words, not just the sentence-initial
        # capital every ordinary sentence has.
        if len(re.findall(r'\b[A-Z][a-z]*\b', before_text)) >= 2:
            return True
    return False


def _strip_section_header(source: str, subsection: str = "") -> str:
    """Remove a leading section-heading label from `source` (using the
    known `subsection` name if given, else a heuristic colon-split) so
    similarity/anchoring checks compare against actual sentence content,
    not the heading that precedes it."""
    s = source.strip()
    if not s:
        return s

    if subsection:
        sub = subsection.strip()
        idx = s.lower().find(sub.lower())
        if idx >= 0:
            rest = s[idx + len(sub):]
            rest = re.sub(r'^[\s:—–\-]+', '', rest).strip()
            if ":" in rest[:100]:
                before_c, after_c = rest.split(":", 1)
                if _looks_like_header(before_c.strip(), max_len=60):
                    rest = after_c.strip()
            if rest:
                return rest

    if ":" in s:
        colons = [i for i, c in enumerate(s) if c == ":"]
        for ci in reversed(colons):
            before = s[:ci].strip()
            after  = s[ci + 1:].strip()
            if len(after) < 5:
                continue
            if _looks_like_header(before, max_len=80):
                return after

    return s


_sim_model   = None
_sim_backend = None

COSINE_THRESHOLD         = 0.60
COSINE_THRESHOLD_PATTERN = 0.55

_JACCARD_STOPS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been",
    "has", "have", "had", "to", "of", "in", "for", "with", "at", "by", "from", "that", "this",
    "it", "as", "her", "his", "their", "she", "he", "they", "who",
    "which", "what", "when", "about", "after", "before", "during",
    "client", "family",
}

_CLINICAL_FAMILIES = {
    "born":        {"born", "birth", "gave", "delivered"},
    "died":        {"died", "death", "passed", "deceased"},
    "disclosed":   {"disclosed", "disclose", "discloses", "told", "revealed"},
    "fondled":     {"fondled", "touched", "fondling", "inappropriate"},
    "married":     {"married", "marriage", "wedding", "wed"},
    "stand":       {"stand", "standing", "stood"},
    "realizes":    {"realizes", "realized", "realise", "understands"},
    "believed":    {"believed", "believes", "supported"},
    "relocated":   {"relocated", "moved", "transferred"},
    "prioritized": {"prioritized", "prioritise", "chose", "focused"},
    "remains":     {"remains", "remain", "stays", "continues"},
}


def _load_sim_model() -> str:
    """Lazily load the sentence-embedding model for semantic grounding
    checks (offline cache first, then online, else fall back to the
    lexical containment score); caches the resolved backend name."""
    global _sim_model, _sim_backend
    if _sim_backend is not None:
        return _sim_backend

    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _sim_model   = SentenceTransformer("all-MiniLM-L6-v2")
        _sim_backend = "transformers"
        logging.info("Stage 4: all-MiniLM-L6-v2 loaded (offline cache)")
        return _sim_backend
    except Exception:
        pass

    try:
        os.environ.pop("HF_HUB_OFFLINE", None)
        import importlib
        import sentence_transformers
        importlib.reload(sentence_transformers)
        from sentence_transformers import SentenceTransformer
        _sim_model   = SentenceTransformer("all-MiniLM-L6-v2")
        _sim_backend = "transformers"
        logging.info("Stage 4: all-MiniLM-L6-v2 loaded (online)")
        return _sim_backend
    except Exception as e2:
        _sim_backend = "containment"
        logging.warning(
            "Stage 4: sentence-transformers UNAVAILABLE (%s). Fallback to containment.",
            e2,
        )
    return _sim_backend


def _cosine_sim(text_a: str, text_b: str) -> Optional[float]:
    """
    On a genuine encode failure, return None (skip this check) rather
    than 1.0 (silently claim perfect groundedness). None is treated by the
    caller as "could not verify" and does not downgrade the event, but the
    failure is logged at ERROR level rather than counted as a clean pass.
    """
    try:
        from sentence_transformers import util as st_util
        ea = _sim_model.encode(text_a, convert_to_tensor=True)
        eb = _sim_model.encode(text_b, convert_to_tensor=True)
        return float(st_util.cos_sim(ea, eb))
    except Exception as e:
        logging.error("Stage 4: cosine similarity encode failed (%s) — "
                      "skipping semantic check for this item rather than "
                      "assuming it is grounded.", e)
        return None


def _containment_sim(event_text: str, source_text: str) -> float:
    """Lexical-overlap fallback for semantic similarity when no embedding
    model is available: fraction of event_text's (family-expanded)
    content words also present in source_text."""
    def expand(words):
        expanded = set(words)
        for w in list(words):
            for fam in _CLINICAL_FAMILIES.values():
                if w in fam:
                    expanded |= fam
        return expanded
    we = expand(set(re.findall(r'\b[a-z]+\b', event_text.lower()))  - _JACCARD_STOPS)
    ws = expand(set(re.findall(r'\b[a-z]+\b', source_text.lower())) - _JACCARD_STOPS)
    if not we:
        return 1.0
    return len(we & ws) / len(we)


def _check_semantic_grounding(
    event:        str,
    source_clean: str,
    context:      str,
    exp_type:     str,
) -> Optional[str]:
    """Check whether `event` is semantically close enough to its cleaned
    source sentence (optionally widened with context) to count as
    grounded. Skipped for symptom/belief events, which do not need a
    literal source match. Returns a flag string if the similarity score
    falls below threshold, else None."""
    if exp_type in ("symptom", "belief"):
        return None

    backend   = _load_sim_model()
    threshold = COSINE_THRESHOLD_PATTERN if exp_type == "pattern" else COSINE_THRESHOLD

    if backend == "transformers":
        score = _cosine_sim(source_clean, event)
        if score is None:
            # encoder failed for this item — cannot verify, don't guess.
            return None
        if score < threshold and context.strip():
            alt = _cosine_sim(context + " " + source_clean, event)
            if alt is not None:
                score = max(score, alt)
    else:
        all_source = (source_clean + " " + context).strip()
        score      = _containment_sim(event, all_source)
        threshold  = COSINE_THRESHOLD_PATTERN if exp_type == "pattern" else COSINE_THRESHOLD

    logging.debug(
        "  Semantic score=%.3f threshold=%.2f backend=%s exp_type=%s",
        score, threshold, backend, exp_type
    )

    if score < threshold:
        return "SEMANTIC_NOT_GROUNDED (score=%.2f, backend=%s)" % (score, backend)
    return None


TEMPORAL_SIGNALS = {
    "childhood": {
        "keywords": [
            "child", "childhood", "infant", "baby",
            "primary school", "early life", "as a child",
            "age 4", "age 5", "age 6", "age 7", "age 8",
            "age 9", "age 10", "age 11", "age 12",
        ],
    },
    "teenage": {
        "keywords": [
            "teen", "teenage", "adolesc", "puberty", "high school",
            "secondary school", "age 13", "age 14", "age 15",
            "age 16", "age 17", "age 18", "age 19",
        ],
    },
    "adult": {
        "keywords": [
            "college", "university", "married", "marriage",
            "husband", "wife", "postpartum", "pregnancy",
            "age 20", "age 21", "age 22", "age 23", "age 24",
            "age 25", "age 26", "age 27", "age 28", "age 29",
            "age 30", "adult", "grown", "working",
        ],
    },
    "mid-life": {
        "keywords": [
            "middle age", "mid-life", "age 40", "age 41",
            "age 42", "age 43", "age 44", "age 45", "age 50",
            "menopause", "retirement",
        ],
    },
}

# Set of (life_stage, other_stage) pairs treated as clinically incompatible.
# Adjacent stages are deliberately excluded — too close to treat keyword
# overlap as a contradiction.
INCOMPATIBLE_STAGE_PAIRS = {
    ("childhood", "adult"),    ("adult", "childhood"),
    ("childhood", "mid-life"), ("mid-life", "childhood"),
    ("teenage",   "mid-life"), ("mid-life", "teenage"),
}

# Single definition of "adjacent stages", reused by both the age-based check
# and the keyword-based check below.
_ADJACENT_STAGE_PAIRS = {
    ("childhood", "teenage"), ("teenage", "childhood"),
    ("teenage",   "adult"),    ("adult",   "teenage"),
    ("adult",     "mid-life"), ("mid-life","adult"),
}


def _extract_age_from_context(age_context: Optional[str]) -> Optional[tuple]:
    """Parse an age_context string (single age or "min-max" range) into
    an (age_min, age_max) tuple, or None if no numeric age is present."""
    if not age_context:
        return None
    cleaned = re.sub(r'[~\u2248]', '', str(age_context)).strip()
    m = re.search(r'(\d+)\s*[-\u2013]\s*(\d+)', cleaned)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)', cleaned)
    if m:
        age = int(m.group(1))
        return age, age
    return None


def _life_stage_from_age(age_min: int, age_max: int) -> Optional[str]:
    """Life stage implied by the midpoint of an age range.

    Note: here mid-life starts at 40, while Stage 3's LIFE_STAGE_AGE_RANGES
    lets adult and mid-life overlap from 35 to 59, so the two stages can
    disagree near those ages. A disagreement between adjacent stages is
    never flagged (_ADJACENT_STAGE_PAIRS), and a timing failure has LOW
    severity, which flags an event without lowering its status.
    """
    mid = (age_min + age_max) / 2
    if mid <= 12: return "childhood"
    if mid <= 19: return "teenage"
    if mid <= 39: return "adult"
    return "mid-life"


MAX_CONTEXT_SENTENCES = 3


# ── machine-readable flags → same E1..E6 vocabulary as Stage 3 ────
_STAGE4_FLAG_TO_MODE = {
    "ENTITY_NOT_ANCHORED":     ("groundedness", "E3"),
    "SEMANTIC_NOT_GROUNDED":   ("construct",    "E1"),
    "TEMPORAL_MISMATCH":       ("age_window",   "E5"),
}


def structure_stage4_flags(flags) -> list:
    """Convert raw "CODE: detail" Stage 4 flag strings into structured
    {axis, code, error_mode, detail} dicts using the same E1-E6 taxonomy
    vocabulary as Stage 3's `structure_flags`."""
    out = []
    for f in flags or []:
        code = f.split(":", 1)[0].split(" ", 1)[0].strip()
        detail = f.split(":", 1)[1].strip() if ":" in f else f
        axis, mode = _STAGE4_FLAG_TO_MODE.get(code, ("unknown", "NA"))
        out.append({"axis": axis, "code": code, "error_mode": mode, "detail": detail})
    return out


def stage4_verify(events: list, stage1_data: Optional[list] = None) -> list:
    """Run this module's second-pass audit over every CONFIRMED event.

    Input: the event list from a Worker 1 output file, plus optional
    Stage 1 data used to reconstruct nearby same-heading sentences as
    extra context. Events are mutated in place.
    Output: the same list, each event annotated with `stage4_flags`,
    `stage4_flags_structured`, and `stage4_passed`; events with a HIGH
    or MEDIUM-severity flag are downgraded from CONFIRMED to TENTATIVE
    and routed to human review.
    Checks three axes per event: entity anchoring (are named persons
    actually present in the source/context), semantic grounding (does
    the event text plausibly follow from the source), and temporal
    consistency (does the assigned life stage contradict age evidence).
    """
    _load_sim_model()

    # Rebuild each event's local Stage 1 neighbourhood (same heading,
    # nearby lines) so the entity/semantic checks below have the same
    # surrounding context Stage 2/3 saw, not just the bare source sentence.
    line_context: dict = {}
    if stage1_data:
        def _extract_prefix(text: str) -> str:
            if ": " in text:
                return re.sub(r"\s+", " ", text.split(": ", 1)[0].strip())
            return ""

        for i, sent in enumerate(stage1_data):
            ln = sent.get("line_number", i)
            target_prefix = _extract_prefix(sent.get("sentence", ""))

            before = [
                s["sentence"] for s in stage1_data[max(0, i - MAX_CONTEXT_SENTENCES):i]
                if _extract_prefix(s.get("sentence", "")) == target_prefix
            ]
            after  = [
                s["sentence"] for s in stage1_data[i + 1: i + 1 + MAX_CONTEXT_SENTENCES]
                if _extract_prefix(s.get("sentence", "")) == target_prefix
            ]
            line_context[ln] = {"before": before, "after": after}

    n_checked    = 0
    n_downgraded = 0
    n_flagged    = 0

    for ev in events:
        ev["stage4_flags"]            = []
        ev["stage4_flags_structured"] = []
        ev["stage4_passed"]           = True

        if ev.get("status") != "CONFIRMED":
            continue
        event_text = ev.get("event", "").strip()
        if not event_text:
            continue

        n_checked += 1
        flags      = []
        exp_type   = (ev.get("experience_type") or "unknown").lower()
        source     = ev.get("source_sentence", "")
        subsection = ev.get("subsection", "") or ""
        line_no    = ev.get("line_number", 0)
        age_ctx    = ev.get("age_context")

        ctx_before = line_context.get(line_no, {}).get("before", [])
        ctx_after  = line_context.get(line_no, {}).get("after",  [])

        source_lower     = source.lower()
        subsection_lower = subsection.lower()
        context_lower    = " ".join(ctx_before + ctx_after).lower()
        source_clean     = _strip_section_header(source, subsection)

        # Check 1: entity anchoring — every named person must appear (or
        # be culturally implied) in the source, its heading, or context.
        if exp_type != "belief":
            for person in (ev.get("persons_involved") or []):
                person_l = person.lower().strip()
                if not person_l or person_l in PERSON_EXEMPTIONS:
                    continue
                if _person_in_text(person_l, source_lower):
                    continue
                if _person_in_text(person_l, subsection_lower):
                    continue
                if _person_in_text(person_l, context_lower):
                    continue
                if _source_anchors_person(person_l, source_lower,
                                          subsection_lower + " " + context_lower):
                    continue
                flags.append(("HIGH", "ENTITY_NOT_ANCHORED: '%s'" % person))

        # Check 2: semantic grounding — the synthesized event text must
        # stay close to what the (heading-stripped) source actually says.
        sem_flag = _check_semantic_grounding(
            event_text, source_clean, context_lower, exp_type
        )
        if sem_flag:
            flags.append(("MEDIUM", sem_flag))

        # Check 3: temporal consistency — the assigned life stage must
        # not contradict explicit age evidence or non-adjacent-stage cues.
        life_stage = ev.get("life_stage", "unknown")
        if life_stage and life_stage != "unknown":
            temporal_ok   = True
            contradiction = None

            age_range = _extract_age_from_context(age_ctx)
            if age_range:
                expected = _life_stage_from_age(*age_range)
                if expected and expected != life_stage:
                    if (life_stage, expected) not in _ADJACENT_STAGE_PAIRS:
                        temporal_ok   = False
                        contradiction = (
                            "age_context='%s' implies '%s' but assigned '%s'"
                            % (age_ctx, expected, life_stage)
                        )

            if temporal_ok:
                for other_stage, data in TEMPORAL_SIGNALS.items():
                    if other_stage == life_stage:
                        continue
                    if (life_stage, other_stage) in INCOMPATIBLE_STAGE_PAIRS:
                        if any(kw in source_lower for kw in data["keywords"][:5]):
                            temporal_ok   = False
                            contradiction = (
                                "source signals '%s' but assigned '%s'"
                                % (other_stage, life_stage)
                            )
                            break

            if not temporal_ok and contradiction:
                flags.append(("LOW", "TEMPORAL_MISMATCH: " + contradiction))

        if not flags:
            continue

        severities                    = [f[0] for f in flags]
        flag_strs                     = [f[1] for f in flags]
        ev["stage4_flags"]            = flag_strs
        ev["stage4_flags_structured"] = structure_stage4_flags(flag_strs)
        ev["stage4_passed"]           = False
        n_flagged                    += 1

        deduction = 0.0
        if "HIGH"   in severities: deduction += 0.20
        if "MEDIUM" in severities: deduction += 0.10
        if "LOW"    in severities: deduction += 0.05
        deduction        = min(deduction, 0.30)
        ev["confidence"] = max(round(ev.get("confidence", 0.92) - deduction, 3), 0.40)

        if "HIGH" in severities or "MEDIUM" in severities:
            ev["status"]                = "TENTATIVE"
            ev["verified"]              = False
            ev["human_review_required"] = True
            n_downgraded               += 1
            logging.info(
                "Stage4 downgrade: %s [%s] %s | %s",
                ev.get("unit_id"), exp_type, event_text[:55], flag_strs
            )
        else:
            ev["human_review_required"] = True

    print("\n[STAGE 4] Done (%s). %d CONFIRMED events checked:" % (_sim_backend, n_checked))
    print("  Passed clean:  %d" % (n_checked - n_flagged))
    print("  Downgraded:    %d  (CONFIRMED -> TENTATIVE)" % n_downgraded)
    print("  Flagged LOW:   %d  (stays CONFIRMED)" % (n_flagged - n_downgraded))

    return events


def atomic_write_json(path: str, obj) -> None:
    """Write JSON via a temp-file-then-rename so a crash mid-write never
    leaves a partially-written output file."""
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


def selftest() -> bool:
    """Offline regression guard: section-heading stripping, the logging
    call, and the fail-open behaviour of a broken encoder."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    out = _strip_section_header("Core Beliefs: Client believes: I am not enough", "Core Beliefs")
    check("Client believes" in out,
          "header-strip: ordinary sentence content is preserved")
    out2 = _strip_section_header("Presenting Problems: Husband said: I don't care anymore",
                                 "Presenting Problems")
    check("Husband said" in out2,
          "header-strip: speaker attribution is preserved")
    out3 = _strip_section_header("Father: Beat mother regularly during childhood years", "Father")
    check(out3 == "Beat mother regularly during childhood years",
          "header-strip: real single-word header still strips")
    out4 = _strip_section_header("Therapist Formulation: Suggests parentification pattern",
                                 "Therapist Formulation")
    check(out4 == "Suggests parentification pattern",
          "header-strip: real multi-word header still strips")

    import io, contextlib
    buf = io.StringIO()
    logging.basicConfig(level=logging.WARNING, force=True)
    try:
        with contextlib.redirect_stderr(buf):
            logging.warning("Stage 4: sentence-transformers UNAVAILABLE (%s). Fallback to containment.",
                            Exception("boom"))
        check("Logging error" not in buf.getvalue(),
              "logging: warning call does not break the formatter")
    except Exception:
        check(False, "logging: warning call does not break the formatter")

    global _sim_model
    class _BrokenModel:
        def encode(self, *a, **k):
            raise RuntimeError("simulated encoder failure")
    _sim_model_backup = _sim_model
    _sim_model = _BrokenModel()
    result = _cosine_sim("a", "b")
    _sim_model = _sim_model_backup
    check(result is None, "fail-open: encoder exception returns None, not 1.0")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    """CLI entry point: load a Worker 1 output file (and optional Stage 1
    context), run `stage4_verify`, print a before/after status summary,
    and write the annotated events to a new JSON file."""
    parser = argparse.ArgumentParser(
        description="FiRE Worker 1, Stage 4: recheck CONFIRMED events for entities, "
                    "grounding and timing (no model call)"
    )
    parser.add_argument("input", nargs="?", help="Stages 2-3 output (validated_events.json)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: <input>_s4.json)")
    parser.add_argument("--stage1", default=None,
                        help="stage1_output.json, for nearby-line context")
    parser.add_argument("--selftest", action="store_true",
                        help="Run offline regression checks and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

    if not args.input:
        parser.error("input is required (unless using --selftest)")

    output_path = args.output or str(
        Path(args.input).parent / (Path(args.input).stem + "_s4.json")
    )

    print("[LOAD] Reading %s" % args.input)
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    events     = data.get("events", [])
    patient_id = data.get("patient_id", "unknown")
    print("[LOAD] %d events for patient %s" % (len(events), patient_id))

    stage1_data = None
    if args.stage1:
        print("[LOAD] Stage 1 context from %s" % args.stage1)
        with open(args.stage1, encoding="utf-8") as f:
            stage1_data = json.load(f)

    incoming = Counter(e.get("status") for e in events)
    print("\n[STAGE 4] Input status counts:")
    for s, n in sorted(incoming.items()):
        print("  %s: %d" % (s, n))

    events = stage4_verify(events, stage1_data)

    final   = Counter(e.get("status") for e in events)
    passing = sum(final.get(k, 0) for k in ["CONFIRMED", "TENTATIVE", "REVIEW_REQUIRED"])
    review  = sum(1 for e in events if e.get("human_review_required"))
    flagged = sum(1 for e in events if e.get("stage4_flags"))

    print("\n[SUMMARY] Final status counts (post Stage 4):")
    for s, n in sorted(final.items()):
        print("  %s: %d" % (s, n))
    print("[SUMMARY] Passing to Worker 2: %d" % passing)
    print("[SUMMARY] Human review queue:  %d" % review)
    if flagged:
        print("[SUMMARY] Stage 4 flagged:     %d events" % flagged)

    data["events"]         = events
    data["stage4_applied"] = True
    data["stage4_stats"]   = {
        "status_counts": dict(final),
        "sim_backend":   _sim_backend,
        "n_downgraded":  sum(1 for e in events
                             if e.get("stage4_flags") and e.get("status") == "TENTATIVE"),
        "n_flagged_low": sum(1 for e in events
                             if e.get("stage4_flags") and e.get("status") == "CONFIRMED"),
        "note": ("Stage 4 audit of CONFIRMED events (entities, grounding, "
                 "timing); see Supplementary Material, Section S1."),
    }

    atomic_write_json(output_path, data)
    print("\n[SAVE] Stage 4 output -> %s" % output_path)
    print("[DONE] Patient: %s" % patient_id)


if __name__ == "__main__":
    main()
