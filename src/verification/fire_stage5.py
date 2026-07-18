"""
FiRE Pipeline — Stage 5: Rescue Verifier + Semantic Deduplication
==================================================================
IMPORTANT — READ BEFORE USING THIS MODULE
------------------------------------------------------------------
1. PAPER ALIGNMENT — MODEL (RESOLVED). The paper's central
   methodological claim is a same-model ablation: "FiRE uses GPT-5.1 as
   its own extraction model, so the comparison between FiRE and GPT-5.1
   is a same-model ablation... Because FiRE and one baseline share the
   same underlying model, the gap is attributable to symbolic
   discipline rather than model strength." The rescue verifier now
   calls RESCUE_MODEL_NAME = "gpt-5.1", the same model Worker 1's
   extractor uses, via the OpenAI Chat Completions endpoint.

2. PAPER ALIGNMENT — STAGES (still open). The paper describes four FiRE
   stages; a rescue-and-deduplicate pass is not one of them and is not
   mentioned anywhere in the manuscript, same caveat as Stage 4. This
   is a documentation decision, not something this file can resolve on
   its own — see the Stage 4 file's docstring for the same discussion.

Runs on the output of Stage 4 (validated_events_s4.json).

TWO JOBS:

JOB 1 — RESCUE VERIFIER
  Events with status=DISCARD are not blindly dropped.
  Any DISCARD event that passed Stage 4 (stage4_passed=True) and has a
  non-null extracted event string gets a final rescue-model verification
  call: "Is this clinically significant for trauma assessment? Y/N"

  If YES              -> TENTATIVE, status_note="rescued_by_stage5"
  If NO               -> DISCARD,   status_note="confirmed_discard_stage5"
  If the call could not be completed (no key / API failure after
  retries) -> TENTATIVE, status_note="rescued_api_unavailable" — this is
  distinct from an actual "Y" verdict so the output is honest about
  what was and wasn't actually checked.

JOB 2 — SEMANTIC DEDUPLICATION
  Events are grouped by (unit_id, subsection) and, WITHIN each group
  only, compared pairwise by semantic similarity. Near-identical events
  (cosine >= 0.88 or Jaccard >= 0.60) within the same group: keep the
  higher-confidence one, mark the other DEDUP.

  This catches the pattern where old per-sentence mode caused one LLM
  to extract several sub-events of the same incident that should be one
  compound event. It must never merge the SAME construct recurring at
  DIFFERENT life periods (e.g. emotional abuse at ages 0-8 and again at
  12-16) — that recurrence is exactly what the paper's algebra encodes
  via a feedback edge (a -> a^3); collapsing it into one event would
  silently delete a signal the trajectory-assembly stage depends on.

Order: rescue first (so rescued events are included in dedup), then dedup.

Usage:
  export OPENAI_API_KEY=...
  python3 fire_stage5.py validated_events_s4.json \
      --stage1 stage1_output.json \
      --output validated_events_s5.json
  python3 fire_stage5.py --selftest
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────

RESCUE_SIM_THRESHOLD  = 0.88   # cosine: near-identical events
RESCUE_JAC_THRESHOLD  = 0.60   # Jaccard fallback
RESCUE_STATUSES       = {"DISCARD"}   # only rescue these statuses

RESCUE_SIM_THRESHOLD  = 0.88   # cosine: near-identical events
RESCUE_JAC_THRESHOLD  = 0.60   # Jaccard fallback
RESCUE_STATUSES       = {"DISCARD"}   # only rescue these statuses

# Same model as Worker 1's extractor, so the paper's same-model
# ablation claim isn't broken by the rescue pass. OpenAI Chat Completions
# endpoint.
RESCUE_MODEL_NAME  = "gpt-5.1"
RESCUE_API_URL     = "https://api.openai.com/v1/chat/completions"
RESCUE_API_KEY_ENV = "OPENAI_API_KEY"

# gpt-5.1 is in the GPT-5 reasoning family, which rejects `temperature`
# and `max_tokens` on Chat Completions and needs `max_completion_tokens`
# instead (same fact established for Worker 1 Stage 2/3).
_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)


def _is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))

_COMMENTARY_PATTERNS = [
    r"^one of the most significant",
    r"^identified as",
    r"^this is identified",
    r"^noted as",
    r"^described as",
    r"^considered",
    r"^this represents",
    r"possible (pattern|schema|dynamic)",
]

RESCUE_SYSTEM = (
    "You are a clinical psychologist reviewing extracted events from therapy "
    "consultation notes for a trauma and mental health assessment pipeline.\n\n"
    "You will be shown:\n"
    "  1. The full clinical section (all notes under one heading)\n"
    "  2. A specific event extracted from those notes\n\n"
    "Decide: Is this event clinically significant?\n\n"
    "An event IS significant if it describes ANY of the following:\n"
    "  - A traumatic or adverse experience (abuse, violence, loss, neglect)\n"
    "  - A psychological symptom or state (anxiety, depression, PTSD, grief)\n"
    "  - A relational pattern with clinical impact (abandonment, coercion)\n"
    "  - A life event with psychological consequences (bereavement, betrayal)\n"
    "  - A core belief or schema formed from adversity\n"
    "  - Any experience that contributes to understanding the patient's\n"
    "    mental health, trauma history, or psychological functioning\n\n"
    "An event is NOT significant only if it is:\n"
    "  - Pure therapist meta-commentary with no patient content\n"
    "  - Administrative or demographic facts (age, location, referral source)\n"
    "  - A clearly positive or neutral lifecycle fact with no clinical bearing\n\n"
    "Respond with EXACTLY one character: Y (significant) or N (not significant). "
    "No other text."
)

DEDUP_STOPWORDS = {
    "the", "a", "an", "of", "by", "to", "in", "for", "with",
    "from", "at", "on", "is", "was", "are", "be", "as", "and",
    "or", "her", "his", "their", "client", "she", "he", "they",
}


# ─────────────────────────────────────────────────────────────
# 2.  UTILITIES
# ─────────────────────────────────────────────────────────────

def _is_commentary(event_text: str) -> bool:
    """Return True if the event looks like clinical commentary, not a patient event."""
    t = event_text.strip().lower()
    for pat in _COMMENTARY_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _content_words(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z][a-z-]+", text.lower())) - DEDUP_STOPWORDS


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_sim(text_a: str, text_b: str, model) -> float:
    try:
        from sentence_transformers import util as st_util
        ea = model.encode(text_a, convert_to_tensor=True)
        eb = model.encode(text_b, convert_to_tensor=True)
        return float(st_util.cos_sim(ea, eb))
    except Exception:
        return _jaccard(_content_words(text_a), _content_words(text_b))


def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("Stage 5: sentence-transformers loaded")
        return m
    except Exception:
        logging.warning("Stage 5: sentence-transformers unavailable, using Jaccard")
        return None


# ─────────────────────────────────────────────────────────────
# 3.  SECTION MAP — from Stage 1 output
# ─────────────────────────────────────────────────────────────

_BRACKET_PREFIX_RE = re.compile(r'^\s*\[([^\]]+)\]\s*(.*)$')


def build_section_paragraphs(stage1_data: list) -> dict:
    """
    Build {subsection -> paragraph_text} from Stage 1 sentences.
    Used to give the rescue verifier full section context.

    Also indexes bracketed timeline entries ("[~0-8] Father was..."),
    keyed by the bracket content, so timeline-sourced DISCARD events (whose
    `subsection` field is an age bracket, per Stage 2/3's grouping) get real
    context instead of always falling through to "(section text unavailable)".
    """
    sections = defaultdict(list)
    for s in stage1_data:
        sentence = s.get("sentence", "")

        bracket_match = _BRACKET_PREFIX_RE.match(sentence)
        if bracket_match:
            key     = bracket_match.group(1).strip()
            content = bracket_match.group(2).strip()
            if content:
                sections[key].append(content)
            continue

        if ": " in sentence:
            prefix  = sentence.split(": ", 1)[0].strip()
            content = sentence.split(": ", 1)[1].strip()
            if len(prefix) <= 80:
                sections[prefix].append(content)

    return {p: ". ".join(c.rstrip(". ") for c in contents)
            for p, contents in sections.items()}


def lookup_section(subsection: str, section_map: dict) -> str:
    """Fuzzy lookup — same logic as Worker 2."""
    if not subsection:
        return ""
    h = re.sub(r"\s+", " ", subsection.strip())
    if h in section_map:
        return section_map[h]
    best_key, best_len = None, 0
    for key in section_map:
        if key.startswith(h) or h.startswith(key):
            if len(key) > best_len:
                best_key, best_len = key, len(key)
            continue
        cp = 0
        while cp < len(h) and cp < len(key) and h[cp] == key[cp]:
            cp += 1
        if cp >= 12 and cp > best_len:
            best_key, best_len = key, cp
    return section_map[best_key] if best_key else ""


# ─────────────────────────────────────────────────────────────
# 4.  RESCUE VERIFIER
# ─────────────────────────────────────────────────────────────

def _build_rescue_payload(model_name: str, user_msg: str) -> dict:
    """
    Payload shape depends on whether model_name is a reasoning model.
    gpt-5.1 rejects temperature/max_tokens and needs max_completion_tokens.
    reasoning_effort="none" suppresses hidden reasoning for this trivial
    Y/N classification, and the token cap has a small safety margin rather
    than the bare minimum a single character would need — a reasoning
    model can otherwise burn its whole budget on internal reasoning before
    emitting any visible content, returning an empty answer.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": RESCUE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
    }
    if _is_reasoning_model(model_name):
        payload["max_completion_tokens"] = 16
        payload["reasoning_effort"] = "none"
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = 4
    return payload


def rescue_call(event_text: str, section_para: str, api_key: str,
                model_name: str = RESCUE_MODEL_NAME) -> Optional[str]:
    """
    Call the rescue model to verify if a DISCARD event is actually
    clinically significant. Returns 'Y', 'N', or None (call could not be
    completed — caller must NOT treat this as equivalent to 'Y').
    """
    user = (
        "CLINICAL SECTION NOTES:\n"
        f"{section_para or '(section text unavailable)'}\n\n"
        "EXTRACTED EVENT:\n"
        f"{event_text}\n\n"
        "Is this event clinically significant for psychological or trauma assessment? "
        "Answer Y or N."
    )
    payload = json.dumps(_build_rescue_payload(model_name, user)).encode()

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                RESCUE_API_URL,
                data=payload,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                })
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"]
            ans = (content or "").strip().upper()
            if ans.startswith("Y"):
                return "Y"
            if ans.startswith("N"):
                return "N"
            logging.warning("Rescue call returned unparseable answer: %r", ans)
            return None
        except Exception as e:
            logging.warning("Rescue attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    return None   # caller must treat this distinctly from "Y"


def run_rescue(events: list, section_map: dict, api_key: Optional[str],
               model_name: str = RESCUE_MODEL_NAME) -> tuple:
    """
    Run rescue verifier on all DISCARD events.
    Returns (events_with_rescue_applied, stats).
    """
    rescued_confirmed  = 0   # an actual "Y" verdict
    rescued_fallback    = 0   # no key / call failed — defaulted, not verified
    confirmed_discard   = 0
    skipped_commentary  = 0

    for ev in events:
        if ev.get("status") not in RESCUE_STATUSES:
            continue
        if not ev.get("stage4_passed", True):
            continue   # Stage 4 explicitly flagged — don't rescue
        event_text = ev.get("event", "")
        if not event_text or not event_text.strip():
            continue

        if _is_commentary(event_text):
            ev["status_note"] = "commentary_not_rescued"
            skipped_commentary += 1
            continue

        subsection = ev.get("subsection", "")
        section_para = lookup_section(subsection, section_map)

        if api_key:
            result = rescue_call(event_text, section_para, api_key, model_name)
            if result == "N":
                ev["status_note"] = "confirmed_discard_stage5"
                confirmed_discard += 1
            elif result == "Y":
                ev["status"]                = "TENTATIVE"
                ev["human_review_required"] = True
                ev["verified"]              = False
                ev["status_note"]           = "rescued_by_stage5"
                rescued_confirmed           += 1
                logging.info("RESCUED (Y): %s — %s", ev.get("unit_id", ""), event_text[:60])
            else:
                # call could not be completed — distinct from a real "Y"
                ev["status"]                = "TENTATIVE"
                ev["human_review_required"] = True
                ev["verified"]              = False
                ev["status_note"]           = "rescued_api_unavailable"
                rescued_fallback            += 1
                logging.warning(
                    "RESCUE FALLBACK (call failed, defaulted to keep): %s — %s",
                    ev.get("unit_id", ""), event_text[:60])
        else:
            ev["status"]                = "TENTATIVE"
            ev["human_review_required"] = True
            ev["status_note"]           = "rescued_no_llm"
            rescued_fallback            += 1

    stats = {
        "discards_examined":   rescued_confirmed + rescued_fallback
                               + confirmed_discard + skipped_commentary,
        "rescued":              rescued_confirmed + rescued_fallback,  # back-compat total
        "rescued_llm_confirmed": rescued_confirmed,                     # actual Y verdicts
        "rescued_api_fallback":  rescued_fallback,                      # defaulted, unverified
        "confirmed_discard":    confirmed_discard,
        "skipped_commentary":   skipped_commentary,
    }
    return events, stats


# ─────────────────────────────────────────────────────────────
# 5.  SEMANTIC DEDUPLICATION
# ─────────────────────────────────────────────────────────────

def run_dedup(events: list, model) -> tuple:
    """
    Deduplicate near-identical events within the same (unit_id, subsection)
    group only. Pairwise comparison never crosses group boundaries, so
    the same construct recurring at a different life period (different
    unit_id/subsection) is never merged away.

    Returns (events_with_dedup_applied, stats).
    """
    groups: dict = defaultdict(list)   # (unit_id, subsection) -> [(idx, text, conf)]
    for i, ev in enumerate(events):
        if ev.get("status") in ("DISCARD", "NO_EVENT", "EXTRACTION_FAILED", "DEDUP"):
            continue
        text = ev.get("event", "")
        conf = float(ev.get("confidence", 0.0))
        if not text:
            continue
        key = (ev.get("unit_id", ""), ev.get("subsection", ""))
        groups[key].append((i, text, conf))

    dedup_count = 0
    dedup_map   = {}   # drop_idx → keep_idx
    marked      = set()

    for key, candidates in groups.items():
        for a in range(len(candidates)):
            if candidates[a][0] in marked:
                continue
            for b in range(a + 1, len(candidates)):
                if candidates[b][0] in marked:
                    continue
                ia, text_a, conf_a = candidates[a]
                ib, text_b, conf_b = candidates[b]

                if model:
                    sim = _cosine_sim(text_a, text_b, model)
                    threshold = RESCUE_SIM_THRESHOLD
                else:
                    sim = _jaccard(_content_words(text_a), _content_words(text_b))
                    threshold = RESCUE_JAC_THRESHOLD

                if sim >= threshold:
                    ev_a = events[ia]
                    ev_b = events[ib]
                    a_is_timeline = str(ev_a.get("subsection", "")).startswith("~")
                    b_is_timeline = str(ev_b.get("subsection", "")).startswith("~")
                    if conf_a > conf_b:
                        keep, drop = ia, ib
                    elif conf_b > conf_a:
                        keep, drop = ib, ia
                    elif b_is_timeline and not a_is_timeline:
                        keep, drop = ia, ib
                    elif a_is_timeline and not b_is_timeline:
                        keep, drop = ib, ia
                    else:
                        keep, drop = ia, ib

                    marked.add(drop)
                    dedup_map[drop] = keep
                    logging.info(
                        "DEDUP sim=%.3f group=%s → keep [%s]: %s",
                        sim, key, events[keep].get("unit_id", ""),
                        events[drop].get("event", "")[:50]
                    )

                    if drop == ia:
                        break

    for drop_idx, keep_idx in dedup_map.items():
        events[drop_idx]["status"]               = "DEDUP"
        events[drop_idx]["human_review_required"] = False
        events[drop_idx]["dedup_of"]             = events[keep_idx].get("unit_id", "")
        events[drop_idx]["dedup_kept_event"]     = events[keep_idx].get("event", "")[:80]
        dedup_count += 1

    return events, {"deduplicated": dedup_count, "groups_checked": len(groups)}


# ─────────────────────────────────────────────────────────────
# 5A. ATOMIC WRITE
# ─────────────────────────────────────────────────────────────

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
# 5B. SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # recurrence at different life periods must NOT be deduplicated
    events = [
        {"unit_id": "p_L10_0001", "subsection": "~0-8",
         "event": "Father emotionally abused client and was controlling",
         "status": "CONFIRMED", "confidence": 0.95},
        {"unit_id": "p_L40_0007", "subsection": "~12-16",
         "event": "Father emotionally abused client and was controlling",
         "status": "CONFIRMED", "confidence": 0.90},
    ]
    out, stats = run_dedup([dict(e) for e in events], model=None)
    check(stats["deduplicated"] == 0 and all(e["status"] == "CONFIRMED" for e in out),
          "dedup: same construct at different life periods is preserved")

    # positive case: genuine within-unit duplication still gets caught
    events2 = [
        {"unit_id": "p_L10_0001", "subsection": "~0-8",
         "event": "Father was present and controlling",
         "status": "CONFIRMED", "confidence": 0.80},
        {"unit_id": "p_L10_0001", "subsection": "~0-8",
         "event": "Father was present and controlling at home",
         "status": "CONFIRMED", "confidence": 0.90},
    ]
    out2, stats2 = run_dedup([dict(e) for e in events2], model=None)
    check(stats2["deduplicated"] == 1,
          "dedup: genuine within-unit near-duplicate is still caught")

    # rescue outcomes must be distinguishable
    events3 = [{"unit_id": "u1", "status": "DISCARD", "stage4_passed": True,
               "event": "Client reports ongoing conflict with sibling",
               "subsection": "X"}]
    out3, stats3 = run_rescue([dict(e) for e in events3], {}, api_key=None)
    check(out3[0]["status_note"] == "rescued_no_llm",
          "rescue: no-API-key path is labelled distinctly from a real Y verdict")
    check("rescued_llm_confirmed" in stats3 and "rescued_api_fallback" in stats3,
          "rescue: stats break out real verdicts from fallbacks")

    # timeline bracket entries are indexed for section context
    stage1 = [
        {"sentence": "[~0-8] Father emotionally abusive and controlling", "line_number": 1},
        {"sentence": "[~12-16] Father beat and pinched her", "line_number": 2},
    ]
    smap = build_section_paragraphs(stage1)
    check(lookup_section("~0-8", smap) != "" and lookup_section("~12-16", smap) != "",
          "section-map: timeline brackets are indexed and retrievable")

    # payload shape must adapt to the reasoning-model family
    rp = _build_rescue_payload("gpt-5.1", "test")
    check("temperature" not in rp and "max_completion_tokens" in rp
          and rp.get("reasoning_effort") == "none",
          "payload: gpt-5.1 drops temperature, sets max_completion_tokens + reasoning_effort=none")
    cp = _build_rescue_payload("gpt-4.1", "test")
    check("temperature" in cp and "max_tokens" in cp,
          "payload: non-reasoning model keeps temperature + max_tokens")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="FiRE Stage 5 — Rescue Verifier + Semantic Deduplication")
    p.add_argument("input", nargs="?",
                   help="Path to validated_events_s4.json (Stage 4 output)")
    p.add_argument("--stage1", default=None,
                   help="Path to stage1_output.json (for section context)")
    p.add_argument("--output", "-o", default="validated_events_s5.json")
    p.add_argument("--rescue-model", default=RESCUE_MODEL_NAME,
                   help=f"Rescue model name (default: {RESCUE_MODEL_NAME}, same "
                        "model as Worker 1's extractor — see module docstring "
                        "before pointing this at a different model)")
    p.add_argument("--no-rescue",  action="store_true",
                   help="Skip rescue verifier (run dedup only)")
    p.add_argument("--no-dedup",   action="store_true",
                   help="Skip deduplication (run rescue only)")
    p.add_argument("--selftest", action="store_true",
                   help="Run offline regression checks and exit")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.input:
        p.error("input is required (unless using --selftest)")

    if args.rescue_model != RESCUE_MODEL_NAME:
        logging.warning(
            "Rescue model overridden to '%s' (default is '%s', the same "
            "model Worker 1's extractor uses). Using a different model "
            "here re-introduces the same-model-ablation problem described "
            "in the module docstring.", args.rescue_model, RESCUE_MODEL_NAME)

    api_key = os.environ.get(RESCUE_API_KEY_ENV)
    if not api_key and not args.no_rescue:
        logging.warning("%s not set — rescue will conservatively "
                        "upgrade all non-commentary DISCARDs to TENTATIVE "
                        "(status_note='rescued_no_llm', not an actual verdict)",
                        RESCUE_API_KEY_ENV)

    print(f"[LOAD] {args.input}")
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    events     = data.get("events", [])
    patient_id = data.get("patient_id", "unknown")
    print(f"[LOAD] {len(events)} events for patient {patient_id}")

    section_map = {}
    if args.stage1 and Path(args.stage1).exists():
        with open(args.stage1, encoding="utf-8") as f:
            stage1_data = json.load(f)
        section_map = build_section_paragraphs(stage1_data)
        print(f"[LOAD] Section map: {len(section_map)} sections from stage1")

    model = None
    if not args.no_dedup:
        model = _load_model()

    from collections import Counter
    status_before = Counter(ev.get("status") for ev in events)
    print(f"\n[BEFORE] Status counts: {dict(status_before)}")

    # ── JOB 1: RESCUE ────────────────────────────────────────────────────────
    rescue_stats = {"rescued": 0, "rescued_llm_confirmed": 0, "rescued_api_fallback": 0,
                    "confirmed_discard": 0, "skipped_commentary": 0}
    if not args.no_rescue:
        n_discard = sum(1 for ev in events if ev.get("status") in RESCUE_STATUSES)
        print(f"\n[RESCUE] {n_discard} DISCARD events to examine...")
        events, rescue_stats = run_rescue(events, section_map, api_key, args.rescue_model)
        print(f"[RESCUE] Done: rescued(LLM-confirmed)={rescue_stats['rescued_llm_confirmed']}  "
              f"rescued(API-fallback)={rescue_stats['rescued_api_fallback']}  "
              f"confirmed_discard={rescue_stats['confirmed_discard']}  "
              f"skipped_commentary={rescue_stats['skipped_commentary']}")

    # ── JOB 2: DEDUP ─────────────────────────────────────────────────────────
    dedup_stats = {"deduplicated": 0, "groups_checked": 0}
    if not args.no_dedup:
        n_candidates = sum(1 for ev in events
                           if ev.get("status") not in
                           ("DISCARD", "NO_EVENT", "EXTRACTION_FAILED", "DEDUP"))
        print(f"\n[DEDUP] {n_candidates} active events to check for duplicates...")
        events, dedup_stats = run_dedup(events, model)
        print(f"[DEDUP] Done: {dedup_stats['deduplicated']} duplicates marked "
              f"across {dedup_stats['groups_checked']} (unit_id, subsection) groups")

    status_after = Counter(ev.get("status") for ev in events)
    print(f"\n[AFTER]  Status counts: {dict(status_after)}")

    data["events"] = events
    data["stage5_stats"] = {
        "rescue_model":  RESCUE_MODEL_NAME if args.no_rescue else args.rescue_model,
        "rescue": rescue_stats,
        "dedup":  dedup_stats,
        "status_before": dict(status_before),
        "status_after":  dict(status_after),
        "note": ("Rescue uses the same model as Worker 1's extractor (see "
                 "rescue_model above); this pass is still not described as "
                 "a separate stage in the paper's four-stage architecture, "
                 "see module docstring."),
    }
    atomic_write_json(args.output, data)
    print(f"\n[SAVE] {args.output}")
    print(f"[DONE] Patient: {patient_id}")


if __name__ == "__main__":
    main()
