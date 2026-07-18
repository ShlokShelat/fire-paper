#!/usr/bin/env python3
"""
LLM-based ACE scoring: direct mode (score from notes) or reconciliation mode 
(revise patient's self-filled form using notes).

DIRECT MODE (default):
  Queries the model to score the 16-item Adverse Childhood Experiences (ACE)
  questionnaire directly from consultation notes.
  
  python3 script.py notes.txt --model gpt-5.1

RECONCILIATION MODE:
  Given both the patient's self-filled ACE form and the clinician's notes,
  the model REVISES the form using the notes — changing an answer only where
  notes provide evidence, leaving it unchanged where notes are silent. Tests
  whether a model can reconcile two sources with discipline rather than
  over-editing.

  python3 script.py notes.txt --form form.json --model gpt-5.1 -o result.json

Supported models: gpt-5.1, gemini-3.1-pro, grok-4.20-0309-reasoning, deepseek-v4-flash

API keys are read from environment variables:
    OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY, DEEPSEEK_API_KEY (as needed)

No third-party packages required — standard library only.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import OrderedDict
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
# ACE ITEMS  (16-item Adverse Childhood Experiences questionnaire)
# ─────────────────────────────────────────────────────────────
ACE_QUESTIONS = OrderedDict([
    (1,  "Did a parent or adult in the household often swear at, insult, humiliate, or threaten the child?"),
    (2,  "Did a parent or adult in the household push, grab, slap, throw things at, or hit the child hard enough to leave marks?"),
    (3,  "Did an adult or person 5+ years older sexually touch the child, or attempt/commit sex with the child?"),
    (4,  "Did the child often feel unloved, unimportant, or that the family lacked closeness and support?"),
    (5,  "Did the child lack food, clean clothes, or protection, or were parents too impaired to provide care?"),
    (6,  "Were the child's parents separated or divorced?"),
    (7,  "Was the child's mother or stepmother physically abused by a partner?"),
    (8,  "Did a household member drink problematically or use street drugs?"),
    (9,  "Was a household member depressed, mentally ill, or did one attempt/die by suicide?"),
    (10, "Did a household member go to prison?"),
    (11, "Did the child live 2+ years in a dangerous neighbourhood or witness community assault?"),
    (12, "Did the child's mother, father, guardian, or primary-caregiver relative die?"),
    (13, "Did other kids or siblings often hit, threaten, pick on, or insult the child?"),
    (14, "Did the child often feel lonely, rejected, or isolated from peers?"),
    (15, "Was the family very poor or on public assistance for 2+ years?"),
    (16, "Did the child's parents have physical or verbal fights with each other?"),
])

# Risk bands and clinical threshold
HIGH_RISK_THRESHOLD = 7
HIGH_RISK_NOTE = (
    "Adults reporting seven or more ACEs show roughly a thirtyfold increase "
    "in the odds of attempted suicide (Dube et al. 2001). This threshold "
    "is clinically significant for escalation decisions."
)

def risk_band(score: int) -> str:
    if score == 0:  return "MINIMAL"
    if score <= 1:  return "LOW"
    if score <= 3:  return "MODERATE"
    if score <= 5:  return "HIGH"
    if score == 6:  return "VERY HIGH"
    return "VERY HIGH — CRITICAL (>=7)"


# ─────────────────────────────────────────────────────────────
# COERCION HELPERS
# ─────────────────────────────────────────────────────────────
def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "present", "1")
    return False


def _valid_confidence(v) -> str:
    """Normalise confidence to one of three allowed values."""
    s = str(v or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    return "low"


# ─────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────
def read_notes(path: str) -> str:
    """Read notes file robustly: handle UTF-8 BOM and fallback to latin-1 
    for encoding compatibility."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _form_val(v) -> int:
    """Coerce one form entry's value to 0/1. Accepts dict with common field names
    or bare scalar values."""
    if isinstance(v, dict):
        for fld in ("score", "present", "value", "answer", "ace", "rating",
                    "endorsed", "yes"):
            if fld in v:
                return 1 if _as_bool(v[fld]) else 0
        return 0
    return 1 if _as_bool(v) else 0


def _form_num(k) -> "int | None":
    m = re.search(r"\d+", str(k))
    return int(m.group()) if m else None


def read_form(path: str) -> "OrderedDict[int,int]":
    """Load the patient's self-filled ACE form -> {1..16: 0 or 1}.

    Tolerant of shape. Accepts:
      dict keyed by item number: {"1": 1, "2": 0, ...}
      dict with nested values: {"1": {"score": 1}, ...}
      list of dicts: [{"item": 1, "score": 1}, ...]
      bare list of scalars: [1, 0, 1, 1, ...] — read positionally as ACE-1..16
    
    Keys may be '1' / 'ACE 1' / 'ACE-1' / 'ACE1' / int; values may be 0/1, '0'/'1',
    true/false, or dict with score/present/value/answer/endorsed/yes field.
    Missing items default to 0.
    """
    data = json.loads(read_notes(path))
    if isinstance(data, dict) and isinstance(data.get("items"), (dict, list)):
        data = data["items"]

    form = OrderedDict((n, 0) for n in range(1, 17))

    if isinstance(data, dict):
        for k, v in data.items():
            n = _form_num(k)
            if n and 1 <= n <= 16:
                form[n] = _form_val(v)

    elif isinstance(data, list):
        if data and all(isinstance(e, dict) for e in data):
            # list-of-objects: each entry names its own item number
            for entry in data:
                n = _form_num(entry.get("item") or entry.get("id") or entry.get("ace")
                              or entry.get("number") or "")
                if n and 1 <= n <= 16:
                    form[n] = _form_val(entry)
        elif data:
            # bare list of scalars: read positionally as ACE-1..ACE-16
            for i, v in enumerate(data[:16]):
                form[i + 1] = _form_val(v)

    return form


# Lines that EXPLICITLY pre-label ACE results
_ACE_LABEL_PREFIX = ("ace burden", "ace score", "ace count", "ace total", "ace items")
_ACE_NUMLIST_RE = re.compile(r"\bACEs?\b[^\n]*?\d+\s*[,/&]\s*\d+", re.IGNORECASE)


def strip_ace_annotations(text: str):
    """Remove lines that explicitly state a pre-computed ACE score or item list, 
    so the model scores from the clinical narrative instead. Returns (cleaned_text, removed_lines)."""
    kept, removed = [], []
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith(_ACE_LABEL_PREFIX) or _ACE_NUMLIST_RE.search(line):
            removed.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept), removed


# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
MODELS = {
    "gpt-5.1": {
        "provider": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": ["OPENAI_API_KEY"],
    },
    "deepseek-v4-flash": {
        "provider": "openai",
        "url": "https://api.deepseek.com/chat/completions",
        "key_env": ["DEEPSEEK_API_KEY"],
    },
    "gemini-3.1-pro": {
        "provider": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-3.1-pro-preview:generateContent",
        "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    },
    "grok-4.20-0309-reasoning": {
        "provider": "openai",
        "url": "https://api.x.ai/v1/chat/completions",
        "key_env": ["XAI_API_KEY", "GROK_API_KEY"],
    },
}

MAX_OUTPUT_TOKENS = 16000

_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)


def _is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))


# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a clinical expert in mental health and childhood trauma, scoring the "
    "Adverse Childhood Experiences (ACE) questionnaire from a patient's consultation "
    "notes."
)

RECONCILE_SYSTEM_PROMPT = (
    "You are a clinical expert in mental health and childhood trauma. A patient has "
    "filled in their own Adverse Childhood Experiences (ACE) questionnaire, and you "
    "also have the clinician's consultation notes for the same patient."
)


def build_user_prompt(notes: str) -> str:
    items = "\n".join(f"  ACE-{n}: {q}" for n, q in ACE_QUESTIONS.items())

    schema = '''{
  "items": {
    "1": {
      "present": true,
      "source_sentence": "exact verbatim quote from the notes that supports this decision, or null if absent",
      "reasoning": "one or two sentences explaining why the quote maps to this ACE item, or why it is absent",
      "confidence": "high | medium | low"
    },
    ... one entry per item, 1 to 16 ...
  },
  "ace_score": <integer 0-16>
}'''

    return (
        "These are the 16 ACE questions:\n\n" + items + "\n\n"
        "Based on the consultation notes below, decide for each question whether it is "
        "present, and give the total ACE score (the number of present items, 0 to 16).\n\n"
        "For each item include:\n"
        "  - source_sentence: verbatim quote from the notes supporting the decision, or null if absent\n"
        "  - reasoning: brief explanation of why the item is present or absent\n"
        "  - confidence: 'high' (explicit evidence), 'medium' (inferred), or 'low' (ambiguous or absent)\n\n"
        "Respond with JSON only, in this shape:\n" + schema + "\n\n"
        "=== CONSULTATION NOTES ===\n" + notes.strip() + "\n=== END NOTES ==="
    )


def build_reconcile_prompt(notes: str, form: dict) -> str:
    """Reconciliation task: revise the patient's self-filled form using the notes.
    Change an answer only on evidence; keep it on silence."""
    items = "\n".join(
        f"  ACE-{n}: {ACE_QUESTIONS[n]}\n          patient's answer: {form[n]}"
        for n in ACE_QUESTIONS
    )

    schema = '''{
  "items": {
    "1": {
      "present": true,
      "source_sentence": "verbatim quote from the notes supporting your final value, or null",
      "reasoning": "one or two sentences explaining your final value for this item",
      "confidence": "high | medium | low",
      "revision_reason": "if you changed the patient's answer, what in the notes justifies the change; otherwise null"
    },
    ... one entry per item, 1 to 16 ...
  },
  "ace_score": <integer 0-16>
}'''

    return (
        "Below is the patient's OWN self-filled ACE questionnaire — each item scored 0 "
        "(no) or 1 (yes) by the patient — followed by the clinician's consultation "
        "notes for the same patient.\n\n"
        "Your task: starting from the patient's own answers, REVISE the questionnaire "
        "using the consultation notes. Change an item's answer ONLY when the notes give "
        "evidence for a different answer. Note that the consultation notes may not "
        "cover every item — they are not guaranteed to mention everything the patient "
        "reported.\n\n"
        "For every item, give your final value (present = true for 1, false for 0) with "
        "a supporting source quote, reasoning, and confidence. For any item whose answer "
        "you changed from the patient's, also give the reason for the change.\n\n"
        "PATIENT'S SELF-FILLED ACE FORM (item — question — patient's answer):\n\n"
        + items + "\n\n"
        "Respond with JSON only, in this shape:\n" + schema + "\n\n"
        "=== CONSULTATION NOTES ===\n" + notes.strip() + "\n=== END NOTES ==="
    )


# ─────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────
def _post_json(url: str, payload: dict, headers: dict, timeout: int = 180):
    """POST JSON, return (status, body_text). Raises only on transport errors."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def _extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ``` fences and prose."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        lo, hi = s.find("{"), s.rfind("}")
        if lo != -1 and hi != -1 and hi > lo:
            return json.loads(s[lo:hi + 1])
        raise


# ─────────────────────────────────────────────────────────────
# PROVIDER CALLS
# ─────────────────────────────────────────────────────────────
def _build_openai_payload(model: str) -> dict:
    """Build API payload with appropriate settings for the model family."""
    payload = {
        "model": model,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    if not _is_reasoning_model(model):
        payload["temperature"] = 0
    return payload


def call_openai_compatible(url: str, key: str, model: str,
                           system: str, user: str, verbose: bool = False) -> str:
    """Call OpenAI-compatible API endpoint (used for gpt-5.1, DeepSeek, and Grok).
    
    Handles parameter adaptation for different model families. Retries on transient errors."""
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = _build_openai_payload(model)
    payload["messages"] = [{"role": "system", "content": system},
                           {"role": "user", "content": user}]

    for attempt in range(4):
        status, body = _post_json(url, payload, headers)
        if status == 200:
            data = json.loads(body)
            return data["choices"][0]["message"]["content"]

        low = body.lower()
        if status == 400 and "temperature" in low and "temperature" in payload:
            if verbose: print("  [adapting] removing unsupported 'temperature'", file=sys.stderr)
            payload.pop("temperature", None); continue
        if status == 400 and "max_tokens" in low and "max_completion_tokens" in payload:
            if verbose: print("  [adapting] switching to 'max_tokens'", file=sys.stderr)
            payload["max_tokens"] = payload.pop("max_completion_tokens"); continue
        if status == 400 and "max_completion_tokens" in low and "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens"); continue
        if status == 400 and "response_format" in low and "response_format" in payload:
            if verbose: print("  [adapting] removing unsupported 'response_format'", file=sys.stderr)
            payload.pop("response_format", None); continue
        if status in (429, 500, 502, 503, 529):
            wait = 2 ** attempt
            if verbose: print(f"  [retry] HTTP {status}, waiting {wait}s", file=sys.stderr)
            time.sleep(wait); continue

        raise RuntimeError(f"API error (HTTP {status}): {body[:600]}")
    raise RuntimeError("API call failed after retries.")


def call_gemini(url: str, key: str, model: str,
                system: str, user: str, verbose: bool = False) -> str:
    """Call Google Gemini generateContent endpoint."""
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }
    for attempt in range(4):
        status, body = _post_json(url, payload, headers)
        if status == 200:
            data = json.loads(body)
            try:
                cand = data["candidates"][0]
                return "".join(p.get("text", "") for p in cand["content"]["parts"])
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Gemini response shape: {e}; body={body[:400]}")
        if status in (429, 500, 502, 503):
            wait = 2 ** attempt
            if verbose: print(f"  [retry] HTTP {status}, waiting {wait}s", file=sys.stderr)
            time.sleep(wait); continue
        raise RuntimeError(f"Gemini error (HTTP {status}): {body[:600]}")
    raise RuntimeError("Gemini call failed after retries.")


def query_model(model_name: str, key: str, system: str, user: str,
                verbose: bool = False) -> str:
    cfg = MODELS[model_name]
    if cfg["provider"] == "gemini":
        return call_gemini(cfg["url"], key, model_name, system, user, verbose)
    return call_openai_compatible(cfg["url"], key, model_name, system, user, verbose)


# ─────────────────────────────────────────────────────────────
# RESULT NORMALISATION
# ─────────────────────────────────────────────────────────────
def normalise(raw: dict) -> dict:
    """DIRECT mode: coerce the model's JSON into a clean result and recompute
    the score from the per-item flags, ensuring internal consistency."""
    raw_items = raw.get("items") or {}
    items = OrderedDict()
    for n in range(1, 17):
        entry = raw_items.get(str(n)) or raw_items.get(n) or {}
        if not isinstance(entry, dict):
            entry = {"present": _as_bool(entry)}

        present = _as_bool(entry.get("present"))
        source  = entry.get("source_sentence")
        source  = source.strip() if isinstance(source, str) and source.strip() else None
        reasoning = str(entry.get("reasoning") or "").strip()
        confidence = _valid_confidence(entry.get("confidence"))

        items[n] = {
            "present":         present,
            "source_sentence": source,
            "reasoning":       reasoning,
            "confidence":      confidence,
            "age":             str(entry.get("age") or "unknown"),
            "evidence":        str(entry.get("evidence") or "").strip(),
            "question":        ACE_QUESTIONS[n],
        }

    recomputed = sum(1 for it in items.values() if it["present"])
    model_score = raw.get("ace_score")
    try:
        model_score = int(model_score)
    except (TypeError, ValueError):
        model_score = None

    low_confidence_items = [n for n, it in items.items() if it["confidence"] == "low"]

    return {
        "mode":                  "direct",
        "ace_score":             recomputed,
        "model_reported_score":  model_score,
        "score_consistent":      (model_score == recomputed),
        "items":                 items,
        "items_present":         [n for n, it in items.items() if it["present"]],
        "low_confidence_items":  low_confidence_items,
        "risk":                  risk_band(recomputed),
        "crosses_high_risk_threshold": recomputed >= HIGH_RISK_THRESHOLD,
        "high_risk_threshold_note": HIGH_RISK_NOTE if recomputed >= HIGH_RISK_THRESHOLD else "",
    }


def normalise_reconcile(raw: dict, form: dict) -> dict:
    """RECONCILE mode: the model returns its FINAL value per item. We compare to
    the patient's self-report and flag any change without a stated reason (over-editing)."""
    raw_items = raw.get("items") or {}
    items = OrderedDict()
    for n in range(1, 17):
        entry = raw_items.get(str(n)) or raw_items.get(n) or {}
        if not isinstance(entry, dict):
            entry = {"present": _as_bool(entry)}

        final_bool = _as_bool(entry.get("present"))
        final_i    = 1 if final_bool else 0
        self_v     = int(form.get(n, 0))
        changed    = (final_i != self_v)
        source  = entry.get("source_sentence")
        source  = source.strip() if isinstance(source, str) and source.strip() else None
        reasoning = str(entry.get("reasoning") or "").strip()
        confidence = _valid_confidence(entry.get("confidence"))
        revision_reason = str(entry.get("revision_reason") or "").strip()

        items[n] = {
            "present":         final_bool,
            "final":           final_i,
            "self_report":     self_v,
            "changed":         changed,
            "direction":       ("1->0" if changed and self_v == 1
                                else "0->1" if changed else None),
            "source_sentence": source,
            "reasoning":       reasoning,
            "confidence":      confidence,
            "revision_reason": revision_reason,
            "question":        ACE_QUESTIONS[n],
        }

    recomputed   = sum(1 for it in items.values() if it["present"])
    self_score   = sum(int(form.get(n, 0)) for n in range(1, 17))
    model_score  = raw.get("ace_score")
    try:
        model_score = int(model_score)
    except (TypeError, ValueError):
        model_score = None

    changed_items = [n for n, it in items.items() if it["changed"]]
    changes = [{"item": n, "direction": items[n]["direction"],
                "self_report": items[n]["self_report"], "final": items[n]["final"],
                "revision_reason": items[n]["revision_reason"] or None}
               for n in changed_items]
    # a change with no stated reason is a red flag for over-editing
    unjustified_changes = [n for n in changed_items if not items[n]["revision_reason"]]
    low_confidence_items = [n for n, it in items.items() if it["confidence"] == "low"]

    return {
        "mode":                  "reconcile",
        "ace_score":             recomputed,
        "self_report_score":     self_score,
        "model_reported_score":  model_score,
        "score_consistent":      (model_score == recomputed),
        "items":                 items,
        "items_present":         [n for n, it in items.items() if it["present"]],
        "changed_items":         changed_items,
        "changes":               changes,
        "unjustified_changes":   unjustified_changes,
        "low_confidence_items":  low_confidence_items,
        "risk":                  risk_band(recomputed),
        "self_report_risk":      risk_band(self_score),
        "crosses_high_risk_threshold": recomputed >= HIGH_RISK_THRESHOLD,
        "high_risk_threshold_note": HIGH_RISK_NOTE if recomputed >= HIGH_RISK_THRESHOLD else "",
    }


# ─────────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────────
CONF_LABEL = {"high": "HIGH  ", "medium": "MEDIUM", "low": "LOW \u26a0 "}

def print_report(model_name: str, patient: str, result: dict):
    sep = "=" * 72
    print("\n" + sep)
    print(f"  LLM ACE BASELINE — {patient}   (model: {model_name})")
    print(sep)

    for n, it in result["items"].items():
        mark = "\u2713" if it["present"] else " "
        conf = CONF_LABEL.get(it["confidence"], "     ")
        print(f"\n  [{mark}] ACE-{n:<2d}  [{conf}]  {it['question']}")
        if it["present"]:
            src = it["source_sentence"]
            print(f"         SOURCE   : \"{src[:120]}{'…' if src and len(src) > 120 else ''}\""
                  if src else "         SOURCE   : (no verbatim quote provided)")
            if it["reasoning"]:
                print(f"         REASONING: {it['reasoning'][:200]}")
        else:
            if it["reasoning"]:
                print(f"         ABSENT   : {it['reasoning'][:160]}")

    print("\n" + "-" * 72)
    print(f"  ACE SCORE         : {result['ace_score']}/16   ({result['risk']} risk)")
    print(f"  Items present     : {result['items_present']}")
    if result["crosses_high_risk_threshold"]:
        print(f"  \u26a0\u26a0 CROSSES ACE>=7 THRESHOLD: {result['high_risk_threshold_note']}")
    if result["low_confidence_items"]:
        print(f"  Low confidence \u26a0  : ACE items {result['low_confidence_items']} — recommend human review")
    if not result["score_consistent"] and result["model_reported_score"] is not None:
        print(f"  NOTE: model self-reported {result['model_reported_score']}; "
              f"recomputed from item flags = {result['ace_score']}")
    print(sep + "\n")


def print_report_reconcile(model_name: str, patient: str, result: dict):
    sep = "=" * 72
    print("\n" + sep)
    print(f"  LLM ACE RECONCILE (self-form + notes) — {patient}   (model: {model_name})")
    print(sep)

    for n, it in result["items"].items():
        final_mark = "1" if it["present"] else "0"
        conf = CONF_LABEL.get(it["confidence"], "     ")
        tag = ""
        if it["changed"]:
            tag = f"   \u27f5 CHANGED {it['direction']}  (patient said {it['self_report']})"
        print(f"\n  ACE-{n:<2d} = {final_mark}  [{conf}]{tag}")
        print(f"        {it['question']}")
        if it["source_sentence"]:
            s = it["source_sentence"]
            print(f"        SOURCE   : \"{s[:120]}{'…' if len(s) > 120 else ''}\"")
        if it["reasoning"]:
            print(f"        REASONING: {it['reasoning'][:200]}")
        if it["changed"]:
            rr = it["revision_reason"] or "(no reason given \u26a0)"
            print(f"        REVISION : {rr[:200]}")

    print("\n" + "-" * 72)
    print(f"  SELF-REPORT SCORE : {result['self_report_score']}/16   ({result['self_report_risk']} risk)")
    print(f"  REVISED SCORE     : {result['ace_score']}/16   ({result['risk']} risk)")
    print(f"  Items present     : {result['items_present']}")
    if result["crosses_high_risk_threshold"]:
        print(f"  \u26a0\u26a0 CROSSES ACE>=7 THRESHOLD: {result['high_risk_threshold_note']}")
    if result["changed_items"]:
        for c in result["changes"]:
            rr = c["revision_reason"] or "(no reason given \u26a0)"
            print(f"    changed ACE-{c['item']}: {c['direction']} — {rr[:120]}")
    else:
        print("  Changes           : none (the notes did not revise the self-report)")
    if result["unjustified_changes"]:
        print(f"  \u26a0 changes without a stated reason: ACE {result['unjustified_changes']}")
    if result["low_confidence_items"]:
        print(f"  Low confidence \u26a0  : ACE items {result['low_confidence_items']}")
    if not result["score_consistent"] and result["model_reported_score"] is not None:
        print(f"  NOTE: model self-reported {result['model_reported_score']}; "
              f"recomputed from item flags = {result['ace_score']}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    check(MODELS["gemini-3.1-pro"]["url"].endswith("gemini-3.1-pro-preview:generateContent"),
          "Gemini registry uses the correct API model ID")
    check("gemini-2.5-flash" not in json.dumps(MODELS),
          "No outdated model references in registry")
    check(set(["gpt-5.1", "gemini-3.1-pro", "grok-4.20-0309-reasoning"]) <= set(MODELS),
          "All expected baseline models are registered")

    check(risk_band(6) == "VERY HIGH" and risk_band(7).startswith("VERY HIGH — CRITICAL"),
          "Risk band distinguishes ACE score 6 from 7")
    check(risk_band(0) == "MINIMAL" and risk_band(1) == "LOW"
          and risk_band(2) == "MODERATE" and risk_band(5) == "HIGH",
          "Other risk band boundaries are correct")

    p1 = _build_openai_payload("gpt-5.1")
    check("temperature" not in p1 and p1.get("max_completion_tokens") == MAX_OUTPUT_TOKENS,
          "Reasoning models omit temperature in payload")
    p2 = _build_openai_payload("deepseek-v4-flash")
    check(p2.get("temperature") == 0,
          "Non-reasoning models include temperature")

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], f)
        bare_list_path = f.name
    form = read_form(bare_list_path)
    check(sum(form.values()) == 5,
          "Bare list of scalars is read positionally, not silently zeroed")
    check(form[1] == 1 and form[2] == 0 and form[12] == 1,
          "Positional mapping is correct (index 0 -> ACE-1, etc.)")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"1": 1, "ACE-2": 0, "ACE3": {"present": True}}, f)
        dict_path = f.name
    form2 = read_form(dict_path)
    check(form2[1] == 1 and form2[2] == 0 and form2[3] == 1,
          "Dict-shaped forms (various key styles) still parse correctly")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"item": 1, "score": 1}, {"item": 5, "present": False}], f)
        listdict_path = f.name
    form3 = read_form(listdict_path)
    check(form3[1] == 1 and form3[5] == 0,
          "List-of-dict forms still parse correctly")

    form4 = OrderedDict((n, 0) for n in range(1, 17))
    form4[1] = 1
    raw = {
        "items": {
            "1": {"present": False, "revision_reason": "notes contradict this"},
            "2": {"present": True},
            **{str(n): {"present": False} for n in range(3, 17)},
        },
        "ace_score": 1,
    }
    res = normalise_reconcile(raw, form4)
    check(res["items"][1]["direction"] == "1->0" and res["items"][2]["direction"] == "0->1",
          "Change direction computed independently of model's claims")
    check(2 in res["unjustified_changes"] and 1 not in res["unjustified_changes"],
          "Changes without a stated reason are flagged; justified ones are not")
    check(res["self_report_score"] == 1 and res["ace_score"] == 1,
          "Self-report and revised scores both recomputed correctly")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Score ACE from consultation notes using one LLM. "
                    "With --form, instead revise the patient's own self-filled ACE form. "
                    "Every item is auditable (source, reasoning, confidence; plus revision reasons).")
    ap.add_argument("notes", nargs="?", help="Path to the consultation-notes text file.")
    ap.add_argument("--model", "-m", choices=sorted(MODELS),
                    help="Which LLM to query.")
    ap.add_argument("--form", default=None,
                    help="Path to the patient's self-filled ACE form (JSON). When given, "
                         "the model REVISES the self-report using the notes instead of "
                         "scoring from scratch.")
    ap.add_argument("--output", "-o", default=None,
                    help="Write the full JSON result here (default: <notes>.<model>.ace[.reconciled].json).")
    ap.add_argument("--print-prompt", action="store_true",
                    help="Print the exact prompt sent to the model and exit.")
    ap.add_argument("--strip-ace-annotations", action="store_true",
                    help="Remove lines in the notes that already state an ACE score / "
                         "burden / item list.")
    ap.add_argument("--selftest", action="store_true",
                    help="Run offline regression checks and exit")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show retry/adaptation diagnostics on stderr.")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.notes or not args.model:
        ap.error("notes and --model are required (unless using --selftest)")

    try:
        notes = read_notes(args.notes)
    except FileNotFoundError:
        print(f"error: notes file not found: {args.notes}", file=sys.stderr); sys.exit(1)
    if not notes.strip():
        print("error: notes file is empty.", file=sys.stderr); sys.exit(1)

    form = None
    if args.form:
        try:
            form = read_form(args.form)
        except FileNotFoundError:
            print(f"error: form file not found: {args.form}", file=sys.stderr); sys.exit(1)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"error: could not parse form JSON: {e}", file=sys.stderr); sys.exit(1)

    if args.strip_ace_annotations:
        notes, removed = strip_ace_annotations(notes)
        if removed:
            print(f"  stripped {len(removed)} pre-marked ACE annotation line(s) "
                  f"from the notes:", file=sys.stderr)
            for r in removed:
                print(f"    - {r}", file=sys.stderr)

    if form is not None:
        system, user = RECONCILE_SYSTEM_PROMPT, build_reconcile_prompt(notes, form)
    else:
        system, user = SYSTEM_PROMPT, build_user_prompt(notes)

    if args.print_prompt:
        print("--- SYSTEM ---\n" + system + "\n\n--- USER ---\n" + user)
        return

    cfg = MODELS[args.model]
    key = next((os.environ[e] for e in cfg["key_env"] if os.environ.get(e)), None)
    if not key:
        envs = " or ".join(cfg["key_env"])
        print(f"error: no API key found for {args.model}. Set {envs} in your environment.",
              file=sys.stderr)
        sys.exit(1)

    patient = os.path.splitext(os.path.basename(args.notes))[0]

    try:
        reply = query_model(args.model, key, system, user, verbose=args.verbose)
    except Exception as e:
        print(f"error: model query failed: {e}", file=sys.stderr); sys.exit(2)

    try:
        raw = _extract_json(reply)
    except Exception as e:
        print(f"error: could not parse model output as JSON: {e}\n--- raw reply ---\n"
              f"{reply[:1500]}", file=sys.stderr)
        sys.exit(3)

    if form is not None:
        result = normalise_reconcile(raw, form)
        print_report_reconcile(args.model, patient, result)
        scoring_method = "llm_reconcile_selfform_plus_notes"
        default_suffix = f".{args.model}.ace_reconciled.json"
    else:
        result = normalise(raw)
        print_report(args.model, patient, result)
        scoring_method = "llm_direct_baseline_grounded"
        default_suffix = f".{args.model}.ace.json"

    out_path = args.output or f"{os.path.splitext(args.notes)[0]}{default_suffix}"
    out = {
        "patient_id":       patient,
        "model":            args.model,
        "provider":         cfg["provider"],
        "scoring_method":   scoring_method,
        "notes_file":       os.path.abspath(args.notes),
        "form_file":        os.path.abspath(args.form) if args.form else None,
        "self_report_form": {str(n): v for n, v in form.items()} if form else None,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        **result,
        "items":            {str(n): it for n, it in result["items"].items()},
        "raw_model_output": raw,
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  written: {out_path}")
    except OSError as e:
        print(f"error: could not write output: {e}", file=sys.stderr); sys.exit(4)


if __name__ == "__main__":
    main()
