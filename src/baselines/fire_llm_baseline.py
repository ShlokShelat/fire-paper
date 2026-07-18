#!/usr/bin/env python3
"""
Direct ACE scoring from consultation notes using a single LLM.

Queries one of four supported models (gpt-5.1, gemini-3.1-pro, 
grok-4.20-0309-reasoning, or deepseek-v4-flash) to score the Adverse Childhood 
Experiences (ACE) questionnaire from clinical notes. Each item is grounded in 
verbatim source text from the notes, with reasoning and confidence level for 
full auditability.

Usage:
    python3 fire_llm_baseline.py notes.txt --model gpt-5.1
    python3 fire_llm_baseline.py notes.txt --model gemini-3.1-pro
    python3 fire_llm_baseline.py notes.txt --model grok-4.20-0309-reasoning
    python3 fire_llm_baseline.py notes.txt --model deepseek-v4-flash -o result.json

API keys are read from environment variables (set one for your chosen model):
    OPENAI_API_KEY          for gpt-5.1
    GEMINI_API_KEY          for gemini-3.1-pro
    XAI_API_KEY             for grok-4.20-0309-reasoning
    DEEPSEEK_API_KEY        for deepseek-v4-flash

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
# NOTES LOADING
# ─────────────────────────────────────────────────────────────
def read_notes(path: str) -> str:
    """Read notes file robustly: handle UTF-8 BOM and fall back to latin-1 
    for encoding compatibility."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# Lines that EXPLICITLY pre-label ACE results (so a reader could just copy them):
#   "ACE Burden: ... ACE 1, 2, 4, 5 ..."   |  "ACE Score: 7"  |  "marks ACE 1,2,4"
_ACE_LABEL_PREFIX = ("ace burden", "ace score", "ace count", "ace total", "ace items")
_ACE_NUMLIST_RE = re.compile(r"\bACEs?\b[^\n]*?\d+\s*[,/&]\s*\d+", re.IGNORECASE)


def strip_ace_annotations(text: str):
    """Remove lines that explicitly state a pre-computed ACE score or item list, 
    so the model scores from the clinical narrative instead of copying a 
    pre-marked answer. Returns (cleaned_text, removed_lines)."""
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

_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)


def _is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))


# ─────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a clinical expert in mental health and childhood trauma, scoring the "
    "Adverse Childhood Experiences (ACE) questionnaire from a patient's consultation "
    "notes."
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


# ─────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────
def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120):
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
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "response_format": {"type": "json_object"},
    }
    if _is_reasoning_model(model):
        payload["max_completion_tokens"] = 8000
    else:
        payload["temperature"] = 0
        payload["max_completion_tokens"] = 8000
    return payload


def call_openai_compatible(url: str, key: str, model: str,
                           system: str, user: str, verbose: bool = False) -> str:
    """Call OpenAI-compatible API endpoint (used for gpt-5.1, DeepSeek, and Grok).
    
    Handles parameter adaptation for different model families (e.g., reasoning models
    that do not accept temperature). Retries on transient errors."""
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
            "maxOutputTokens": 8000,
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


def query_model(model_name: str, key: str, notes: str, verbose: bool = False) -> str:
    cfg = MODELS[model_name]
    system, user = SYSTEM_PROMPT, build_user_prompt(notes)
    if cfg["provider"] == "gemini":
        return call_gemini(cfg["url"], key, model_name, system, user, verbose)
    return call_openai_compatible(cfg["url"], key, model_name, system, user, verbose)


# ─────────────────────────────────────────────────────────────
# RESULT NORMALISATION
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


def normalise(raw: dict) -> dict:
    """Coerce the model's JSON into a clean 16-item result and recompute the score
    from the per-item flags, ensuring internal consistency."""
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


# ─────────────────────────────────────────────────────────────
# REPORT
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
            if src:
                print(f"         SOURCE   : \"{src[:120]}{'…' if len(src) > 120 else ''}\"")
            else:
                print(f"         SOURCE   : (no verbatim quote provided)")
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
    check("temperature" not in p1 and p1.get("max_completion_tokens") == 8000,
          "Reasoning models omit temperature in payload")
    p2 = _build_openai_payload("deepseek-v4-flash")
    check(p2.get("temperature") == 0,
          "Non-reasoning models include temperature")

    raw = {"items": {str(n): {"present": n in (1, 2, 4, 9, 12, 13, 16)} for n in range(1, 17)},
          "ace_score": 7}
    res = normalise(raw)
    check(res["ace_score"] == 7 and res["crosses_high_risk_threshold"] is True,
          "Score normalisation and risk threshold detection works")
    check(res["risk"].startswith("VERY HIGH — CRITICAL"),
          "Risk label reflects high-risk threshold")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Score ACE directly from consultation notes using one LLM. "
                    "Each item includes a source sentence, reasoning, and confidence level.")
    ap.add_argument("notes", nargs="?", help="Path to the consultation-notes text file.")
    ap.add_argument("--model", "-m", choices=sorted(MODELS),
                    help="Which LLM to query.")
    ap.add_argument("--output", "-o", default=None,
                    help="Write the full JSON result here (default: <notes>.<model>.ace.json).")
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

    if args.strip_ace_annotations:
        notes, removed = strip_ace_annotations(notes)
        if removed:
            print(f"  stripped {len(removed)} pre-marked ACE annotation line(s) "
                  f"from the notes:", file=sys.stderr)
            for r in removed:
                print(f"    - {r}", file=sys.stderr)

    if args.print_prompt:
        print("--- SYSTEM ---\n" + SYSTEM_PROMPT + "\n\n--- USER ---\n" + build_user_prompt(notes))
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
        reply = query_model(args.model, key, notes, verbose=args.verbose)
    except Exception as e:
        print(f"error: model query failed: {e}", file=sys.stderr); sys.exit(2)

    try:
        raw = _extract_json(reply)
    except Exception as e:
        print(f"error: could not parse model output as JSON: {e}\n--- raw reply ---\n"
              f"{reply[:1500]}", file=sys.stderr)
        sys.exit(3)

    result = normalise(raw)
    print_report(args.model, patient, result)

    out_path = args.output or f"{os.path.splitext(args.notes)[0]}.{args.model}.ace.json"
    out = {
        "patient_id":       patient,
        "model":            args.model,
        "provider":         cfg["provider"],
        "scoring_method":   "llm_direct_baseline_grounded",
        "notes_file":       os.path.abspath(args.notes),
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
