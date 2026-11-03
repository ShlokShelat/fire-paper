#!/usr/bin/env python3
"""
Structure-guided baseline: one model call told to carry out FiRE's steps itself.

Paper concept: the structure-guided rows of the main paper's Table II
(Supplementary Material, Section S4). The plain baseline's prompt (same
system message plus one sentence, same 14 items, same per-item fields) is
extended with FiRE's steps:
  (1) extract every event with a verbatim quote, the people involved with
      the actor first, and the age, inferring nothing the notes do not state;
  (2) map each event to one item, or to none for symptoms and outcomes;
  (3) check each mapping for construct, actor and role, an age window of
      0 to 18, and the cited span, keeping it only if all four pass (a
      simplification of FiRE, which keeps but flags an event failing one);
  (4) order the kept events by age;
  (5) score the items with at least one kept event;
and, with --form, (6) start from the individual's answers, change one only
on evidence, and keep it where the notes are silent. The prompt includes the
Indian clinical glossary (prompts/indian_cultural_glossary.txt), and the
model returns its events, check results and timeline as well as the items.
No cluster-to-item table or state space is given; those belong to the
knowledge-matched baseline (fire_llm_knowledge_baseline.py). The model does
every step in one call, with no separate verification.

The item texts, model registry, API calls, payload settings and per-item
normalisation are imported from fire_llm_baseline_self_form_included.py, so
the items part of the output has the same shape as the plain baseline's.
The output adds `procedure`: the model's events, its check results, and
whether its present items match the items of its kept events.

Model-call settings: as the plain baseline (GPT-5.1 without temperature,
so the API defaults apply, including reasoning effort "none", the same as
FiRE; Gemini 3.1 Pro and Grok-4.20 at temperature 0), with the output limit and HTTP timeout raised below
because the reply is much longer.

DIRECT MODE (notes only):
  python3 fire_llm_structured_baseline.py notes.txt --model gpt-5.1 -o out.json
RECONCILIATION MODE (notes + self-filled form):
  python3 fire_llm_structured_baseline.py notes.txt --form form.json --model gpt-5.1 -o out.json
BATCH (every patient_NN.txt in a folder; existing outputs are skipped):
  python3 fire_llm_structured_baseline.py --batch DATA_DIR --model gpt-5.1 --out-dir results_structured
  python3 fire_llm_structured_baseline.py --batch DATA_DIR --with-forms --model gpt-5.1 --out-dir results_structured
PROMPT ONLY (no API call):
  python3 fire_llm_structured_baseline.py notes.txt --print-prompt

API keys: the same environment variables as the plain baselines.
Standard library only.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import OrderedDict, Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fire_llm_baseline_self_form_included as base  # noqa: E402

# Output file tag and scoring_method prefix; the knowledge-matched script
# overrides both so its files never overwrite these.
OUTPUT_TAG = "ace_structured"
SCORING_PREFIX = "llm_structured"

# The structured reply (an events list plus the 14 items) is several times
# longer than the plain baseline's, and GPT-5.1 counts its hidden reasoning
# against the same budget, so the output limit and the HTTP timeout are
# raised to 32000 tokens, as in Supplementary Section S7. Nothing else in the
# request changes. The knowledge-matched script inherits this limit.
base.MAX_OUTPUT_TOKENS = 32000
_base_post_json = base._post_json
base._post_json = lambda url, payload, headers, timeout=600: _base_post_json(
    url, payload, headers, timeout=timeout)

ACE_QUESTIONS = base.ACE_QUESTIONS
MODELS = base.MODELS

GLOSSARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "prompts", "indian_cultural_glossary.txt")


def load_glossary(path: str = GLOSSARY_PATH) -> str:
    """The term/meaning table from prompts/indian_cultural_glossary.txt,
    without the file's header lines."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("term"))
    return "\n".join(lines[start:]).strip()


# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────
_FOLLOW = " Follow the procedure given in the task exactly, step by step."
SYSTEM_PROMPT = base.SYSTEM_PROMPT + _FOLLOW
RECONCILE_SYSTEM_PROMPT = base.RECONCILE_SYSTEM_PROMPT + _FOLLOW

PROCEDURE = """Follow this procedure exactly, in order.

STEP 1 — EXTRACT EVENTS
Read the notes and list every experience they describe: events, ongoing patterns, symptoms, and beliefs. For each, record:
  - event: a short description
  - quote: the exact verbatim sentence(s) from the notes
  - persons_involved: who was involved and in what role, listing the person who acted (the perpetrator or active subject) first
  - age: the client's age or age range when it happened, as the notes give it, or "unknown"
Do not infer: only describe what is explicitly in the notes. Do not add persons, places, actions, or ages that the notes do not state.

STEP 2 — MAP EACH EVENT
Assign each event to the one ACE item whose defining element it describes, or to "none". Symptoms and outcomes the client developed (for example low mood, anxiety, or overeating) are not adverse experiences and map to "none".

STEP 3 — VERIFY EACH MAPPING
Check every event mapped to an item against four checks:
  (a) construct: the event matches the item's defining element, not merely something related to it;
  (b) actor and role: the people involved are the ones the item's question asks about (for example, whether it is the child's parent, an adult in the household, another household member, or other children);
  (c) age window: the event happened between birth and age 18; an event after age 18 fails, and an event with no stated age passes only if the notes clearly place it in childhood;
  (d) grounding: the quote directly supports the mapping.
Record each check as "pass" or "fail". Keep the event only if it passes all four; otherwise drop it and give the reason.

STEP 4 — BUILD THE TIMELINE
Order the kept events by age. Note an item that repeats within one continuous period (recurrence) and one that returns after a gap (reactivation). The timeline records the history; it does not change the score.

STEP 5 — SCORE
An ACE item is present if at least one kept event maps to it. Each item counts once, however many events map to it. The score is the number of present items (0 to 14).
"""

RECONCILE_STEP = """
STEP 6 — RECONCILE WITH THE PATIENT'S FORM
Start from the patient's own answers. Change an item's answer ONLY when the notes give evidence for a different answer: change "no" to "yes" only when a kept event from Step 3 maps to the item, and change "yes" to "no" only when the notes give explicit evidence against it. If the notes are silent on an item, keep the patient's answer. The final score is the number of items whose final value is yes.
"""

_EVENTS_SCHEMA = '''  "events": [
    {
      "event": "short description",
      "quote": "exact verbatim quote from the notes",
      "persons_involved": "actor first, with roles",
      "age": "age or age range, or unknown",
      "item": "ACE-N or none",
      "checks": {"construct": "pass | fail", "actor_role": "pass | fail",
                 "age_window": "pass | fail", "grounding": "pass | fail"},
      "kept": true,
      "drop_reason": "why the event was dropped, or null"
    },
    ... one entry per extracted event ...
  ],
  "timeline": "the kept events in age order, with any recurrence or reactivation noted",'''


def _glossary_block() -> str:
    return "INDIAN CLINICAL CONTEXT (terms that may appear in the notes):\n\n" + load_glossary()


def build_user_prompt(notes: str) -> str:
    """Direct mode: the plain baseline's question list and per-item
    fields, with FiRE's procedure and the glossary added before the notes."""
    items = "\n".join(f"  ACE-{n}: {q}" for n, q in ACE_QUESTIONS.items())

    schema = '{\n' + _EVENTS_SCHEMA + '''
  "items": {
    "1": {
      "present": true,
      "source_sentence": "exact verbatim quote from the notes that supports this decision, or null if absent",
      "reasoning": "one or two sentences explaining why the quote maps to this ACE item, or why it is absent",
      "confidence": "high | medium | low"
    },
    ... one entry per item, 1 to 14 ...
  },
  "ace_score": <integer 0-14>
}'''

    return (
        "These are the 14 ACE questions:\n\n" + items + "\n\n"
        "Based on the consultation notes below, decide for each question whether it is "
        "present, and give the total ACE score (the number of present items, 0 to 14).\n\n"
        + PROCEDURE + "\n"
        + _glossary_block() + "\n\n"
        "For each item include:\n"
        "  - source_sentence: verbatim quote from the notes supporting the decision, or null if absent\n"
        "  - reasoning: brief explanation of why the item is present or absent\n"
        "  - confidence: 'high' (explicit evidence), 'medium' (inferred), or 'low' (ambiguous or absent)\n\n"
        "Respond with JSON only, in this shape:\n" + schema + "\n\n"
        "=== CONSULTATION NOTES ===\n" + notes.strip() + "\n=== END NOTES ==="
    )


def build_reconcile_prompt(notes: str, form: dict) -> str:
    """Reconciliation mode: the published reconciliation instructions, word
    for word, with FiRE's procedure, Step 6 and the glossary added."""
    items = "\n".join(
        f"  ACE-{n}: {ACE_QUESTIONS[n]}\n          patient's answer: {form[n]}"
        for n in ACE_QUESTIONS
    )

    schema = '{\n' + _EVENTS_SCHEMA + '''
  "items": {
    "1": {
      "present": true,
      "source_sentence": "verbatim quote from the notes supporting your final value, or null",
      "reasoning": "one or two sentences explaining your final value for this item",
      "confidence": "high | medium | low",
      "revision_reason": "if you changed the patient's answer, what in the notes justifies the change; otherwise null"
    },
    ... one entry per item, 1 to 14 ...
  },
  "ace_score": <integer 0-14>
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
        + PROCEDURE + RECONCILE_STEP + "\n"
        + _glossary_block() + "\n\n"
        "For every item, give your final value (present = true for 1, false for 0) with "
        "a supporting source quote, reasoning, and confidence. For any item whose answer "
        "you changed from the patient's, also give the reason for the change.\n\n"
        "PATIENT'S SELF-FILLED ACE FORM (item — question — patient's answer):\n\n"
        + items + "\n\n"
        "Respond with JSON only, in this shape:\n" + schema + "\n\n"
        "=== CONSULTATION NOTES ===\n" + notes.strip() + "\n=== END NOTES ==="
    )


# ─────────────────────────────────────────────────────────────
# RESULT NORMALISATION
# ─────────────────────────────────────────────────────────────
_CHECKS = ("construct", "actor_role", "age_window", "grounding")


def _item_num(v):
    """'ACE-7' / 'ACE 7' / 7 -> 7; 'none' / anything else -> None."""
    m = re.search(r"\d+", str(v or ""))
    n = int(m.group()) if m and str(v).strip().lower() != "none" else None
    return n if n and 1 <= n <= 14 else None


def normalise_events(raw: dict) -> dict:
    """Clean the model's `events` list and summarise what each step did.

    `procedure_consistent` is False when the items the model marks present
    differ from the items its own kept events map to (direct mode only:
    in reconciliation a kept "yes" can come from the form, not an event).
    """
    events = []
    for e in raw.get("events") or []:
        if not isinstance(e, dict):
            continue
        checks = e.get("checks") if isinstance(e.get("checks"), dict) else {}
        checks = {c: str(checks.get(c) or "").strip().lower() or None for c in _CHECKS}
        events.append({
            "event":            str(e.get("event") or "").strip(),
            "quote":            str(e.get("quote") or "").strip() or None,
            "persons_involved": str(e.get("persons_involved") or "").strip(),
            "age":              str(e.get("age") or "unknown"),
            "item":             _item_num(e.get("item")),
            "checks":           checks,
            "kept":             base._as_bool(e.get("kept")),
            "drop_reason":      (str(e.get("drop_reason")).strip()
                                 if e.get("drop_reason") not in (None, "", "null") else None),
        })

    mapped = [e for e in events if e["item"] is not None]
    kept = [e for e in mapped if e["kept"]]
    failed_by = Counter(c for e in mapped for c in _CHECKS if e["checks"][c] == "fail")
    kept_but_failed = [i for i, e in enumerate(events)
                       if e["kept"] and any(e["checks"][c] == "fail" for c in _CHECKS)]
    return {
        "events":               events,
        "timeline":             raw.get("timeline"),
        "n_events":             len(events),
        "n_mapped":             len(mapped),
        "n_kept":               len(kept),
        "n_dropped":            len(mapped) - len(kept),
        "fails_by_check":       {c: failed_by.get(c, 0) for c in _CHECKS},
        "kept_event_items":     sorted({e["item"] for e in kept}),
        "kept_despite_failed_check": kept_but_failed,
    }


def normalise(raw: dict, form=None) -> dict:
    """The plain baselines' per-item normalisation, plus the event audit."""
    result = base.normalise_reconcile(raw, form) if form is not None else base.normalise(raw)
    audit = normalise_events(raw)
    if form is None:
        audit["procedure_consistent"] = (audit["kept_event_items"] == result["items_present"])
    result["procedure"] = audit
    return result


def print_report(model_name: str, patient: str, result: dict, form=None):
    """The plain baseline's report, then a short summary of the steps."""
    if form is not None:
        base.print_report_reconcile(model_name, patient, result)
    else:
        base.print_report(model_name, patient, result)
    p = result["procedure"]
    print(f"  PROCEDURE: {p['n_events']} events, {p['n_mapped']} mapped to an item, "
          f"{p['n_kept']} kept, {p['n_dropped']} dropped")
    print(f"  Fails by check   : {p['fails_by_check']}")
    if p["kept_despite_failed_check"]:
        print(f"  Kept despite a failed check: events {p['kept_despite_failed_check']}")
    if p.get("procedure_consistent") is False:
        print(f"  NOTE: items present {result['items_present']} differ from items of kept "
              f"events {p['kept_event_items']}")
    print()


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
def run_one(notes_path: str, model: str, key: str, form_path=None, out_path=None,
            strip=False, verbose=False) -> str:
    """Score one individual and write the result JSON; returns its path."""
    notes = base.read_notes(notes_path)
    if not notes.strip():
        raise ValueError(f"notes file is empty: {notes_path}")
    form = base.read_form(form_path) if form_path else None
    if strip:
        notes, _ = base.strip_ace_annotations(notes)

    if form is not None:
        system, user = RECONCILE_SYSTEM_PROMPT, build_reconcile_prompt(notes, form)
        scoring_method = f"{SCORING_PREFIX}_reconcile_selfform_plus_notes"
        suffix = f".{model}.{OUTPUT_TAG}_reconciled.json"
    else:
        system, user = SYSTEM_PROMPT, build_user_prompt(notes)
        scoring_method = f"{SCORING_PREFIX}_direct"
        suffix = f".{model}.{OUTPUT_TAG}.json"

    reply = base.query_model(model, key, system, user, verbose=verbose)
    raw = base._extract_json(reply)
    result = normalise(raw, form)

    patient = os.path.splitext(os.path.basename(notes_path))[0]
    print_report(model, patient, result, form)

    out_path = out_path or f"{os.path.splitext(notes_path)[0]}{suffix}"
    out = {
        "patient_id":       patient,
        "model":            model,
        "provider":         MODELS[model]["provider"],
        "scoring_method":   scoring_method,
        "notes_file":       os.path.abspath(notes_path),
        "form_file":        os.path.abspath(form_path) if form_path else None,
        "self_report_form": {str(n): v for n, v in form.items()} if form else None,
        "max_output_tokens": base.MAX_OUTPUT_TOKENS,
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        **result,
        "items":            {str(n): it for n, it in result["items"].items()},
        "raw_model_output": raw,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  written: {out_path}")
    return out_path


_NOTES_RE = re.compile(r"^patient_\d+\.txt$")


def batch_inputs(data_dir: str, with_forms: bool):
    """(notes_path, form_path or None) for every patient_NN.txt in data_dir,
    the same files the plain baselines read."""
    pairs = []
    for p in sorted(glob.glob(os.path.join(data_dir, "patient_*.txt"))):
        if not _NOTES_RE.match(os.path.basename(p)):
            continue
        form = None
        if with_forms:
            form = os.path.splitext(p)[0] + "_ace_form.json"
            if not os.path.exists(form):
                raise FileNotFoundError(f"no form for {p}: expected {form}")
        pairs.append((p, form))
    return pairs


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────
def selftest() -> bool:
    """Offline checks: the prompts carry every step and the published
    baseline text unchanged, and the event audit counts correctly."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    form = OrderedDict((n, 1 if n in (1, 4) else 0) for n in range(1, 15))
    d = build_user_prompt("NOTES")
    r = build_reconcile_prompt("NOTES", form)
    for step in ("STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"):
        check(step in d and step in r, f"{step} is in both prompts")
    check("STEP 6" in r and "STEP 6" not in d, "Step 6 only in the reconciliation prompt")
    check("between birth and age 18" in d, "The 0-18 age window is stated explicitly")
    check("kanyadaan" in d and "kanyadaan" in r, "The glossary is included")
    check(all(q in d and q in r for q in ACE_QUESTIONS.values()),
          "All 14 item texts are included word for word")
    published = base.build_reconcile_prompt("N", form).split("PATIENT'S SELF-FILLED")[0]
    check(all(par in r for par in published.split("\n\n") if par.strip()),
          "The published reconciliation instructions are included word for word")
    check("cluster" not in d.lower() and "state space" not in d.lower(),
          "No cluster table or state space is given")
    check(SYSTEM_PROMPT.startswith(base.SYSTEM_PROMPT), "System prompt extends the published one")
    check(base.MAX_OUTPUT_TOKENS == 32000, "Output cap raised for the longer reply")

    raw = {
        "events": [
            {"event": "father hit", "item": "ACE-2", "kept": True,
             "checks": {c: "pass" for c in _CHECKS}},
            {"event": "friend took money", "item": "ACE-12", "kept": False,
             "checks": {"construct": "fail", "actor_role": "pass",
                        "age_window": "fail", "grounding": "pass"},
             "drop_reason": "adult event"},
            {"event": "low mood", "item": "none", "kept": False, "checks": {}},
        ],
        "items": {str(n): {"present": n == 2} for n in range(1, 15)},
        "ace_score": 1,
    }
    res = normalise(raw)
    p = res["procedure"]
    check(p["n_events"] == 3 and p["n_mapped"] == 2 and p["n_kept"] == 1 and p["n_dropped"] == 1,
          "Event counts")
    check(p["fails_by_check"] == {"construct": 1, "actor_role": 0, "age_window": 1, "grounding": 0},
          "Fails counted per check")
    check(p["procedure_consistent"] is True and res["ace_score"] == 1,
          "Items present match the kept events")
    raw["items"]["5"] = {"present": True}
    check(normalise(raw)["procedure"]["procedure_consistent"] is False,
          "A present item with no kept event is flagged")
    rr = normalise(raw, form)
    check(rr["mode"] == "reconcile" and "procedure_consistent" not in rr["procedure"],
          "Reconciliation uses the published normalisation")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Structure-guided ACE baseline: one LLM call told to follow FiRE's "
                    "procedure. With --form, revise the patient's self-filled form instead.")
    ap.add_argument("notes", nargs="?", help="Path to one consultation-notes text file.")
    ap.add_argument("--model", "-m", choices=sorted(MODELS), help="Which LLM to query.")
    ap.add_argument("--form", default=None, help="Self-filled ACE form JSON (reconciliation).")
    ap.add_argument("--output", "-o", default=None, help="Output JSON path (single run).")
    ap.add_argument("--batch", default=None, metavar="DATA_DIR",
                    help="Run every patient_NN.txt in DATA_DIR.")
    ap.add_argument("--with-forms", action="store_true",
                    help="Batch: reconciliation, reading patient_NN_ace_form.json.")
    ap.add_argument("--out-dir", default="results_structured",
                    help="Batch: folder for the outputs (existing outputs are skipped).")
    ap.add_argument("--print-prompt", action="store_true",
                    help="Print the exact prompt for the given notes (and form) and exit.")
    ap.add_argument("--strip-ace-annotations", action="store_true",
                    help="Remove lines that already state an ACE score / item list.")
    ap.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.print_prompt:
        if not args.notes:
            ap.error("--print-prompt needs a notes file")
        notes = base.read_notes(args.notes)
        if args.strip_ace_annotations:
            notes, _ = base.strip_ace_annotations(notes)
        if args.form:
            system, user = RECONCILE_SYSTEM_PROMPT, build_reconcile_prompt(notes, base.read_form(args.form))
        else:
            system, user = SYSTEM_PROMPT, build_user_prompt(notes)
        print("--- SYSTEM ---\n" + system + "\n\n--- USER ---\n" + user)
        return

    if not args.model or not (args.notes or args.batch):
        ap.error("--model and either a notes file or --batch are required")

    cfg = MODELS[args.model]
    key = next((os.environ[e] for e in cfg["key_env"] if os.environ.get(e)), None)
    if not key:
        print(f"error: no API key found for {args.model}. Set {' or '.join(cfg['key_env'])}.",
              file=sys.stderr)
        sys.exit(1)

    if not args.batch:
        run_one(args.notes, args.model, key, args.form, args.output,
                args.strip_ace_annotations, args.verbose)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{OUTPUT_TAG}_reconciled" if args.with_forms else OUTPUT_TAG
    failed = []
    for notes_path, form_path in batch_inputs(args.batch, args.with_forms):
        stem = os.path.splitext(os.path.basename(notes_path))[0]
        out_path = os.path.join(args.out_dir, f"{stem}.{args.model}.{tag}.json")
        if os.path.exists(out_path):
            print(f"  skip (exists): {out_path}")
            continue
        try:
            run_one(notes_path, args.model, key, form_path, out_path,
                    args.strip_ace_annotations, args.verbose)
        except Exception as e:
            print(f"error: {stem}: {e}", file=sys.stderr)
            failed.append(stem)
    if failed:
        print(f"FAILED ({len(failed)}): {failed} — rerun the same command to retry them.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
