#!/usr/bin/env python3
"""
Compute-matched baseline: GPT-5.1 asked about each ACE item separately,
11 times, keeping the most common answer.

Paper concept: the compute-matched baseline of Supplementary Section S8
(main paper, Section VI). FiRE makes about 156 GPT-5.1 calls per
individual; this baseline gives GPT-5.1 about as many (14 items x 11
answers = 154 calls) to test whether the number of calls explains the gap.

Each call receives the full notes and ONE item, in the plain prompt's
wording (fire_llm_baseline_self_form_included.py): the same system message,
the same item text, the same per-item fields (a verbatim source sentence, a
short reasoning, a confidence) and, with --form, the same reconciliation
instruction with the individual's own answer to that item. Reasoning is off
(effort "none") but no temperature is set, so repeated answers can differ.

For each item the final answer is the most common of the first k answers,
for k = 1, 3, 5, 7, 9 and 11 (odd, so there is never a tie). The cited
sentence, reasoning and confidence come from the FIRST call that gave the
most common answer. The output keeps every answer, so accuracy for each k
can be computed against the reference (scripts/collect_answers.py --k).

Usage:
  python3 fire_llm_compute_matched.py notes.txt -o out.json
  python3 fire_llm_compute_matched.py notes.txt --form form.json -o out.json
  python3 fire_llm_compute_matched.py notes.txt --item 6 --print-prompt   # no API call
  python3 fire_llm_compute_matched.py --selftest

Answers are saved to <output>.partial.json after every call, so an
interrupted run resumes where it stopped when the same command is rerun.

API key: OPENAI_API_KEY. Standard library only.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fire_llm_baseline_self_form_included as base  # noqa: E402

MODEL = "gpt-5.1"
VOTES = 11
K_VALUES = (1, 3, 5, 7, 9, 11)
# Output limits of the plain prompt's scripts (Supplementary Table SIII):
# 8000 tokens with notes alone, 16000 with the form.
MAX_OUTPUT_TOKENS_NOTES = 8000
MAX_OUTPUT_TOKENS_FORM = 16000
MAX_ATTEMPTS_PER_ANSWER = 5

ACE_QUESTIONS = base.ACE_QUESTIONS


# ─────────────────────────────────────────────────────────────
# PROMPTS (one item, in the plain prompt's wording)
# ─────────────────────────────────────────────────────────────
def build_item_prompt(notes: str, item: int) -> str:
    """Notes only: the plain direct prompt, restricted to one item."""
    schema = '''{
  "present": true,
  "source_sentence": "exact verbatim quote from the notes that supports this decision, or null if absent",
  "reasoning": "one or two sentences explaining why the quote maps to this ACE item, or why it is absent",
  "confidence": "high | medium | low"
}'''
    return (
        f"This is ACE question {item}:\n\n  ACE-{item}: {ACE_QUESTIONS[item]}\n\n"
        "Based on the consultation notes below, decide whether it is present.\n\n"
        "Include:\n"
        "  - source_sentence: verbatim quote from the notes supporting the decision, or null if absent\n"
        "  - reasoning: brief explanation of why the item is present or absent\n"
        "  - confidence: 'high' (explicit evidence), 'medium' (inferred), or 'low' (ambiguous or absent)\n\n"
        "Respond with JSON only, in this shape:\n" + schema + "\n\n"
        "=== CONSULTATION NOTES ===\n" + notes.strip() + "\n=== END NOTES ==="
    )


def build_item_reconcile_prompt(notes: str, item: int, answer: int) -> str:
    """Notes plus form: the plain reconciliation prompt, restricted to one item."""
    schema = '''{
  "present": true,
  "source_sentence": "verbatim quote from the notes supporting your final value, or null",
  "reasoning": "one or two sentences explaining your final value for this item",
  "confidence": "high | medium | low",
  "revision_reason": "if you changed the patient's answer, what in the notes justifies the change; otherwise null"
}'''
    return (
        "Below is one item of the patient's OWN self-filled ACE questionnaire — scored 0 "
        "(no) or 1 (yes) by the patient — followed by the clinician's consultation "
        "notes for the same patient.\n\n"
        "Your task: starting from the patient's own answer, REVISE it using the "
        "consultation notes. Change the answer ONLY when the notes give evidence for a "
        "different answer. Note that the consultation notes may not cover every item — "
        "they are not guaranteed to mention everything the patient reported.\n\n"
        "Give your final value (present = true for 1, false for 0) with a supporting "
        "source quote, reasoning, and confidence. If you changed the patient's answer, "
        "also give the reason for the change.\n\n"
        "PATIENT'S SELF-FILLED ACE ITEM (item — question — patient's answer):\n\n"
        f"  ACE-{item}: {ACE_QUESTIONS[item]}\n          patient's answer: {answer}\n\n"
        "Respond with JSON only, in this shape:\n" + schema + "\n\n"
        "=== CONSULTATION NOTES ===\n" + notes.strip() + "\n=== END NOTES ==="
    )


def prompts_for(notes: str, item: int, form: dict = None) -> tuple:
    """(system, user) for one item, notes only or with the form."""
    if form is None:
        return base.SYSTEM_PROMPT, build_item_prompt(notes, item)
    return base.RECONCILE_SYSTEM_PROMPT, build_item_reconcile_prompt(notes, item, form[item])


# ─────────────────────────────────────────────────────────────
# MODEL CALL
# ─────────────────────────────────────────────────────────────
def build_payload(system: str, user: str, with_form: bool) -> dict:
    """GPT-5.1 with reasoning off and NO temperature (Supplementary
    Section S8), so that repeated answers can differ."""
    return {
        "model": MODEL,
        "reasoning_effort": "none",
        "max_completion_tokens": MAX_OUTPUT_TOKENS_FORM if with_form else MAX_OUTPUT_TOKENS_NOTES,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }


def parse_answer(raw: dict) -> dict:
    """One call's reply as {present, source_sentence, reasoning, confidence,
    revision_reason}. Accepts the item object itself or {"items": {"n": ...}}."""
    if isinstance(raw, dict) and "present" not in raw:
        items = raw.get("items")
        if isinstance(items, dict) and len(items) == 1:
            raw = next(iter(items.values()))
    if not isinstance(raw, dict) or "present" not in raw:
        raise ValueError("reply has no 'present' field")
    src = raw.get("source_sentence")
    return {
        "present": base._as_bool(raw.get("present")),
        "source_sentence": (str(src).strip() or None) if src not in (None, "null") else None,
        "reasoning": str(raw.get("reasoning") or "").strip(),
        "confidence": base._valid_confidence(raw.get("confidence")),
        "revision_reason": (str(raw.get("revision_reason")).strip() or None)
        if raw.get("revision_reason") not in (None, "null", "") else None,
    }


def ask_once(key: str, system: str, user: str, with_form: bool, post=None) -> dict:
    """One answer, retried on transport errors and unparsable replies."""
    post = post or base._post_json
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = build_payload(system, user, with_form)
    last = None
    for attempt in range(MAX_ATTEMPTS_PER_ANSWER):
        try:
            status, body = post(base.MODELS[MODEL]["url"], payload, headers)
        except Exception as e:  # network error
            last = f"transport error: {e}"
            time.sleep(2 ** attempt)
            continue
        if status == 200:
            try:
                content = json.loads(body)["choices"][0]["message"]["content"]
                return parse_answer(base._extract_json(content))
            except Exception as e:
                last = f"unparsable reply: {e}"
                continue
        last = f"HTTP {status}: {body[:300]}"
        if status in (429, 500, 502, 503, 529):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(last)
    raise RuntimeError(f"no valid answer after {MAX_ATTEMPTS_PER_ANSWER} attempts ({last})")


# ─────────────────────────────────────────────────────────────
# VOTING
# ─────────────────────────────────────────────────────────────
def majority(answers: list, k: int) -> bool:
    """Most common 'present' value among the first k answers (k odd)."""
    if k % 2 == 0 or k > len(answers):
        raise ValueError(f"k must be odd and at most {len(answers)}, got {k}")
    c = Counter(a["present"] for a in answers[:k])
    return c[True] > c[False]


def decide(answers_by_item: dict, k: int, form: dict = None) -> "OrderedDict":
    """Per-item decision from the first k answers. The cited sentence,
    reasoning and confidence are those of the first call giving the
    majority answer."""
    out = OrderedDict()
    for n in ACE_QUESTIONS:
        ans = answers_by_item[n]
        present = majority(ans, k)
        first = next(a for a in ans[:k] if a["present"] == present)
        votes = sum(1 for a in ans[:k] if a["present"])
        entry = {
            "present": present,
            "source_sentence": first["source_sentence"],
            "reasoning": first["reasoning"],
            "confidence": first["confidence"],
            "votes_present": votes,
            "votes": k,
        }
        if form is not None:
            entry["self_report"] = form[n]
            entry["changed"] = int(present) != form[n]
            entry["revision_reason"] = first["revision_reason"] if entry["changed"] else None
        out[n] = entry
    return out


def answer_string(decisions: dict) -> str:
    """The 14 decisions as a 0/1 string, item 1 first."""
    return "".join("1" if decisions[n]["present"] else "0" for n in ACE_QUESTIONS)


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
def fingerprint(notes: str, form: dict = None) -> str:
    """Short hash of the notes and form, so a resume never mixes inputs."""
    blob = json.dumps([notes, form], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run(notes: str, key: str, form: dict = None, votes: int = VOTES,
        partial_path: str = None, ask=None, verbose: bool = False) -> dict:
    """Collect `votes` answers per item (resuming from partial_path), then
    the decisions for every k in K_VALUES up to `votes`."""
    ask = ask or ask_once
    answers = {n: [] for n in ACE_QUESTIONS}
    if partial_path and os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("with_form") != (form is not None):
            raise SystemExit(f"error: {partial_path} was made in the other condition")
        if saved.get("input") not in (None, fingerprint(notes, form)):
            raise SystemExit(f"error: {partial_path} was made from different notes or form; "
                             "delete it or choose another --output")
        answers.update({int(n): a for n, a in saved["answers"].items()})
    calls = 0
    for round_ in range(votes):                 # round-robin over items
        for n in ACE_QUESTIONS:
            if len(answers[n]) > round_:
                continue
            system, user = prompts_for(notes, n, form)
            answers[n].append(ask(key, system, user, form is not None))
            calls += 1
            if verbose:
                print(f"  ACE-{n} answer {round_ + 1}/{votes}: "
                      f"{'yes' if answers[n][-1]['present'] else 'no'}", file=sys.stderr)
            if partial_path:
                tmp = partial_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"with_form": form is not None, "input": fingerprint(notes, form),
                               "answers": {str(k): v for k, v in answers.items()}}, f)
                os.replace(tmp, partial_path)
    ks = [k for k in K_VALUES if k <= votes]
    by_k = OrderedDict((str(k), decide(answers, k, form)) for k in ks)
    final = by_k[str(ks[-1])]
    return {
        "items": OrderedDict((str(n), e) for n, e in final.items()),
        "ace_score": sum(1 for e in final.values() if e["present"]),
        "answers_by_k": OrderedDict((k, answer_string(d)) for k, d in by_k.items()),
        "calls_made_this_run": calls,
        "calls_total": sum(len(a) for a in answers.values()),
        "all_answers": OrderedDict((str(n), a) for n, a in answers.items()),
    }


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────
def selftest() -> bool:
    """Offline checks of the prompts, payload, voting and resume logic."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    notes = "Age 0-8: father hit the client often."
    s, u = prompts_for(notes, 2)
    check(s == base.SYSTEM_PROMPT and f"ACE-2: {ACE_QUESTIONS[2]}" in u
          and "ACE-3:" not in u and notes in u, "notes-only prompt: plain wording, one item")
    form = {n: 0 for n in ACE_QUESTIONS}
    s, u = prompts_for(notes, 6, form)
    check(s == base.RECONCILE_SYSTEM_PROMPT and "patient's answer: 0" in u
          and "ACE-7:" not in u and "ONLY when the notes give evidence" in u,
          "form prompt: plain reconciliation wording, one item")
    p = build_payload("s", "u", False)
    check(p["reasoning_effort"] == "none" and "temperature" not in p
          and p["max_completion_tokens"] == 8000, "payload: reasoning off, no temperature, 8000")
    check(build_payload("s", "u", True)["max_completion_tokens"] == 16000,
          "payload with the form: 16000")
    check(len(ACE_QUESTIONS) * VOTES == 154, "14 items x 11 answers = 154 calls")

    check(parse_answer({"items": {"3": {"present": "yes", "confidence": "HIGH"}}})["present"]
          is True, "parser accepts a wrapped single item")

    # Deterministic fake model: item n says yes on calls whose index i
    # satisfies (i + n) % 3 == 0, so majorities differ by k.
    counter = Counter()

    def fake_ask(key, system, user, with_form):
        n = int(user.split("ACE-")[1].split(":")[0])
        i = counter[n]
        counter[n] += 1
        return {"present": (i + n) % 3 == 0, "source_sentence": f"s{n}.{i}",
                "reasoning": "r", "confidence": "high", "revision_reason": None}

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        part = os.path.join(d, "x.partial.json")
        res = run(notes, "k", None, VOTES, part, ask=fake_ask)
        check(res["calls_total"] == 154 and res["calls_made_this_run"] == 154,
              "154 calls made")
        # item 3: answers yes at i = 0, 3, 6, 9 -> 1 yes of first 1, 4 of 11
        check(res["answers_by_k"]["1"][2] == "1" and res["answers_by_k"]["11"][2] == "0",
              "majority over the first k answers differs by k as expected")
        it3 = res["items"]["3"]
        check(it3["present"] is False and it3["source_sentence"] == "s3.1"
              and it3["votes_present"] == 4, "cited sentence from the first majority call")
        res2 = run(notes, "k", None, VOTES, part, ask=fake_ask)
        check(res2["calls_made_this_run"] == 0 and res2["answers_by_k"] == res["answers_by_k"],
              "rerun resumes from the saved answers without new calls")
        try:
            run(notes + " Changed.", "k", None, VOTES, part, ask=fake_ask)
            check(False, "a resume with different notes is refused")
        except SystemExit:
            check(True, "a resume with different notes is refused")
    try:
        majority([{"present": True}] * 4, 4)
        check(False, "even k rejected")
    except ValueError:
        check(True, "even k rejected")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Compute-matched GPT-5.1 baseline: each ACE item asked separately "
                    "11 times; the most common answer is kept.")
    ap.add_argument("notes", nargs="?", help="Consultation-notes text file.")
    ap.add_argument("--form", default=None, help="Self-filled ACE form JSON (notes plus form).")
    ap.add_argument("--output", "-o", default=None,
                    help="Output JSON (default: <notes>.gpt-5.1.ace_compute_matched[_reconciled].json)")
    ap.add_argument("--votes", type=int, default=VOTES,
                    help="Answers per item (default 11, as in the paper; must be odd)")
    ap.add_argument("--item", type=int, default=1, help="Item for --print-prompt")
    ap.add_argument("--print-prompt", action="store_true", help="Print one item's prompt and exit")
    ap.add_argument("--strip-ace-annotations", action="store_true",
                    help="Remove lines that already state an ACE score / item list")
    ap.add_argument("--selftest", action="store_true", help="Run offline checks and exit")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.notes:
        ap.error("a notes file is required (unless using --selftest)")
    if args.votes < 1 or args.votes % 2 == 0:
        ap.error("--votes must be a positive odd number")

    notes = base.read_notes(args.notes)
    if not notes.strip():
        sys.exit("error: notes file is empty")
    if args.strip_ace_annotations:
        notes, _ = base.strip_ace_annotations(notes)
    form = base.read_form(args.form) if args.form else None

    if args.print_prompt:
        system, user = prompts_for(notes, args.item, form)
        print("--- SYSTEM ---\n" + system + "\n\n--- USER ---\n" + user)
        return

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("error: set OPENAI_API_KEY")

    tag = "ace_compute_matched_reconciled" if form else "ace_compute_matched"
    out_path = args.output or f"{os.path.splitext(args.notes)[0]}.{MODEL}.{tag}.json"
    partial = out_path + ".partial.json"
    result = run(notes, key, form, args.votes, partial, verbose=args.verbose)

    out = {
        "patient_id": os.path.splitext(os.path.basename(args.notes))[0],
        "model": MODEL,
        "scoring_method": "llm_compute_matched" + ("_reconciliation" if form else "_direct"),
        "settings": {"reasoning_effort": "none", "temperature": "not set",
                     "answers_per_item": args.votes,
                     "max_output_tokens": MAX_OUTPUT_TOKENS_FORM if form else MAX_OUTPUT_TOKENS_NOTES},
        "notes_file": os.path.abspath(args.notes),
        "form_file": os.path.abspath(args.form) if args.form else None,
        "self_report_form": {str(n): v for n, v in form.items()} if form else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.remove(partial)
    print(f"  ACE score (majority of {args.votes}): {result['ace_score']}/14")
    print(f"  answers by k: {dict(result['answers_by_k'])}")
    print(f"  written: {out_path}")


if __name__ == "__main__":
    main()
