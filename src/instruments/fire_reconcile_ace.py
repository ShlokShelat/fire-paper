"""FiRE Pipeline — ACE reconciliation: the notes-plus-form condition.

Paper concept: in the notes-plus-form condition FiRE is given the
individual's self-filled ACE form and "alters an answer only where the
notes provide evidence" (main paper; Supplementary Material, Section S1,
"Notes plus form"). This file is that step. It does not re-run or
re-decide anything in the pipeline: it reads FiRE's finished ACE output
and the form, and decides each of the 14 items from them. It makes no
model call.

Pipeline position: after Worker 4 ACE (fire_worker4_ace.py).
Inputs: the Worker 3 ACE output (evidence: resolved_states,
excluded_events, age_unverified, verification_report), the Worker 4 ACE
score card, and the self-filled form ({"1": {"score": 0|1}, ...} or
{"1": 0|1, ...}; only the yes/no answer is read, never the frequency; an
item missing from the form has no form answer and is never "changed").
Output: the reconciled form, one record per item with the final answer,
where it came from, the evidence, whether it changed, and whether a
psychotherapist must review it.

Rules, per item (Supplementary Material, Section S1, "Notes plus form"):
  1. FiRE counted the item on verified evidence -> yes, citing that evidence.
  2. FiRE held the item (a required condition or the age is not stated in
     the notes)                                   -> the form's answer, flagged.
  3. FiRE's only evidence for the item failed a check whose removal the
     second GPT-5.1 call did not confirm          -> the form's answer, flagged.
  4. FiRE has evidence the item does not count    -> no, citing that evidence.
     Such evidence is an ACE event FiRE left out only because it starts
     after 18, or a removal the second GPT-5.1 call confirmed.
  5. The notes are silent on the item             -> the form's answer, flagged.
In 1 and 4 the answer changes when it differs from the form, and the change
cites FiRE's evidence; in 2, 3 and 5 the form's answer is never changed.

Usage:
  python3 fire_reconcile_ace.py worker3_ace_output.json ace_scorecard.json \
      --form form.json -o ace_reconciled.json
  python3 fire_reconcile_ace.py --selftest
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fire_worker4_ace import ACE_QUESTIONS  # noqa: E402

ITEMS = range(1, 15)


def _evidence_row(rec: dict, kind: str) -> dict:
    return {"kind": kind, "state_code": rec.get("state_code"), "event": rec.get("event"),
            "source_sentence": rec.get("source_sentence"),
            "onset_age": rec.get("onset_age"), "end_age": rec.get("end_age")}


def read_form(form: dict) -> dict:
    """Self-filled form -> {item: 0 | 1 | None}. Only the yes/no answer is read."""
    answers = {}
    for i in ITEMS:
        v = form.get(str(i), form.get(i))
        if isinstance(v, dict):
            v = v.get("score")
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ("", "na", "n/a", "none", "null"):
                v = None
            elif t in ("yes", "y", "true", "1"):
                v = 1
            elif t in ("no", "n", "false", "0"):
                v = 0
            else:
                raise ValueError(f"form item {i}: unreadable answer {v!r}")
        answers[i] = None if v is None else (1 if int(v) else 0)
    return answers


def collect_evidence(w3: dict, score: dict) -> dict:
    """Sort FiRE's output into, per item: verified evidence for the item,
    evidence against it, evidence found but not confirmed, and whether
    FiRE held the item."""
    ev = {i: {"for": [], "against": [], "unconfirmed": [], "review": [],
              "held": False, "verified_uids": set(), "kept_uids": set()} for i in ITEMS}
    unconfirmed_uids = {f.get("event_uid") for f in (w3.get("verification_report") or [])
                        if f.get("severity") == "fail" and not f.get("record_removed")}

    for st in w3.get("resolved_states") or []:
        i = st.get("ace_item")
        if i in ev:
            uids = set(st.get("event_uids") or [])
            ev[i]["kept_uids"] |= uids
            verified = (uids - unconfirmed_uids) if uids else {None}
            ev[i]["verified_uids"] |= verified
            if verified:
                ev[i]["for"].append(_evidence_row(st, "verified state"))

    for x in w3.get("excluded_events") or []:
        i = x.get("ace_item")
        if x.get("reason") == "out_of_window" and i in ev:
            ev[i]["against"].append(_evidence_row(x, "starts after age 18"))

    for x in w3.get("age_unverified") or []:
        i = x.get("ace_item")
        if i in ev:
            ev[i]["held"] = True
            ev[i]["unconfirmed"].append(_evidence_row(x, "no age: held for age review"))
            ev[i]["review"].append("FiRE found this item but the event has no age; "
                                   "a psychotherapist must confirm the age")

    for h in w3.get("held_for_review") or []:
        i = h.get("ace_item")
        if i in ev:
            ev[i]["held"] = True
            ev[i]["unconfirmed"].append(_evidence_row(h, "held: " + str(h.get("reason"))))
            ev[i]["review"].append("FiRE held this item: " + str(h.get("reason")))

    for f in w3.get("verification_report") or []:
        i = f.get("ace_item")
        if i not in ev:
            continue
        if f.get("record_removed"):
            ev[i]["against"].append(_evidence_row(f, "adjudicator confirmed the notes do not support it"))
        elif f.get("severity") == "fail":
            ev[i]["unconfirmed"].append(_evidence_row(f, "failed a check; removal not confirmed"))
            ev[i]["review"].append("a check failed on FiRE's evidence and the second call "
                                   "did not confirm its removal")

    card = score.get("ace_scorecard") or {}
    for i in ITEMS:
        row = card.get(str(i), card.get(i)) or {}
        if row.get("held_for_review") and not ev[i]["held"]:
            ev[i]["held"] = True
            ev[i]["review"].append("FiRE held this item: the notes do not state a "
                                   "required condition")
    return ev


def reconcile(w3: dict, score: dict, form: dict) -> dict:
    """Decide all 14 items from FiRE's evidence and the self-filled form."""
    answers = read_form(form)
    ev = collect_evidence(w3, score)
    present = set(score.get("items_present") or [])
    items = []
    for i in ITEMS:
        f, e = answers[i], ev[i]
        fire = 1 if i in present else 0
        if i in present and e["verified_uids"]:
            final, source = 1, "FiRE evidence"
            evidence = e["for"] + e["unconfirmed"]
            reason = "FiRE found the item in the notes"
        elif e["held"] and i not in present:
            final, source = f, "self-filled form"
            evidence = e["unconfirmed"]
            reason = "FiRE held this item for review; the form's answer is kept"
        elif i in present:
            final, source = f, "self-filled form"
            evidence = e["unconfirmed"]
            reason = ("FiRE's evidence failed a check whose removal was not confirmed; "
                      "the form's answer is kept")
        elif e["against"]:
            final, source = 0, "FiRE evidence"
            evidence = e["against"] + e["unconfirmed"]
            reason = "the notes place this item outside what it counts (" + e["against"][0]["kind"] + ")"
        else:
            final, source = f, "self-filled form"
            evidence = e["unconfirmed"]
            reason = "the notes are silent on this item; the form's answer is kept"
        review = list(e["review"])
        if source == "self-filled form" and not review:
            review.insert(0, "taken from the self-filled form: no evidence in the notes")
        changed = f is not None and final is not None and final != f
        if changed:
            reason += f"; changed from the form's {'yes' if f else 'no'}"
        items.append({
            "ace_item": i, "question": ACE_QUESTIONS.get(i), "form_answer": f, "fire_answer": fire,
            "final_answer": final, "source": source, "changed": changed, "reasoning": reason,
            "evidence": evidence, "needs_review": bool(review), "review_reasons": review,
        })
    return {
        "patient_id": score.get("patient_id") or w3.get("patient_id"),
        "condition": "notes + self-filled form",
        "reconciled_ace_score": sum(1 for r in items if r["final_answer"] == 1),
        "form_ace_score": sum(1 for v in answers.values() if v == 1),
        "fire_ace_score": score.get("canonical_ace_score"),
        "items_changed": [r["ace_item"] for r in items if r["changed"]],
        "items_from_form": [r["ace_item"] for r in items if r["source"] == "self-filled form"],
        "items_for_review": [r["ace_item"] for r in items if r["needs_review"]],
        "items": items,
    }


def selftest() -> bool:
    """Offline checks of every rule on synthetic FiRE output."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    src = "Age 0 to 8: father emotionally abusive"
    w3 = {
        "resolved_states": [
            {"state_code": "q-a", "ace_item": 1, "event": "Father emotionally abusive", "source_sentence": src,
             "onset_age": 0, "end_age": 8, "event_uids": ["u-a"]},
            {"state_code": "q-a2", "ace_item": 1, "event": "Father shouted", "source_sentence": "shouted at 9",
             "onset_age": 9, "end_age": 9, "event_uids": ["u-a2"]},
            {"state_code": "q-p", "ace_item": 12, "event": "Felt lonely", "source_sentence": "lonely at 12",
             "onset_age": 12, "end_age": 12, "event_uids": ["u-p"]}],
        "excluded_events": [
            {"reason": "out_of_window", "ace_item": 8, "state_code": "q-n1", "event": "Father began drinking",
             "source_sentence": "father began drinking heavily when the client was 25", "onset_age": 25},
            {"reason": "not_ace_relevant", "state_code": "q-d", "event": "Mother died"}],
        "age_unverified": [{"ace_item": 3, "state_code": "q-i3", "event": "Inappropriate touch",
                            "source_sentence": "inappropriate touch"}],
        "held_for_review": [
            {"ace_item": 13, "reason": "two or more years not stated", "state_code": "q-t1",
             "event": "Family very poor", "source_sentence": "poor at 10", "onset_age": 10, "end_age": 10},
            {"ace_item": 11, "reason": "frequency not stated", "state_code": "q-c",
             "event": "Teased once", "source_sentence": "teased once at 7", "onset_age": 7, "end_age": 7}],
        "verification_report": [
            {"ace_item": 6, "state_code": "q-l1", "event": "Parents separated", "severity": "fail",
             "record_removed": True, "event_uid": "u-l1"},
            {"ace_item": 1, "state_code": "q-a2", "event": "Father shouted", "severity": "fail",
             "record_removed": False, "drop_unconfirmed": True, "event_uid": "u-a2"},
            {"ace_item": 12, "state_code": "q-p", "event": "Felt lonely", "severity": "fail",
             "record_removed": False, "drop_unconfirmed": False, "event_uid": "u-p",
             "adjudicator": {"confirm_drop": False}}],
    }
    score = {"items_present": [1, 12], "canonical_ace_score": 2,
             "ace_scorecard": {"13": {"present": False, "held_for_review": True},
                               "11": {"present": False, "held_for_review": True}}}
    form = {str(i): {"score": 0} for i in ITEMS}
    form.update({"6": {"score": 1}, "8": {"score": 1}, "9": {"score": 1}, "13": {"score": 1}})
    out = reconcile(w3, score, form)
    it = {r["ace_item"]: r for r in out["items"]}

    check(it[1]["final_answer"] == 1 and it[1]["changed"] and it[1]["evidence"],
          "form no + FiRE yes on verified evidence -> yes, changed, evidence attached")
    check(it[1]["needs_review"], "a second, unconfirmed finding on a found item -> marked for review")
    check(it[12]["final_answer"] == 0 and not it[12]["changed"] and it[12]["needs_review"],
          "only evidence failed a check whose removal was not confirmed -> form's answer kept")
    check(it[13]["final_answer"] == 1 and not it[13]["changed"] and it[13]["needs_review"],
          "held item (duration not stated), form yes -> yes kept, unchanged, flagged")
    check(it[11]["final_answer"] == 0 and not it[11]["changed"] and it[11]["needs_review"],
          "held item (frequency not stated), form no -> no kept, flagged")
    check(it[8]["final_answer"] == 0 and it[8]["changed"] and it[8]["source"] == "FiRE evidence",
          "form yes + event only after 18 -> no, changed")
    check(it[6]["final_answer"] == 0 and it[6]["changed"],
          "form yes + confirmed removal -> no, changed")
    check(it[9]["final_answer"] == 1 and it[9]["source"] == "self-filled form" and not it[9]["evidence"]
          and it[9]["needs_review"], "notes silent -> form's yes kept, no evidence, marked for review")
    check(it[2]["final_answer"] == 0 and it[2]["needs_review"], "notes silent -> form's no kept, marked for review")
    check(it[3]["final_answer"] == 0 and it[3]["evidence"] and it[3]["needs_review"] and not it[3]["changed"],
          "event with no age -> held, form's answer kept, evidence attached")
    check(out["reconciled_ace_score"] == 3 and out["items_changed"] == [1, 6, 8],
          "reconciled score and changed items")
    check(all(not r["changed"] for r in out["items"] if r["source"] == "self-filled form"),
          "the form's answer is never changed without FiRE evidence")
    check(read_form({"1": 1, "2": {"score": 4}})[2] == 1 and read_form({})[1] is None,
          "form: any non-zero answer is yes; a missing answer is None")
    print("\n" + ("SELF-TEST: ALL PASS" if ok else "SELF-TEST: FAILURES ABOVE"))
    return ok


def main():
    p = argparse.ArgumentParser(
        description="FiRE notes-plus-form step: decide each ACE item from FiRE's finished "
                    "output and the self-filled form (no model call)")
    p.add_argument("ace", nargs="?", help="Worker 3 (ACE) output (worker3_ace_output.json)")
    p.add_argument("score", nargs="?", help="Worker 4 (ACE) score card (ace_scorecard.json)")
    p.add_argument("--form", help="Self-filled ACE form JSON")
    p.add_argument("--output", "-o", default="ace_reconciled.json",
                   help="Output path (default: ace_reconciled.json)")
    p.add_argument("--selftest", action="store_true", help="Run offline checks and exit")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not (args.ace and args.score and args.form):
        p.error("ace, score and --form are required (unless using --selftest)")

    loaded = []
    for path in (args.ace, args.score, args.form):
        try:
            with open(path, encoding="utf-8") as f:
                loaded.append(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error("Cannot load %s: %s", path, e)
            sys.exit(1)
    out = reconcile(*loaded)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Reconciled ACE score: {out['reconciled_ace_score']} "
          f"(form {out['form_ace_score']}, FiRE {out['fire_ace_score']})")
    print(f"Changed from the form: {out['items_changed']}")
    print(f"Taken from the form (no evidence): {out['items_from_form']}")
    print(f"For psychotherapist review: {out['items_for_review']}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
