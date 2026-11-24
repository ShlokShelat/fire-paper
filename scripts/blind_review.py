#!/usr/bin/env python3
"""
Blind, shuffled review sheets for the psychotherapists' item-by-item
judging, and their conversion back into a judgements CSV.

Supplementary Section S5: "every output was converted to one plain layout
(for each item, the decision, the cited sentence, and a short reasoning),
with FiRE's trajectory and flags removed, then stripped of any label and
shuffled." The reviewers then record, per item, one agreed judgment and its
error mode (E1 to E6, or empty when correct and grounded).

  export  Reads every system's outputs and writes
            sheet.csv  one row per (output, item), labels removed, rows in a
                       seeded random order; the reviewers fill `error_mode`
                       (and may add notes)
            key.csv    blind_id -> system, condition, individual; keep it away
                       from the reviewers until judging is finished
  import  Joins the filled sheet with the key and writes the judgements CSV
          that scripts/evaluate_judgements.py, make_tables.py and
          aggregate_runs.py read.

Usage:
  python3 scripts/blind_review.py export --reference ref.json \\
      --add "FiRE" notes "runs/*/" \\
      --add "FiRE" notes+form "runs/*/" \\
      --add "GPT-5.1" notes "out/*.gpt-5.1.ace.json" \\
      --sheet sheet.csv --key key.csv --seed 20260501
  python3 scripts/blind_review.py import sheet_filled.csv key.csv -o judgements.csv
  python3 scripts/blind_review.py --selftest

For FiRE, --add takes the individuals' working folders (as written by
scripts/run_fire.py): the decision comes from ace_scorecard.json (notes) or
ace_reconciled.json (notes + form), the cited sentence and a short reasoning
from the verified states in worker3_ace_output.json. Other systems' files
are read as in scripts/collect_answers.py. The reference JSON is used only
to fill reference_answer in the imported judgements; it is never written to
the sheet. Standard library only.
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_answers as ca  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src", "instruments"))
import fire_reconcile_ace as rc  # noqa: E402

N_ITEMS = 14
ITEM_TEXT = {
    1: "A parent or other adult in the household often swore at, insulted, put down or humiliated "
       "you, or acted in a way that made you afraid you might be physically hurt.",
    2: "A parent or other adult in the household often pushed, grabbed, slapped or threw something "
       "at you, or ever hit you so hard that you had marks or were injured.",
    3: "An adult or person at least 5 years older ever touched you sexually, or tried to or had sex with you.",
    4: "You often felt no one in your family loved you or thought you important or special, or your "
       "family did not look out for each other, feel close or support each other.",
    5: "You often had not enough to eat, had to wear dirty clothes and had no one to protect you, or "
       "your parents were too drunk or high to take care of you.",
    6: "Your parents were ever separated or divorced.",
    7: "Your mother or stepmother was often pushed, grabbed, slapped or had something thrown at her; "
       "sometimes or often kicked, bitten or hit with a fist or something hard; or ever repeatedly "
       "hit over at least a few minutes or threatened with a gun or knife.",
    8: "You lived with a problem drinker or alcoholic, or someone who used street drugs.",
    9: "A household member was depressed or mentally ill, or attempted suicide.",
    10: "You lived for 2 or more years in a dangerous neighborhood, or where you saw people being assaulted.",
    11: "Other kids, including brothers or sisters, often or very often hit, threatened, picked on or insulted you.",
    12: "You often or very often felt lonely, rejected or that nobody liked you (by peer group).",
    13: "There was a period of 2 or more years when your family was very poor or on public assistance.",
    14: "Your parents had physical or verbal fights all the time.",
}
SHEET_COLUMNS = ("blind_id", "row", "condition", "item", "item_text", "form_answer",
                 "decision", "cited_sentence", "reasoning", "error_mode", "reviewer_notes")
KEY_COLUMNS = ("blind_id", "system", "condition", "individual", "system_answer")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _span(rec: dict) -> str:
    age, end = rec.get("onset_age"), rec.get("end_age")
    if age is None:
        return "age unknown"
    return f"age {age}" if end in (None, age) else f"age {age}-{end}"


NO_EVIDENCE = "no supporting evidence in the notes"
FORM_KEPT = "the notes do not establish this item; the form's answer is kept"


def fire_items(folder: str, condition: str) -> tuple:
    """(individual, {item: (decision, cited, reasoning, form_answer)}) for one
    FiRE working folder, with its trajectory and flags left out. A cited
    sentence always comes from verified evidence (the same evidence the
    notes-plus-form step accepts)."""
    folder = Path(folder)
    w3a = _load(folder / "worker3_ace_output.json")
    card = _load(folder / "ace_scorecard.json")
    ev = rc.collect_evidence(w3a, card)
    out = {}
    if condition == "notes":
        present = {int(i) for i in card.get("items_present") or []}
        for i in range(1, N_ITEMS + 1):
            support = ev[i]["for"][0] if ev[i]["for"] else {}
            if i in present:
                out[i] = ("yes", support.get("source_sentence") or "",
                          f"{support.get('event') or 'event in the notes'} ({_span(support)})", "")
            else:
                out[i] = ("no", "", NO_EVIDENCE, "")
        ind = card.get("patient_id") or folder.name
    else:
        rec = _load(folder / "ace_reconciled.json")
        for r in rec["items"]:
            i = int(r["ace_item"])
            first = (r.get("evidence") or [{}])[0]
            yes = r["final_answer"] == 1
            form = "" if r.get("form_answer") is None else str(r["form_answer"])
            if r.get("source") == "FiRE evidence" and yes:
                cited = first.get("source_sentence") or ""
                why = f"{first.get('event') or 'event in the notes'} ({_span(first)})"
            elif r.get("source") == "FiRE evidence":
                cited = first.get("source_sentence") or ""
                why = ("the event starts after age 18" if first.get("kind") == "starts after age 18"
                       else "the notes do not support this item")
            else:
                cited, why = "", FORM_KEPT
            out[i] = ("yes" if yes else "no", cited, why, form)
        ind = rec.get("patient_id") or folder.name
    return str(ind), out


_NAMES = re.compile(r"\b(FiRE|GPT-?5(?:\.1)?|Gemini(?: 3\.1)?(?: Pro)?|Grok(?:-4\.20)?|TF-IDF|NegEx)\b",
                    re.IGNORECASE)


def plain(cited: str, reason: str) -> tuple:
    """One plain layout for every system: fixed phrases a system always uses
    are rewritten in common words, and system names are removed."""
    cited, reason = str(cited or ""), str(reason or "")
    m = re.match(r'matched "(.+?)" \(', reason)
    if m:
        reason = f'the notes mention "{m.group(1)}"'
    elif reason == "no matching term":
        reason = NO_EVIDENCE
    elif reason.startswith("form says yes") or reason == "form unchanged":
        reason = FORM_KEPT
    elif reason.startswith("p(yes)=") or reason.endswith("other individuals have this item"):
        reason = ""
    if cited in ("(form answer kept)", "(form answer)"):
        cited = ""
    return cited, _NAMES.sub("the system", reason)


def baseline_items(path: str) -> tuple:
    d = _load(path)
    ind = ca.individual_of(path, d)
    form = d.get("self_report_form") or {}
    out = {}
    for i in range(1, N_ITEMS + 1):
        it = (d.get("items") or {}).get(str(i)) or {}
        cited = it.get("source_sentence", it.get("source")) or ""
        cited, reason = plain("" if cited in (None, "null") else cited, it.get("reasoning"))
        f = form.get(str(i), it.get("form"))
        out[i] = ("yes" if it.get("present") else "no", cited, reason,
                  "" if f is None else str(int(bool(f))))
    return ind, out


def export(additions: list, sheet: str, key: str, seed: int) -> int:
    rows, keys, n = [], [], 0
    for system, condition, pattern in additions:
        condition = ca.CONDITIONS[ca.CONDITIONS.index(condition)]
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"error: nothing matches {pattern!r}")
        for p in paths:
            ind, items = fire_items(p, condition) if os.path.isdir(p) else baseline_items(p)
            n += 1
            for i, (decision, cited, reason, form) in items.items():
                if condition == "notes+form" and reason == NO_EVIDENCE:
                    reason = FORM_KEPT      # with the form, silent notes keep its answer
                rows.append({"condition": condition, "item": i, "item_text": ITEM_TEXT[i],
                             "form_answer": form if condition == "notes+form" else "",
                             "decision": decision, "cited_sentence": cited, "reasoning": reason,
                             "error_mode": "", "reviewer_notes": "",
                             "_key": (system, condition, ind, int(decision == "yes"))})
    rng = random.Random(seed)
    rng.shuffle(rows)
    ids = rng.sample(range(10 ** 6, 10 ** 7), len(rows))
    with open(sheet, "w", newline="", encoding="utf-8") as f, \
            open(key, "w", newline="", encoding="utf-8") as g:
        ws, wk = csv.DictWriter(f, fieldnames=SHEET_COLUMNS), csv.writer(g)
        ws.writeheader()
        wk.writerow(KEY_COLUMNS)
        for k, (r, bid) in enumerate(zip(rows, ids), start=1):
            system, cond, ind, ans = r.pop("_key")
            ws.writerow({"blind_id": f"B{bid}", "row": k, **r})
            wk.writerow([f"B{bid}", system, cond, ind, ans])
    return n


def import_sheet(sheet: str, key: str, out: str, reference: str = None) -> int:
    with open(key, newline="", encoding="utf-8") as g:
        keymap = {r["blind_id"]: r for r in csv.DictReader(g)}
    refs = {}
    if reference:
        for cond in sorted({r["condition"] for r in keymap.values()}):
            refs[cond] = ca.read_reference(reference, cond)
    n = 0
    with open(sheet, newline="", encoding="utf-8-sig") as f, \
            open(out, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["system", "condition", "individual", "item", "error_mode",
                    "system_answer", "reference_answer"])
        for r in csv.DictReader(f):
            k = keymap.get(r["blind_id"])
            if k is None:
                raise SystemExit(f"error: {r['blind_id']} is not in the key")
            mode = str(r.get("error_mode") or "").strip().upper()
            if mode and mode not in ("E1", "E2", "E3", "E4", "E5", "E6"):
                raise SystemExit(f"error: {r['blind_id']}: error_mode {mode!r}")
            item = int(r["item"])
            ref = refs.get(k["condition"], {}).get(k["individual"])
            w.writerow([k["system"], k["condition"], k["individual"], item, mode,
                        k["system_answer"], "" if ref is None else ref[item - 1]])
            n += 1
    return n


def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fire = d / "runs" / "P1"
        fire.mkdir(parents=True)
        (fire / "worker3_ace_output.json").write_text(json.dumps({"resolved_states": [
            {"ace_item": 2, "source_sentence": "[12-16] Father beat the client.",
             "event": "Father beat the client", "onset_age": 12, "end_age": 16}]}))
        (fire / "ace_scorecard.json").write_text(json.dumps(
            {"patient_id": "P1", "items_present": [2], "ace_scorecard": {},
             "fire_expression": "[C] e", "held_items": [11]}))
        base = d / "P1.gpt-5.1.ace.json"
        base.write_text(json.dumps({"patient_id": "P1", "items": {
            str(i): {"present": i == 6, "source_sentence": "divorced" if i == 6 else None,
                     "reasoning": "r"} for i in range(1, 15)}}))
        ref = d / "ref.json"
        ref.write_text(json.dumps({"P1": {"notes": "01" + "0" * 12}}))
        n = export([("FiRE", "notes", str(d / "runs" / "*")),
                    ("GPT-5.1", "notes", str(d / "*.ace.json"))],
                   str(d / "sheet.csv"), str(d / "key.csv"), seed=7)
        sheet = list(csv.DictReader(open(d / "sheet.csv")))
        text = (d / "sheet.csv").read_text()
        check(n == 2 and len(sheet) == 28, "two outputs, 28 rows")
        check("FiRE" not in text and "GPT" not in text and "[C]" not in text
              and "held" not in text, "no system label, trajectory or flag on the sheet")
        check([r["item"] for r in sheet[:14]] != [str(i) for i in range(1, 15)],
              "rows are shuffled")
        fire_row = next(r for r in sheet if r["cited_sentence"].startswith("[12-16]"))
        check(fire_row["decision"] == "yes" and "age 12-16" in fire_row["reasoning"],
              "FiRE row: decision, cited sentence and a short reasoning")
        for r in sheet:
            if r["decision"] == "yes" and r["cited_sentence"] == "divorced":
                r["error_mode"] = "E1"
            if r["decision"] == "no" and r["item"] == "2":      # GPT-5.1 misses item 2
                r["error_mode"] = "E2"
        with open(d / "filled.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
            w.writeheader()
            w.writerows(sheet)
        import_sheet(str(d / "filled.csv"), str(d / "key.csv"), str(d / "j.csv"), str(ref))
        j = list(csv.DictReader(open(d / "j.csv")))
        gpt6 = next(r for r in j if r["system"] == "GPT-5.1" and r["item"] == "6")
        check(gpt6["error_mode"] == "E1" and gpt6["system_answer"] == "1"
              and gpt6["reference_answer"] == "0", "import restores system, answers and mode")
        import evaluate_judgements as ev
        rows = ev.read_judgements(str(d / "j.csv"))
        check(not ev.validate(rows), "the imported CSV passes evaluate_judgements' checks")

        # Notes plus form: FiRE's reconcile output and the keyword baseline's.
        (fire / "ace_reconciled.json").write_text(json.dumps({"patient_id": "P1", "items": [
            {"ace_item": i, "form_answer": int(i in (2, 5)),
             "final_answer": int(i in (2, 5)),
             "source": "FiRE evidence" if i == 2 else "self-filled form",
             "reasoning": ("FiRE found the item in the notes" if i == 2 else
                           "FiRE held this item for review; the form's answer is kept"),
             "evidence": [{"kind": "verified state", "event": "Father beat the client",
                           "source_sentence": "[12-16] Father beat the client.",
                           "onset_age": 12, "end_age": 16}] if i == 2 else []}
            for i in range(1, 15)]}))
        kw = d / "kw"
        kw.mkdir()
        (kw / "P1.json").write_text(json.dumps({"items": {str(i): (
            {"present": True, "source": "(form answer kept)", "form": 1, "changed": False,
             "reasoning": "form says yes; notes also matched: x"} if i == 5 else
            {"present": True, "source": "Family: Parents divorced [Age 9]", "form": 0,
             "changed": True, "reasoning": 'matched "divorced" (item text)'} if i == 6 else
            {"present": False, "source": "", "form": 0, "reasoning": "no matching term"})
            for i in range(1, 15)}}))
        export([("FiRE", "notes+form", str(d / "runs" / "*")),
                ("Keyword", "notes+form", str(kw / "*.json"))],
               str(d / "sheet2.csv"), str(d / "key2.csv"), seed=3)
        text2 = (d / "sheet2.csv").read_text()
        sheet2 = list(csv.DictReader(open(d / "sheet2.csv")))
        check(not re.search(r"FiRE|held|matched|form says|form answer kept|no matching term", text2),
              "notes+form: no system name, flag or fixed phrase of one system")
        check(sum(r["reasoning"] == FORM_KEPT for r in sheet2) == 26 and
              sum(r["reasoning"] == NO_EVIDENCE for r in sheet2) == 0,
              "kept form answers and empty findings read the same for every system")
        check(sum(r["form_answer"] == "1" for r in sheet2) == 3,
              "the form answer is shown for both systems")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Blind review sheets and their import.")
    ap.add_argument("cmd", nargs="?", choices=["export", "import"])
    ap.add_argument("paths", nargs="*", help="import: SHEET KEY")
    ap.add_argument("--add", nargs=3, action="append", metavar=("SYSTEM", "CONDITION", "GLOB"),
                    help="export: one system's outputs (FiRE: working folders)")
    ap.add_argument("--sheet", default="sheet.csv")
    ap.add_argument("--key", default="key.csv")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed (record it)")
    ap.add_argument("--reference", help="Reference JSON (import: fills reference_answer)")
    ap.add_argument("-o", "--output", default="judgements.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.cmd == "export":
        if not a.add or a.seed is None:
            ap.error("export needs at least one --add and a --seed")
        n = export(a.add, a.sheet, a.key, a.seed)
        print(f"{n} outputs -> {a.sheet} (shuffled, seed {a.seed}); key -> {a.key}")
    elif a.cmd == "import":
        if len(a.paths) != 2:
            ap.error("import needs SHEET KEY")
        n = import_sheet(a.paths[0], a.paths[1], a.output, a.reference)
        print(f"{n} judgements -> {a.output}")
    else:
        ap.error("choose export or import")


if __name__ == "__main__":
    main()
