#!/usr/bin/env python3
"""
Collect every system's 14 answers per individual, next to the reference,
into one answers CSV.

The CSV has one row per decision with the columns that
scripts/evaluate_judgements.py and scripts/clinical_measures.py read:

  system, condition, individual, item, system_answer, reference_answer,
  error_mode (left empty; the psychotherapists fill it when judging), run

It reads the output files the repository writes:

  FiRE, notes only      ace_scorecard.json        (items_present)
  FiRE, notes + form    ace_reconciled.json       (items[].final_answer)
  LLM baselines         *.json with "items": {"1": {"present": ...}, ...}
                        (plain, structure-guided, knowledge-matched)
  compute-matched       *.ace_compute_matched*.json; --k picks the
                        majority of the first k answers (default 11)
  non-LLM baselines     src/baselines/nonllm/ outputs (same "items" shape)

The individual's id is the output's "patient_id" or, failing that, the
file name up to its first dot (FiRE: the name of the individual's working
folder). Rows from several calls can be appended into one file (--append).

Reference JSON, one entry per individual, item 1 first:
  {"P01": {"notes": "10010000100101", "notes+form": "...", "form": "..."}}
"notes" is the reference with notes alone and "notes+form" the reference
with the form (Supplementary Section S5). A single "reference" key is used
for both conditions when the two keys are absent.

Usage:
  python3 scripts/collect_answers.py --reference ref.json --system FiRE \\
      --condition notes runs/*/ace_scorecard.json -o answers.csv
  python3 scripts/collect_answers.py --reference ref.json --system FiRE \\
      --condition notes+form runs/*/ace_reconciled.json -o answers.csv --append
  python3 scripts/collect_answers.py --reference ref.json --system "GPT-5.1" \\
      --condition notes out/*.gpt-5.1.ace.json -o answers.csv --append
  python3 scripts/collect_answers.py --reference ref.json \\
      --system "GPT-5.1, compute-matched (k=5)" --k 5 --condition notes \\
      out/*.ace_compute_matched.json -o answers.csv --append
  python3 scripts/collect_answers.py --selftest

Standard library only. It makes no model call.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

N_ITEMS = 14
COLUMNS = ("system", "condition", "individual", "item", "system_answer",
           "reference_answer", "error_mode", "run")
CONDITIONS = ("notes", "notes+form")


def _bits(value) -> list:
    """A 14-answer vector from "1001...", [1, 0, ...] or {"1": 1, ...}."""
    if isinstance(value, str):
        v = [int(c) for c in value.strip()]
    elif isinstance(value, dict):
        v = [int(bool(value.get(str(i), value.get(i, 0)))) for i in range(1, N_ITEMS + 1)]
    else:
        v = [int(bool(x)) for x in value]
    if len(v) != N_ITEMS or any(x not in (0, 1) for x in v):
        raise ValueError(f"expected {N_ITEMS} answers of 0 or 1, got {value!r}")
    return v


def read_reference(path: str, condition: str) -> dict:
    """{individual: [14 answers]} for the condition."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for ind, v in data.items():
        if isinstance(v, dict):
            val = v.get(condition, v.get("reference"))
            if val is None:
                raise ValueError(f"{path}: {ind} has no '{condition}' or 'reference' answers")
        else:
            val = v
        out[str(ind)] = _bits(val)
    return out


def answers_from_output(d: dict, k: int = None) -> list:
    """The 14 answers in any of the repository's output formats."""
    if "answers_by_k" in d:                                    # compute-matched
        key = str(k) if k is not None else max(d["answers_by_k"], key=int)
        if key not in d["answers_by_k"]:
            raise ValueError(f"no answers for k={key}; have {sorted(d['answers_by_k'], key=int)}")
        return _bits(d["answers_by_k"][key])
    if k is not None:
        raise ValueError("--k applies only to compute-matched outputs")
    if isinstance(d.get("items"), list) and d["items"] and "final_answer" in d["items"][0]:
        by = {int(r["ace_item"]): r["final_answer"] for r in d["items"]}  # FiRE + form
        if any(by.get(i) is None for i in range(1, N_ITEMS + 1)):
            raise ValueError("reconciled output has an item with no final answer (no form answer)")
        return [int(by[i]) for i in range(1, N_ITEMS + 1)]
    if "items_present" in d and "ace_scorecard" in d:          # FiRE, notes only
        present = {int(i) for i in d["items_present"]}
        return [int(i in present) for i in range(1, N_ITEMS + 1)]
    if isinstance(d.get("items"), dict):                       # baselines
        return [int(bool((d["items"].get(str(i)) or {}).get("present")))
                for i in range(1, N_ITEMS + 1)]
    raise ValueError("unrecognised output format")


def individual_of(path: str, d: dict) -> str:
    if d.get("patient_id"):
        return str(d["patient_id"])
    p = Path(path)
    if p.name in ("ace_scorecard.json", "ace_reconciled.json"):
        return p.parent.name
    return p.name.split(".")[0]


def collect(paths: list, reference: dict, system: str, condition: str,
            k: int = None, run: str = "1") -> list:
    rows, seen = [], set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        ind = individual_of(path, d)
        if ind not in reference:
            raise ValueError(f"{path}: individual {ind!r} is not in the reference")
        if ind in seen:
            raise ValueError(f"{path}: a second output for {ind!r}")
        seen.add(ind)
        got = answers_from_output(d, k)
        for i in range(N_ITEMS):
            rows.append({"system": system, "condition": condition, "individual": ind,
                         "item": i + 1, "system_answer": got[i],
                         "reference_answer": reference[ind][i], "error_mode": "", "run": run})
    missing = sorted(set(reference) - seen)
    if missing:
        print(f"warning: {system} / {condition}: no output for {len(missing)} "
              f"individual(s): {missing}; measures will cover {len(seen)} of "
              f"{len(reference)} individuals", file=sys.stderr)
    return rows


def write_rows(rows: list, out: str, append: bool) -> None:
    exists = append and os.path.exists(out)
    with open(out, "a" if exists else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as d:
        ref = Path(d) / "ref.json"
        ref.write_text(json.dumps({"P1": {"notes": "1101" + "0" * 10,
                                          "notes+form": "1111" + "0" * 10},
                                   "P2": {"reference": "0" * 14}}))
        r_notes, r_form = read_reference(ref, "notes"), read_reference(ref, "notes+form")
        check(r_notes["P1"][2] == 0 and r_form["P1"][2] == 1 and r_form["P2"] == [0] * 14,
              "reference: condition keys, with 'reference' as the fallback")
        (Path(d) / "P1").mkdir()
        card = Path(d) / "P1" / "ace_scorecard.json"
        card.write_text(json.dumps({"items_present": [1, 2, 4], "ace_scorecard": {}}))
        check(answers_from_output(json.loads(card.read_text()))[:4] == [1, 1, 0, 1]
              and individual_of(str(card), {}) == "P1", "FiRE score card, id from its folder")
        rec = {"items": [{"ace_item": i, "final_answer": int(i in (1, 3))} for i in range(1, 15)]}
        check(answers_from_output(rec)[:3] == [1, 0, 1], "FiRE notes-plus-form output")
        base = {"patient_id": "P2", "items": {str(i): {"present": i == 9} for i in range(1, 15)}}
        check(answers_from_output(base)[8] == 1, "baseline output")
        cm = {"answers_by_k": {"1": "1" + "0" * 13, "11": "0" * 14}}
        check(answers_from_output(cm)[0] == 0 and answers_from_output(cm, 1)[0] == 1,
              "compute-matched: k = 11 by default, any k on request")
        p2 = Path(d) / "P2.gpt-5.1.ace.json"
        p2.write_text(json.dumps(base))
        rows = collect([str(card)], r_notes, "FiRE", "notes") + \
            collect([str(p2)], r_notes, "GPT-5.1", "notes")
        out = Path(d) / "answers.csv"
        write_rows(rows[:14], str(out), False)
        write_rows(rows[14:], str(out), True)
        back = list(csv.DictReader(open(out)))
        check(len(back) == 28 and back[0]["reference_answer"] == "1"
              and back[0]["error_mode"] == "" and set(back[0]) == set(COLUMNS),
              "append writes one header and 28 rows")
        try:
            collect([str(card), str(card)], r_notes, "FiRE", "notes")
            check(False, "a duplicate individual is refused")
        except ValueError:
            check(True, "a duplicate individual is refused")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Collect system answers next to the reference.")
    ap.add_argument("outputs", nargs="*", help="Output JSON files, one per individual")
    ap.add_argument("--reference", help="Reference JSON (see the module docstring)")
    ap.add_argument("--system", help="System label, e.g. 'FiRE' or 'GPT-5.1, structure-guided'")
    ap.add_argument("--condition", choices=CONDITIONS)
    ap.add_argument("--k", type=int, default=None, help="Compute-matched: answers per item (odd)")
    ap.add_argument("--run", default="1", help="Run label (default 1, the main run)")
    ap.add_argument("-o", "--output", default="answers.csv")
    ap.add_argument("--append", action="store_true", help="Append to an existing CSV")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.outputs and a.reference and a.system and a.condition):
        ap.error("outputs, --reference, --system and --condition are required")
    try:
        rows = collect(a.outputs, read_reference(a.reference, a.condition),
                       a.system, a.condition, a.k, a.run)
    except ValueError as e:
        sys.exit(f"error: {e}")
    write_rows(rows, a.output, a.append)
    print(f"{a.system} / {a.condition}: {len(rows) // N_ITEMS} individuals -> {a.output}")


if __name__ == "__main__":
    main()
