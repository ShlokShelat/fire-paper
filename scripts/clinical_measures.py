#!/usr/bin/env python3
"""
Clinical measures from system answers and the reference (main paper,
Section VI and Table II; Supplementary Tables SV and SVI).

Accuracy rewards correct absences, and most of the 14 items are absent for
any one individual, so the paper also reports, per system and condition:

  sensitivity   share of present items (reference yes) the system finds
  specificity   share of absent items (reference no) it correctly leaves out
  MAE           mean absolute error of the individual's total score
  >= 4 correct  individuals on the correct side of the clinical cutoff
                ACE >= 4 (system total >= 4 exactly when reference total >= 4)

and, per item, sensitivity / specificity (Tables SV and SVI).

Input: a CSV with the columns system, condition, individual, item,
system_answer, reference_answer (and optionally run), as written by
scripts/collect_answers.py; a judgements CSV that carries the two answer
columns works too. Rows with another run label are ignored unless --run
selects them (default: run 1, the main run; files with no run column are
read whole).

Usage:
  python3 scripts/clinical_measures.py answers.csv
  python3 scripts/clinical_measures.py answers.csv --per-item --condition notes
  python3 scripts/clinical_measures.py answers.csv --json measures.json
  python3 scripts/clinical_measures.py --selftest

Numbers are rounded half up, as in the paper. Standard library only.
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_judgements import num, normalise_condition, JudgementError  # noqa: E402

N_ITEMS = 14
CUTOFF = 4


def read_answers(path: str, run: str = "1") -> list:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        need = {"system", "condition", "individual", "item", "system_answer", "reference_answer"}
        if not need <= cols:
            raise JudgementError(f"{path}: missing column(s): {', '.join(sorted(need - cols))}")
        for lineno, r in enumerate(reader, start=2):
            if "run" in cols and run is not None and str(r.get("run", "")).strip() != str(run):
                continue
            try:
                sa, ra = int(r["system_answer"]), int(r["reference_answer"])
                item = int(r["item"])
            except ValueError:
                raise JudgementError(f"{path}:{lineno}: answers and item must be integers")
            if sa not in (0, 1) or ra not in (0, 1) or not 1 <= item <= N_ITEMS:
                raise JudgementError(f"{path}:{lineno}: answers must be 0/1, item 1-{N_ITEMS}")
            rows.append({"system": r["system"].strip(),
                         "condition": normalise_condition(r["condition"]),
                         "individual": r["individual"].strip(), "item": item,
                         "sys": sa, "ref": ra})
    return rows


def _rate(k: int, n: int):
    return None if n == 0 else 100.0 * k / n


def measures(rows: list) -> "OrderedDict":
    """{(system, condition): {...}} in first-appearance order."""
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["system"], r["condition"]), []).append(r)
    out = OrderedDict()
    for (system, cond), rs in groups.items():
        seen = defaultdict(set)
        for r in rs:
            if r["item"] in seen[r["individual"]]:
                raise JudgementError(f"duplicate decision: {system} / {cond} / "
                                     f"{r['individual']} / item {r['item']}")
            seen[r["individual"]].add(r["item"])
        incomplete = [i for i, items in seen.items() if len(items) != N_ITEMS]
        if incomplete:
            raise JudgementError(f"{system} / {cond}: not all {N_ITEMS} items for {incomplete}")
        tp = sum(1 for r in rs if r["sys"] == 1 and r["ref"] == 1)
        fn = sum(1 for r in rs if r["sys"] == 0 and r["ref"] == 1)
        fp = sum(1 for r in rs if r["sys"] == 1 and r["ref"] == 0)
        tn = sum(1 for r in rs if r["sys"] == 0 and r["ref"] == 0)
        tot_s, tot_r = defaultdict(int), defaultdict(int)
        for r in rs:
            tot_s[r["individual"]] += r["sys"]
            tot_r[r["individual"]] += r["ref"]
        inds = sorted(seen)
        abs_err = [abs(tot_s[i] - tot_r[i]) for i in inds]
        cutoff_ok = sum(1 for i in inds if (tot_s[i] >= CUTOFF) == (tot_r[i] >= CUTOFF))
        per_item = OrderedDict()
        for item in range(1, N_ITEMS + 1):
            it = [r for r in rs if r["item"] == item]
            p = sum(1 for r in it if r["ref"] == 1)
            n = len(it) - p
            per_item[item] = {
                "present": p, "absent": n,
                "sensitivity": _rate(sum(1 for r in it if r["ref"] == 1 and r["sys"] == 1), p),
                "specificity": _rate(sum(1 for r in it if r["ref"] == 0 and r["sys"] == 0), n),
            }
        out[(system, cond)] = {
            "system": system, "condition": cond, "individuals": len(inds),
            "decisions": len(rs), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": 100.0 * (tp + tn) / len(rs),
            "sensitivity": _rate(tp, tp + fn), "specificity": _rate(tn, tn + fp),
            "mae_total": sum(abs_err) / len(inds),
            "cutoff_correct": cutoff_ok, "cutoff": CUTOFF,
            "per_item": per_item,
        }
    return out


def _pct(x, digits=1):
    return "--" if x is None else num(x, digits)


def _item_cell(x):
    """Per-item cells as in Tables SV and SVI: 100 without a decimal."""
    if x is None:
        return "--"
    s = num(x, 1)
    return "100" if s == "100.0" else s


def format_table(m: "OrderedDict") -> str:
    head = (f"{'system':<36} {'condition':<10} {'acc%':>6} {'sens%':>6} {'spec%':>6} "
            f"{'MAE':>5} {'>=4 ok':>7}")
    lines = [head, "-" * len(head)]
    for v in m.values():
        lines.append(f"{v['system'][:36]:<36} {v['condition']:<10} {_pct(v['accuracy']):>6} "
                     f"{_pct(v['sensitivity']):>6} {_pct(v['specificity']):>6} "
                     f"{num(v['mae_total'], 1):>5} {v['cutoff_correct']:>3}/{v['individuals']:<3}")
    return "\n".join(lines)


def format_per_item(m: "OrderedDict", condition: str) -> str:
    systems = [v for v in m.values() if v["condition"] == condition]
    head = "Item    " + "".join(f"{v['system'][:16]:>18}" for v in systems)
    lines = [f"Sensitivity / specificity (%) per item, {condition}", head, "-" * len(head)]
    for item in range(1, N_ITEMS + 1):
        cells = []
        for v in systems:
            pi = v["per_item"][item]
            cells.append(f"{_item_cell(pi['sensitivity'])}/{_item_cell(pi['specificity'])}")
        lines.append(f"ACE-{item:<4}" + "".join(f"{c:>18}" for c in cells))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────
def _synthetic(system, cond, tp, fn, fp, tn, n_ind=40):
    """Rows with the given confusion counts spread over n_ind individuals."""
    cells = [(1, 1)] * tp + [(0, 1)] * fn + [(1, 0)] * fp + [(0, 0)] * tn
    assert len(cells) == n_ind * N_ITEMS
    rows = []
    for k, (s, r) in enumerate(cells):
        rows.append({"system": system, "condition": cond, "individual": f"S{k % n_ind:02d}",
                     "item": k // n_ind + 1, "sys": s, "ref": r})
    return rows


def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # Table II, notes only: with 213 present and 347 absent reference
    # decisions, every missed item is an E2 error and every other error is a
    # false yes, so sensitivity and specificity follow from the error counts.
    table2_notes = {  # system: (E2, total errors, sensitivity, specificity)
        "FiRE": (27, 31, "87.3", "98.8"), "GPT-5.1": (16, 170, "92.5", "55.6"),
        "Grok-4.20": (49, 192, "77.0", "58.8"), "Gemini 3.1 Pro": (38, 180, "82.2", "59.1"),
        "GPT-5.1, SG": (108, 210, "49.3", "70.6"), "Grok-4.20, SG": (124, 221, "41.8", "72.0"),
        "Gemini 3.1 Pro, SG": (104, 202, "51.2", "71.8"), "GPT-5.1, KM": (55, 143, "74.2", "74.6"),
        "Grok-4.20, KM": (66, 155, "69.0", "74.4"), "Gemini 3.1 Pro, KM": (52, 139, "75.6", "74.9"),
    }
    rows = []
    for s, (e2, tot, _, _) in table2_notes.items():
        fn, fp = e2, tot - e2
        rows += _synthetic(s, "notes", 213 - fn, fn, fp, 347 - fp)
    m = measures(rows)
    bad = [s for s, (_, _, se, sp) in table2_notes.items()
           if (_pct(m[(s, "notes")]["sensitivity"]), _pct(m[(s, "notes")]["specificity"])) != (se, sp)]
    check(not bad, f"Table II, notes only: sensitivity and specificity from the error counts {bad}")
    check(_pct(m[("FiRE", "notes")]["accuracy"]) == "94.5", "FiRE accuracy 94.5% (529 of 560)")

    # MAE and the cutoff on a small hand-checked example.
    rows = []
    totals = {"A": (5, 3), "B": (2, 4), "C": (0, 0)}   # (system total, reference total)
    for ind, (ts, tr) in totals.items():
        for item in range(1, N_ITEMS + 1):
            rows.append({"system": "X", "condition": "notes", "individual": ind, "item": item,
                         "sys": int(item <= ts), "ref": int(item <= tr)})
    v = measures(rows)[("X", "notes")]
    # A: 5 vs 3 (wrong side), B: 2 vs 4 (wrong side), C: 0 vs 0 (right side)
    check(abs(v["mae_total"] - (2 + 2 + 0) / 3) < 1e-12 and v["cutoff_correct"] == 1,
          "MAE of the totals and agreement on ACE >= 4")
    # item 4: present only for B, whom the system misses; item 5: absent for
    # all three, wrongly yes for A
    check(v["per_item"][4]["sensitivity"] == 0.0 and v["per_item"][4]["present"] == 1
          and abs(v["per_item"][5]["specificity"] - 200 / 3) < 1e-9,
          "per-item sensitivity and specificity")
    check(_item_cell(100.0) == "100" and _item_cell(40.0) == "40.0" and _item_cell(None) == "--",
          "per-item cells print as in Tables SV and SVI")
    try:
        measures(rows + rows[:1])
        check(False, "a duplicate decision is refused")
    except JudgementError:
        check(True, "a duplicate decision is refused")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Sensitivity, specificity, MAE and the ACE >= 4 cutoff.")
    ap.add_argument("csv", nargs="?", help="Answers CSV (scripts/collect_answers.py)")
    ap.add_argument("--run", default="1", help="Run label to read (default 1, the main run)")
    ap.add_argument("--per-item", action="store_true", help="Also print Tables SV/SVI")
    ap.add_argument("--condition", default="notes", help="Condition for --per-item")
    ap.add_argument("--json", help="Write every measure to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.csv:
        ap.error("a CSV is required (unless using --selftest)")
    try:
        m = measures(read_answers(a.csv, a.run))
    except JudgementError as e:
        sys.exit(f"error: {e}")
    print(format_table(m))
    if a.per_item:
        print()
        print(format_per_item(m, normalise_condition(a.condition)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump([dict(v, per_item={str(k): x for k, x in v["per_item"].items()})
                       for v in m.values()], f, indent=2)


if __name__ == "__main__":
    main()
