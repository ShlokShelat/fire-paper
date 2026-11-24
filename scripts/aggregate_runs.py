#!/usr/bin/env python3
"""
Run-to-run stability: accuracy and grounding over repeated runs, as mean
plus or minus the SAMPLE standard deviation (Supplementary Table SIV,
"Three runs (main run included)"; Section S8, "Run-to-run stability").

The paper made three runs of every system with the same settings and the
same judging; the main run is one of them. Each run is judged item by item
like the main run, giving one judgements CSV per run (or one CSV with a
`run` column). Accuracy and grounding of each run follow the definitions
of scripts/evaluate_judgements.py; this script only combines the runs.

Usage:
  python3 scripts/aggregate_runs.py run1.csv run2.csv run3.csv
  python3 scripts/aggregate_runs.py all_runs.csv --run-column run
  python3 scripts/aggregate_runs.py run1.csv run2.csv run3.csv --json stability.json
  python3 scripts/aggregate_runs.py --selftest

Numbers are rounded half up, as in the paper. Standard library only.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_judgements as ev  # noqa: E402


def runs_from_files(paths: list) -> "OrderedDict":
    """{run label: judgement rows}, one run per file (label = file name)."""
    if len(set(paths)) != len(paths):
        raise ev.JudgementError("the same file is given twice")
    names = [os.path.basename(p) for p in paths]
    labels = names if len(set(names)) == len(names) else list(paths)
    out = OrderedDict()
    for p, label in zip(paths, labels):
        rows = ev.read_judgements(p)
        errors = ev.validate(rows)
        if errors:
            raise ev.JudgementError(f"{p}: " + "; ".join(errors[:5]))
        out[label] = rows
    return out


def runs_from_column(path: str, column: str) -> "OrderedDict":
    """{run label: judgement rows}, split on a run column."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ev.JudgementError(f"{path}: no '{column}' column")
        by_run = OrderedDict()
        for r in reader:
            by_run.setdefault(str(r[column]).strip(), []).append(r)
        fields = reader.fieldnames
    out = OrderedDict()
    with tempfile.TemporaryDirectory() as d:
        for run, raw in by_run.items():
            tmp = os.path.join(d, f"run_{len(out)}.csv")
            with open(tmp, "w", newline="", encoding="utf-8") as g:
                w = csv.DictWriter(g, fieldnames=fields)
                w.writeheader()
                w.writerows(raw)
            rows = ev.read_judgements(tmp)
            errors = ev.validate(rows)
            if errors:
                raise ev.JudgementError(f"{path}, run {run}: " + "; ".join(errors[:5]))
            out[run] = rows
    return out


def aggregate(runs: "OrderedDict") -> "OrderedDict":
    """{(system, condition): {...}} with each run's accuracy and grounding
    and their mean and sample standard deviation. Every (system,
    condition) must appear in every run."""
    per_run = OrderedDict((label, ev.summarise(rows)) for label, rows in runs.items())
    keys = []
    for s in per_run.values():
        for k in s:
            if k not in keys:
                keys.append(k)
    out = OrderedDict()
    for k in keys:
        missing = [label for label, s in per_run.items() if k not in s]
        if missing:
            raise ev.JudgementError(f"{k[0]} / {k[1]} is missing from run(s) {missing}")
        acc = [per_run[label][k]["accuracy"] for label in per_run]
        grd = [per_run[label][k]["grounded"] for label in per_run]
        n = len(acc)
        out[k] = {
            "system": k[0], "condition": k[1], "runs": n,
            "accuracy_by_run": acc, "grounded_by_run": grd,
            "accuracy_mean": statistics.fmean(acc),
            "accuracy_sd": statistics.stdev(acc) if n > 1 else 0.0,
            "grounded_mean": statistics.fmean(grd),
            "grounded_sd": statistics.stdev(grd) if n > 1 else 0.0,
        }
    return out


def format_table(agg: "OrderedDict") -> str:
    head = f"{'system':<36} {'condition':<10} {'runs':>4} {'accuracy':>14} {'grounding':>14}"
    lines = [head, "-" * len(head)]
    for v in agg.values():
        a = f"{ev.num(v['accuracy_mean'], 1)} ± {ev.num(v['accuracy_sd'], 1)}"
        g = f"{ev.num(v['grounded_mean'], 1)} ± {ev.num(v['grounded_sd'], 1)}"
        lines.append(f"{v['system'][:36]:<36} {v['condition']:<10} {v['runs']:>4} {a:>14} {g:>14}")
    return "\n".join(lines)


def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    def rows(n_errors, mode="E1"):
        out = []
        for k in range(40 * 14):
            out.append({"system": "X", "condition": "notes", "individual": f"S{k // 14:02d}",
                        "item": k % 14 + 1, "error_mode": mode if k < n_errors else None,
                        "system_answer": None, "reference_answer": None})
        return out

    runs = OrderedDict([("r1", rows(31)), ("r2", rows(33)), ("r3", rows(29))])
    v = aggregate(runs)[("X", "notes")]
    accs = [100 * (560 - e) / 560 for e in (31, 33, 29)]
    check(abs(v["accuracy_mean"] - sum(accs) / 3) < 1e-9, "mean accuracy over the runs")
    check(abs(v["accuracy_sd"] - statistics.stdev(accs)) < 1e-9
          and v["accuracy_sd"] > statistics.pstdev(accs), "sample (not population) s.d.")
    check(ev.num(v["accuracy_mean"], 1) == "94.5" and ev.num(v["accuracy_sd"], 1) == "0.4",
          "94.5 ± 0.4 for 31, 33 and 29 errors")
    check(v["grounded_mean"] == v["accuracy_mean"], "E1 errors count against grounding too")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "all.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "condition", "individual", "item", "error_mode", "run"])
            for label, rs in runs.items():
                for r in rs:
                    w.writerow([r["system"], r["condition"], r["individual"], r["item"],
                                r["error_mode"] or "", label])
        v2 = aggregate(runs_from_column(p, "run"))[("X", "notes")]
        check(v2["accuracy_by_run"] == v["accuracy_by_run"], "one CSV with a run column")
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, label in enumerate(runs):
            os.makedirs(os.path.join(d, f"run{k + 1}"))
            p = os.path.join(d, f"run{k + 1}", "judgements.csv")
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["system", "condition", "individual", "item", "error_mode"])
                for r in runs[label]:
                    w.writerow([r["system"], r["condition"], r["individual"], r["item"],
                                r["error_mode"] or ""])
            paths.append(p)
        v3 = aggregate(runs_from_files(paths))[("X", "notes")]
        check(v3["runs"] == 3 and v3["accuracy_by_run"] == v["accuracy_by_run"],
              "files with the same name in different folders are separate runs")
    try:
        aggregate(OrderedDict([("r1", rows(31)), ("r2", [])]))
        check(False, "a system missing from a run is refused")
    except ev.JudgementError:
        check(True, "a system missing from a run is refused")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Accuracy and grounding over repeated runs.")
    ap.add_argument("csv", nargs="*", help="Judgements CSVs, one per run (or one with --run-column)")
    ap.add_argument("--run-column", help="Split one CSV on this column")
    ap.add_argument("--json", help="Write the result to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.csv or (a.run_column and len(a.csv) != 1):
        ap.error("give one CSV per run, or one CSV with --run-column")
    try:
        runs = runs_from_column(a.csv[0], a.run_column) if a.run_column else runs_from_files(a.csv)
        agg = aggregate(runs)
    except ev.JudgementError as e:
        sys.exit(f"error: {e}")
    if len(runs) < 2:
        print("warning: only one run; the standard deviation needs two or more", file=sys.stderr)
    print(format_table(agg))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(list(agg.values()), f, indent=2)


if __name__ == "__main__":
    main()
