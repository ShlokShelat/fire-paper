#!/usr/bin/env python3
"""
Accuracy and grounding from per-item expert judgements, as defined in the paper.

The paper's figures were not computed by a model or a script: the
psychotherapists judged every output item by item, double-blind
(Supplementary Material, Section S5). This script turns those judgements
into the reported metrics, so that anyone holding judgements in the format
below can recompute them. It makes no model call and needs only the
standard library.

INPUT: a CSV file with one row per item decision and these columns:

  system        label of the system, e.g. "FiRE", "GPT-5.1",
                "GPT-5.1, structure-guided" (free text)
  condition     "notes" (notes only) or "notes+form" (notes plus form)
  individual    coded identifier of the individual
  item          ACE item, 1 to 14
  error_mode    empty when the decision was judged correct and grounded;
                otherwise one of E1 to E6 (main paper, Section VI):
                  E1 over-inference              E4 surface keyword matching
                  E2 construct under-recognition E5 age-window violation
                  E3 referent and relational     E6 reconciliation instability
                     misattribution
  system_answer, reference_answer
                optional, 0 or 1. When present, a row whose answers differ
                must carry an error mode (checked).

DEFINITIONS (main paper, Section VI; Supplementary Material, S5 and S8):

  accuracy   share of decisions with no error mode. In the paper every
             judged error carries exactly one mode, so the total of Table
             III equals the number of decisions that do not match the
             reference, and accuracy = 1 - total / decisions.
  grounded   share of decisions with no E1, E3, E4 or E6 error. E2 (an
             omission) and E5 (a scope violation) count against accuracy
             but not against grounding.
  interval   Wilson 95% interval on accuracy, treating decisions as
             independent (Table SV), and, for the clustered extreme
             of Section S8, the same proportion with an effective sample
             of one per individual.
  McNemar    against a reference system (default FiRE) in the same
             condition. From counts alone the continuity-corrected
             statistic is bounded below by (e_B - e_F - 1)^2 / (e_B + e_F),
             reported rounded down as in Table SV. When both systems
             have judgements for the same (individual, item) pairs, the
             exact statistic (|b - c| - 1)^2 / (b + c) and its p-value are
             also reported.

Usage:
  python3 scripts/evaluate_judgements.py judgements.csv               # table to stdout
  python3 scripts/evaluate_judgements.py judgements.csv --json out.json
  python3 scripts/evaluate_judgements.py judgements.csv --reference "FiRE"
  python3 scripts/evaluate_judgements.py --selftest

The script checks that each (system, condition) has 14 decisions per
individual, no duplicate rows and valid modes, and exits with status 1 on
any error. scripts/make_tables.py renders the paper's tables from the same
CSV.
"""

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from decimal import ROUND_HALF_UP, Decimal

MODES = ("E1", "E2", "E3", "E4", "E5", "E6")
UNGROUNDED_MODES = ("E1", "E3", "E4", "E6")
CONDITIONS = ("notes", "notes+form")
N_ITEMS = 14
Z_95 = 1.959963984540054

REQUIRED_COLUMNS = ("system", "condition", "individual", "item", "error_mode")

_CONDITION_ALIASES = {
    "notes": "notes", "notes only": "notes", "notes-only": "notes",
    "notes+form": "notes+form", "notes + form": "notes+form", "form": "notes+form",
    "notes plus form": "notes+form", "notes-plus-form": "notes+form", "+form": "notes+form",
}


class JudgementError(ValueError):
    """Raised when the judgements file is malformed."""


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def num(x: float, digits: int) -> str:
    """Format with half-up rounding, as the paper does (56.25 -> 56.3);
    Python's round() and format() round half to even."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-digits),
                                                rounding=ROUND_HALF_UP))


def wilson_interval(successes: float, n: float, z: float = Z_95) -> tuple:
    """Wilson score interval for a proportion, as (low, high) in [0, 1].
    `successes` may be fractional (used for the clustered extreme, where the
    proportion is kept and n is the number of individuals)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar_lower_bound(errors_b: int, errors_f: int) -> float:
    """Lower bound on the continuity-corrected McNemar statistic from the two
    systems' error counts alone (Supplementary Material, Section S8)."""
    if errors_b + errors_f == 0:
        return 0.0
    return max(errors_b - errors_f - 1, 0) ** 2 / (errors_b + errors_f)


def chi2_sf_1df(x: float) -> float:
    """Upper tail of the chi-square distribution with one degree of freedom."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def mcnemar_exact(b: int, c: int) -> tuple:
    """Continuity-corrected McNemar statistic and p-value from the discordant
    counts b (only the reference system wrong) and c (only the other wrong)."""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return stat, chi2_sf_1df(stat)


# ─────────────────────────────────────────────────────────────
# READING
# ─────────────────────────────────────────────────────────────

def normalise_condition(value: str) -> str:
    key = " ".join(str(value or "").strip().lower().split())
    if key not in _CONDITION_ALIASES:
        raise JudgementError(f"unknown condition {value!r}; use 'notes' or 'notes+form'")
    return _CONDITION_ALIASES[key]


def _answer(value):
    v = str(value if value is not None else "").strip().lower()
    if v == "":
        return None
    if v in ("1", "yes", "y", "true"):
        return 1
    if v in ("0", "no", "n", "false"):
        return 0
    raise JudgementError(f"answer must be 0 or 1, got {value!r}")


def read_judgements(path: str) -> list:
    """Read and validate a judgements CSV; returns a list of row dicts with
    keys system, condition, individual, item (int), error_mode (str or None),
    system_answer, reference_answer (int or None)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise JudgementError(f"{path}: missing column(s): {', '.join(missing)}")
        rows = []
        for lineno, raw in enumerate(reader, start=2):
            r = {k.strip(): v for k, v in raw.items() if k is not None}
            try:
                item = int(str(r["item"]).strip())
            except ValueError:
                raise JudgementError(f"{path}:{lineno}: item must be an integer")
            if not 1 <= item <= N_ITEMS:
                raise JudgementError(f"{path}:{lineno}: item {item} is outside 1-{N_ITEMS}")
            mode = str(r.get("error_mode") or "").strip().upper() or None
            if mode is not None and mode not in MODES:
                raise JudgementError(f"{path}:{lineno}: error_mode {mode!r} is not one of "
                                     f"{', '.join(MODES)} (or empty)")
            try:
                row = {
                    "system": str(r["system"]).strip(),
                    "condition": normalise_condition(r["condition"]),
                    "individual": str(r["individual"]).strip(),
                    "item": item,
                    "error_mode": mode,
                    "system_answer": _answer(r.get("system_answer")),
                    "reference_answer": _answer(r.get("reference_answer")),
                }
            except JudgementError as e:
                raise JudgementError(f"{path}:{lineno}: {e}")
            if not row["system"] or not row["individual"]:
                raise JudgementError(f"{path}:{lineno}: system and individual must not be empty")
            rows.append(row)
    return rows


def validate(rows: list) -> list:
    """Structural checks; returns a list of error messages (empty if clean)."""
    errors = []
    seen = set()
    per_individual = defaultdict(lambda: defaultdict(set))
    for r in rows:
        key = (r["system"], r["condition"], r["individual"], r["item"])
        if key in seen:
            errors.append(f"duplicate decision: {key}")
        seen.add(key)
        per_individual[(r["system"], r["condition"])][r["individual"]].add(r["item"])
        sa, ra = r["system_answer"], r["reference_answer"]
        if sa is not None and ra is not None and sa != ra and r["error_mode"] is None:
            errors.append(f"answer differs from the reference but no error mode: {key}")
    for (system, cond), by_ind in per_individual.items():
        for ind, items in by_ind.items():
            if len(items) != N_ITEMS:
                errors.append(f"{system} / {cond} / {ind}: {len(items)} of {N_ITEMS} items judged")
    return errors


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def summarise(rows: list, reference: str = "FiRE") -> "OrderedDict":
    """Metrics per (system, condition), in first-appearance order:
    {(system, condition): {...}}. See the module docstring."""
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["system"], r["condition"]), []).append(r)

    out = OrderedDict()
    for (system, cond), rs in groups.items():
        n = len(rs)
        counts = {m: sum(1 for r in rs if r["error_mode"] == m) for m in MODES}
        total = sum(counts.values())
        ungrounded = sum(counts[m] for m in UNGROUNDED_MODES)
        individuals = sorted({r["individual"] for r in rs})
        lo, hi = wilson_interval(n - total, n)
        clo, chi = wilson_interval((n - total) / n * len(individuals), len(individuals))
        out[(system, cond)] = {
            "system": system, "condition": cond,
            "decisions": n, "individuals": len(individuals),
            "errors": dict(counts, total=total),
            "ungrounded": ungrounded,
            "accuracy": 100.0 * (n - total) / n,
            "grounded": 100.0 * (n - ungrounded) / n,
            "accuracy_ci95": [100.0 * lo, 100.0 * hi],
            "accuracy_ci95_one_per_individual": [100.0 * clo, 100.0 * chi],
            "answer_matches_but_judged_error": sum(
                1 for r in rs if r["error_mode"] is not None
                and r["system_answer"] is not None and r["reference_answer"] is not None
                and r["system_answer"] == r["reference_answer"]),
        }

    for (system, cond), m in out.items():
        ref = out.get((reference, cond))
        if ref is None or system == reference:
            m["mcnemar_vs_reference"] = None
            continue
        e_b, e_f = m["errors"]["total"], ref["errors"]["total"]
        bound = mcnemar_lower_bound(e_b, e_f)
        res = {"reference": reference, "chi2_lower_bound": bound,
               "chi2_lower_bound_floor": math.floor(bound),
               "p_upper_bound": chi2_sf_1df(bound)}
        ref_err = {(r["individual"], r["item"]): r["error_mode"] is not None
                   for r in groups[(reference, cond)]}
        sys_err = {(r["individual"], r["item"]): r["error_mode"] is not None
                   for r in groups[(system, cond)]}
        if set(ref_err) == set(sys_err):
            b = sum(1 for k in ref_err if ref_err[k] and not sys_err[k])
            c = sum(1 for k in ref_err if sys_err[k] and not ref_err[k])
            stat, p = mcnemar_exact(b, c)
            res.update({"paired": True, "only_reference_wrong": b, "only_system_wrong": c,
                        "chi2": stat, "p": p})
        else:
            res["paired"] = False
        m["mcnemar_vs_reference"] = res
    return out


def format_summary(summary: "OrderedDict") -> str:
    """A plain-text table of the main metrics."""
    head = (f"{'system':<36} {'condition':<10} {'n':>4} "
            + " ".join(f"{m:>3}" for m in MODES)
            + f" {'tot':>4} {'acc%':>6} {'grd%':>6} {'95% CI':>13} {'chi2>=':>7}")
    lines = [head, "-" * len(head)]
    for m in summary.values():
        e = m["errors"]
        mc = m["mcnemar_vs_reference"]
        ci = m["accuracy_ci95"]
        lines.append(
            f"{m['system'][:36]:<36} {m['condition']:<10} {m['decisions']:>4} "
            + " ".join(f"{e[k]:>3}" for k in MODES)
            + f" {e['total']:>4} {num(m['accuracy'], 2):>6} {num(m['grounded'], 2):>6} "
            + f"{num(ci[0], 1):>6}-{num(ci[1], 1):<6} "
            + (f"{mc['chi2_lower_bound_floor']:>7d}" if mc else f"{'--':>7}"))
    return "\n".join(lines)


def _jsonable(summary: "OrderedDict") -> list:
    return list(summary.values())


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    """Offline checks of the definitions on small synthetic judgements."""
    import os
    import tempfile
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    lo, hi = wilson_interval(529, 560)
    check((round(100 * lo, 1), round(100 * hi, 1)) == (92.2, 96.1),
          "Wilson interval reproduces FiRE's notes-only interval (92.2, 96.1)")
    check(math.floor(mcnemar_lower_bound(170, 31)) == 94,
          "McNemar lower bound reproduces GPT-5.1 notes-only (>= 94)")
    stat, p = mcnemar_exact(10, 0)
    check(abs(stat - 8.1) < 1e-9 and abs(p - chi2_sf_1df(8.1)) < 1e-12 and p < 0.005,
          "exact McNemar from discordant counts")

    header = "system,condition,individual,item,error_mode,system_answer,reference_answer\n"
    rows = []
    for ind in ("I01", "I02"):
        for item in range(1, 15):
            rows.append(f"A,notes,{ind},{item},,1,1")
            mode = {1: "E1", 2: "E2", 3: "E5"}.get(item, "") if ind == "I01" else ""
            rows.append(f"B,notes,{ind},{item},{mode},{0 if mode else 1},1")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "j.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(rows) + "\n")
        data = read_judgements(path)
        check(not validate(data), "a complete synthetic file validates")
        s = summarise(data, reference="A")
        b = s[("B", "notes")]
        check(b["errors"]["total"] == 3 and abs(b["accuracy"] - 100 * 25 / 28) < 1e-9,
              "accuracy counts every judged error (E1, E2, E5)")
        check(b["ungrounded"] == 1 and abs(b["grounded"] - 100 * 27 / 28) < 1e-9,
              "grounding counts only E1, E3, E4 and E6")
        mc = b["mcnemar_vs_reference"]
        check(mc["paired"] and mc["only_system_wrong"] == 3 and mc["only_reference_wrong"] == 0,
              "paired McNemar uses the same (individual, item) decisions")

        bad = os.path.join(tmp, "bad.csv")
        with open(bad, "w", encoding="utf-8") as f:
            f.write(header + "A,notes,I01,1,,0,1\nA,notes,I01,1,,1,1\n")
        errs = validate(read_judgements(bad))
        check(any("no error mode" in e for e in errs) and any("duplicate" in e for e in errs)
              and any("of 14 items" in e for e in errs),
              "inconsistent answers, duplicates and missing items are reported")
        bad2 = os.path.join(tmp, "bad2.csv")
        with open(bad2, "w", encoding="utf-8") as f:
            f.write(header + "A,notes,I01,1,E7,,\n")
        try:
            read_judgements(bad2)
            check(False, "an unknown error mode is refused")
        except JudgementError:
            check(True, "an unknown error mode is refused")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Accuracy, grounding, error counts, Wilson intervals and McNemar "
                    "statistics from per-item expert judgements (no model call).")
    ap.add_argument("judgements", nargs="?", help="Judgements CSV (see the module docstring)")
    ap.add_argument("--reference", default="FiRE",
                    help="System the McNemar statistics compare against (default: FiRE)")
    ap.add_argument("--json", metavar="PATH", help="Also write the metrics as JSON")
    ap.add_argument("--selftest", action="store_true", help="Run offline checks and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.judgements:
        ap.error("judgements is required (unless using --selftest)")

    try:
        rows = read_judgements(args.judgements)
    except (OSError, JudgementError) as e:
        sys.exit(f"error: {e}")
    problems = validate(rows)
    if problems:
        for p in problems[:50]:
            print(f"error: {p}", file=sys.stderr)
        if len(problems) > 50:
            print(f"error: ... and {len(problems) - 50} more", file=sys.stderr)
        sys.exit(1)

    summary = summarise(rows, reference=args.reference)
    print(format_summary(summary))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2)
            f.write("\n")
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
