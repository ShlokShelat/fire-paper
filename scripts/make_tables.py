#!/usr/bin/env python3
"""
Regenerate the paper's result tables from a per-item judgements CSV.

Tables (LaTeX, as in the paper, or Markdown):
  main      Table II   accuracy and grounding, notes and notes + form
  errors    Table III  error counts by mode, notes and notes + form
  stats     Table SV accuracy with Wilson 95% intervals and the McNemar
                       lower bound against the reference system
  ablation  the Supplementary ablation table (Section S9): error counts,
                       accuracy and grounding per run, both conditions

The judgements CSV format and the metric definitions are those of
scripts/evaluate_judgements.py. For the ablation table, pass a CSV whose
`system` column holds the run names ("FiRE", "Construct off", ...).

The per-item judgements behind the paper cannot be released (consultation
notes and reference scores are protected; see the README). What is public
are the aggregate counts, kept in data/paper_results/published_results.json.
--published renders the tables from those counts, and --self-test checks
that every derived number the paper reports (accuracy, grounding, Wilson
intervals, McNemar bounds, the clustered extreme of Section S8, and the
ablation table) follows from the error counts under the definitions in
evaluate_judgements.py.

Usage:
  python3 scripts/make_tables.py judgements.csv --table main
  python3 scripts/make_tables.py judgements.csv --table all --format markdown
  python3 scripts/make_tables.py ablation_judgements.csv --table ablation
  python3 scripts/make_tables.py --published --table all
  python3 scripts/make_tables.py --example-csv synthetic_judgements.csv
  python3 scripts/make_tables.py --selftest

With a judgements CSV, --table all prints Tables II, III and SV; the
ablation table is printed only with --table ablation, from a CSV of
ablation runs. --example-csv (and --example-ablation-csv) write a SYNTHETIC
judgements file whose counts equal the published counts (individuals are
named S01..S40 and errors are placed in a fixed order). It shows the input
format; it is not the study's data.

Standard library only; no model call.
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_judgements as ev  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(REPO, "data", "paper_results", "published_results.json")
TABLES = ("main", "errors", "stats", "ablation")
COND_LABEL = {"notes": "Notes", "notes+form": "Notes + form"}


# ─────────────────────────────────────────────────────────────
# PUBLISHED COUNTS -> SYNTHETIC JUDGEMENTS
# ─────────────────────────────────────────────────────────────

def load_published(path: str = PUBLISHED) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def synthetic_rows(counts: "OrderedDict", n_individuals: int = 40) -> list:
    """Judgement rows whose per-mode counts equal `counts`
    ({(system, condition): {"E1": n, ...}}). Errors fill the decisions in a
    fixed order; the placement is arbitrary and carries no information."""
    rows = []
    for (system, cond), errs in counts.items():
        modes = [m for m in ev.MODES for _ in range(int(errs.get(m, 0)))]
        slots = [(f"S{i + 1:02d}", item) for i in range(n_individuals)
                 for item in range(1, ev.N_ITEMS + 1)]
        if len(modes) > len(slots):
            raise ValueError(f"{system} / {cond}: more errors than decisions")
        for k, (ind, item) in enumerate(slots):
            rows.append({"system": system, "condition": cond, "individual": ind,
                         "item": item, "error_mode": modes[k] if k < len(modes) else None,
                         "system_answer": None, "reference_answer": None})
    return rows


def published_counts(pub: dict, section: str = "systems") -> "OrderedDict":
    key = "system" if section == "systems" else "run"
    out = OrderedDict()
    for entry in pub[section]:
        for cond in ev.CONDITIONS:
            out[(entry[key], cond)] = entry[cond]["errors"]
    return out


def write_csv(rows: list, path: str) -> None:
    cols = ("system", "condition", "individual", "item", "error_mode")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["system"], r["condition"], r["individual"], r["item"],
                        r["error_mode"] or ""])


# ─────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────

num = ev.num   # half-up rounding, as in the paper


def short_label(system: str) -> str:
    """Table III and SV abbreviate the prompted baselines."""
    return (system.replace(", structure-guided", ", SG")
                  .replace(", knowledge-matched", ", KM"))


def _systems(summary: "OrderedDict") -> list:
    seen = []
    for (system, _cond) in summary:
        if system not in seen:
            seen.append(system)
    return seen


def _get(summary, system, cond):
    m = summary.get((system, cond))
    if m is None:
        raise KeyError(f"no judgements for {system!r} in condition {cond!r}")
    return m


def _bold(text: str, on: bool, fmt: str) -> str:
    if not on:
        return text
    return f"\\textbf{{{text}}}" if fmt == "latex" else f"**{text}**"


def _table(rows: list, header: list, fmt: str) -> str:
    if fmt == "markdown":
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    out = [" & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    return "\n".join(out)


def render_main(summary, systems, fmt="latex", highlight="FiRE") -> str:
    """Table II: accuracy and grounding (%)."""
    rows = []
    for s in systems:
        n, f = _get(summary, s, "notes"), _get(summary, s, "notes+form")
        cells = [num(n["accuracy"], 2), num(f["accuracy"], 2),
                 num(n["grounded"], 2), num(f["grounded"], 2)]
        on = s == highlight
        rows.append([_bold(s, on, fmt)] + [_bold(c, on, fmt) for c in cells])
    header = ["Model", "Accuracy, notes", "Accuracy, + form",
              "Grounded, notes", "Grounded, + form"]
    return _table(rows, header, fmt)


def render_errors(summary, systems, fmt="latex", highlight="FiRE") -> str:
    """Table III: error counts by mode, one row per system and condition."""
    rows = []
    for s in systems:
        for cond in ev.CONDITIONS:
            e = _get(summary, s, cond)["errors"]
            on = s == highlight
            rows.append([_bold(short_label(s), on, fmt), COND_LABEL[cond]]
                        + [str(e[m]) for m in ev.MODES]
                        + [_bold(str(e["total"]), on, fmt)])
    header = ["Model", "Condition"] + list(ev.MODES) + ["Tot."]
    return _table(rows, header, fmt)


def render_stats(summary, systems, fmt="latex") -> str:
    """Table SV: accuracy (Wilson 95% CI) and McNemar lower bound."""
    rows = []
    for s in systems:
        cells = [short_label(s)]
        for cond in ev.CONDITIONS:
            m = _get(summary, s, cond)
            lo, hi = m["accuracy_ci95"]
            cells.append(f"{num(m['accuracy'], 1)} ({num(lo, 1)}, {num(hi, 1)})")
            mc = m["mcnemar_vs_reference"]
            cells.append(str(mc["chi2_lower_bound_floor"]) if mc else "--")
        rows.append(cells)
    header = ["System", "Notes: accuracy (CI)", "chi2 >=",
              "Notes + form: accuracy (CI)", "chi2 >="]
    return _table(rows, header, fmt)


def render_ablation(summary, runs, fmt="latex") -> str:
    """The ablation table: E1-E6, total, accuracy and grounding, both conditions."""
    rows = []
    for r in runs:
        cells = [r]
        for cond in ev.CONDITIONS:
            m = _get(summary, r, cond)
            e = m["errors"]
            cells += [str(e[k]) for k in ev.MODES] + [str(e["total"]),
                                                      num(m["accuracy"], 2),
                                                      num(m["grounded"], 2)]
        rows.append(cells)
    per = list(ev.MODES) + ["Tot.", "Acc. (%)", "Grd. (%)"]
    header = (["Run"] + [f"{c} (notes)" for c in per]
              + [f"{c} (+ form)" for c in per]) if fmt == "markdown" else (["Run"] + per + per)
    return _table(rows, header, fmt)


def render(summary, table, fmt, systems=None) -> str:
    systems = systems or _systems(summary)
    if table == "main":
        return render_main(summary, systems, fmt)
    if table == "errors":
        return render_errors(summary, systems, fmt)
    if table == "stats":
        return render_stats(summary, systems, fmt)
    if table == "ablation":
        return render_ablation(summary, systems, fmt)
    raise ValueError(table)


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest(path: str = PUBLISHED) -> bool:
    """Every derived number the paper reports follows from its error counts."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    pub = load_published(path)
    n_ind = pub["n_individuals"]

    def summary_of(section):
        rows = synthetic_rows(published_counts(pub, section), n_ind)
        check(not ev.validate(rows), f"{section}: synthetic judgements are complete "
                                     f"({n_ind} individuals x {ev.N_ITEMS} items)")
        return ev.summarise(rows, reference="FiRE")

    s = summary_of("systems")
    bad_acc, bad_grd, bad_ci, bad_chi, bad_tot = [], [], [], [], []
    for entry in pub["systems"]:
        for cond in ev.CONDITIONS:
            m, p = s[(entry["system"], cond)], entry[cond]
            label = f"{entry['system']} / {cond}"
            if sum(p["errors"][k] for k in ev.MODES) != p["errors"]["total"]:
                bad_tot.append(label)
            if num(m["accuracy"], 2) != num(p["accuracy"], 2):
                bad_acc.append(label)
            if num(m["grounded"], 2) != num(p["grounded"], 2):
                bad_grd.append(label)
            lo, hi = m["accuracy_ci95"]
            if [num(m["accuracy"], 1), num(lo, 1), num(hi, 1)] != [num(v, 1) for v in p["accuracy_ci"]]:
                bad_ci.append(label)
            mc = m["mcnemar_vs_reference"]
            got = mc["chi2_lower_bound_floor"] if mc else None
            if got != p["chi2_lower_bound"]:
                bad_chi.append(label)
    n_rows = 2 * len(pub["systems"])
    check(not bad_tot, f"Table III: each total is the sum of E1-E6 ({n_rows} rows) {bad_tot}")
    check(not bad_acc, f"Table II: accuracy = 1 - total/560 for all {n_rows} rows {bad_acc}")
    check(not bad_grd, f"Table II: grounded = 1 - (E1+E3+E4+E6)/560 for all {n_rows} rows {bad_grd}")
    check(not bad_ci, f"Table SV: every Wilson 95% interval {bad_ci}")
    check(not bad_chi, f"Table SV: every McNemar lower bound (rounded down) {bad_chi}")
    check(all(ev.chi2_sf_1df(s[(e['system'], c)]["mcnemar_vs_reference"]["chi2_lower_bound"])
              < 1e-15 for e in pub["systems"][1:] for c in ev.CONDITIONS),
          "Section S8: every bound gives p < 1e-15")

    bad_cl = []
    for system, ci in pub["clustered_extreme_notes_only"].items():
        if system.startswith("_"):
            continue
        lo, hi = s[(system, "notes")]["accuracy_ci95_one_per_individual"]
        if [num(lo, 1), num(hi, 1)] != [num(v, 1) for v in ci]:
            bad_cl.append(system)
    check(not bad_cl, f"Section S8: the clustered extreme (one decision per individual) {bad_cl}")

    a = summary_of("ablation")
    bad_ab = []
    for entry in pub["ablation"]:
        for cond in ev.CONDITIONS:
            m, p = a[(entry["run"], cond)], entry[cond]
            if (num(m["accuracy"], 2) != num(p["accuracy"], 2)
                    or num(m["grounded"], 2) != num(p["grounded"], 2)
                    or m["errors"]["total"] != p["errors"]["total"]):
                bad_ab.append(f"{entry['run']} / {cond}")
    check(not bad_ab, f"ablation table: accuracy, grounding and totals for "
                      f"{2 * len(pub['ablation'])} rows {bad_ab}")
    fire_main = pub["systems"][0]
    fire_abl = pub["ablation"][0]
    check(all(fire_main[c]["errors"] == fire_abl[c]["errors"] for c in ev.CONDITIONS),
          "the ablation table's FiRE row is the main run")

    latex = render(s, "main", "latex")
    check(r"\textbf{FiRE} & \textbf{94.46} & \textbf{95.54} & \textbf{100.00} & \textbf{100.00} \\"
          in latex and r"Gemini 3.1 Pro, knowledge-matched & 75.18 & 65.00 & 88.93 & 77.86 \\" in latex,
          "Table II renders as in the paper (FiRE and last rows)")
    err = render(s, "errors", "latex")
    check(r"GPT-5.1 & Notes + form & 38 & 18 & 47 & 31 & 49 & 57 & 240 \\" in err,
          "Table III renders as in the paper (GPT-5.1, notes + form)")
    st = render(s, "stats", "latex")
    check(r"FiRE & 94.5 (92.2, 96.1) & -- & 95.5 (93.5, 97.0) & -- \\" in st
          and r"GPT-5.1 & 69.6 (65.7, 73.3) & 94 & 57.1 (53.0, 61.2) & 172 \\" in st,
          "Table SV renders as in the paper (FiRE and GPT-5.1)")
    ab = render(a, "ablation", "latex")
    check(r"All four off & 13 & 17 & 10 & 10 & 21 & 0 & 71 & 87.32 & 94.11 & "
          r"11 & 15 & 10 & 9 & 17 & 13 & 75 & 86.61 & 92.32 \\" in ab,
          "the ablation table renders as in the paper (all four checks off)")
    md = render(s, "main", "markdown")
    check(md.splitlines()[2] == "| **FiRE** | **94.46** | **95.54** | **100.00** | **100.00** |",
          "Markdown output")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Render the paper's Tables II and III, Table SV and the "
                    "ablation table from a judgements CSV (no model call).")
    ap.add_argument("judgements", nargs="?", help="Judgements CSV (see evaluate_judgements.py)")
    ap.add_argument("--table", choices=TABLES + ("all",), default="all",
                    help="Which table to print (default: all)")
    ap.add_argument("--format", choices=("latex", "markdown"), default="latex",
                    help="Output format (default: latex, the rows of the paper's tabular)")
    ap.add_argument("--systems", nargs="+", metavar="NAME",
                    help="Systems (or runs) in the order to print; default: CSV order")
    ap.add_argument("--reference", default="FiRE",
                    help="System the McNemar bounds compare against (default: FiRE)")
    ap.add_argument("--published", action="store_true",
                    help="Render from the published counts "
                         "(data/paper_results/published_results.json)")
    ap.add_argument("--example-csv", metavar="PATH",
                    help="Write a SYNTHETIC judgements CSV with the published Table III "
                         "counts and exit")
    ap.add_argument("--example-ablation-csv", metavar="PATH",
                    help="Write a SYNTHETIC judgements CSV with the published ablation "
                         "counts and exit")
    ap.add_argument("--selftest", "--self-test", action="store_true",
                    help="Check every published derived number against the counts, and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.example_csv or args.example_ablation_csv:
        pub = load_published()
        for path, section in ((args.example_csv, "systems"),
                              (args.example_ablation_csv, "ablation")):
            if path:
                write_csv(synthetic_rows(published_counts(pub, section), pub["n_individuals"]),
                          path)
                print(f"written (synthetic, published counts): {path}")
        return

    if args.published:
        pub = load_published()
        main_summary = ev.summarise(synthetic_rows(published_counts(pub, "systems"),
                                                   pub["n_individuals"]), args.reference)
        abl_summary = ev.summarise(synthetic_rows(published_counts(pub, "ablation"),
                                                  pub["n_individuals"]), args.reference)
        tables = TABLES if args.table == "all" else (args.table,)
        for t in tables:
            summ = abl_summary if t == "ablation" else main_summary
            print(f"% {t}\n{render(summ, t, args.format)}\n")
        return

    if not args.judgements:
        ap.error("a judgements CSV is required (or --published, --example-csv, --selftest)")
    try:
        rows = ev.read_judgements(args.judgements)
    except (OSError, ev.JudgementError) as e:
        sys.exit(f"error: {e}")
    problems = ev.validate(rows)
    if problems:
        for p in problems[:50]:
            print(f"error: {p}", file=sys.stderr)
        sys.exit(1)
    summary = ev.summarise(rows, reference=args.reference)
    tables = TABLES if args.table == "all" else (args.table,)
    for t in tables:
        if t == "ablation" and args.table == "all":
            continue
        try:
            print(f"% {t}\n{render(summary, t, args.format, args.systems)}\n")
        except KeyError as e:
            sys.exit(f"error: {e.args[0]}")


if __name__ == "__main__":
    main()
