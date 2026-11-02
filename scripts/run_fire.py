#!/usr/bin/env python3
"""
Run the FiRE pipeline end to end, for one individual or for a folder.

This script only chains the pipeline's own command-line scripts, in the
order and with the options the paper's runs used (Supplementary Material,
Section S1); it adds no logic of its own. Each step runs as a separate
process in the individual's working folder, exactly as the README's
step-by-step commands do, so every intermediate file is kept.

Steps (names for --from and --to):
  preprocess   src/preprocessing/fire_preprocessor.py   -> <id>_flat.txt
  stage1       src/extraction/fire_stage1.py            -> stage1_output.json
  stage2_3     src/extraction/fire_stage2_3_single_model.py -> validated_events.json   [GPT-5.1]
  stage4       src/verification/fire_stage4.py          -> validated_events_s4.json
  stage5       src/verification/fire_stage5.py          -> validated_events_s5.json  [GPT-5.1]
  worker2      src/state_space/fire_worker2.py          -> final_mapped_events.json  [GPT-5.1]
  worker3      src/algebra/fire_worker3_general.py      -> worker3_output.json (+ .html)
  worker3_ace  src/instruments/fire_worker3_ace.py      -> worker3_ace_output.json   [GPT-5.1]
               (--verify --verify-llm, as in the paper)
  worker4      src/instruments/fire_worker4_ace.py      -> ace_scorecard.json
  reconcile    src/instruments/fire_reconcile_ace.py    -> ace_reconciled.json
               (notes-plus-form condition; only when a form is given)

Steps marked [GPT-5.1] need OPENAI_API_KEY; the runner checks that it is set
before starting and stops otherwise, because Stage 5 and Worker 3 (ACE)
would silently behave differently without it. Worker 2 also needs the
sentence-transformers package. --offline-item-check runs Worker 3 (ACE)
with --verify only (deterministic part of the item check, no model call);
the paper's results used the model call.

Usage, one individual:
  export OPENAI_API_KEY=...
  python3 scripts/run_fire.py notes/patient_01.txt \\
      --state-space runs/state_space.json --workdir runs/patient_01
  python3 scripts/run_fire.py notes/patient_01.txt --form notes/patient_01_ace_form.json \\
      --state-space runs/state_space.json --workdir runs/patient_01

Usage, a folder (every patient_NN.txt, one individual at a time, in name
order, sharing one state space; with --with-forms each needs
patient_NN_ace_form.json; an individual whose final output exists is
skipped unless --force):
  python3 scripts/run_fire.py --batch notes/ --out-dir runs --state-space runs/state_space.json

Other options:
  --from STEP / --to STEP   run part of the chain (inputs must already be in
                            the working folder)
  --print-commands          print the commands and exit; nothing is run
  --selftest                offline check of the command plan

Every run writes run_log.json in the working folder: the steps, commands,
exit codes and durations. If the state-space file does not exist it is
created empty ({"domains": {}}), and every state is then created by Worker 2
as provisional (see the README for the psychotherapist review step).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

STEPS = ("preprocess", "stage1", "stage2_3", "stage4", "stage5", "worker2",
         "worker3", "worker3_ace", "worker4", "reconcile")
MODEL_STEPS = {"stage2_3", "stage5", "worker2", "worker3_ace"}
SCRIPTS = {
    "preprocess":  SRC / "preprocessing" / "fire_preprocessor.py",
    "stage1":      SRC / "extraction" / "fire_stage1.py",
    "stage2_3":    SRC / "extraction" / "fire_stage2_3_single_model.py",
    "stage4":      SRC / "verification" / "fire_stage4.py",
    "stage5":      SRC / "verification" / "fire_stage5.py",
    "worker2":     SRC / "state_space" / "fire_worker2.py",
    "worker3":     SRC / "algebra" / "fire_worker3_general.py",
    "worker3_ace": SRC / "instruments" / "fire_worker3_ace.py",
    "worker4":     SRC / "instruments" / "fire_worker4_ace.py",
    "reconcile":   SRC / "instruments" / "fire_reconcile_ace.py",
}
_NOTES_RE = re.compile(r"^patient_\d+\.txt$")


def plan(patient_id: str, state_space: str = None, form: str = None,
         offline_item_check: bool = False, concurrency: int = None) -> list:
    """The (step, argv) list for one individual, run with the working
    folder as the current directory. Paths of files outside the working
    folder (state space, form) must be absolute."""
    w3a = ["worker3_output.json", "-o", "worker3_ace_output.json", "--verify"]
    if not offline_item_check:
        w3a.append("--verify-llm")
    s23 = ["stage1_output.json", "--patient-id", patient_id, "--output", "validated_events.json"]
    w2 = ["validated_events_s5.json", "--stage1", "stage1_output.json",
          "--state-space", str(state_space), "--output", "final_mapped_events.json"]
    if concurrency is not None:
        s23 += ["--concurrency", str(concurrency)]
        w2 += ["--concurrency", str(concurrency)]
    args = {
        "preprocess":  [f"{patient_id}.txt"],
        "stage1":      [f"{patient_id}_flat.txt", "-o", "stage1_output.json"],
        "stage2_3":    s23,
        "stage4":      ["validated_events.json", "--stage1", "stage1_output.json",
                        "-o", "validated_events_s4.json"],
        "stage5":      ["validated_events_s4.json", "--stage1", "stage1_output.json",
                        "-o", "validated_events_s5.json"],
        "worker2":     w2,
        "worker3":     ["final_mapped_events.json", "-o", "worker3_output.json"],
        "worker3_ace": w3a,
        "worker4":     ["worker3_ace_output.json", "-o", "ace_scorecard.json"],
        "reconcile":   ["worker3_ace_output.json", "ace_scorecard.json",
                        "--form", str(form), "-o", "ace_reconciled.json"],
    }
    steps = [s for s in STEPS if s != "reconcile" or form]
    return [(s, [sys.executable, str(SCRIPTS[s])] + args[s]) for s in steps]


def select(steps: list, start: str = None, stop: str = None) -> list:
    names = [s for s, _ in steps]
    i = names.index(start) if start else 0
    j = names.index(stop) + 1 if stop else len(names)
    if i >= j:
        raise ValueError(f"--from {start} comes after --to {stop}")
    return steps[i:j]


def _quote(argv: list) -> str:
    return " ".join(a if re.fullmatch(r"[\w@%+=:,./-]+", a) else repr(a) for a in argv)


def run_individual(notes: str, workdir: str, state_space: str = None, form: str = None,
                   patient_id: str = None, start: str = None, stop: str = None,
                   offline_item_check: bool = False, concurrency: int = None,
                   print_only: bool = False) -> int:
    """Run (or print) the chain for one individual. Returns 0 on success,
    otherwise the exit code of the failing step."""
    wd = Path(workdir).resolve()
    if patient_id is None:
        if not notes:
            raise ValueError("--patient-id is needed when no notes file is given")
        patient_id = Path(notes).name[:-4] if notes.endswith(".txt") else Path(notes).name
    ss = str(Path(state_space).resolve()) if state_space else None
    fm = str(Path(form).resolve()) if form else None
    steps = select(plan(patient_id, ss, fm, offline_item_check, concurrency), start, stop)
    names = [s for s, _ in steps]

    if print_only:
        print(f"# {patient_id}: run in {wd}")
        if "preprocess" in names and notes:
            print(f"# copy {Path(notes).resolve()} -> {wd / (patient_id + '.txt')}")
        for _s, argv in steps:
            print(_quote(argv))
        return 0

    if "preprocess" in names and not notes:
        raise ValueError("a notes file is needed to run the preprocess step")
    if "worker2" in names and not ss:
        raise ValueError("--state-space is needed to run Worker 2")
    need_key = (set(names) & MODEL_STEPS) - ({"worker3_ace"} if offline_item_check else set())
    if need_key and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; it is needed by "
                           + ", ".join(s for s in names if s in need_key))

    wd.mkdir(parents=True, exist_ok=True)
    if "preprocess" in names:
        target = wd / f"{patient_id}.txt"
        if Path(notes).resolve() != target:
            shutil.copyfile(notes, target)
    if ss and "worker2" in names and not Path(ss).exists():
        Path(ss).parent.mkdir(parents=True, exist_ok=True)
        Path(ss).write_text('{"domains": {}}\n', encoding="utf-8")
        print(f"[run_fire] created an empty state space: {ss}")

    log = {"patient_id": patient_id, "workdir": str(wd),
           "started_utc": datetime.now(timezone.utc).isoformat(),
           "offline_item_check": offline_item_check, "steps": []}
    code = 0
    for step, argv in steps:
        print(f"\n[run_fire] {patient_id}: {step}\n[run_fire] $ {_quote(argv)}", flush=True)
        t0 = time.time()
        rc = subprocess.run(argv, cwd=wd).returncode
        log["steps"].append({"step": step, "argv": argv, "returncode": rc,
                             "seconds": round(time.time() - t0, 1)})
        if rc != 0:
            print(f"[run_fire] {patient_id}: {step} failed (exit {rc})", file=sys.stderr)
            code = rc
            break
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (wd / "run_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return code


def batch_inputs(data_dir: str, with_forms: bool) -> list:
    """(notes, form or None) for every patient_NN.txt in data_dir, in name order."""
    pairs = []
    for p in sorted(Path(data_dir).glob("patient_*.txt")):
        if not _NOTES_RE.match(p.name):
            continue
        form = None
        if with_forms:
            form = p.with_name(p.stem + "_ace_form.json")
            if not form.exists():
                raise FileNotFoundError(f"no form for {p}: expected {form}")
            form = str(form)
        pairs.append((str(p), form))
    return pairs


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    """Offline checks of the command plan (nothing is run)."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    full = plan("p01", "/abs/state_space.json", "/abs/form.json")
    names = [s for s, _ in full]
    check(names == list(STEPS), "the full chain has the ten steps in pipeline order")
    check(all(Path(argv[1]).exists() for _s, argv in full), "every step's script exists")
    argv = dict(full)
    check(argv["worker3_ace"][-2:] == ["--verify", "--verify-llm"],
          "Worker 3 (ACE) runs with --verify --verify-llm, as in the paper")
    check("--fresh-start" not in argv["worker2"] and "/abs/state_space.json" in argv["worker2"],
          "Worker 2 reads the given state space and never deletes it")
    check([s for s, _ in plan("p01")] == list(STEPS[:-1]), "without a form there is no reconcile step")
    check(plan("p01", offline_item_check=True)[7][1][-1] == "--verify",
          "--offline-item-check drops --verify-llm")
    check([s for s, _ in select(full, "worker3", "worker4")] == ["worker3", "worker3_ace", "worker4"],
          "--from and --to select a contiguous part of the chain")
    outputs = {"stage1": "stage1_output.json", "stage2_3": "validated_events.json",
               "stage4": "validated_events_s4.json", "stage5": "validated_events_s5.json",
               "worker2": "final_mapped_events.json", "worker3": "worker3_output.json",
               "worker3_ace": "worker3_ace_output.json", "worker4": "ace_scorecard.json"}
    order = list(STEPS)
    chained = all(any(outputs[prev] in a for a in argv[nxt])
                  for prev, nxt in zip(order[1:-2], order[2:-1]))
    check(chained, "each step reads the file the previous step writes")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Run the FiRE pipeline end to end for one individual or a folder.")
    ap.add_argument("notes", nargs="?", help="Consultation notes (plain text) of one individual")
    ap.add_argument("--workdir", help="Working folder for one individual "
                                      "(default: runs/<patient id>)")
    ap.add_argument("--patient-id", help="Identifier (default: notes file name without .txt)")
    ap.add_argument("--form", help="Self-filled ACE form JSON: also run the notes-plus-form step")
    ap.add_argument("--state-space", help="Shared state-space JSON, read and updated by Worker 2 "
                                          "(created empty if missing)")
    ap.add_argument("--batch", metavar="DATA_DIR", help="Run every patient_NN.txt in DATA_DIR")
    ap.add_argument("--with-forms", action="store_true",
                    help="Batch: also reconcile with patient_NN_ace_form.json")
    ap.add_argument("--out-dir", default="runs",
                    help="Batch: parent folder of the working folders (default: runs)")
    ap.add_argument("--force", action="store_true",
                    help="Batch: rerun individuals whose final output already exists")
    ap.add_argument("--from", dest="start", choices=STEPS, help="First step to run")
    ap.add_argument("--to", dest="stop", choices=STEPS, help="Last step to run")
    ap.add_argument("--offline-item-check", action="store_true",
                    help="Worker 3 (ACE) with --verify only (no model call; not the paper's setting)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Parallel calls in Stages 2-3 and Worker 2 (default: each script's own)")
    ap.add_argument("--print-commands", action="store_true",
                    help="Print the commands and exit without running anything")
    ap.add_argument("--selftest", action="store_true", help="Run offline checks and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    common = dict(state_space=args.state_space, start=args.start, stop=args.stop,
                  offline_item_check=args.offline_item_check,
                  concurrency=args.concurrency, print_only=args.print_commands)
    try:
        if not args.batch:
            if not args.notes and not args.patient_id:
                ap.error("give a notes file, --patient-id with --from, or --batch")
            pid = args.patient_id or (Path(args.notes).name[:-4] if args.notes.endswith(".txt")
                                      else Path(args.notes).name)
            workdir = args.workdir or str(Path("runs") / pid)
            sys.exit(run_individual(args.notes, workdir, form=args.form, patient_id=pid, **common))

        final = "ace_reconciled.json" if args.with_forms else "ace_scorecard.json"
        failed = []
        for notes, form in batch_inputs(args.batch, args.with_forms):
            pid = Path(notes).stem
            wd = Path(args.out_dir) / pid
            if (wd / final).exists() and not args.force and not args.print_commands:
                print(f"[run_fire] skip (exists): {wd / final}")
                continue
            rc = run_individual(notes, str(wd), form=form, patient_id=pid, **common)
            if rc:
                failed.append(pid)
        if failed:
            print(f"[run_fire] FAILED ({len(failed)}): {failed}; rerun the same command "
                  "to retry them", file=sys.stderr)
            sys.exit(2)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
