#!/usr/bin/env python3
"""
Review burden: the entries FiRE leaves for a psychotherapist to check
(main paper, Section VI; Supplementary Section S5, "Review burden").

Per individual, from the working folder written by scripts/run_fire.py
(notes-only run):

  flagged events   kept events that failed one Stage 2-3 check
                   (REVIEW_REQUIRED) or that Stage 4 marked uncertain;
                   an event that is both counts once
  rescued events   events Stage 5 restored ("rescued_by_stage5") or kept
                   unverified after a failed call ("rescued_api_unavailable",
                   "rescued_no_llm")
  held items       items held for review and not counted: a required
                   condition (how often, how long, how much older) or the age
                   is not stated in the notes
  new states       provisional states Worker 2 mapped the individual's events
                   to; a state is approved once, so the totals count each
                   state once however many individuals use it

The total is the number of entries to check. Every entry comes with its
source line in the outputs. The script also reports events extracted at
Stages 2-3 and the discarded events Stage 5 examined.

Usage:
  python3 scripts/review_burden.py runs/*/
  python3 scripts/review_burden.py runs/*/ --json burden.json
  python3 scripts/review_burden.py --selftest

Standard library only. It makes no model call.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

DROPPED = {"DISCARD", "DEDUP"}
RESTORED = {"rescued_by_stage5"}
UNVERIFIED = {"rescued_api_unavailable", "rescued_no_llm"}


def _events(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["events"] if isinstance(d, dict) else d


def _key(e: dict) -> tuple:
    return (e.get("unit_id"), e.get("event"), e.get("source_sentence"))


def burden(folder: str) -> dict:
    f = Path(folder)
    s23 = _events(f / "validated_events.json")
    s5_path = f / "validated_events_s5.json"
    final = _events(s5_path)
    with open(s5_path, encoding="utf-8") as h:
        s5_meta = json.load(h)
    s5_stats = (s5_meta.get("stage5_stats") or {}) if isinstance(s5_meta, dict) else {}
    s23_status = {_key(e): e.get("status") for e in s23}

    kept = [e for e in final if e.get("status") not in DROPPED]
    flagged_s23 = {_key(e) for e in kept if s23_status.get(_key(e)) == "REVIEW_REQUIRED"}
    uncertain_s4 = {_key(e) for e in kept if e.get("stage4_uncertain")}
    flagged = flagged_s23 | uncertain_s4
    # an event is one entry: a rescued event already flagged is not counted twice
    restored = [e for e in kept if e.get("status_note") in RESTORED and _key(e) not in flagged]
    unverified = [e for e in kept if e.get("status_note") in UNVERIFIED and _key(e) not in flagged]

    with open(f / "ace_scorecard.json", encoding="utf-8") as h:
        card = json.load(h)
    held = sorted(set(card.get("held_items") or []) |
                  set(card.get("pending_age_review_items") or []))

    with open(f / "final_mapped_events.json", encoding="utf-8") as h:
        w2 = json.load(h)
    pieces = []
    for event in w2.get("events") or []:
        pieces.append(event)
        pieces.extend(event.get("sub_events") or [])
    new_states = sorted({p.get("state_code") or p.get("mapped_state_code") for p in pieces
                         if p.get("mapped_state_status") == "provisional"} - {None})

    total = len(flagged) + len(restored) + len(unverified) + len(held) + len(new_states)
    return {
        "individual": card.get("patient_id") or f.name,
        "events_extracted": len(s23),
        "events_kept": len(kept),
        "flagged_events": len(flagged),
        "flagged_stage2_3": len(flagged_s23),
        "uncertain_stage4": len(uncertain_s4),
        "flagged_both": len(flagged_s23 & uncertain_s4),
        "rescued_restored": len(restored),
        "rescued_unverified": len(unverified),
        "discards_examined": s5_stats.get("discards_examined"),
        "held_items": held,
        "new_states": new_states,
        "entries_to_check": total,
    }


def summarise(rows: list) -> dict:
    n = len(rows)
    tot = {k: sum(r[k] for r in rows) for k in (
        "events_extracted", "flagged_events", "flagged_stage2_3", "uncertain_stage4",
        "flagged_both", "rescued_restored", "rescued_unverified", "entries_to_check")}
    tot["held_items"] = sum(len(r["held_items"]) for r in rows)
    # a new state is approved once for everyone, so it is one entry overall
    unique_states = {s for r in rows for s in r["new_states"]}
    tot["new_states"] = len(unique_states)
    tot["entries_to_check"] -= sum(len(r["new_states"]) for r in rows) - len(unique_states)
    tot["individuals"] = n
    tot["entries_per_individual"] = tot["entries_to_check"] / n if n else 0.0
    tot["events_per_individual"] = tot["events_extracted"] / n if n else 0.0
    return tot


def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    def ev(i, status, **kw):
        return {"unit_id": f"u{i}", "event": f"e{i}", "source_sentence": f"s{i}",
                "status": status, **kw}

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "P1"
        f.mkdir()
        s23 = [ev(1, "CONFIRMED"), ev(2, "REVIEW_REQUIRED"), ev(3, "REVIEW_REQUIRED"),
               ev(4, "DISCARD"), ev(5, "DISCARD"), ev(6, "CONFIRMED"), ev(7, "DISCARD")]
        final = [ev(1, "CONFIRMED"), ev(2, "TENTATIVE", stage4_uncertain=True),
                 ev(3, "REVIEW_REQUIRED"), ev(4, "TENTATIVE", status_note="rescued_by_stage5"),
                 ev(5, "TENTATIVE", status_note="rescued_api_unavailable"),
                 ev(6, "TENTATIVE", stage4_uncertain=True), ev(7, "DISCARD")]
        (f / "validated_events.json").write_text(json.dumps({"events": s23}))
        (f / "validated_events_s5.json").write_text(json.dumps(
            {"events": final, "stage5_stats": {"discards_examined": 3}}))
        (f / "ace_scorecard.json").write_text(json.dumps(
            {"patient_id": "P1", "held_items": [10], "pending_age_review_items": [3]}))
        (f / "final_mapped_events.json").write_text(json.dumps({"events": [
            {"sub_events": [{"state_code": "new-1", "mapped_state_status": "provisional"},
                            {"state_code": "q-a", "mapped_state_status": "permanent"}]}]}))
        r = burden(str(f))
        check(r["flagged_stage2_3"] == 2 and r["uncertain_stage4"] == 2 and r["flagged_both"] == 1
              and r["flagged_events"] == 3, "flagged events: Stage 2-3 and Stage 4, both counted once")
        check(r["rescued_restored"] == 1 and r["rescued_unverified"] == 1,
              "rescued events: restored and kept unverified")
        check(r["held_items"] == [3, 10] and r["new_states"] == ["new-1"],
              "held items (condition or age) and new states")
        check(r["entries_to_check"] == 3 + 1 + 1 + 2 + 1 and r["events_extracted"] == 7,
              "entries to check")
        s = summarise([r, r])
        check(s["entries_to_check"] == 15 and s["new_states"] == 1 and s["held_items"] == 4,
              "totals: a new state shared by two individuals is one entry")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Entries FiRE leaves for psychotherapist review.")
    ap.add_argument("folders", nargs="*", help="FiRE working folders (notes-only runs)")
    ap.add_argument("--json", help="Write per-individual rows and totals to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.folders:
        ap.error("give FiRE working folders")
    rows = [burden(p) for p in a.folders if os.path.isdir(p)]
    s = summarise(rows)
    for r in rows:
        print(f"{r['individual']:<16} events {r['events_extracted']:>4}  flagged {r['flagged_events']:>3}  "
              f"rescued {r['rescued_restored'] + r['rescued_unverified']:>2}  held {len(r['held_items']):>2}  "
              f"new states {len(r['new_states']):>2}  entries {r['entries_to_check']:>3}")
    print(f"\n{s['individuals']} individuals: {s['events_extracted']} events "
          f"({s['events_per_individual']:.1f} each); {s['entries_to_check']} entries to check "
          f"({s['entries_per_individual']:.1f} each): {s['flagged_events']} flagged events "
          f"({s['flagged_stage2_3']} at Stages 2-3, {s['uncertain_stage4']} uncertain at Stage 4, "
          f"{s['flagged_both']} both), {s['rescued_restored'] + s['rescued_unverified']} rescued "
          f"({s['rescued_restored']} restored, {s['rescued_unverified']} unverified), "
          f"{s['held_items']} held items, {s['new_states']} new states")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"individuals": rows, "totals": s}, f, indent=2)


if __name__ == "__main__":
    main()
