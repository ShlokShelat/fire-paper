"""FiRE Pipeline — Worker 4 (ACE): parses the ACE expression and counts
distinct items (0-14).

Paper concept: the last step of stage (iv). Scoring parses the projected
expression and resolves each symbol through its cluster to an item with
the fixed cluster -> item table (ace14.py), giving the chain
symbol -> cluster -> item. The ACE score is the number of distinct items
present. Exponents (self-loop and feedback) are carried for reference and
never change the count.

Flags raised here, none of which changes the score (Supplementary
Material, Section S1): unconfirmed duration on items 10 and 13 (the item
still counts) and findings Worker 3 (ACE) left out for missing age
evidence. The score card reports the score and the items behind it, and
nothing else: it gives no risk level and no clinical interpretation.

The instrument is the 14-item ACE questionnaire validated in India
(Trivedi et al., 2025), which extends nine of the ten original ACE items
(Felitti et al., 1998); see Supplementary Material, Section S2.

Pipeline position: final stage, after Worker 3 (ACE).
Input: a Worker 3 (ACE) output JSON. Output: the per-item score card.

Usage:
  python3 fire_worker4_ace.py worker3_ace_output.json -o ace_scorecard.json
  python3 fire_worker4_ace.py --selftest
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ace14  # noqa: E402  (shared ACE-14 table, see ace14.py)


# ─────────────────────────────────────────────────────────────
# ACE DEFINITIONS (canonical 14-item, India-adapted set)
# ─────────────────────────────────────────────────────────────

ACE_QUESTIONS = {
    1:  "Did a parent or adult in the household often swear at, insult, humiliate, or threaten the child?",
    2:  "Did a parent or adult in the household push, grab, slap, throw things at, or hit the child hard enough to leave marks?",
    3:  "Did an adult or person 5+ years older sexually touch the child, or attempt/commit sex with the child?",
    4:  "Did the child often feel unloved, unimportant, or that the family lacked closeness and support?",
    5:  "Did the child lack food, clean clothes, or protection, or were parents too impaired to provide care?",
    6:  "Were the child's parents separated or divorced?",
    7:  "Was the child's mother or stepmother physically abused by a partner?",
    8:  "Did a household member drink problematically or use street drugs?",
    9:  "Was a household member depressed, mentally ill, or did one attempt/die by suicide?",
    10: "Did the child live 2+ years in a dangerous neighbourhood or witness community assault?",
    11: "Did other kids or siblings often hit, threaten, pick on, or insult the child?",
    12: "Did the child often feel lonely, rejected, or isolated from peers?",
    13: "Was the family very poor or on public assistance for 2+ years?",
    14: "Did the child's parents have physical or verbal fights with each other?",
}

# The three factors of the 14-item version (Supplementary Material, Section S2):
# CM, childhood maltreatment; RS, resource scarcity and environmental
# distress; HC, household challenges. Reported only; never used in scoring.
ACE_CATEGORIES = {
    "CM": [1, 2, 3, 4, 11, 12, 14],
    "RS": [5, 10, 13],
    "HC": [6, 7, 8, 9],
}

PRESENCE_ONLY_ITEMS = {6}
# Items 10 and 13 ask for a duration of two or more years. If no contributing
# state shows it, the item still counts and is flagged (Section S2).
DURATION_ITEMS      = {10, 13}

# A state spanning onset..end with end - onset >= 1 covers two calendar years.
MIN_DURATION_SPAN     = 1
MIN_DURATION_EXPONENT = 2


# ─────────────────────────────────────────────────────────────
# CLUSTER -> ACE ITEM BINDING
# ─────────────────────────────────────────────────────────────
# One table, shared with Worker 3 ACE (projection) via ace14.py, so the two
# stages cannot disagree about which item a cluster scores as.

CLUSTER_ACE_ITEM = ace14.CLUSTER_ITEM
NON_ACE_CLUSTERS = ace14.NON_ACE_CLUSTERS


def cluster_to_ace_item(cluster_name: str):
    """Resolve a cluster name to its ACE item number, or None. Longest token wins."""
    return ace14.cluster_to_item(cluster_name)


# ─────────────────────────────────────────────────────────────
# EXPRESSION PARSER
# ─────────────────────────────────────────────────────────────

def parse_expression_symbols(fire_expression: str) -> dict:
    """Walk the FiRE expression and return {symbol: [exponent, ...]} per symbol."""
    appearances: "OrderedDict[str, list]" = OrderedDict()
    cleaned = re.sub(r"\[[^\]]*\]", " ", fire_expression or "")
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9]*)(?:\^(\d+))?", cleaned):
        sym = m.group(1)
        exp = int(m.group(2)) if m.group(2) else 1
        appearances.setdefault(sym, []).append(exp)
    return appearances


def accumulate_exponent_weights(appearances: dict) -> "OrderedDict":
    """Sum of exponents per symbol — reference only, never scored."""
    return OrderedDict((sym, sum(exps)) for sym, exps in appearances.items())


# ─────────────────────────────────────────────────────────────
# DURATION CHECK
# ─────────────────────────────────────────────────────────────

def _check_duration(item: int, state_codes: list, states_by_code: dict) -> dict:
    """Check whether a DURATION_ITEMS entry has evidence of a 2+ year span.
    Returns a review signal only — never gates the score."""
    if item not in DURATION_ITEMS:
        return {"applicable": False, "confirmed": True, "detail": ""}

    best_span, best_exp = None, 0
    for code in state_codes:
        st = states_by_code.get(code)
        if not st:
            continue
        onset, end = st.get("onset_age"), st.get("end_age")
        if onset is not None and end is not None:
            span = end - onset
            if best_span is None or span > best_span:
                best_span = span
        exp = st.get("exponent") or 1
        best_exp = max(best_exp, exp)

    if best_span is not None and best_span >= MIN_DURATION_SPAN:
        return {"applicable": True, "confirmed": True,
                "detail": f"contributing state spans {best_span + 1} year(s)"}
    if best_exp >= MIN_DURATION_EXPONENT:
        return {"applicable": True, "confirmed": True,
                "detail": f"contributing state has exponent {best_exp} "
                          "(repetition/chronic language detected upstream)"}
    return {"applicable": True, "confirmed": False,
            "detail": "no contributing state shows an explicit 2+ year span "
                      "or repetition/chronic language; this item's own "
                      "defining text requires a sustained duration — review "
                      "recommended, not auto-dropped"}


# ─────────────────────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────────────────────

def score_from_expression(w3: dict) -> dict:
    """Classic ACE score (0..14) from Worker 3 ACE output. Each distinct ACE
    item present counts 1. Exponents are reference weights only.

    Runs in two passes: first, resolve every alphabet symbol in the
    expression to an ACE item number (preferring the deterministic
    cluster-name mapping over Worker 3's own hint, and logging any
    disagreement as a mapping_conflict); second, build one scorecard row
    per ACE item (1-14) from the symbols that resolved to it, including
    a duration check for the two items that require a sustained span.
    """
    fire_expression = str(w3.get("fire_expression") or "")
    alphabet_mapping = w3.get("alphabet_mapping") or {}
    inventory = w3.get("alphabet_inventory") or {}
    resolved = w3.get("resolved_states") or []
    states_by_code = {s.get("state_code"): s for s in resolved if s.get("state_code")}

    appearances = parse_expression_symbols(fire_expression)
    weights = accumulate_exponent_weights(appearances)

    symbol_rows = []
    item_to_symbols = defaultdict(list)
    unmapped_symbols = []
    mapping_conflicts = []

    # Pass 1: resolve each symbol to an ACE item number.
    for sym in appearances:
        cluster = alphabet_mapping.get(sym) or \
                  (inventory.get(sym, {}) or {}).get("cluster_name") or ""
        if not cluster and sym not in inventory and sym not in alphabet_mapping:
            continue
        det_item = cluster_to_ace_item(cluster)
        # Items Worker 3 ACE projected this symbol to (already ACE-14), used only
        # to report a disagreement with the fixed table.
        w3_items = [int(i) for i in ((inventory.get(sym, {}) or {}).get("ace_items") or [])
                    if str(i).isdigit() and 1 <= int(i) <= ace14.N_ITEMS]
        if ace14.is_non_ace_cluster(cluster):
            w3_items = []

        item = det_item
        if det_item is not None and w3_items and det_item not in w3_items:
            mapping_conflicts.append({
                "symbol": sym, "cluster": cluster,
                "deterministic_item": det_item, "worker3_items": w3_items})
        # A cluster the fixed table does not resolve scores nothing and is
        # reported as unmapped; it is never resolved from another source.

        row = {
            "symbol": sym,
            "cluster_name": cluster,
            "ace_item": item,
            "exponent_weight": weights.get(sym, 0),
            "exponent_appearances": appearances.get(sym, []),
            "contributing_states": (inventory.get(sym, {}) or {}).get("contributing_states", []),
            "confidence": (inventory.get(sym, {}) or {}).get("confidence"),
        }
        symbol_rows.append(row)
        if item is None:
            unmapped_symbols.append({"symbol": sym, "cluster": cluster})
        else:
            item_to_symbols[item].append(sym)

    # Pass 2: build one scorecard row per ACE item from its resolved symbols.
    scorecard: "OrderedDict[int, dict]" = OrderedDict()
    duration_unconfirmed_items = []
    for n in range(1, 15):
        syms = item_to_symbols.get(n, [])
        present = len(syms) > 0
        item_weight = sum(weights.get(s, 0) for s in syms)
        clusters = sorted({alphabet_mapping.get(s) or
                           (inventory.get(s, {}) or {}).get("cluster_name") or s
                           for s in syms})
        states = []
        for s in syms:
            states += (inventory.get(s, {}) or {}).get("contributing_states", [])
        states = sorted(set(states))

        dur = _check_duration(n, states, states_by_code) if present else \
              {"applicable": n in DURATION_ITEMS, "confirmed": True, "detail": ""}
        if present and dur["applicable"] and not dur["confirmed"]:
            duration_unconfirmed_items.append(n)

        scorecard[n] = {
            "ace_item": n,
            "question": ACE_QUESTIONS[n],
            "present": present,
            "category": _category_of(n),
            "presence_only": n in PRESENCE_ONLY_ITEMS,
            "duration_expected": n in DURATION_ITEMS,
            "duration_confirmed": dur["confirmed"] if dur["applicable"] else None,
            "duration_detail": dur["detail"],
            "symbols": syms,
            "clusters": clusters,
            "contributing_states": states,
            "exponent_weight_reference": item_weight,
        }

    canonical = sum(1 for r in scorecard.values() if r["present"])

    category_breakdown = {
        cat: sorted(n for n in items if scorecard[n]["present"])
        for cat, items in ACE_CATEGORIES.items()
    }

    return {
        "canonical_ace_score": canonical,
        "items_present": sorted(n for n, r in scorecard.items() if r["present"]),
        "category_breakdown": category_breakdown,
        "ace_scorecard": {str(n): r for n, r in scorecard.items()},
        "symbol_mapping": symbol_rows,
        "exponent_weights_reference": dict(weights),
        "unmapped_symbols": unmapped_symbols,
        "mapping_conflicts": mapping_conflicts,
        "duration_unconfirmed_items": duration_unconfirmed_items,
        **_pending_age_review(w3, canonical, scorecard),
        "fire_expression": fire_expression,
    }


def _pending_age_review(w3: dict, canonical: int, scorecard: dict) -> dict:
    """Surface ACE findings Worker 3 ACE left out of the score because they
    have no age evidence (pending psychotherapist age confirmation), so a
    withheld item is visible on the score card rather than silently absent."""
    pending = w3.get("age_unverified") or []
    withheld = sorted({int(x["ace_item"]) for x in pending
                       if x.get("ace_item") is not None
                       and not scorecard[int(x["ace_item"])]["present"]})
    return {
        "pending_age_review": pending,
        "pending_age_review_items": withheld,
    }


def _category_of(n: int) -> str:
    """Return which ACE_CATEGORIES bucket item n belongs to."""
    for cat, items in ACE_CATEGORIES.items():
        if n in items:
            return cat
    return "other"


def feedback_notes(feedback_loops: list) -> list:
    """Feedback edges from Worker 3, listed for reference. They add weight
    to the recurring state's exponent and never change the ACE count."""
    return [{
        "cluster": fb.get("cluster"),
        "cluster_name": fb.get("cluster_name", ""),
        "state_code": fb.get("state_code"),
        "earlier_interval": fb.get("earlier_interval"),
        "later_interval": fb.get("later_interval"),
        "evidence": fb.get("evidence"),
    } for fb in (feedback_loops or [])]


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    """Offline regression guard: exercises the paper's worked example
    scorecard, duration-check confirmation/flagging, and the ACE-10/-12
    dropped-item guard and the shared table,
    against synthetic Worker 3 ACE output."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # The main paper's worked example (Table I, Fig. 5):
    # E_ACE = (a + b + c^2) . (e + a^3), score = 4, items {1, 2, 4, 11}.
    # a = emotional abuse (ACE-1), b = felt unloved (ACE-4),
    # c = peer bullying (ACE-11), e = physical abuse (ACE-2).
    # Mother's death (d, caregiver death) is verified but has no counterpart
    # among the 14 items, so it must NOT be scored.
    w3 = {
        "fire_expression": "[C] (a+b+c^2) . (e+a^3)",
        "alphabet_mapping": {
            "a": "Emotional Abuse", "b": "Felt Unloved", "c": "Peer Bullying",
            "d": "Death of Parent or Guardian", "e": "Physical Abuse",
        },
        "alphabet_inventory": {
            "a": {"cluster_name": "Emotional Abuse", "ace_items": [1],
                 "contributing_states": ["st-a"], "confidence": 0.9},
            "b": {"cluster_name": "Felt Unloved", "ace_items": [4],
                 "contributing_states": ["st-b"], "confidence": 0.9},
            "c": {"cluster_name": "Peer Bullying", "ace_items": [11],
                 "contributing_states": ["st-c"], "confidence": 0.9},
            "d": {"cluster_name": "Death of Parent or Guardian", "ace_items": [],
                 "contributing_states": ["st-d"], "confidence": 0.9},
            "e": {"cluster_name": "Physical Abuse", "ace_items": [2],
                 "contributing_states": ["st-e"], "confidence": 0.9},
        },
        "resolved_states": [
            {"state_code": "st-a", "onset_age": 0, "end_age": 16, "exponent": 3},
            {"state_code": "st-b", "onset_age": 0, "end_age": 8, "exponent": 1},
            {"state_code": "st-c", "onset_age": 0, "end_age": 8, "exponent": 2},
            {"state_code": "st-d", "onset_age": 11, "end_age": 11, "exponent": 1},
            {"state_code": "st-e", "onset_age": 12, "end_age": 16, "exponent": 1},
        ],
        "feedback_loops": [],
    }
    result = score_from_expression(w3)
    check(result["canonical_ace_score"] == 4,
          "PAPER MATCH (worked example): worked example scores exactly 4 (items 1,2,4,11)")
    check(set(result["items_present"]) == {1, 2, 4, 11},
          "PAPER MATCH: exactly the items {1,2,4,11} of the paper's worked example (Fig. 5)")
    check(result["exponent_weights_reference"]["a"] == 4
          and result["exponent_weights_reference"]["c"] == 2,
          "exponent weights carried as reference (a=4, c=2), not affecting the count")
    with_d = score_from_expression(dict(w3, fire_expression="[C] (a+b+c^2) . d . (e+a^3)"))
    check(with_d["canonical_ace_score"] == 4
          and next(r["ace_item"] for r in with_d["symbol_mapping"] if r["symbol"] == "d") is None
          and "d" in [u["symbol"] for u in with_d["unmapped_symbols"]],
          "caregiver death (d), even if left in the expression, resolves to no item and "
          "is reported as unmapped, not scored")

    banned = ("risk", "PTSD", "dissociation", "hypervigilance", "burden")
    check(not any(b.lower() in k.lower() for k in result for b in banned),
          "score card carries no risk level or clinical interpretation")

    # Duration: a single-year ACE-10 state is flagged but still counts
    w3_dur = {
        "fire_expression": "[C] k",
        "alphabet_mapping": {"k": "Community Violence"},
        "alphabet_inventory": {"k": {"cluster_name": "Community Violence", "ace_items": [10],
                                    "contributing_states": ["dyn-005"], "confidence": 0.8}},
        "resolved_states": [{"state_code": "dyn-005", "onset_age": 10, "end_age": 10, "exponent": 1}],
        "feedback_loops": [],
    }
    res_dur = score_from_expression(w3_dur)
    check(res_dur["canonical_ace_score"] == 1,
          "single-year ACE-10 state is NOT silently dropped from the count")
    check(10 in res_dur["duration_unconfirmed_items"],
          "single-year ACE-10 state IS flagged as duration-unconfirmed")
    check(res_dur["ace_scorecard"]["10"]["duration_confirmed"] is False,
          "scorecard row carries duration_confirmed=False")

    w3_dur2 = {
        "fire_expression": "[C] k",
        "alphabet_mapping": {"k": "Community Violence"},
        "alphabet_inventory": {"k": {"cluster_name": "Community Violence", "ace_items": [10],
                                    "contributing_states": ["dyn-006"], "confidence": 0.8}},
        "resolved_states": [{"state_code": "dyn-006", "onset_age": 5, "end_age": 9, "exponent": 1}],
        "feedback_loops": [],
    }
    res_dur2 = score_from_expression(w3_dur2)
    check(10 not in res_dur2["duration_unconfirmed_items"],
          "genuine multi-year span (5-9) is confirmed, no flag raised")

    # Dropped items: incarceration and death-of-parent clusters must never
    # be scored, even if an ignored hint field carries a different item number.
    w3_dropped = {
        "fire_expression": "[C] m + n",
        "alphabet_mapping": {"m": "Household Incarceration", "n": "Death of Father"},
        "alphabet_inventory": {
            "m": {"cluster_name": "Household Incarceration", "ace_items": [10],
                 "contributing_states": ["st-m"], "confidence": 0.9},
            "n": {"cluster_name": "Death of Father", "ace_items": [12],
                 "contributing_states": ["st-n"], "confidence": 0.9},
        },
        "resolved_states": [
            {"state_code": "st-m", "onset_age": 8, "end_age": 9, "exponent": 1},
            {"state_code": "st-n", "onset_age": 6, "end_age": 6, "exponent": 1},
        ],
        "feedback_loops": [],
    }
    res_dropped = score_from_expression(w3_dropped)
    check(res_dropped["canonical_ace_score"] == 0,
          "incarceration and death-of-parent score ZERO under the 14-item instrument")

    # Items 10 and 12 are valid ACE-14 items and must score.
    w3_10_12 = {
        "fire_expression": "[C] k . r",
        "alphabet_mapping": {"k": "Community Violence", "r": "Peer Rejection"},
        "alphabet_inventory": {
            "k": {"cluster_name": "Community Violence", "ace_items": [10],
                  "contributing_states": ["s-k"], "confidence": 0.9},
            "r": {"cluster_name": "Peer Rejection", "ace_items": [12],
                  "contributing_states": ["s-r"], "confidence": 0.9},
        },
        "resolved_states": [
            {"state_code": "s-k", "onset_age": 5, "end_age": 9, "exponent": 1},
            {"state_code": "s-r", "onset_age": 8, "end_age": 12, "exponent": 1},
        ],
        "feedback_loops": [],
    }
    check(score_from_expression(w3_10_12)["items_present"] == [10, 12],
          "ACE-14 items 10 (dangerous neighbourhood) and 12 (peer rejection) are scored")

    # The scorer and Worker 3 ACE resolve every table entry identically.
    check(all(cluster_to_ace_item(k) == v for k, v in ace14.CLUSTER_ITEM.items())
          and len(ace14.CLUSTER_ITEM) == 69,
          "the shared 69-entry table resolves every entry to its own item")

    # Findings withheld for missing age are surfaced, never silently absent.
    w3_three = dict(w3, fire_expression="[C] (a+b+c^2)")
    w3_age = dict(w3_three, age_unverified=[
        {"event_uid": "x1", "state_code": "st-x", "event": "hit by father", "ace_item": 2},
        {"event_uid": "x2", "state_code": "st-y", "event": "teased", "ace_item": 1}])
    res_age = score_from_expression(w3_age)
    check(res_age["canonical_ace_score"] == 3 and res_age["pending_age_review_items"] == [2]
          and len(res_age["pending_age_review"]) == 2,
          "no-age findings are listed; only items not otherwise present count as withheld")

    # Empty FE scores clean 0
    w3_empty = {
        "fire_expression": "(no ACE-valid events — empty FE)",
        "alphabet_mapping": {}, "alphabet_inventory": {}, "resolved_states": [],
        "feedback_loops": [],
    }
    res_empty = score_from_expression(w3_empty)
    check(res_empty["canonical_ace_score"] == 0 and not res_empty["unmapped_symbols"],
          "empty-FE sentinel scores a clean 0 with no junk unmapped symbols")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    """CLI entry point: load a Worker 3 ACE output file, compute the
    scorecard and clinical flags, print the full report, and write it
    to --output."""
    p = argparse.ArgumentParser(
        description="FiRE Worker 4 (ACE): parse the ACE expression and count distinct "
                    "items (no model call)")
    p.add_argument("input", nargs="?", help="Worker 3 ACE output (worker3_ace_output.json)")
    p.add_argument("--output", "-o", default="ace_scorecard.json",
                   help="Output path (default: ace_scorecard.json)")
    p.add_argument("--patient-id", default=None,
                   help="Identifier for the report (default: the input's patient_id)")
    p.add_argument("--selftest", action="store_true",
                   help="Run offline regression checks and exit")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.input:
        p.error("input is required (unless using --selftest)")

    try:
        with open(args.input, encoding="utf-8") as f:
            w3 = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error("Cannot load input: %s", e); sys.exit(1)

    if "fire_expression" not in w3:
        logging.error("Input has no 'fire_expression' — this scorer reads the "
                      "Worker 3 ACE output (e.g. worker3_ace_output.json), not Worker 2.")
        sys.exit(1)
    if w3.get("signal") and w3.get("signal") != "ACE":
        logging.warning("Input signal is '%s', expected 'ACE'. Proceeding anyway.",
                        w3.get("signal"))

    patient_id = args.patient_id or w3.get("patient_id", "unknown")
    logging.info("Patient: %s", patient_id)

    result = score_from_expression(w3)
    feedback = feedback_notes(w3.get("feedback_loops", []))

    sep = "=" * 64
    print("\n" + sep)
    print(f"  WORKER 4 ACE — SCORE CARD — {patient_id}")
    print(sep)
    print(f"  Expression : {result['fire_expression']}")
    print(sep)
    print("  Symbol -> Cluster -> ACE item  (exponent weight = reference only):")
    for row in result["symbol_mapping"]:
        item = row["ace_item"]
        item_str = f"ACE-{item}" if item is not None else "UNMAPPED"
        print(f"    {row['symbol']:>3s}  {row['cluster_name']:<34s} -> {item_str:<9s}"
              f"  weight={row['exponent_weight']}  states={','.join(row['contributing_states'][:3])}")
    if result["unmapped_symbols"]:
        print("  ! Unmapped symbols (cluster matched no ACE category — NOT scored):")
        for u in result["unmapped_symbols"]:
            print(f"      {u['symbol']}: {u['cluster']}")
    if result["mapping_conflicts"]:
        print("  ! Mapping conflicts (deterministic vs Worker 3 hint):")
        for c in result["mapping_conflicts"]:
            print(f"      {c['symbol']} ({c['cluster']}): det=ACE-{c['deterministic_item']} "
                  f"vs W3={c['worker3_items']}")
    print(sep)
    print("  Per-item presence (1 point each if present):")
    for n in range(1, 15):
        rec = result["ace_scorecard"][str(n)]
        mark = "\u2713" if rec["present"] else " "
        cl = ", ".join(rec["clusters"]) if rec["clusters"] else ""
        cl = f"  <- {cl}" if cl else ""
        wt = f"  [w={rec['exponent_weight_reference']}]" if rec["present"] else ""
        dur_flag = ""
        if rec["present"] and rec["duration_expected"] and rec["duration_confirmed"] is False:
            dur_flag = "  [DURATION UNCONFIRMED — review]"
        print(f"    [{mark}] ACE-{n:<2d} {rec['category']:<3s}{wt}{cl}{dur_flag}")
    print("-" * 64)
    print(f"  ACE SCORE           : {result['canonical_ace_score']}/14")
    print(f"  Items present       : {result['items_present']}")
    print("  Factors present     : " +
          (", ".join(f"{cat}={v}" for cat, v in result["category_breakdown"].items() if v)
           or "none"))
    if result["duration_unconfirmed_items"]:
        print(f"  ! Duration-unconfirmed items (present, but no contributing state "
              f"shows a 2+ year span or chronic language): {result['duration_unconfirmed_items']}")
    if result["pending_age_review"]:
        print(f"  ! Pending age review (no age in the notes; NOT scored until a "
              f"psychotherapist confirms the age): "
              f"{len(result['pending_age_review'])} finding(s); items not otherwise present: "
              f"{result['pending_age_review_items'] or 'none'}")
    if feedback:
        print("  Feedback (same construct recurring after a gap; reference only, not scored):")
        for fb in feedback:
            print(f"    - {fb['cluster_name']} ({fb.get('state_code')}): "
                  f"{fb.get('earlier_interval')} -> {fb.get('later_interval')}")
    print(sep + "\n")

    out = {
        "patient_id": patient_id,
        "input_file": str(args.input),
        "scoring_method": "expression_driven_classic_ace",
        **result,
        "feedback_notes": feedback,
    }
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logging.info("Score card written: %s", args.output)
    except OSError as e:
        logging.error("Write failed: %s", e); sys.exit(1)


if __name__ == "__main__":
    main()
