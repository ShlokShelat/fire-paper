"""FiRE Pipeline — Worker 4 ACE: expression-driven classic ACE scorer (0..16).

Usage:
  python3 fire_worker4_ace.py p10_ace_w3.json -o p10_ace_score.json
  python3 fire_worker4_ace.py p10_ace_w3.json -o out.json --verbose
  python3 fire_worker4_ace.py --selftest
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# ACE DEFINITIONS (canonical 16-item set)
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
    10: "Did a household member go to prison?",
    11: "Did the child live 2+ years in a dangerous neighbourhood or witness community assault?",
    12: "Did the child's mother, father, guardian, or primary-caregiver relative die?",
    13: "Did other kids or siblings often hit, threaten, pick on, or insult the child?",
    14: "Did the child often feel lonely, rejected, or isolated from peers?",
    15: "Was the family very poor or on public assistance for 2+ years?",
    16: "Did the child's parents have physical or verbal fights with each other?",
}

ACE_CATEGORIES = {
    "abuse":     [1, 2, 3],
    "neglect":   [4, 5],
    "household": [6, 7, 8, 9, 10],
    "expanded":  [11, 12, 13, 14, 15, 16],
}

PRESENCE_ONLY_ITEMS = {6, 10, 12}
DURATION_ITEMS      = {11, 15}

MIN_DURATION_SPAN     = 1
MIN_DURATION_EXPONENT = 2

HIGH_RISK_THRESHOLD = 7
HIGH_RISK_NOTE = (
    "Adults reporting seven or more ACEs show roughly a thirtyfold increase "
    "in the odds of attempted suicide (Dube et al. 2001). This is the "
    "specific threshold the paper treats as clinically pivotal for escalation."
)

# ─────────────────────────────────────────────────────────────
# CLUSTER -> ACE ITEM BINDING
# ─────────────────────────────────────────────────────────────

CLUSTER_ACE_ITEM = {
    "emotional abuse":               1,
    "verbal abuse":                  1,
    "verbal/emotional abuse":        1,
    "emotional/verbal abuse":        1,
    "psychological abuse":           1,
    "physical abuse":                2,
    "sexual abuse":                  3,
    "sexual molestation":            3,
    "inappropriate touch":           3,
    "emotional neglect":             4,
    "emotional unavailability":      4,
    "felt unloved":                  4,
    "feeling unwanted":              4,
    "feeling unloved":               4,
    "unwanted child":                4,
    "physical neglect":              5,
    "material neglect":              5,
    "parental separation":           6,
    "separation or divorce":         6,
    "parental divorce":              6,
    "divorce":                       6,
    "domestic violence against mother": 7,
    "domestic violence":             7,
    "abuse of mother":               7,
    "violence against mother":       7,
    "household substance":           8,
    "substance abuse":               8,
    "substance use":                 8,
    "alcoholism":                    8,
    "alcoholic":                     8,
    "parental alcoholism":           8,
    "problem drinking":              8,
    "drug use":                      8,
    "household mental illness":      9,
    "household mental":              9,
    "mental illness or suicide":     9,
    "mental illness":                9,
    "parental depression":           9,
    "household suicide":             9,
    "household incarceration":       10,
    "incarceration":                 10,
    "imprisonment":                  10,
    "prison":                        10,
    "jail":                          10,
    "community violence":            11,
    "dangerous neighbourhood":       11,
    "dangerous neighborhood":        11,
    "neighbourhood violence":        11,
    "neighborhood violence":         11,
    "death of parent":               12,
    "death of guardian":             12,
    "death of parent or guardian":   12,
    "death of caregiver":            12,
    "death of mother":               12,
    "death of father":               12,
    "parental death":                12,
    "loss of parent":                12,
    "loss of mother":                12,
    "loss of father":                12,
    "peer or sibling bullying":      13,
    "sibling bullying":              13,
    "peer bullying":                 13,
    "bullying":                      13,
    "peer rejection":                14,
    "peer rejection or isolation":   14,
    "rejection or isolation":        14,
    "social isolation":              14,
    "peer isolation":                14,
    "loneliness":                    14,
    "isolation":                     14,
    "childhood poverty":             15,
    "material deprivation":          15,
    "financial hardship":            15,
    "economic hardship":             15,
    "poverty":                       15,
    "parental conflict":             16,
    "interparental conflict":        16,
    "parental fighting":             16,
    "parental fights":               16,
    "parents fighting":              16,
    "parents fought":                16,
    "verbal conflict":               16,
    "marital conflict":              16,
    "spousal conflict":              16,
}


def cluster_to_ace_item(cluster_name: str):
    """Resolve a cluster name to its ACE item number, or None. Longest token wins."""
    cn = str(cluster_name or "").lower().strip()
    if not cn:
        return None
    for token in sorted(CLUSTER_ACE_ITEM, key=len, reverse=True):
        if token in cn:
            return CLUSTER_ACE_ITEM[token]
    _death = any(w in cn for w in ("death", "died", "passed away", "passing", "deceased"))
    _parent = any(w in cn for w in ("parent", "mother", "father", "guardian",
                                    "caregiver", "mom", "dad"))
    if _death and _parent:
        return 12
    _conflict = any(w in cn for w in ("conflict", "fight", "fought", "arguing",
                                      "argument", "quarrel"))
    _between_parents = ("parent" in cn or "marital" in cn or "spousal" in cn
                        or "between" in cn)
    if _conflict and _between_parents:
        return 16
    return None


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
    """Classic ACE score (0..16) from Worker 3 ACE output. Each distinct ACE
    category present counts 1. Exponents are reference weights only."""
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

    for sym in appearances:
        cluster = alphabet_mapping.get(sym) or \
                  (inventory.get(sym, {}) or {}).get("cluster_name") or ""
        if not cluster and sym not in inventory and sym not in alphabet_mapping:
            continue
        det_item = cluster_to_ace_item(cluster)
        w3_items = (inventory.get(sym, {}) or {}).get("ace_items") or []
        w3_item = w3_items[0] if len(w3_items) == 1 else (w3_items or None)

        item = det_item
        if det_item is not None and w3_items and det_item not in w3_items:
            mapping_conflicts.append({
                "symbol": sym, "cluster": cluster,
                "deterministic_item": det_item, "worker3_items": w3_items})
        if item is None and w3_items:
            if isinstance(w3_item, int):
                item = w3_item
            elif isinstance(w3_items, list) and w3_items:
                item = int(w3_items[0])
                mapping_conflicts.append({
                    "symbol": sym, "cluster": cluster,
                    "deterministic_item": None, "worker3_items": w3_items,
                    "note": "cluster matched no ACE token; resolved to first hint "
                            "(multi-item hint — verify construct purity upstream)"})

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

    scorecard: "OrderedDict[int, dict]" = OrderedDict()
    duration_unconfirmed_items = []
    for n in range(1, 17):
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
        "canonical_risk": canonical_risk(canonical),
        "crosses_high_risk_threshold": canonical >= HIGH_RISK_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "high_risk_threshold_note": HIGH_RISK_NOTE if canonical >= HIGH_RISK_THRESHOLD else "",
        "items_present": sorted(n for n, r in scorecard.items() if r["present"]),
        "category_breakdown": category_breakdown,
        "ace_scorecard": {str(n): r for n, r in scorecard.items()},
        "symbol_mapping": symbol_rows,
        "exponent_weights_reference": dict(weights),
        "unmapped_symbols": unmapped_symbols,
        "mapping_conflicts": mapping_conflicts,
        "duration_unconfirmed_items": duration_unconfirmed_items,
        "fire_expression": fire_expression,
    }


def _category_of(n: int) -> str:
    for cat, items in ACE_CATEGORIES.items():
        if n in items:
            return cat
    return "other"


def canonical_risk(ace_count: int) -> str:
    """Standard ACE-count risk bands."""
    if ace_count == 0:  return "MINIMAL"
    if ace_count <= 1:  return "LOW"
    if ace_count <= 3:  return "MODERATE"
    if ace_count <= 5:  return "HIGH"
    if ace_count == 6:  return "VERY HIGH"
    return "VERY HIGH — CRITICAL (>=7, see paper's cited threshold)"


# ─────────────────────────────────────────────────────────────
# CLINICAL REPORTING
# ─────────────────────────────────────────────────────────────

def build_flags(scorecard: dict, feedback_loops: list) -> dict:
    """Interaction flags for known high-risk ACE co-occurrences, plus
    feedback loop annotations (reference only, not added to ACE total)."""
    def present(n):
        return scorecard[str(n)]["present"]

    interaction = []
    if present(1) and present(4):
        interaction.append({"rule": "ACE-1 + ACE-4",
            "implication": "Emotional abuse + emotional neglect — elevated complex-PTSD risk"})
    if present(3) and present(7):
        interaction.append({"rule": "ACE-3 + ACE-7",
            "implication": "Sexual abuse + witnessed domestic violence — elevated dissociation risk"})
    if present(7) and present(16):
        interaction.append({"rule": "ACE-7 + ACE-16",
            "implication": "Pervasive household violence — chronic hypervigilance expected"})
    if present(1) and present(7) and present(16):
        interaction.append({"rule": "ACE-1 + 7 + 16",
            "implication": "Triple household adversity — very high complex-trauma burden"})

    feedback_f = []
    for fb in (feedback_loops or []):
        feedback_f.append({
            "cluster": fb.get("cluster"),
            "cluster_name": fb.get("cluster_name", ""),
            "state_code": fb.get("state_code"),
            "earlier_interval": fb.get("earlier_interval"),
            "later_interval": fb.get("later_interval"),
            "evidence": fb.get("evidence"),
            "clinical_significance":
                "Childhood trauma reactivated at a later age — reflected as added "
                "weight on the recurring node (exponent), not as an extra point in "
                "the ACE total (ACE is a childhood-only count).",
        })
    return {"interaction_flags": interaction, "feedback_loop_flags": feedback_f}


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # Paper's worked example: EACE = (a+b+c^2)·d·(e+a^3), score = 5
    w3 = {
        "fire_expression": "[C] (a+b+c^2) . d . (a^3+e)",
        "alphabet_mapping": {
            "a": "Emotional Abuse", "b": "Emotional Neglect", "c": "Peer Bullying",
            "d": "Death of Parent or Guardian", "e": "Physical Abuse",
        },
        "alphabet_inventory": {
            "a": {"cluster_name": "Emotional Abuse", "ace_items": [1],
                 "contributing_states": ["st-a"], "confidence": 0.9},
            "b": {"cluster_name": "Emotional Neglect", "ace_items": [4],
                 "contributing_states": ["st-b"], "confidence": 0.9},
            "c": {"cluster_name": "Peer Bullying", "ace_items": [13],
                 "contributing_states": ["st-c"], "confidence": 0.9},
            "d": {"cluster_name": "Death of Parent or Guardian", "ace_items": [12],
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
    check(result["canonical_ace_score"] == 5,
          "PAPER MATCH: worked example scores exactly 5 (items 1,4,13,12,2)")
    check(set(result["items_present"]) == {1, 2, 4, 12, 13},
          "PAPER MATCH: exactly the items {1,2,4,12,13} Table 1 lists")
    check(result["exponent_weights_reference"]["a"] == 4
          and result["exponent_weights_reference"]["c"] == 2,
          "exponent weights carried as reference (a=4, c=2), not affecting the count")

    check(result["crosses_high_risk_threshold"] is False,
          "score of 5 does not cross the ACE>=7 threshold")
    check(canonical_risk(6) == "VERY HIGH" and canonical_risk(7).startswith("VERY HIGH — CRITICAL"),
          "6 and 7 are distinguishable risk bands")

    # Duration: single-year state flagged but not dropped
    w3_dur = {
        "fire_expression": "[C] k",
        "alphabet_mapping": {"k": "Community Violence"},
        "alphabet_inventory": {"k": {"cluster_name": "Community Violence", "ace_items": [11],
                                    "contributing_states": ["dyn-005"], "confidence": 0.8}},
        "resolved_states": [{"state_code": "dyn-005", "onset_age": 10, "end_age": 10, "exponent": 1}],
        "feedback_loops": [],
    }
    res_dur = score_from_expression(w3_dur)
    check(res_dur["canonical_ace_score"] == 1,
          "single-year ACE-11 state is NOT silently dropped from the count")
    check(11 in res_dur["duration_unconfirmed_items"],
          "single-year ACE-11 state IS flagged as duration-unconfirmed")
    check(res_dur["ace_scorecard"]["11"]["duration_confirmed"] is False,
          "scorecard row carries duration_confirmed=False")

    w3_dur2 = {
        "fire_expression": "[C] k",
        "alphabet_mapping": {"k": "Community Violence"},
        "alphabet_inventory": {"k": {"cluster_name": "Community Violence", "ace_items": [11],
                                    "contributing_states": ["dyn-006"], "confidence": 0.8}},
        "resolved_states": [{"state_code": "dyn-006", "onset_age": 5, "end_age": 9, "exponent": 1}],
        "feedback_loops": [],
    }
    res_dur2 = score_from_expression(w3_dur2)
    check(11 not in res_dur2["duration_unconfirmed_items"],
          "genuine multi-year span (5-9) is confirmed, no flag raised")

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
    p = argparse.ArgumentParser(
        description="FiRE Worker 4 ACE — expression-driven classic ACE scorer")
    p.add_argument("input", nargs="?", help="Worker 3 ACE output (p10_ace_w3.json)")
    p.add_argument("--output", "-o", default="ace_scorecard.json")
    p.add_argument("--patient-id", default=None)
    p.add_argument("--selftest", action="store_true",
                   help="Run offline regression checks and exit")
    p.add_argument("--verbose", action="store_true")
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
                      "Worker 3 ACE output (e.g. p10_ace_w3.json), not Worker 2.")
        sys.exit(1)
    if w3.get("signal") and w3.get("signal") != "ACE":
        logging.warning("Input signal is '%s', expected 'ACE'. Proceeding anyway.",
                        w3.get("signal"))

    patient_id = args.patient_id or w3.get("patient_id", "unknown")
    logging.info("Patient: %s", patient_id)

    result = score_from_expression(w3)
    flags = build_flags(result["ace_scorecard"], w3.get("feedback_loops", []))

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
        print("  \u26a0 Unmapped symbols (cluster matched no ACE category — NOT scored):")
        for u in result["unmapped_symbols"]:
            print(f"      {u['symbol']}: {u['cluster']}")
    if result["mapping_conflicts"]:
        print("  \u26a0 Mapping conflicts (deterministic vs Worker 3 hint):")
        for c in result["mapping_conflicts"]:
            print(f"      {c['symbol']} ({c['cluster']}): det=ACE-{c['deterministic_item']} "
                  f"vs W3={c['worker3_items']}")
    print(sep)
    print("  Per-item presence (1 point each if present):")
    for n in range(1, 17):
        rec = result["ace_scorecard"][str(n)]
        mark = "\u2713" if rec["present"] else " "
        cl = ", ".join(rec["clusters"]) if rec["clusters"] else ""
        cl = f"  <- {cl}" if cl else ""
        wt = f"  [w={rec['exponent_weight_reference']}]" if rec["present"] else ""
        dur_flag = ""
        if rec["present"] and rec["duration_expected"] and rec["duration_confirmed"] is False:
            dur_flag = "  [DURATION UNCONFIRMED — review]"
        print(f"    [{mark}] ACE-{n:<2d} {rec['category']:<10s}{wt}{cl}{dur_flag}")
    print("-" * 64)
    print(f"  CANONICAL ACE SCORE : {result['canonical_ace_score']}/16   "
          f"({result['canonical_risk']} risk)")
    print(f"  Items present       : {result['items_present']}")
    print("  Categories present  : " +
          (", ".join(f"{cat}={v}" for cat, v in result["category_breakdown"].items() if v)
           or "none"))
    if result["crosses_high_risk_threshold"]:
        print(f"  \u26a0\u26a0 CROSSES ACE>=7 THRESHOLD: {result['high_risk_threshold_note']}")
    if result["duration_unconfirmed_items"]:
        print(f"  \u26a0 Duration-unconfirmed items (present, but no contributing state "
              f"shows a 2+ year span or chronic language): {result['duration_unconfirmed_items']}")
    if flags["interaction_flags"]:
        print("  Interaction flags:")
        for fl in flags["interaction_flags"]:
            print(f"    - {fl['rule']}: {fl['implication']}")
    if flags["feedback_loop_flags"]:
        print("  Feedback (childhood wounds reactivated — reference, not scored):")
        for fb in flags["feedback_loop_flags"]:
            print(f"    - {fb['cluster_name']} ({fb.get('state_code')}): "
                  f"{fb.get('earlier_interval')} -> {fb.get('later_interval')}")
    print(sep + "\n")

    out = {
        "patient_id": patient_id,
        "input_file": str(args.input),
        "scoring_method": "expression_driven_classic_ace",
        **result,
        **flags,
    }
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logging.info("Score card written: %s", args.output)
    except OSError as e:
        logging.error("Write failed: %s", e); sys.exit(1)


if __name__ == "__main__":
    main()
