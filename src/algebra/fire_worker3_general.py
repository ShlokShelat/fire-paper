"""
FiRE Pipeline — Worker 3: FiRE Expression Builder

Paper concept: this file implements the trajectory-assembly half of the
paper's stage (iv), i.e. Expression Construction — turning the resolved
state-space codes from stage (ii) into the symbolic FiRE Expression
(the paper's algebra of union `+`, concatenation `.`, self-loop
exponents, and feedback) via the finite-automaton-style graph built
here. Instrument-specific Projection and Scoring are separate,
downstream stages (see fire_worker3_ace.py / fire_worker4_ace.py).

Pipeline position: run after Worker 2 (fire_worker2.py) has resolved
events to state-space codes, and before an instrument-specific
projection (fire_worker3_ace.py for ACE).
Builds the complete patient FiRE Expression (all resolved clinical states)
using a finite-automaton-style graph. This is the general Worker 3 file,
generalised to run over all clinical states without instrument-specific gating.

Main input: a Worker 2 output file (events with resolved state_codes,
onset/end ages). Main output: a FiRE Expression string plus the DFA
(nodes/edges) it was traversed from, written to the --output JSON.
Key assumption: state resolution (mapping vs. new-state creation) is
already final by this point — this stage only assembles temporal
structure (tiers, unions, self-loops, feedback) from already-resolved
states; it does not re-decide what a state maps to.

The run also writes <output>.html, an interactive view of the automaton.
The page loads the vis-network library from cdnjs.cloudflare.com when it is
opened, unless a file vis-network-bundle.js is placed next to this script,
in which case the library is inlined and the page works offline. Nothing is
fetched while the script runs.

Usage:
  python3 fire_worker3_general.py final_mapped_events.json -o worker3_output.json
  python3 fire_worker3_general.py final_mapped_events.json --exclude-provisional-states --verbose
  python3 fire_worker3_general.py --selftest
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 0.  CONSTANTS
# ─────────────────────────────────────────────────────────────

# Full life-stage range (not windowed to any one instrument), plus a dedicated
# tier for states with no age evidence — placed at the end in document order.
TIER_ORDER = ["childhood", "teenage", "adult", "mid-life", "no-age"]
TIER_MARK  = {"childhood": "[C]", "teenage": "[T]", "adult": "[A]",
              "mid-life": "[M]", "no-age": "[N]"}

FREQ_EXPONENT = {
    "single": 1, "recurring": 2, "repeated": 3,
    "chronic": 4, "throughout": 4, "unknown": 1,
}
EXPONENT_CAP = 4

# Repetition (self-loop) language
REPETITION_LANGUAGE = [
    "repeated", "repeatedly", "multiple times", "many times", "over and over",
    "again and again", "numerous times", "several times", "time and again",
    "constantly", "continually", "persistently", "chronic", "throughout",
    "every day", "daily", "regularly", "frequently", "on and off",
]

# Reactivation language
REACTIVATION_LANGUAGE = [
    "reminded her of", "reminded him of", "reminds her of", "reminds him of",
    "triggered memories of", "brought back", "felt like childhood again",
    "felt like childhood", "reactivat", "resurfac", "echoes of",
    "same feeling as", "just like when",
]

SIMULTANEITY_MARKERS = [
    "at the same time", "simultaneously", "also during this period",
    "during the same years", "during the same period", "meanwhile",
    "in parallel", "alongside",
]

_NUM_WORDS = {
    "one": 1, "once": 1, "twice": 2, "two": 2, "three": 3, "thrice": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_LATIN = list("abcdefghijklmnopqrstuvwxyz")
_GREEK = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
]
SYMBOL_POOL = _LATIN + _GREEK

# Worker 2 sub-event status mapping to match types
_W2_STATUS_MAP = {
    "MAPPED":                      "RESOLVED",
    "MAPPED_DEDUP_JUDGE":          "RESOLVED",
    "MAPPED_GLOBAL_DEDUP":         "RESOLVED",
    "MAPPED_CONCURRENT":           "RESOLVED",
    "NEW_STATE_CREATED":           "RESOLVED",
    "NEUTRAL":                     "NOT_ADVERSE",
    "UNRESOLVED":                  "UNMATCHED",
    "JUDGE_FAILED":                "UNMATCHED",
    "UNKNOWN":                     "UNMATCHED",
    "DRY_RUN":                     "UNMATCHED",
    "UNGROUNDED":                  "UNMATCHED",
    "REJECTED_STATE_NEEDS_REMAP":  "UNMATCHED",
}


# ─────────────────────────────────────────────────────────────
# AGE / TIER HELPERS
# ─────────────────────────────────────────────────────────────

def _coerce_age(val) -> Optional[int]:
    """Coerce an age value to int, or None if not usable. Handles ints, numeric
    strings ('22', ' 5 '), and floats (5.0)."""
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        m = re.match(r"^-?\d+", s)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
    return None


def age_to_tier(age: int) -> str:
    """Map age to life stage tier."""
    if age <= 12:
        return "childhood"
    if age <= 19:
        return "teenage"
    if age <= 59:
        return "adult"
    return "mid-life"


# ─────────────────────────────────────────────────────────────
# COMPONENT 0 — INPUT ADAPTER
# ─────────────────────────────────────────────────────────────

def infer_frequency(text: str) -> str:
    """Classify how often an event recurred (single/recurring/repeated/
    throughout) from its combined source+event text, via keyword cues.
    Feeds `component2_exponent`'s self-loop/repetition decision."""
    t = text.lower()
    if any(w in t for w in [
        "throughout", "entire childhood", "all through", "whole childhood",
        "chronic", "always", "constant", "persistent", "ongoing", "lifelong",
        "never stopped", "for many years", "for several years",
        "years and years", "from age"]):
        return "throughout"
    if any(w in t for w in [
        "repeated", "repeatedly", "multiple times", "many times", "over and over",
        "again and again", "numerous times", "several times", "three times",
        "four times", "five times"]):
        return "repeated"
    if any(w in t for w in [
        "recurring", "recurred", "sometimes", "periodically", "occasionally",
        "on several", "more than once", "twice", "two occasions"]):
        return "recurring"
    return "single"


def _own_text(e: dict) -> str:
    """The text that belongs to one sub-event alone: its supporting quote and
    its own event text, plus the source sentence only when no other
    sub-event shares that sentence. Worker 2 gives every sub-event of a
    source line the same source_sentence, so reading repetition or
    reactivation language from the shared sentence would leak one
    sub-event's wording (e.g. "bullied frequently") onto its siblings."""
    parts = [e.get("supporting_quote") or "", e.get("event") or ""]
    if not e.get("source_shared"):
        parts.append(e.get("source_sentence") or "")
    return " ".join(p for p in parts if p)


def parse_worker2_output(data: dict) -> tuple:
    """
    Pull every resolved sub-event from a Worker 2 output.

    Returns (resolved, new_states, others):
      resolved   — RESOLVED states mapped to an existing state
      new_states — RESOLVED states Worker 2 created as provisional; graphed by
                   default (the paper uses a provisional state immediately in
                   the current individual's trajectory), excluded only with
                   --exclude-provisional-states
      others     — NOT_ADVERSE / UNMATCHED: reported, never graphed
    """
    resolved, new_states, others = [], [], []
    for ev_idx, ev in enumerate(data.get("events") or []):
        if ev.get("status") != "COMPOUND":
            continue
        life_stage  = str(ev.get("life_stage") or "unknown").lower()
        line_number = int(ev.get("line_number") or 0)
        source      = str(ev.get("source_sentence") or ev.get("event") or "")
        source_shared = sum(1 for s in (ev.get("sub_events") or [])
                            if str(s.get("event") or "").strip()) > 1

        for sub_idx, sub in enumerate(ev.get("sub_events") or []):
            sub_status   = str(sub.get("status") or "UNKNOWN")
            match_type   = _W2_STATUS_MAP.get(sub_status, "UNMATCHED")
            cluster_name = str(sub.get("cluster_name") or "Unknown")
            state_code   = str(sub.get("state_code") or "")
            similarity   = float(sub.get("similarity") or 0.0)
            event_text   = str(sub.get("event") or "")
            exp_type     = str(sub.get("experience_type") or "unknown")
            reason       = str(sub.get("mapping_reason") or "")
            neutral_r    = str(sub.get("neutral_reason") or "")
            is_new_state = (sub_status == "NEW_STATE_CREATED")

            if is_new_state:
                confidence = 0.80
            elif match_type == "RESOLVED":
                confidence = round(similarity, 3) if similarity > 0 else 0.50
            else:
                confidence = 0.0

            own_text = _own_text({"supporting_quote": sub.get("supporting_quote"),
                                  "event": event_text, "source_sentence": source,
                                  "source_shared": source_shared})
            onset_i = _coerce_age(sub.get("onset_age"))
            end_i   = _coerce_age(sub.get("end_age"))

            # Unique per sub-event: Worker 2's sub_id is not unique in practice
            # (several sub-events of one unit can share it), and verification
            # removes findings by this id, so it must name exactly one sub-event.
            _sub_id = str(sub.get("sub_id") or "").strip()
            event_uid = f"{_sub_id or state_code or 'sub'}@e{ev_idx}.s{sub_idx}"

            record = {
                "state_code":          state_code,
                "event_uid":           event_uid,
                "sub_id":              _sub_id or None,
                "cluster_name":        cluster_name,
                "cluster":             "",
                "match_type":          match_type,
                "combined_confidence": confidence,
                "is_new_state":        is_new_state,
                "mapped_state_status": sub.get("mapped_state_status"),
                "life_stage":          life_stage,
                "line_number":         line_number,
                "onset_age":           onset_i,
                "end_age":             end_i,
                "is_ongoing":          bool(sub.get("is_ongoing", False)),
                "is_recollection":     bool(sub.get("is_recollection", False)),
                "recollection_reference": sub.get("recollection_reference"),
                "source_sentence":     source,
                "source_shared":       source_shared,
                "event":               event_text,
                "experience_type":     exp_type,
                "frequency":           infer_frequency(own_text),
                "mapping_reason":      reason,
                "sub_status":          sub_status,
                "unmatched_reason":    reason if match_type == "UNMATCHED" else "",
                "not_adverse_reason":  neutral_r,
                "supporting_quote":    sub.get("supporting_quote"),
                "quote_grounded":      sub.get("quote_grounded"),
                "signal":              sub.get("signal"),
                "signals":             sub.get("signals") or ([sub.get("signal")] if sub.get("signal") else []),
                "ace_items_hint":      sub.get("ace_items_hint"),
                "ace14_items_hint":    sub.get("ace14_items_hint"),
                "perpetrator":         sub.get("perpetrator"),
                "perpetrator_class":   sub.get("perpetrator_class"),
                "recency":             sub.get("recency"),
                "role":                sub.get("role"),
            }

            if match_type == "RESOLVED" and is_new_state:
                new_states.append(record)
            elif match_type == "RESOLVED":
                resolved.append(record)
            else:
                others.append(record)
    return resolved, new_states, others


# ─────────────────────────────────────────────────────────────
# COMPONENT 0.5 — ALPHABET ASSIGNER
# ─────────────────────────────────────────────────────────────

def assign_alphabets(records: list) -> dict:
    """One stable symbol per cluster, ordered by first chronological onset."""
    def _key(r):
        tier_idx = TIER_ORDER.index(r["life_stage"]) if r["life_stage"] in TIER_ORDER else 99
        onset = r.get("onset_age")
        onset = onset if onset is not None else 9999
        return (onset, tier_idx, r.get("line_number", 0))
    mapping, pool_idx = {}, 0
    for r in sorted(records, key=_key):
        cn = r["cluster_name"]
        if cn not in mapping:
            sym = (SYMBOL_POOL[pool_idx] if pool_idx < len(SYMBOL_POOL)
                   else f"z{pool_idx - len(SYMBOL_POOL)}")
            mapping[cn] = sym
            pool_idx += 1
    return mapping


# ─────────────────────────────────────────────────────────────
# INTERVAL ALGEBRA
# ─────────────────────────────────────────────────────────────

def resolve_onset_end(r: dict) -> tuple:
    """
    Returns (onset, end, has_real_span).
    Resolves explicit onset/end boundaries and ongoing states.
    """
    onset = r.get("onset_age")
    end   = r.get("end_age")

    if onset is None:
        return None, None, False

    if isinstance(end, int):
        pass
    elif r.get("is_ongoing"):
        end = onset
    else:
        end = onset

    try:
        o, e = int(onset), int(end)
        if e < o:
            o, e = e, o
        return o, e, (e > o)
    except (TypeError, ValueError):
        return None, None, False


def _has_repetition_language(entries: list) -> bool:
    """True if any entry's own text (see `_own_text`) uses repetition language."""
    text = " ".join(_own_text(e) for e in entries).lower()
    return any(w in text for w in REPETITION_LANGUAGE)


def _self_loops(entries: list) -> bool:
    """Self-loop fires only on explicit repetition language."""
    return _has_repetition_language(entries)


def _collapse_same_state_to_widest(records: list) -> list:
    """De-duplicate records sharing a state code only when one span is nested
    in another (true duplicate of one continuous exposure). Disjoint spans of
    the same state code both survive as separate nodes."""
    def _nested_or_equal(inner, outer):
        return outer["_onset"] <= inner["_onset"] and inner["_end"] <= outer["_end"]

    groups: "OrderedDict[str, list]" = OrderedDict()
    passthrough = []
    for r in records:
        code = r.get("state_code")
        if not code or r.get("_onset") is None or r.get("_end") is None:
            passthrough.append(r)
            continue
        groups.setdefault(code, []).append(r)

    out = []
    for code, recs in groups.items():
        recs_sorted = sorted(
            recs,
            key=lambda x: (-(x["_end"] - x["_onset"]),
                           -float(x.get("combined_confidence") or 0),
                           int(x.get("line_number") or 0)),
        )
        kept = []
        for r in recs_sorted:
            subsumed = any(_nested_or_equal(r, k) for k in kept)
            if not subsumed:
                kept.append(r)
        out.extend(kept)
    return out + passthrough


def build_interval_timeline(records: list) -> list:
    """Build atomic-interval timeline from span records."""
    valid = [r for r in records
             if r.get("_onset") is not None and r.get("_end") is not None
             and r["_end"] >= r["_onset"]]
    if not valid:
        return []

    valid = _collapse_same_state_to_widest(valid)

    cutset = set()
    for r in valid:
        cutset.add(r["_onset"])
        cutset.add(r["_end"] + 1)
    cut = sorted(cutset)
    atomic = [(cut[i], cut[i + 1] - 1) for i in range(len(cut) - 1)]

    raw_segments = []
    for (lo, hi) in atomic:
        active = [r for r in valid if r["_onset"] <= hi and r["_end"] >= lo]
        if not active:
            continue
        # One member per cluster symbol: each cluster owns exactly one symbol,
        # so two states of the same cluster active in the same interval are one
        # occurrence of that symbol (their entries are kept), not "(a+a)".
        members: "OrderedDict[str, dict]" = OrderedDict()
        for r in sorted(active, key=lambda x: (x["_onset"], x.get("line_number", 0),
                                               x.get("_idx", 0))):
            sym = r["cluster"]
            mkey = sym
            if mkey not in members:
                members[mkey] = {"symbol": sym, "cluster_name": r["cluster_name"],
                                 "state_code": r["state_code"], "codes": [], "entries": []}
            if r["state_code"] not in members[mkey]["codes"]:
                members[mkey]["codes"].append(r["state_code"])
            members[mkey]["entries"].append(r)
        raw_segments.append({"interval": (lo, hi), "tier": age_to_tier(lo), "members": members})

    # Merge neighbouring slices with identical members, but only when they
    # touch in time: a year with no active state between them is a gap, and
    # a gap is what makes a later occurrence a reactivation (feedback).
    collapsed = []
    for seg in raw_segments:
        if collapsed:
            prev = collapsed[-1]
            if set(prev["members"].keys()) == set(seg["members"].keys()) \
               and prev["tier"] == seg["tier"] \
               and seg["interval"][0] == prev["interval"][1] + 1:
                prev["interval"] = (prev["interval"][0], seg["interval"][1])
                for mkey, m in seg["members"].items():
                    pm = prev["members"][mkey]
                    for c in m["codes"]:
                        if c not in pm["codes"]:
                            pm["codes"].append(c)
                    pm["entries"].extend(m["entries"])
                continue
        collapsed.append({
            "interval": seg["interval"], "tier": seg["tier"],
            "members": OrderedDict(
                (mkey, {"symbol": m["symbol"], "cluster_name": m["cluster_name"],
                        "state_code": m.get("state_code"),
                        "codes": list(m["codes"]), "entries": list(m["entries"])})
                for mkey, m in seg["members"].items()),
        })
    return collapsed


def build_union_aware_tiers(working: list, alphabet_map: dict) -> "OrderedDict":
    """Assemble the ordered, tier-grouped timeline the DFA is built from.

    Input: flat resolved-state records and the cluster->symbol map.
    Output: an OrderedDict of tier -> list of timeline blocks (each a
    union group of co-occurring states, or a single state), in
    TIER_ORDER, ready for `component35_build_dfa`.
    Runs in three phases: resolve each record's onset/end span (placing
    undated states sequentially after the latest known age), cut the
    timeline into atomic intervals via `build_interval_timeline`, then
    group intervals into per-tier union blocks.
    """
    # Phase 1: resolve spans; states with no age evidence are placed
    # sequentially after the latest known age so they still get a
    # deterministic position in the timeline.
    no_age = []
    for idx, r in enumerate(working):
        r["_idx"] = idx  # sub-event order within a line, for stable union ordering
        r["cluster"] = alphabet_map.get(r["cluster_name"], r["cluster_name"][:4])
        o, e, real = resolve_onset_end(r)
        r["_onset"], r["_end"], r["_has_real_span"] = o, e, real
        if o is None:
            no_age.append(r)

    max_age = 0
    for r in working:
        if r["_end"] is not None:
            max_age = max(max_age, r["_end"])
    no_age.sort(key=lambda x: x.get("line_number", 0))
    for i, r in enumerate(no_age):
        r["_onset"] = max_age + 1 + i
        r["_end"]   = r["_onset"]
        r["_has_real_span"] = False
        r["_no_age"] = True

    # Phase 2: cut the resolved spans into atomic, non-overlapping intervals.
    segments = build_interval_timeline(working)

    # Phase 3: group each interval's co-occurring states into a tier-scoped
    # timeline block, marking multi-state intervals as union groups.
    tiers: "OrderedDict[str, list]" = OrderedDict()
    union_counter = 0
    for seg in segments:
        member_list = list(seg["members"].values())
        ugid = None
        if len(member_list) >= 2:
            ugid = f"u{union_counter}"
            union_counter += 1
        seg_is_no_age = all(
            all(e.get("_no_age") for e in m["entries"])
            for m in member_list
        )
        seg_tier = "no-age" if seg_is_no_age else seg["tier"]
        for m in member_list:
            tiers.setdefault(seg_tier, []).append({
                "cluster": m["symbol"], "cluster_name": m["cluster_name"],
                "state_code": m.get("state_code"),
                "codes": m["codes"], "entries": m["entries"],
                "tier": seg_tier, "interval": seg["interval"],
                "union_group": ugid,
            })

    ordered = OrderedDict()
    for t in TIER_ORDER:
        if t in tiers:
            ordered[t] = tiers[t]
    return ordered


# ─────────────────────────────────────────────────────────────
# COMPONENT 2 — EXPONENT (self-loop strength)
# ─────────────────────────────────────────────────────────────

def _explicit_numeric(text: str) -> Optional[int]:
    """Extract the largest explicit occurrence count in text (e.g. "three
    times", "4-5 occasions"), or None if none is stated."""
    t, best = text.lower(), None
    for m in re.finditer(r"(\d+)\s*(?:[-–]|to)\s*(\d+)\s*(?:occasions?|times?|incidents?)", t):
        best = max(best or 0, int(m.group(2)))
    num_pat = "|".join(re.escape(k) for k in _NUM_WORDS)
    for m in re.finditer(r"\b(\d+|%s)\s*(?:times?|occasions?|incidents?)\b" % num_pat, t):
        v = m.group(1)
        n = int(v) if v.isdigit() else _NUM_WORDS.get(v, 0)
        best = max(best or 0, n)
    return best


def component2_exponent(block: dict) -> dict:
    """Self-loop exponent of one timeline block.

    The self-loop fires only on explicit repetition language in an entry's
    own text (REPETITION_LANGUAGE, e.g. "frequently"), never on the length
    of a span (Supplementary Material, Section S1): exponent 2 with a
    self-loop, else 1. Feedback adds +2 separately (detect_feedbacks). The
    frequency tag, span length and explicit counts are recorded under
    "signals" for inspection only; they do not change the exponent."""
    entries = block["entries"]
    is_loop = _self_loops(entries)

    freq_exps = [FREQ_EXPONENT.get((e.get("frequency") or "unknown").lower(), 1) for e in entries]
    sig_freq  = max(freq_exps) if freq_exps else 1

    span_strength = 1
    for e in entries:
        if e.get("_has_real_span") and e.get("_onset") is not None and e.get("_end") is not None:
            yrs = e["_end"] - e["_onset"] + 1
            span_strength = max(span_strength, min(yrs, EXPONENT_CAP))

    sig_numeric = 0
    for e in entries:
        n = _explicit_numeric(_own_text(e))
        if n:
            sig_numeric = max(sig_numeric, n)

    exponent = 2 if is_loop else 1
    rng = [2, 2] if is_loop else [1, 1]

    confs = [e.get("combined_confidence", 0.5) for e in entries]
    return {
        "exponent": exponent, "range": rng, "is_self_loop": is_loop,
        "confidence": round(sum(confs) / len(confs), 3) if confs else 0.5,
        "signals": {"frequency_tag": sig_freq, "span_strength": span_strength,
                    "explicit_numeric": sig_numeric,
                    "language": _has_repetition_language(entries)},
    }


# ─────────────────────────────────────────────────────────────
# EXPLICIT-MARKER UNION (augments interval union)
# ─────────────────────────────────────────────────────────────

def _block_text(block: dict) -> str:
    """Concatenate a timeline block's entries' source+event text for keyword matching."""
    return " ".join((e.get("source_sentence") or "") + " " + (e.get("event") or "")
                    for e in block["entries"]).lower()


def detect_marker_union(block_a: dict, block_b: dict) -> bool:
    """Explicit simultaneity language between two same-tier blocks forces a union."""
    text = _block_text(block_a) + " " + _block_text(block_b)
    return any(mk in text for mk in SIMULTANEITY_MARKERS)


# ─────────────────────────────────────────────────────────────
# FEEDBACK DETECTOR — structural and language-based
# ─────────────────────────────────────────────────────────────

def _reactivation_reference(block: dict) -> Optional[str]:
    """Extract what a reactivation block's text says was reactivated
    (e.g. a relation or "childhood"), for the review-only report built by
    `detect_reactivation_mentions`."""
    text = _block_text(block)
    patterns = [
        r"remind(?:ed|s)? (?:him|her|them) of (?:the |her |his |their )?([a-z][a-z\s-]{2,40})",
        r"triggered memories of (?:the |her |his |their )?([a-z][a-z\s-]{2,40})",
        r"brought back (?:memories of )?(?:the |her |his |their )?([a-z][a-z\s-]{2,40})",
        r"echoes of (?:the |her |his |their )?([a-z][a-z\s-]{2,40})",
        r"same feeling as (?:when )?([a-z][a-z\s-]{2,40})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    for kw in ("childhood", "being a child", "growing up"):
        if kw in text:
            return kw
    return None


def _block_is_no_age(b: dict) -> bool:
    """A block is no-age if all entries lack a real onset (internal `_no_age`,
    or `is_no_age` once serialised into `blocks`)."""
    return all(e.get("_no_age") or e.get("is_no_age") for e in b.get("entries", []))


def _cluster_episodes(flat: list) -> "dict[str, list]":
    """Group dated blocks by cluster symbol into episodes: maximal runs of
    blocks whose intervals are contiguous or overlapping. A gap of at least
    one year with the cluster absent starts a new episode."""
    by_cluster: "dict[str, list]" = defaultdict(list)
    for b in flat:
        if not _block_is_no_age(b):
            by_cluster[b["cluster"]].append(b)
    episodes = {}
    for sym, blocks in by_cluster.items():
        runs = []
        for b in sorted(blocks, key=lambda x: (x["interval"][0], x["interval"][1])):
            if runs and b["interval"][0] <= runs[-1]["end"] + 1:
                runs[-1]["blocks"].append(b)
                runs[-1]["end"] = max(runs[-1]["end"], b["interval"][1])
            else:
                runs.append({"start": b["interval"][0], "end": b["interval"][1], "blocks": [b]})
        episodes[sym] = runs
    return episodes


def detect_feedbacks(flat: list) -> list:
    """Detect feedback: the same cluster recurring after a temporal gap.

    Each later episode of a cluster (a disjoint span after a gap) is one
    reactivation. It is linked to the cluster's earliest episode by one
    bidirectional edge pair, anchored on the first block of each episode, so
    a reactivation adds +2 exactly once, however many interval slices the
    later episode is cut into. Feedback is keyed by cluster, not by state
    code: two different states of one cluster (e.g. two betrayals) are the
    same construct recurring."""
    feedbacks = []
    for sym, runs in _cluster_episodes(flat).items():
        if len(runs) < 2:
            continue
        first = runs[0]
        earliest = first["blocks"][0]
        for run in runs[1:]:
            later = run["blocks"][0]
            later_codes = list(dict.fromkeys(c for b in run["blocks"] for c in b["codes"]))
            feedbacks.append({
                "cluster":              earliest["cluster"],
                "cluster_name":         earliest["cluster_name"],
                "trigger_cluster":      later["cluster"],
                "trigger_cluster_name": later["cluster_name"],
                "earlier_node":         node_name(earliest),
                "later_node":           node_name(later),
                "earlier_interval":     (first["start"], first["end"]),
                "later_interval":       (run["start"], run["end"]),
                "state_code":           earliest["codes"][0] if earliest["codes"] else None,
                "later_state_codes":    later_codes,
                "evidence":             ["same_cluster_recurrence_after_gap"],
                "link_confidence":      "high",
                "needs_review":         False,
                "source_sentences":     [e.get("source_sentence") for b in run["blocks"]
                                         for e in b["entries"]],
            })
    return feedbacks


def detect_reactivation_mentions(flat: list) -> list:
    """Report blocks whose own text says an earlier experience was
    re-triggered ("reminded her of ..."). Reported for review only: the
    paper's feedback operator is defined structurally (same cluster after a
    gap), so language alone never adds an edge or changes an exponent."""
    flat = [b for b in flat if not _block_is_no_age(b)]
    mentions = []
    for later in flat:
        text = " ".join(_own_text(e) for e in later["entries"]).lower()
        if not any(p in text for p in REACTIVATION_LANGUAGE):
            continue
        mentions.append({
            "node":         node_name(later),
            "cluster":      later["cluster"],
            "cluster_name": later["cluster_name"],
            "interval":     later["interval"],
            "reference":    _reactivation_reference(later),
            "source_sentences": [e.get("source_sentence") for e in later["entries"]],
        })
    return mentions


# ─────────────────────────────────────────────────────────────
# COMPONENT 3.5 — DFA GRAPH CONSTRUCTOR
# ─────────────────────────────────────────────────────────────

class DFA:
    """Minimal finite-automaton-style graph container: a node name list
    and a labeled, typed edge list (sequential/branch/join/self_loop/
    feedback_back/feedback_return/terminal), built by
    `component35_build_dfa` and consumed by `component4_traverse`."""
    def __init__(self):
        self.nodes = ["q_born", "q_end"]
        self.edges = []
    def add_node(self, name):
        """Register a node name once (no-op if already present)."""
        if name not in self.nodes:
            self.nodes.append(name)
    def add_edge(self, src, dst, label, kind):
        """Append one labeled, typed edge to the automaton."""
        self.edges.append({"src": src, "dst": dst, "label": label, "kind": kind})


def node_name(block: dict) -> str:
    """Unique node identity per (cluster occurrence, interval). Once assigned
    (`node_id`), the identity is fixed, so an instrument projection that
    clamps a block's interval still refers to the same node its feedback
    edges were built against."""
    if block.get("node_id"):
        return block["node_id"]
    tier_init = block["tier"][0].upper()
    code = block["codes"][0].replace("-", "") if block["codes"] else "x"
    iv = block.get("interval", (0, 0))
    return "q_%s_%s_%d_%d" % (code, tier_init, iv[0], iv[1])


def _flatten_ordered(tiers: "OrderedDict") -> list:
    """Flatten the tier-grouped timeline into one chronologically ordered
    list of blocks, in TIER_ORDER then by interval/line number — the
    traversal order the DFA is built and read in."""
    flat = []
    for tier in TIER_ORDER:
        if tier not in tiers:
            continue
        blocks = sorted(tiers[tier], key=lambda b: (b["interval"][0],
                                                    b["entries"][0].get("line_number", 0)))
        flat.extend(blocks)
    return flat


def component35_build_dfa(tiers, exponents, feedbacks) -> DFA:
    """Build the patient's full DFA from the tiered timeline.

    Input: tiers (from `build_union_aware_tiers`), each block's
    self-loop exponent (from `component2_exponent`), and detected
    feedback links (from `detect_feedbacks`).
    Output: a DFA whose edge sequence, once traversed, yields the FiRE
    Expression: union-group blocks branch from a shared predecessor and
    rejoin at an epsilon join node, feedback pairs get a back-edge and a
    return-edge, and each block's self-loop count follows its exponent.
    """
    dfa = DFA()
    fb_by_later = defaultdict(list)
    for f in feedbacks:
        fb_by_later[f["later_node"]].append(f)
    flat = _flatten_ordered(tiers)

    prev = "q_born"
    dfa.add_node(prev)
    i = 0
    while i < len(flat):
        block = flat[i]
        ug = block.get("union_group")

        if ug is not None:
            group = [block]
            j = i + 1
            while j < len(flat) and flat[j].get("union_group") == ug:
                group.append(flat[j]); j += 1
            if len(group) >= 2:
                prevs = prev if isinstance(prev, list) else [prev]
                member_nodes = []
                for b in group:
                    n = node_name(b)
                    dfa.add_node(n)
                    member_nodes.append(n)
                    for p in prevs:
                        kind = "self_loop" if p == n else "branch"
                        dfa.add_edge(p, n, b["cluster"], kind)
                    exp = exponents[id(b)]["exponent"]
                    for _ in range(exp - 1):
                        dfa.add_edge(n, n, b["cluster"], "self_loop")
                    for fb in fb_by_later.get(n, []):
                        dfa.add_edge(n, fb["earlier_node"], fb["cluster"], "feedback_back")
                        dfa.add_edge(fb["earlier_node"], n, b["cluster"], "feedback_return")
                join = "j_" + "_".join(sorted(m for m in member_nodes))[:40]
                dfa.add_node(join)
                for n in member_nodes:
                    dfa.add_edge(n, join, "\u03b5", "join")
                prev = join
                i = j
                continue

        n = node_name(block)
        dfa.add_node(n)
        prevs = prev if isinstance(prev, list) else [prev]
        for p in prevs:
            kind = "self_loop" if p == n else "sequential"
            dfa.add_edge(p, n, block["cluster"], kind)
        exp = exponents[id(block)]["exponent"]
        for _ in range(exp - 1):
            dfa.add_edge(n, n, block["cluster"], "self_loop")
        for fb in fb_by_later.get(n, []):
            dfa.add_edge(n, fb["earlier_node"], fb["cluster"], "feedback_back")
            dfa.add_edge(fb["earlier_node"], n, block["cluster"], "feedback_return")
        prev = n
        i += 1

    dfa.add_node("q_end")
    for p in (prev if isinstance(prev, list) else [prev]):
        dfa.add_edge(p, "q_end", "END", "terminal")
    return dfa


# ─────────────────────────────────────────────────────────────
# COMPONENT 4 — DFA TRAVERSAL
# ─────────────────────────────────────────────────────────────

def component4_traverse(dfa: DFA) -> list:
    """Walk the DFA from q_born to q_end, taking each unused outgoing
    edge once, to produce the ordered (label, kind) trace that
    `component5_build_string` renders into the FiRE Expression string.
    A guard counter prevents an unexpected cycle from looping forever."""
    by_src = defaultdict(list)
    for e in dfa.edges:
        by_src[e["src"]].append(e)
    used, trace, node, guard = set(), [], "q_born", 0
    while node != "q_end" and guard < 10_000:
        guard += 1
        nxt = None
        for e in by_src[node]:
            eid = id(e)
            if eid in used:
                continue
            used.add(eid)
            if e["label"] != "END":
                trace.append((e["label"], e["kind"]))
            nxt = e["dst"]
            break
        if nxt is None:
            break
        node = nxt
    return trace


# ─────────────────────────────────────────────────────────────
# COMPONENT 5 — FE STRING BUILDER & VALIDATOR
# ─────────────────────────────────────────────────────────────

def _fmt_exp(sym: str, exp: int) -> str:
    """Render a symbol with its exponent (omitted when exponent is 1)."""
    return sym if exp <= 1 else "%s^%d" % (sym, exp)


def component5_build_string(tiers, exponents, feedbacks) -> str:
    """Render the FiRE Expression string."""
    fb_count = defaultdict(int)
    for f in feedbacks:
        fb_count[f["later_node"]] += 1

    def _eff_exp(block) -> int:
        """Self-loop exponent plus 2 for each feedback that reactivates this node."""
        return exponents[id(block)]["exponent"] + 2 * fb_count[node_name(block)]

    parts = []
    for tier in TIER_ORDER:
        if tier not in tiers:
            continue
        blocks = sorted(tiers[tier], key=lambda b: (b["interval"][0],
                                                    b["entries"][0].get("line_number", 0)))
        seg = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            ug = b.get("union_group")
            if ug is not None:
                group = [b]
                j = i + 1
                while j < len(blocks) and blocks[j].get("union_group") == ug:
                    group.append(blocks[j]); j += 1
                if len(group) >= 2:
                    inner = "+".join(_fmt_exp(g["cluster"], _eff_exp(g)) for g in group)
                    seg.append("(%s)" % inner)
                    i = j
                    continue
            seg.append(_fmt_exp(b["cluster"], _eff_exp(b)))
            i += 1
        if seg:
            parts.append(TIER_MARK[tier] + " " + " . ".join(seg))
    return " . ".join(parts)


def component5_validate(fe, tiers, exponents, feedbacks) -> dict:
    """Run consistency checks over the built expression/DFA: every symbol
    has a resolved state (alphabet_coverage), each block's tier matches
    its interval's derived tier (tier_consistency), every feedback
    target actually precedes its trigger (feedback_integrity), no
    cluster's states are uniformly low-confidence (confidence_floor,
    informational only), and at least two distinct symbols exist
    (minimum_events). Returns a dict of named check results used both
    for `human_review_flags` and for the confidence-asterisk pass."""
    flat = _flatten_ordered(tiers)
    checks = {}
    empty = [b["cluster"] for b in flat if not b["codes"]]
    checks["alphabet_coverage"] = {"pass": not empty, "orphans": empty}

    bad = []
    for b in flat:
        if b["tier"] == "no-age":
            continue
        derived = age_to_tier(b["interval"][0])
        if derived != b["tier"]:
            bad.append({"cluster": b["cluster"], "placed": b["tier"], "interval_tier": derived})
    checks["tier_consistency"] = {"pass": not bad, "mismatches": bad}

    node_order = [node_name(b) for b in flat]
    fb_bad = []
    for f in feedbacks:
        li = node_order.index(f["later_node"]) if f["later_node"] in node_order else -1
        earlier = {flat[j]["cluster"] for j in range(li)} if li > 0 else set()
        if f["cluster"] not in earlier:
            fb_bad.append(f["cluster"])
    checks["feedback_integrity"] = {"pass": not fb_bad, "errors": fb_bad}

    by_cluster = defaultdict(list)
    for b in flat:
        by_cluster[b["cluster"]].append(exponents[id(b)]["confidence"])
    flagged = [sym for sym, confs in by_cluster.items()
               if confs and all(c < 0.60 for c in confs)]
    checks["confidence_floor"] = {"pass": True, "flagged": flagged}

    n_alpha = len({b["cluster"] for b in flat})
    checks["minimum_events"] = {"pass": n_alpha >= 2, "alphabet_count": n_alpha}
    return checks


def apply_confidence_asterisks(fe: str, flagged: list) -> str:
    """Mark every occurrence of a low-confidence symbol in the rendered
    expression string with a trailing "*", without altering its exponent."""
    for sym in flagged:
        fe = re.sub(r"(?<![\w*])%s(\^\d+)?(?![\w*])" % re.escape(sym),
                    lambda m: m.group(0) + "*", fe)
    return fe


# ─────────────────────────────────────────────────────────────
# REPORTING HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt_new_states(records: list) -> list:
    """Format newly created (provisional) states for the output report."""
    return [{
        "state_code": r["state_code"], "cluster_name": r["cluster_name"],
        "event": r["event"], "source_sentence": r["source_sentence"],
        "confidence": r["combined_confidence"],
        "mapped_state_status": r.get("mapped_state_status"),
        "note": "Newly created state — pending clinical validation.",
    } for r in records]


def _fmt_unmatched(records: list) -> list:
    """Format sub-events that never resolved to a state, for the output report."""
    return [{"event": r["event"], "reason": r["unmatched_reason"],
             "life_stage": r["life_stage"], "source_sentence": r["source_sentence"],
             "sub_status": r["sub_status"]}
            for r in records if r["match_type"] == "UNMATCHED"]


def _fmt_not_adverse(records: list) -> list:
    """Format sub-events explicitly ruled NOT_ADVERSE, for the output report."""
    return [{"event": r["event"], "reason": r["not_adverse_reason"]}
            for r in records if r["match_type"] == "NOT_ADVERSE"]


# ─────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────

def _serialise_entry(e: dict) -> dict:
    """JSON-safe copy of one timeline entry: drop the internal span keys
    (leading underscore) and expose the no-age marker under a public name."""
    out = {k: v for k, v in e.items() if not k.startswith("_")}
    out["is_no_age"] = bool(e.get("_no_age"))
    return out


def _serialise_blocks(tiers: "OrderedDict", exponents: dict, fb_count: dict) -> list:
    """The timeline blocks an instrument projection works from, in traversal
    order: one per (cluster, interval) occurrence, with its node id, union
    group, self-loop exponent, feedback count and the entries behind it."""
    out = []
    for b in _flatten_ordered(tiers):
        ex = exponents[id(b)]
        nid = node_name(b)
        out.append({
            "node_id":        nid,
            "cluster":        b["cluster"],
            "cluster_name":   b["cluster_name"],
            "state_code":     b.get("state_code"),
            "codes":          list(b["codes"]),
            "tier":           b["tier"],
            "interval":       list(b["interval"]),
            "union_group":    b.get("union_group"),
            "exponent":       ex["exponent"],
            "is_self_loop":   ex["is_self_loop"],
            "feedback_count": fb_count.get(nid, 0),
            "confidence":     ex["confidence"],
            "entries":        [_serialise_entry(e) for e in b["entries"]],
        })
    return out


def build_expression(resolved_records, new_state_records, other_records,
                     alphabet_map, include_new_states=True) -> dict:
    """Top-level assembly: turn parsed Worker 2 records into the full
    FiRE Expression output.

    Input: resolved/new-state/other records from `parse_worker2_output`
    and the cluster->symbol alphabet map.
    Output: a dict with the expression string, its DFA, per-symbol
    inventory, feedback/simultaneity logs, validation results, and
    human-review flags — the complete Worker 3 output record.
    Pipeline: build the tiered timeline, detect simultaneity unions and
    feedback links, construct and traverse the DFA, render the
    expression string, then validate it and assemble the reporting dict.
    """
    working = list(resolved_records)
    if include_new_states:
        working += list(new_state_records)

    tiers = build_union_aware_tiers(working, alphabet_map)

    if not tiers:
        return {
            "fire_expression": "(no eligible RESOLVED events — empty FE)",
            "confidence_band": 0.0, "alphabet_inventory": {},
            "alphabet_mapping": alphabet_map, "feedback_loops": [],
            "simultaneity": [], "validation": {},
            "human_review_flags": [{"check": "minimum_events", "details": "0 eligible events"}],
            "raw_traversal": [], "dfa": {"nodes": [], "edges": []},
            "node_metadata": {}, "resolved_states": [], "blocks": [],
            "reactivation_mentions": [],
            "new_states_created": _fmt_new_states(new_state_records),
            "unmatched_events": _fmt_unmatched(other_records),
            "not_adverse_events": _fmt_not_adverse(other_records),
        }

    exponents = {id(b): component2_exponent(b)
                 for blocks in tiers.values() for b in blocks}

    # Explicit-marker union pass: force adjacent same-tier blocks into a
    # union group when the text says they co-occurred, even if their
    # intervals didn't already overlap enough to be grouped by age alone.
    union_counter = sum(1 for blocks in tiers.values() for b in blocks
                        if b.get("union_group")) + 100
    sim_log = []
    for tier, blocks in tiers.items():
        ordered = sorted(blocks, key=lambda b: (b["interval"][0],
                                                b["entries"][0].get("line_number", 0)))
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            if a.get("union_group") or b.get("union_group"):
                continue
            if detect_marker_union(a, b):
                ug = f"m{union_counter}"; union_counter += 1
                a["union_group"] = ug
                b["union_group"] = ug
                sim_log.append({"tier": tier, "pair": [a["cluster"], b["cluster"]],
                                "pair_names": [a["cluster_name"], b["cluster_name"]],
                                "signals": ["explicit_marker"]})

    for tier, blocks in tiers.items():
        groups = defaultdict(list)
        for b in blocks:
            if b.get("union_group"):
                groups[b["union_group"]].append(b)
        for ug, members in groups.items():
            if len(members) >= 2 and not any(s.get("pair") == [members[0]["cluster"], members[1]["cluster"]]
                                             for s in sim_log):
                sim_log.append({
                    "tier": tier,
                    "pair": [m["cluster"] for m in members],
                    "pair_names": [m["cluster_name"] for m in members],
                    "interval": members[0]["interval"],
                    "signals": ["age_overlap"],
                })

    # Fix every block's node identity before anything refers to it.
    flat = _flatten_ordered(tiers)
    for b in flat:
        b["node_id"] = node_name(b)

    # Feedback detection, then DFA construction/traversal, then rendering
    # the final expression string from the same tiers/exponents/feedbacks.
    feedbacks = detect_feedbacks(flat)
    reactivation_mentions = detect_reactivation_mentions(flat)
    fb_count = defaultdict(int)
    for f in feedbacks:
        fb_count[f["later_node"]] += 1

    dfa = component35_build_dfa(tiers, exponents, feedbacks)
    raw_trace = component4_traverse(dfa)

    fe = component5_build_string(tiers, exponents, feedbacks)
    checks = component5_validate(fe, tiers, exponents, feedbacks)
    fe = apply_confidence_asterisks(fe, checks["confidence_floor"]["flagged"])

    inventory, node_metadata = {}, {}
    for tier, blocks in tiers.items():
        for b in blocks:
            ex = exponents[id(b)]
            sym = b["cluster"]
            uid = node_name(b)
            iv = b["interval"]
            is_no_age = all(e.get("_no_age") for e in b["entries"])
            display_age = "no age recorded" if is_no_age else (
                f"{iv[0]}-{iv[1]}" if iv[0] != iv[1] else str(iv[0]))
            rep_event = b["entries"][0].get("source_sentence", "") or b["entries"][0].get("event", "")
            statuses = {e.get("mapped_state_status") for e in b["entries"] if e.get("mapped_state_status")}
            node_metadata[uid] = {
                "age": display_age, "is_no_age": is_no_age, "context": rep_event,
                "cluster_name": b["cluster_name"], "state_codes": b["codes"],
                "experience_type": b["entries"][0].get("experience_type", "unknown"),
                "is_self_loop": ex["is_self_loop"], "exponent": ex["exponent"],
                "confidence": ex["confidence"],
                "state_status": sorted(statuses)[0] if len(statuses) == 1 else
                               ("mixed" if statuses else None),
            }
            inv = inventory.setdefault(sym, {
                "symbol": sym, "cluster_name": b["cluster_name"], "exponent": 0,
                "exponent_range": [99, 0], "confidence": [],
                "contributing_states": [], "tiers": [],
            })
            inv["exponent"] = max(inv["exponent"], ex["exponent"])
            inv["exponent_range"][0] = min(inv["exponent_range"][0], ex["range"][0])
            inv["exponent_range"][1] = max(inv["exponent_range"][1], ex["range"][1])
            inv["confidence"].append(ex["confidence"])
            inv["contributing_states"] += [c for c in b["codes"]
                                           if c not in inv["contributing_states"]]
            if b["tier"] not in inv["tiers"]:
                inv["tiers"].append(b["tier"])

    total_w = total_c = 0.0
    for inv in inventory.values():
        c = sum(inv["confidence"]) / len(inv["confidence"])
        inv["confidence"] = round(c, 3)
        total_w += inv["exponent"]
        total_c += c * inv["exponent"]
    overall = round(total_c / total_w, 3) if total_w else 0.0

    review = []
    for name, res in checks.items():
        if not res.get("pass", True):
            review.append({"check": name, "details": res})
    if checks["confidence_floor"]["flagged"]:
        review.append({"check": "confidence_floor",
                       "details": checks["confidence_floor"]["flagged"]})
    provisional_in_fe = sorted({e.get("state_code") for b in flat for e in b["entries"]
                                if e.get("is_new_state")
                                or e.get("mapped_state_status") == "provisional"})
    if provisional_in_fe:
        review.append({"check": "provisional_states_in_expression",
                       "details": {"state_codes": provisional_in_fe,
                                   "note": "used immediately in this trajectory; promoted to "
                                           "the shared state space only after psychotherapist "
                                           "review"}})

    resolved_states = {}
    for tier, blocks in tiers.items():
        for b in blocks:
            for e in b["entries"]:
                code = e.get("state_code")
                if not code:
                    continue
                onset, end = e.get("_onset"), e.get("_end")
                prev = resolved_states.get(code)
                if prev is None:
                    resolved_states[code] = {
                        "state_code":      code,
                        "cluster":         b["cluster"],
                        "cluster_name":    b["cluster_name"],
                        "onset_age":       onset,
                        "end_age":         end,
                        "tier":            b["tier"],
                        "exponent":        exponents[id(b)]["exponent"],
                        "confidence":      round(e.get("combined_confidence", 0.0), 3),
                        "experience_type": e.get("experience_type", "unknown"),
                        "event":           e.get("event", ""),
                        "source_sentence": e.get("source_sentence", ""),
                        "is_recollection": e.get("is_recollection", False),
                        "mapped_state_status": e.get("mapped_state_status"),
                        "signals":         e.get("signals") or ([e.get("signal")] if e.get("signal") else []),
                        "ace_items_hint":  e.get("ace_items_hint"),
                        "perpetrator_class": e.get("perpetrator_class"),
                        "recency":         e.get("recency"),
                        "role":            e.get("role"),
                    }
                else:
                    if onset is not None and (prev["onset_age"] is None or onset < prev["onset_age"]):
                        prev["onset_age"] = onset
                    if end is not None and (prev["end_age"] is None or end > prev["end_age"]):
                        prev["end_age"] = end
                    prev["exponent"] = max(prev["exponent"], exponents[id(b)]["exponent"])
                    for _sg in (e.get("signals") or ([e.get("signal")] if e.get("signal") else [])):
                        if _sg and _sg not in prev["signals"]:
                            prev["signals"].append(_sg)

    return {
        "fire_expression": fe, "confidence_band": overall,
        "alphabet_inventory": inventory,
        "alphabet_mapping": {sym: inv["cluster_name"] for sym, inv in inventory.items()},
        "feedback_loops": feedbacks, "simultaneity": sim_log,
        "validation": checks, "human_review_flags": review,
        "raw_traversal": [t[0] for t in raw_trace],
        "dfa": {"nodes": dfa.nodes,
                "edges": [{k: e[k] for k in ("src", "dst", "label", "kind")} for e in dfa.edges]},
        "node_metadata": node_metadata,
        "resolved_states": list(resolved_states.values()),
        "blocks": _serialise_blocks(tiers, exponents, fb_count),
        "reactivation_mentions": reactivation_mentions,
        "new_states_created": _fmt_new_states(new_state_records),
        "unmatched_events": _fmt_unmatched(other_records),
        "not_adverse_events": _fmt_not_adverse(other_records),
    }


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def _mk_record(state_code, cluster_name, onset, end, event, exp_type="event",
               is_ongoing=False, sub_id=None, life_stage="childhood",
               source=None, quote=None, line_number=1, is_new_state=False):
    """Build a minimal synthetic resolved-state record for self-test fixtures.
    `source` is the (possibly shared) source sentence and `quote` the
    sub-event's own supporting quote; without `source`, the record is the
    only sub-event of its line, as before."""
    return {
        "state_code": state_code, "event_uid": sub_id or state_code,
        "cluster_name": cluster_name, "cluster": "",
        "match_type": "RESOLVED", "combined_confidence": 0.9,
        "is_new_state": is_new_state,
        "mapped_state_status": "provisional" if is_new_state else "permanent",
        "life_stage": life_stage, "line_number": line_number,
        "onset_age": onset, "end_age": end, "is_ongoing": is_ongoing,
        "is_recollection": False, "recollection_reference": None,
        "source_sentence": source if source is not None else event,
        "source_shared": source is not None,
        "event": event, "experience_type": exp_type,
        "frequency": "single", "mapping_reason": "", "sub_status": "MAPPED",
        "unmatched_reason": "", "not_adverse_reason": "",
        "supporting_quote": quote, "quote_grounded": True if quote else None,
        "signal": None, "signals": [], "ace_items_hint": None, "ace14_items_hint": None,
        "perpetrator": None, "perpetrator_class": None,
        "recency": None, "role": None,
    }


# The main paper's worked example (Section "The Worked Example Through FiRE"),
# shaped the way Worker 2 emits it: every sub-event of a note line carries
# that line's full source sentence, plus its own supporting quote.
WORKED_EXAMPLE_LINES = {
    1: "Age 0 to 8: father emotionally abusive and controlling; client felt unloved and "
       "did not belong; bullied frequently at school for a speech difficulty and wearing glasses.",
    2: "Age 11: mother died of cancer.",
    3: "Age 12 to 16: father beat and pinched her, with continuing emotional abuse; "
       "depressed mood and overeating.",
    4: "Age 22 to 24: a trusted friend took the client's money and left, her first major betrayal.",
    5: "Age 35 to 37: the client discovered her husband's affair.",
    6: "Age 40 to 42 (current): divorced and living with family, with persistent low mood "
       "and chronic insomnia.",
}


def worked_example_records() -> list:
    """Resolved-state records for the main paper's worked example (Table I)."""
    L = WORKED_EXAMPLE_LINES
    def rec(code, cluster, onset, end, event, quote, line, sub_id=None, stage="childhood",
            exp_type="event"):
        return _mk_record(code, cluster, onset, end, event, exp_type=exp_type,
                          sub_id=sub_id or code, life_stage=stage, source=L[line],
                          quote=quote, line_number=line)
    return [
        rec("q-a", "Emotional Abuse", 0, 8, "Father was emotionally abusive and controlling",
            "father emotionally abusive and controlling", 1),
        rec("q-b", "Felt Unloved", 0, 8, "Client felt unloved and did not belong",
            "client felt unloved and did not belong", 1, exp_type="symptom"),
        rec("q-c", "Peer Bullying", 0, 8, "Client was bullied at school by peers",
            "bullied frequently at school", 1),
        rec("q-d", "Caregiver Death", 11, 11, "Mother died of cancer",
            "mother died of cancer", 2),
        rec("q-e", "Physical Abuse", 12, 16, "Father beat and pinched client",
            "father beat and pinched her", 3, stage="teenage"),
        rec("q-a", "Emotional Abuse", 12, 16, "Father's emotional abuse continued",
            "continuing emotional abuse", 3, sub_id="q-a-2", stage="teenage"),
        rec("q-p", "Depressive Symptoms", 12, 16, "Client had depressed mood and overeating",
            "depressed mood and overeating", 3, stage="teenage", exp_type="symptom"),
        rec("q-f1", "Betrayal", 22, 24, "A trusted friend took client's money and left",
            "a trusted friend took the client's money and left", 4, stage="adult"),
        rec("q-f2", "Betrayal", 35, 37, "Client discovered husband's affair",
            "the client discovered her husband's affair", 5, stage="adult"),
        rec("q-g", "Depression", 40, 42, "Client has persistent low mood",
            "persistent low mood", 6, stage="adult", exp_type="symptom"),
        rec("q-h", "Sleep Disturbance", 40, 42, "Client has chronic insomnia",
            "chronic insomnia", 6, stage="adult", exp_type="symptom"),
    ]


def selftest() -> bool:
    """Offline regression guard: rebuilds the paper's worked example from
    Worker 2-shaped records (shared source sentences) and checks union,
    self-loop, feedback and concatenation exactly, then exercises the edge
    cases each operator rule exists for."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    records = worked_example_records()
    alphabet_map = assign_alphabets(records)
    result = build_expression(records, [], [], alphabet_map)
    fe = result["fire_expression"]
    print(f"  FE produced: {fe}")
    flat_fe = fe.replace(" ", "")

    check(flat_fe.startswith("[C](a+b+c^2).d.(e+a^3+f)"),
          "PAPER MATCH: childhood union (a+b+c^2), d, then (e+a^3+p) with a^3 from feedback")
    check("(a+b+c^2)" in flat_fe,
          "self-loop stays on the bullying sub-event; 'frequently' in the shared "
          "source sentence does not leak onto a or b")
    a_fb = [f for f in result["feedback_loops"] if f["cluster"] == "a"]
    check(len(a_fb) == 1 and a_fb[0]["evidence"] == ["same_cluster_recurrence_after_gap"],
          "one structural feedback for the emotional-abuse reactivation")
    betrayal = alphabet_map["Betrayal"]
    b_fb = [f for f in result["feedback_loops"] if f["cluster"] == betrayal]
    check(len(b_fb) == 1 and f"{betrayal}^3" in flat_fe,
          "cluster-level feedback: two different betrayal states (f1, f2) are one "
          "construct recurring, so the second carries ^3")
    check(len(result["blocks"]) > 0 and all("node_id" in b and "entries" in b
                                            for b in result["blocks"]),
          "output carries the timeline blocks an instrument projection reads")

    # A later episode cut into several interval slices is still ONE reactivation.
    r_split = [
        _mk_record("s1", "Cluster A", 0, 4, "A early"),
        _mk_record("s1", "Cluster A", 8, 14, "A later", sub_id="s1-2"),
        _mk_record("s2", "Cluster B", 10, 14, "B overlaps the later A"),
    ]
    res_split = build_expression(r_split, [], [], assign_alphabets(r_split))
    fe_split = res_split["fire_expression"].replace(" ", "")
    print(f"  Split-episode FE: {fe_split}")
    check(len(res_split["feedback_loops"]) == 1 and fe_split.count("a^3") == 1,
          "feedback adds +2 once per reactivation, not once per interval slice")

    # Same cluster, no gap: persistence, not reactivation.
    r_cont = [
        _mk_record("s1", "Cluster A", 0, 4, "A first part"),
        _mk_record("s3", "Cluster A", 5, 9, "A continues as another state"),
    ]
    res_cont = build_expression(r_cont, [], [], assign_alphabets(r_cont))
    check(not res_cont["feedback_loops"],
          "same cluster continuing without a gap is not feedback")

    # Reactivation wording is reported, never turned into a feedback edge.
    r_react = [
        _mk_record("s1", "Cluster A", 2, 4, "A in childhood"),
        _mk_record("s2", "Cluster B", 30, 30, "B, which reminded her of her father"),
    ]
    res_react = build_expression(r_react, [], [], assign_alphabets(r_react))
    check(not res_react["feedback_loops"] and len(res_react["reactivation_mentions"]) == 1
          and "b^3" not in res_react["fire_expression"],
          "reactivation language alone is reported, adds no feedback across clusters")

    r_span_only = [_mk_record("s1", "Long Condition", 0, 10,
                              "state present continuously, no repetition wording")]
    res_span = build_expression(r_span_only, [], [], assign_alphabets(r_span_only))
    check(res_span["fire_expression"].replace(" ", "") == "[C]a",
          "Long span with no repetition language stays exponent 1")

    r_lang = [_mk_record("s1", "Repeated Condition", 0, 3,
                         "this happened repeatedly over the period")]
    res_lang = build_expression(r_lang, [], [], assign_alphabets(r_lang))
    check("a^2" in res_lang["fire_expression"].replace(" ", ""),
          "Repetition language triggers ^2")

    r2 = [
        _mk_record("s1", "Cluster A", 0, 5, "state A spans 0-5"),
        _mk_record("s2", "Cluster B", 0, 7, "state B spans 0-7"),
    ]
    res2 = build_expression(r2, [], [], assign_alphabets(r2))
    fe2 = res2["fire_expression"].replace(" ", "")
    print(f"  Union-test FE: {fe2}")
    check("(a+b)" in fe2 or "(b+a)" in fe2,
          "Overlapping spans create union block (a+b)")
    check(fe2.count(".") >= 1 and fe2.rstrip().endswith("b"),
          "Remainder of longer span continues afterward")

    r_same = [
        _mk_record("s1", "Cluster A", 0, 5, "first state of cluster A"),
        _mk_record("s2", "Cluster A", 0, 5, "second state of cluster A"),
    ]
    res_same = build_expression(r_same, [], [], assign_alphabets(r_same))
    check(res_same["fire_expression"].replace(" ", "") == "[C]a",
          "two states of one cluster in the same period are one symbol, not (a+a)")

    new_rec = _mk_record("st-new", "Novel State", 5, 5, "something new", is_new_state=True)
    res_incl = build_expression([], [new_rec], [], {"Novel State": "z"})
    res_excl = build_expression([], [new_rec], [], {"Novel State": "z"},
                                include_new_states=False)
    check("z" in res_incl["fire_expression"] and "z" not in res_excl["fire_expression"],
          "provisional states enter the FE by default; excluded only on request")
    check(any(f["check"] == "provisional_states_in_expression"
              for f in res_incl["human_review_flags"]),
          "a provisional state in the FE is flagged for psychotherapist review")

    r3 = [
        _mk_record("s1", "Known Age", 5, 5, "event at age 5"),
        _mk_record("s2", "Unknown Age", None, None, "event with no age given"),
    ]
    res3 = build_expression(r3, [], [], assign_alphabets(r3))
    check("[N]" in res3["fire_expression"], "No-age state placed in dedicated tier")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: load a Worker 2 output file, parse it into
    resolved/new-state/other records, assign the cluster alphabet, build
    the FiRE Expression via `build_expression`, print a summary, and
    write the full result to --output."""
    parser = argparse.ArgumentParser(
        description="FiRE Worker 3 (all instruments): build the trajectory and its "
                    "expression from Worker 2's mapped events (no model call)")
    parser.add_argument("input", nargs="?", help="Worker 2 output (final_mapped_events.json)")
    parser.add_argument("--output", "-o", default="worker3_output.json",
                        help="Output file path (default: worker3_output.json)")
    parser.add_argument("--exclude-provisional-states", action="store_true",
                        help="Leave states Worker 2 created as provisional out of the FE. "
                             "Default: included, since the paper uses a provisional state "
                             "immediately in the current individual's trajectory.")
    parser.add_argument("--include-new-states", action="store_true",
                        help="Has no effect: provisional states are always included.")
    parser.add_argument("--selftest", action="store_true",
                        help="Run regression checks and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if not args.input:
        parser.error("input is required (unless using --selftest)")

    try:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error("Input file not found: %s", args.input); sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON: %s", e); sys.exit(1)

    if not isinstance(data, dict) or "events" not in data:
        logging.error("Input does not look like a Worker 2 output: %s", args.input); sys.exit(1)

    patient_id = data.get("patient_id") or Path(args.input).stem.replace("_final", "")
    logging.info("Patient: %s", patient_id)

    include_new = not args.exclude_provisional_states
    resolved, new_states, others = parse_worker2_output(data)
    logging.info("Records parsed: %d RESOLVED | %d NEW_STATE (provisional) | %d other",
                 len(resolved), len(new_states), len(others))

    if not resolved and not (include_new and new_states):
        logging.warning("No RESOLVED events found.")

    all_for_alpha = resolved + (new_states if include_new else [])
    alphabet_map = assign_alphabets(all_for_alpha)
    logging.info("Alphabet: %d unique clusters", len(alphabet_map))

    result = build_expression(
        resolved_records=resolved, new_state_records=new_states,
        other_records=others, alphabet_map=alphabet_map,
        include_new_states=include_new)

    sep = "\u2550" * 60
    print(f"\n{sep}\n  FiRE Expression — {patient_id}\n{sep}\n")
    print(f"  {result['fire_expression']}\n")
    print(f"  Confidence band : {result['confidence_band']:.3f}")
    print(f"  Alphabets       : {len(result['alphabet_inventory'])}")
    print(f"  Feedback loops  : {len(result['feedback_loops'])}")
    print(f"  Simultaneity    : {len(result['simultaneity'])}")
    print(f"  Review flags    : {len(result['human_review_flags'])}\n")

    if result["alphabet_inventory"]:
        print("  Alphabet key:")
        for sym, inv in result["alphabet_inventory"].items():
            tiers_str = "+".join(t[0].upper() for t in inv["tiers"])
            states = ",".join(inv["contributing_states"][:3]) + ("..." if len(inv["contributing_states"]) > 3 else "")
            print(f"    {sym+':':8s} {sym}^{inv['exponent']}  [{tiers_str}]  conf={inv['confidence']:.2f}  states={states}")
        print("\n  Symbol \u2192 Cluster name:")
        for sym, name in result["alphabet_mapping"].items():
            print(f"    {sym+':':8s} {name}")
    print()

    if result["simultaneity"]:
        print("  Simultaneous (union) pairs:")
        for s in result["simultaneity"]:
            iv = s.get("interval")
            iv_str = f" age {iv[0]}-{iv[1]}" if iv else ""
            print(f"    [{s['tier'][0].upper()}] ({' + '.join(s['pair'])}){iv_str}  via {'+'.join(s['signals'])}")
        print()

    if result["feedback_loops"]:
        print("  Feedback loops (encoded as +2 on the reactivated (later) node's exponent):")
        for fb in result["feedback_loops"]:
            print(f"    {fb['trigger_cluster']}@{fb['later_interval']} \u2192 "
                  f"{fb['cluster']}@{fb['earlier_interval']}  ({', '.join(fb['evidence'])})")
        print()

    if result["human_review_flags"]:
        print("  ! Human review flags:")
        for flag in result["human_review_flags"]:
            print(f"    - {flag['check']}")
        print()

    prov_note = "excluded" if not include_new else "included, flagged for review"
    print(f"  Provisional states created by Worker 2 ({prov_note}) : "
          f"{len(result['new_states_created'])}")
    print("  Excluded from FE:")
    print(f"    UNMATCHED                                          : {len(result['unmatched_events'])}")
    print(f"    NOT_ADVERSE (neutral)                              : {len(result['not_adverse_events'])}")
    print(f"\n{sep}\n")

    output = {
        "patient_id": patient_id, "input_file": str(args.input),
        "worker2_stats": data.get("worker2_stats", {}),
        "include_new_states": include_new, **result,
    }
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logging.info("Output written: %s", args.output)
    except OSError as e:
        logging.error("Failed to write output: %s", e); sys.exit(1)

    graph_path = str(Path(args.output).with_suffix(".html"))
    generate_interactive_html(result=result, patient_id=patient_id, output_path=graph_path)
    print(f"  Interactive Graph  : {graph_path}")


# ─────────────────────────────────────────────────────────────
# INTERACTIVE HTML GRAPH (vis-network)
# ─────────────────────────────────────────────────────────────

def _symbol_for_node(result, node_id):
    """Map a DFA node to its alphabet symbol."""
    md = result.get("node_metadata", {}).get(node_id, {})
    codes = md.get("state_codes", [])
    inv = result.get("alphabet_inventory", {})
    for sym, info in inv.items():
        if any(c in info.get("contributing_states", []) for c in codes):
            return sym
    return None


def generate_interactive_html(result, patient_id, output_path) -> None:
    """Render the automaton as an interactive HTML graph."""
    dfa = result.get("dfa", {})
    nodes_in = dfa.get("nodes", [])
    edges_in = dfa.get("edges", [])
    meta = result.get("node_metadata", {})
    fe_raw = result.get("fire_expression", "")
    if not nodes_in:
        return

    tier_colors = {"C": "#fef08a", "T": "#fde047", "A": "#facc15", "M": "#eab308",
                   "N": "#cbd5e1"}
    tier_names  = {"C": "Childhood", "T": "Teenage", "A": "Adulthood",
                   "M": "Mid-life", "N": "No age recorded"}

    order = [n for n in nodes_in if n not in ("q_born", "q_end")]
    level = {"q_born": 0}
    for i, n in enumerate(order):
        level[n] = i + 1
    level["q_end"] = len(order) + 1

    def tier_of(node_id):
        """Extract a node's tier-marker letter from its generated node_name."""
        m = re.match(r"q_.+_([CTAMN])_\d+_\d+$", node_id)
        return m.group(1) if m else None

    node_objs = []
    node_objs.append({"id": "q_born", "label": "q\u2080 (Start)", "level": 0,
                      "color": "#16a34a", "shape": "box",
                      "font": {"color": "white", "size": 16}, "borderWidth": 2})

    for n in order:
        if n.startswith("j_"):
            ivs = re.findall(r"_(\d+)_(\d+)(?=_|$)", n)
            if ivs:
                los = [int(a) for a, _ in ivs]
                his = [int(b) for _, b in ivs]
                union_age = (f"{min(los)}-{max(his)}" if min(los) != max(his)
                             else str(min(los)))
            else:
                union_age = "?"
            member_ids = re.findall(r"(q_[A-Za-z0-9]+_[CTAMN]_\d+_\d+)", n)
            mem_syms, mem_clusters = [], []
            for mid in member_ids:
                mm = meta.get(mid, {})
                s = _symbol_for_node(result, mid)
                if s and s not in mem_syms:
                    mem_syms.append(s)
                cn = mm.get("cluster_name")
                if cn and cn not in mem_clusters:
                    mem_clusters.append(cn)
            j_title = (
                "Union join (\u03b5)\n"
                + "Age (overlap): " + union_age + "\n"
                + "Members: " + (" + ".join(mem_syms) if mem_syms else "?") + "\n"
                + "Clusters: " + (", ".join(mem_clusters) if mem_clusters else "?") + "\n"
                + "-----------------------------------\n"
                + "These states overlap in age and run in parallel; the\n"
                + "graph forks into them and rejoins here before continuing."
            )
            node_objs.append({
                "id": n, "label": "\u03b5", "title": j_title, "level": level[n],
                "color": "#cbd5e1", "shape": "dot", "size": 12,
                "font": {"size": 14},
                "_meta": {
                    "symbol": "\u03b5 (union join)",
                    "cluster_name": ", ".join(mem_clusters) if mem_clusters else "-",
                    "state_codes": [], "age": union_age, "interval_slice": union_age,
                    "tier": "union join", "experience_type": "-", "self_loop": "-",
                    "confidence": "-", "state_status": "-",
                    "context": "Overlapping states that run in parallel over age "
                               + union_age + ", then rejoin.",
                },
            })
            continue
        md = meta.get(n, {})
        t = tier_of(n) or "C"
        loop = "yes (^%d)" % md.get("exponent", 1) if md.get("is_self_loop") else "no"
        tier_label = "no age recorded" if md.get("is_no_age") else tier_names.get(t, "?")
        status = md.get("state_status")
        status_line = (f"Status: {status}"
                       + (" (PENDING CLINICIAN REVIEW)" if status == "provisional" else "")) \
                       if status else None
        title_lines = [
            "Cluster: " + str(md.get("cluster_name", "?")),
            "State: " + ", ".join(md.get("state_codes", [])),
            "Age: " + str(md.get("age", "?")) + "   (" + tier_label + ")",
            "Type: " + str(md.get("experience_type", "?")),
            "Self-loop: " + loop,
            "Confidence: " + str(md.get("confidence", "?")),
        ]
        if status_line:
            title_lines.append(status_line)
        title_lines += ["-----------------------------------",
                        'Context: "' + str(md.get("context", "")) + '"']
        title = "\n".join(title_lines)
        label = _symbol_for_node(result, n) or "?"
        node_objs.append({
            "id": n, "label": label, "title": title, "level": level[n],
            "color": tier_colors.get(t, "#FFD700"), "shape": "dot",
            "size": 24, "borderWidth": 2, "font": {"size": 18},
            "_meta": {
                "symbol": label, "cluster_name": md.get("cluster_name", "?"),
                "state_codes": md.get("state_codes", []), "age": md.get("age", "?"),
                "interval_slice": md.get("age", "?"), "tier": tier_label,
                "experience_type": md.get("experience_type", "?"),
                "self_loop": loop, "confidence": md.get("confidence", "?"),
                "state_status": status or "-", "context": md.get("context", ""),
            },
        })
    node_objs.append({"id": "q_end", "label": "q_f (End)", "level": level["q_end"],
                      "color": "#ea580c", "shape": "box",
                      "font": {"color": "white", "size": 16}, "borderWidth": 2})

    edge_colors = {
        "sequential": "#000000", "branch": "#000000", "join": "#94a3b8",
        "self_loop": "#6b7280", "feedback_back": "#ef4444",
        "feedback_return": "#ef4444", "terminal": "#000000",
    }
    edge_objs = []
    for e in edges_in:
        kind = e.get("kind", "sequential")
        label = e.get("label", "")
        if label == "END":
            label = ""
        obj = {"from": e["src"], "to": e["dst"], "label": label,
               "color": edge_colors.get(kind, "#000000"), "arrows": "to",
               "font": {"align": "top"}}
        if kind == "self_loop":
            obj["dashes"] = [2, 2]
            obj["color"] = "#6b7280"
        elif kind in ("feedback_back", "feedback_return"):
            obj["dashes"] = [5, 5]
            obj["width"] = 2
            obj["smooth"] = {"type": "curvedCW", "roundness": 0.35}
        elif kind == "join":
            obj["label"] = "\u03b5"
        edge_objs.append(obj)

    nodes_js = ",".join(json.dumps(o, ensure_ascii=False) for o in node_objs)
    edges_js = ",".join(json.dumps(o, ensure_ascii=False) for o in edge_objs)

    _here = Path(__file__).resolve().parent
    _bundle = _here / "vis-network-bundle.js"
    if _bundle.exists():
        visjs_inline = _bundle.read_text(encoding="utf-8")
    else:
        visjs_inline = (
            "document.write('<scr'+'ipt src=\"https://cdnjs.cloudflare.com/ajax/"
            "libs/vis-network/9.1.2/dist/vis-network.min.js\"><\\/scr'+'ipt>');"
        )
        logging.warning("vis-network-bundle.js not found; "
                        "graph will need internet.")

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FiRE trajectory — @@PID@@</title>
<script>@@VISJS@@</script>
<style>
  html,body{margin:0;padding:0;height:100%;font-family:Helvetica,Arial,sans-serif;background:#f8fafc;}
  #hdr{padding:10px 16px;background:#1e293b;color:#fff;font-size:15px;font-weight:bold;}
  #hdr small{font-weight:normal;color:#cbd5e1;margin-left:10px;}
  #net{width:100%;height:calc(100vh - 84px);border-top:1px solid #cbd5e1;background:#f8fafc;}
  #bar{padding:6px 16px;background:#f1f5f9;border-top:1px solid #cbd5e1;font-size:13px;color:#334155;}
  #bar button{margin-right:6px;padding:4px 10px;border:1px solid #94a3b8;background:#fff;border-radius:4px;cursor:pointer;}
  #bar button:hover{background:#e2e8f0;}
  .legend{display:inline-block;margin-left:14px;}
  .legend span{display:inline-block;width:12px;height:0;border-top:3px solid;margin:0 4px 0 10px;vertical-align:middle;}
  div.vis-tooltip{
    white-space:pre-wrap !important; max-width:380px !important;
    font-family:Helvetica,Arial,sans-serif !important; font-size:12px !important;
    line-height:1.45 !important; padding:10px 12px !important;
    background:#ffffff !important; border:1px solid #94a3b8 !important;
    border-radius:6px !important; color:#1e293b !important;
    box-shadow:0 4px 14px rgba(0,0,0,0.15) !important;
  }
  #pin{position:absolute; top:54px; right:16px; width:340px; max-height:calc(100vh - 160px);
       overflow:auto; background:#fff; border:1px solid #94a3b8; border-radius:8px;
       box-shadow:0 6px 20px rgba(0,0,0,0.18); padding:14px 16px; font-size:13px;
       line-height:1.5; color:#1e293b; display:none; z-index:50;}
  #pin h3{margin:0 0 8px 0; font-size:15px;}
  #pin .row{margin:3px 0;}
  #pin .k{color:#64748b; display:inline-block; min-width:96px; vertical-align:top;}
  #pin .v{color:#0f172a; white-space:pre-wrap;}
  #pin .close{position:absolute; top:8px; right:10px; cursor:pointer; color:#64748b; font-weight:bold;}
  #pin .close:hover{color:#0f172a;}
</style>
</head>
<body>
<div id="hdr">FiRE Automaton — @@PID@@ <small>@@FE@@</small></div>
<div id="net"></div>
<div id="pin"><span class="close" onclick="document.getElementById('pin').style.display='none'">\u00d7</span><div id="pin-body"></div></div>
<div id="bar">
  <button onclick="window.net && window.net.fit()">Fit</button>
  <button onclick="window.zoom && window.zoom(1.2)">Zoom +</button>
  <button onclick="window.zoom && window.zoom(0.8)">Zoom \u2212</button>
  <span class="legend">
    <span style="border-color:#6b7280;border-top-style:dashed;"></span>self-loop
    <span style="border-color:#ef4444;border-top-style:dashed;"></span>feedback
    <span style="border-color:#94a3b8;"></span>\u03b5 join
  </span>
  <span style="margin-left:12px;">hover a node for details \u00b7 double-click to pin \u00b7 drag nodes freely</span>
</div>
<script>
  var rawNodes = [@@NODES@@];
  var nodes = new vis.DataSet(rawNodes);
  var edges = new vis.DataSet([@@EDGES@@]);
  var metaById = {};
  rawNodes.forEach(function(n){ if(n._meta){ metaById[n.id] = n._meta; } });
  var container = document.getElementById('net');
  var options = {
    layout:{hierarchical:{enabled:true,direction:"LR",sortMethod:"directed",
      levelSeparation:240,nodeSpacing:150,treeSpacing:220,blockShifting:true,
      edgeMinimization:true,parentCentralization:true}},
    physics:{enabled:false},
    interaction:{hover:true,zoomView:true,dragView:true,dragNodes:true,
      tooltipDelay:100,navigationButtons:false,multiselect:false},
    edges:{smooth:{type:"cubicBezier",forceDirection:"horizontal",roundness:0.5}},
    nodes:{shadow:true,margin:12,fixed:{x:false,y:false}}
  };
  var net;
  if (typeof vis === "undefined" || !vis.Network) {
    container.innerHTML = '<div style="padding:30px;color:#b91c1c;font-size:15px;">'
      + 'Graph library failed to load. Re-run Worker 3 with '
      + 'vis-network-bundle.js next to the script for a fully offline file.</div>';
  } else {
    net = new vis.Network(container, {nodes:nodes, edges:edges}, options);
    window.net = net;
    window.zoom = function(f){ net.moveTo({scale: net.getScale()*f}); };
    net.once("afterDrawing", function(){ try{ net.fit(); }catch(e){} });
    window.addEventListener("resize", function(){ try{ net.fit(); }catch(e){} });

    function esc(s){ return String(s==null?"":s)
      .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
    net.on("doubleClick", function(params){
      if(!params.nodes || !params.nodes.length){ return; }
      var m = metaById[params.nodes[0]];
      if(!m){ document.getElementById('pin').style.display='none'; return; }
      var rows = [
        ["Symbol", m.symbol], ["Cluster", m.cluster_name],
        ["State codes", (m.state_codes||[]).join(", ")],
        ["Age", m.age], ["Life stage", m.tier], ["Type", m.experience_type],
        ["Self-loop", m.self_loop], ["Confidence", m.confidence],
        ["Status", m.state_status], ["Context", m.context]
      ];
      var body = '<h3>Node ' + esc(m.symbol) + '</h3>';
      rows.forEach(function(r){
        body += '<div class="row"><span class="k">'+esc(r[0])+
                ':</span> <span class="v">'+esc(r[1])+'</span></div>';
      });
      document.getElementById('pin-body').innerHTML = body;
      document.getElementById('pin').style.display='block';
    });
  }
</script>
</body>
</html>"""
    html = (html
            .replace("@@PID@@", str(patient_id))
            .replace("@@FE@@", fe_raw.replace("<", "&lt;").replace(">", "&gt;"))
            .replace("@@NODES@@", nodes_js)
            .replace("@@EDGES@@", edges_js)
            .replace("@@VISJS@@", visjs_inline))
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info("Interactive graph written: %s", output_path)
    except OSError as e:
        logging.error("HTML write failed: %s", e)


if __name__ == "__main__":
    main()
