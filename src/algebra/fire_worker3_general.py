"""
FiRE Pipeline — Worker 3: FiRE Expression Builder

Builds the complete patient FiRE Expression (all resolved clinical states)
using a deterministic finite automaton. This is the general Worker 3 file,
generalised to run over all clinical states without instrument-specific gating.

Usage:
  python3 fire_worker3_general.py patient01_final.json --output patient01_w3.json
  python3 fire_worker3_general.py patient01_final.json --include-new-states --verbose
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

RELATION_TOKENS = {
    "father", "mother", "husband", "wife", "cousin", "uncle", "aunt",
    "brother", "sister", "grandmother", "grandfather", "stepmother",
    "stepfather", "teacher", "partner", "boyfriend", "girlfriend",
    "friend", "classmate", "neighbour", "neighbor", "parents", "sibling",
    "colleague", "manager", "employer",
}

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


def parse_worker2_output(data: dict) -> tuple:
    """
    Pull every resolved sub-event from a Worker 2 output.

    Returns (resolved, new_states, others):
      resolved   — RESOLVED states: graphed by default
      new_states — RESOLVED states flagged is_new_state: only graphed with --include-new-states
      others     — NOT_ADVERSE / UNMATCHED: reported, never graphed
    """
    resolved, new_states, others = [], [], []
    for ev in (data.get("events") or []):
        if ev.get("status") != "COMPOUND":
            continue
        life_stage  = str(ev.get("life_stage") or "unknown").lower()
        line_number = int(ev.get("line_number") or 0)
        source      = str(ev.get("source_sentence") or ev.get("event") or "")

        for sub in (ev.get("sub_events") or []):
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

            combined_text = source + " " + event_text
            onset_i = _coerce_age(sub.get("onset_age"))
            end_i   = _coerce_age(sub.get("end_age"))

            _sub_id = str(sub.get("sub_id") or "").strip()
            if _sub_id:
                event_uid = _sub_id
            else:
                import hashlib as _hashlib
                _h = _hashlib.sha1(event_text.strip().lower().encode("utf-8")).hexdigest()[:8]
                event_uid = f"{state_code}:{_h}"

            record = {
                "state_code":          state_code,
                "event_uid":           event_uid,
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
                "event":               event_text,
                "experience_type":     exp_type,
                "frequency":           infer_frequency(combined_text),
                "mapping_reason":      reason,
                "sub_status":          sub_status,
                "unmatched_reason":    reason if match_type == "UNMATCHED" else "",
                "not_adverse_reason":  neutral_r,
                "supporting_quote":    sub.get("supporting_quote"),
                "quote_grounded":      sub.get("quote_grounded"),
                "signal":              sub.get("signal"),
                "signals":             sub.get("signals") or ([sub.get("signal")] if sub.get("signal") else []),
                "ace_items_hint":      sub.get("ace_items_hint"),
                "perpetrator":         sub.get("perpetrator"),
                "perpetrator_class":   sub.get("perpetrator_class"),
                "mdi_items_hint":      sub.get("mdi_items_hint"),
                "mdi_frequency":       sub.get("mdi_frequency"),
                "gad_items_hint":      sub.get("gad_items_hint"),
                "gad_frequency":       sub.get("gad_frequency"),
                "isi_items_hint":      sub.get("isi_items_hint"),
                "isi_severity":        sub.get("isi_severity"),
                "asq_items_hint":      sub.get("asq_items_hint"),
                "acuity":              sub.get("acuity"),
                "safety_flag":         sub.get("safety_flag"),
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
    text = " ".join((e.get("source_sentence") or "") + " " + (e.get("event") or "")
                    for e in entries).lower()
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
        members: "OrderedDict[str, dict]" = OrderedDict()
        for r in sorted(active, key=lambda x: (x["_onset"], x.get("line_number", 0))):
            mkey = (r["state_code"], r.get("_onset"), r.get("_end"))
            sym = r["cluster"]
            if mkey not in members:
                members[mkey] = {"symbol": sym, "cluster_name": r["cluster_name"],
                                 "state_code": r["state_code"], "codes": [], "entries": []}
            if r["state_code"] not in members[mkey]["codes"]:
                members[mkey]["codes"].append(r["state_code"])
            members[mkey]["entries"].append(r)
        raw_segments.append({"interval": (lo, hi), "tier": age_to_tier(lo), "members": members})

    collapsed = []
    for seg in raw_segments:
        if collapsed:
            prev = collapsed[-1]
            if set(prev["members"].keys()) == set(seg["members"].keys()) \
               and prev["tier"] == seg["tier"]:
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
    """Resolve windows, place no-age states at the end sequentially."""
    no_age = []
    for r in working:
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

    segments = build_interval_timeline(working)

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
    """Compute self-loop strength from language and span signals."""
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
        n = _explicit_numeric((e.get("source_sentence") or "") + " " + (e.get("event") or ""))
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
    return " ".join((e.get("source_sentence") or "") + " " + (e.get("event") or "")
                    for e in block["entries"]).lower()


def detect_marker_union(block_a: dict, block_b: dict) -> bool:
    """Explicit simultaneity language between two same-tier blocks forces a union."""
    text = _block_text(block_a) + " " + _block_text(block_b)
    return any(mk in text for mk in SIMULTANEITY_MARKERS)


# ─────────────────────────────────────────────────────────────
# FEEDBACK DETECTOR — structural and language-based
# ─────────────────────────────────────────────────────────────

def _has_reactivation_language(block: dict) -> bool:
    return any(p in _block_text(block) for p in REACTIVATION_LANGUAGE)


def _reactivation_reference(block: dict) -> Optional[str]:
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
    """A block is no-age if all entries lack a real onset."""
    return all(e.get("_no_age") for e in b.get("entries", []))


def detect_feedbacks(flat: list) -> list:
    """Detect feedback (later reactivates earlier) via structure and language."""
    flat = [b for b in flat if not _block_is_no_age(b)]
    feedbacks = []
    seen_pairs = set()

    # Structural: same state recurs after temporal gap
    by_code: "dict[str, list]" = defaultdict(list)
    for b in flat:
        code = b["codes"][0] if b["codes"] else node_name(b)
        by_code[code].append(b)

    for code, occs in by_code.items():
        if len(occs) < 2:
            continue
        occs_sorted = sorted(occs, key=lambda b: b["interval"][0])
        earliest = occs_sorted[0]
        earliest_end = earliest["interval"][1]
        for later in occs_sorted[1:]:
            later_onset = later["interval"][0]
            if later_onset <= earliest_end + 1:
                earliest_end = max(earliest_end, later["interval"][1])
                continue
            key = (node_name(later), node_name(earliest))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            feedbacks.append({
                "cluster":              earliest["cluster"],
                "cluster_name":         earliest["cluster_name"],
                "trigger_cluster":      later["cluster"],
                "trigger_cluster_name": later["cluster_name"],
                "earlier_node":         node_name(earliest),
                "later_node":           node_name(later),
                "earlier_interval":     earliest["interval"],
                "later_interval":       later["interval"],
                "state_code":           code,
                "evidence":             ["same_state_recurrence_after_gap"],
                "link_confidence":      "high",
                "needs_review":         False,
                "source_sentences":     [e.get("source_sentence") for e in later["entries"]],
            })

    # Language-based: reactivation language in later block
    for i, later in enumerate(flat):
        if not _has_reactivation_language(later):
            continue
        earlier_candidates = flat[:i]
        if not earlier_candidates:
            continue
        ref = _reactivation_reference(later)
        target = None
        reason = None
        if ref:
            ref_words = set(re.findall(r"[a-z]+", ref))
            rel = ref_words & RELATION_TOKENS
            if rel:
                for e in reversed(earlier_candidates):
                    if rel & set(re.findall(r"[a-z]+", _block_text(e))):
                        target, reason = e, f"reference_relation:{sorted(rel)}"
                        break
        if target is None:
            same = [e for e in earlier_candidates if e["cluster"] == later["cluster"]]
            if same:
                target, reason = same[0], "language_same_cluster_earliest"
        if target is None:
            target, reason = earlier_candidates[-1], "language_nearest_earlier"

        key = (node_name(later), node_name(target))
        if target is later or key in seen_pairs:
            continue
        later_code = later["codes"][0] if later["codes"] else None
        target_code = target["codes"][0] if target["codes"] else None
        if (later_code and later_code == target_code
                and later["interval"][0] <= target["interval"][1] + 1):
            continue
        seen_pairs.add(key)
        feedbacks.append({
            "cluster":              target["cluster"],
            "cluster_name":         target["cluster_name"],
            "trigger_cluster":      later["cluster"],
            "trigger_cluster_name": later["cluster_name"],
            "earlier_node":         node_name(target),
            "later_node":           node_name(later),
            "earlier_interval":     target["interval"],
            "later_interval":       later["interval"],
            "evidence":             ["reactivation_language", f"link:{reason}"],
            "link_confidence":      "high" if ref else "medium",
            "needs_review":         False,
            "source_sentences":     [e.get("source_sentence") for e in later["entries"]],
        })

    return feedbacks


# ─────────────────────────────────────────────────────────────
# COMPONENT 3.5 — DFA GRAPH CONSTRUCTOR
# ─────────────────────────────────────────────────────────────

class DFA:
    def __init__(self):
        self.nodes = ["q_born", "q_end"]
        self.edges = []
    def add_node(self, name):
        if name not in self.nodes:
            self.nodes.append(name)
    def add_edge(self, src, dst, label, kind):
        self.edges.append({"src": src, "dst": dst, "label": label, "kind": kind})


def node_name(block: dict) -> str:
    """Unique node identity per (state_code, interval)."""
    tier_init = block["tier"][0].upper()
    code = block["codes"][0].replace("-", "") if block["codes"] else "x"
    iv = block.get("interval", (0, 0))
    return "q_%s_%s_%d_%d" % (code, tier_init, iv[0], iv[1])


def _flatten_ordered(tiers: "OrderedDict") -> list:
    flat = []
    for tier in TIER_ORDER:
        if tier not in tiers:
            continue
        blocks = sorted(tiers[tier], key=lambda b: (b["interval"][0],
                                                    b["entries"][0].get("line_number", 0)))
        flat.extend(blocks)
    return flat


def component35_build_dfa(tiers, exponents, feedbacks) -> DFA:
    """Build DFA with union groups, feedback edges, and self-loops."""
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
    return sym if exp <= 1 else "%s^%d" % (sym, exp)


def component5_build_string(tiers, exponents, feedbacks) -> str:
    """Render the FiRE Expression string."""
    fb_initiator_nodes = {f["later_node"] for f in feedbacks}

    def _eff_exp(block) -> int:
        base = exponents[id(block)]["exponent"]
        if node_name(block) in fb_initiator_nodes:
            base += 2
        return base

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
    for sym in flagged:
        fe = re.sub(r"(?<![\w*])%s(\^\d+)?(?![\w*])" % re.escape(sym),
                    lambda m: m.group(0) + "*", fe)
    return fe


# ─────────────────────────────────────────────────────────────
# REPORTING HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt_new_states(records: list) -> list:
    return [{
        "state_code": r["state_code"], "cluster_name": r["cluster_name"],
        "event": r["event"], "source_sentence": r["source_sentence"],
        "confidence": r["combined_confidence"],
        "mapped_state_status": r.get("mapped_state_status"),
        "note": "Newly created state — pending clinical validation.",
    } for r in records]


def _fmt_unmatched(records: list) -> list:
    return [{"event": r["event"], "reason": r["unmatched_reason"],
             "life_stage": r["life_stage"], "source_sentence": r["source_sentence"],
             "sub_status": r["sub_status"]}
            for r in records if r["match_type"] == "UNMATCHED"]


def _fmt_not_adverse(records: list) -> list:
    return [{"event": r["event"], "reason": r["not_adverse_reason"]}
            for r in records if r["match_type"] == "NOT_ADVERSE"]


# ─────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────

def build_expression(resolved_records, new_state_records, other_records,
                     alphabet_map, include_new_states=False) -> dict:
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
            "node_metadata": {}, "resolved_states": [],
            "new_states_created": _fmt_new_states(new_state_records),
            "unmatched_events": _fmt_unmatched(other_records),
            "not_adverse_events": _fmt_not_adverse(other_records),
        }

    exponents = {id(b): component2_exponent(b)
                 for blocks in tiers.values() for b in blocks}

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

    flat = _flatten_ordered(tiers)
    feedbacks = detect_feedbacks(flat)

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
                        "mdi_items_hint":  e.get("mdi_items_hint"),
                        "mdi_frequency":   e.get("mdi_frequency"),
                        "gad_items_hint":  e.get("gad_items_hint"),
                        "gad_frequency":   e.get("gad_frequency"),
                        "isi_items_hint":  e.get("isi_items_hint"),
                        "isi_severity":    e.get("isi_severity"),
                        "asq_items_hint":  e.get("asq_items_hint"),
                        "acuity":          e.get("acuity"),
                        "safety_flag":     e.get("safety_flag"),
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
        "new_states_created": _fmt_new_states(new_state_records),
        "unmatched_events": _fmt_unmatched(other_records),
        "not_adverse_events": _fmt_not_adverse(other_records),
    }


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

def _mk_record(state_code, cluster_name, onset, end, event, exp_type="event",
               is_ongoing=False, sub_id=None, life_stage="childhood"):
    return {
        "state_code": state_code, "event_uid": sub_id or state_code,
        "cluster_name": cluster_name, "cluster": "",
        "match_type": "RESOLVED", "combined_confidence": 0.9,
        "is_new_state": False, "mapped_state_status": "permanent",
        "life_stage": life_stage, "line_number": 1,
        "onset_age": onset, "end_age": end, "is_ongoing": is_ongoing,
        "is_recollection": False, "recollection_reference": None,
        "source_sentence": event, "event": event, "experience_type": exp_type,
        "frequency": "single", "mapping_reason": "", "sub_status": "MAPPED",
        "unmatched_reason": "", "not_adverse_reason": "",
        "supporting_quote": None, "quote_grounded": None,
        "signal": None, "signals": [], "ace_items_hint": None,
        "perpetrator": None, "perpetrator_class": None,
        "mdi_items_hint": None, "mdi_frequency": None, "gad_items_hint": None,
        "gad_frequency": None, "isi_items_hint": None, "isi_severity": None,
        "asq_items_hint": None, "acuity": None, "safety_flag": None,
        "recency": None, "role": None,
    }


def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    records = [
        _mk_record("st-a", "Emotional Abuse", 0, 8,
                  "Father emotionally abusive and controlling"),
        _mk_record("st-b", "Unloved", 0, 8, "Client felt unloved"),
        _mk_record("st-c", "Peer Bullying", 0, 8,
                  "Bullied frequently and repeatedly at school", exp_type="event"),
        _mk_record("st-d", "Caregiver Death", 11, 11, "Mother died of cancer"),
        _mk_record("st-e", "Physical Abuse", 12, 16, "Father beat and pinched her"),
        _mk_record("st-a", "Emotional Abuse", 12, 16,
                  "Father's emotional abuse continued", sub_id="st-a-2"),
    ]
    alphabet_map = assign_alphabets(records)
    result = build_expression(records, [], [], alphabet_map, include_new_states=False)
    fe = result["fire_expression"]
    print(f"  FE produced: {fe}")

    flat_fe = fe.replace(" ", "")
    check(re.search(r"\(a\+b\+c\^2\)|\(a\+c\^2\+b\)|\(b\+a\+c\^2\)|\(c\^2\+a\+b\)|"
                    r"\(b\+c\^2\+a\)|\(c\^2\+b\+a\)", flat_fe) is not None,
          "Union with language-only self-loop: (a+b+c^2)")
    check("a^3" in flat_fe,
          "Feedback: recurring state gets a^3 (base 1, +2 feedback)")
    check(len(result["feedback_loops"]) >= 1
          and result["feedback_loops"][0]["evidence"] == ["same_state_recurrence_after_gap"],
          "Structural feedback detected for recurring state")
    check("[C]" in fe and " . " in fe,
          "Concatenation joins sequential segments")

    r_span_only = [_mk_record("s1", "Long Condition", 0, 10,
                              "state present continuously, no repetition wording")]
    am_span = assign_alphabets(r_span_only)
    res_span = build_expression(r_span_only, [], [], am_span, include_new_states=False)
    check(res_span["fire_expression"].replace(" ", "") == "[C]a",
          "Long span with no repetition language stays exponent 1")

    r_lang = [_mk_record("s1", "Repeated Condition", 0, 3,
                         "this happened repeatedly over the period")]
    am_lang = assign_alphabets(r_lang)
    res_lang = build_expression(r_lang, [], [], am_lang, include_new_states=False)
    check("a^2" in res_lang["fire_expression"].replace(" ", ""),
          "Repetition language triggers ^2")

    r2 = [
        _mk_record("s1", "Cluster A", 0, 5, "state A spans 0-5"),
        _mk_record("s2", "Cluster B", 0, 7, "state B spans 0-7"),
    ]
    am2 = assign_alphabets(r2)
    res2 = build_expression(r2, [], [], am2, include_new_states=False)
    fe2 = res2["fire_expression"].replace(" ", "")
    print(f"  Union-test FE: {fe2}")
    check("(a+b)" in fe2 or "(b+a)" in fe2,
          "Overlapping spans create union block (a+b)")
    check(fe2.count(".") >= 1 and fe2.rstrip().endswith("b"),
          "Remainder of longer span continues afterward")

    new_rec = _mk_record("st-new", "Novel State", 5, 5, "something new")
    new_rec["is_new_state"] = True
    new_rec["mapped_state_status"] = "provisional"
    resolved, new_states, others = [], [new_rec], []
    check(len(new_states) == 1, "Provisional record correctly bucketed")
    res_excl = build_expression(resolved, new_states, others, {"Novel State": "z"},
                                include_new_states=False)
    res_incl = build_expression(resolved, new_states, others, {"Novel State": "z"},
                                include_new_states=True)
    check("z" not in res_excl["fire_expression"] and "z" in res_incl["fire_expression"],
          "--include-new-states controls whether provisional states enter the FE")

    r3 = [
        _mk_record("s1", "Known Age", 5, 5, "event at age 5"),
        _mk_record("s2", "Unknown Age", None, None, "event with no age given"),
    ]
    am3 = assign_alphabets(r3)
    res3 = build_expression(r3, [], [], am3, include_new_states=False)
    check("[N]" in res3["fire_expression"], "No-age state placed in dedicated tier")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FiRE Worker 3 — FiRE Expression Builder")
    parser.add_argument("input", nargs="?", help="Worker 2 output JSON")
    parser.add_argument("--output", "-o", default="worker3_output.json",
                        help="Output file path (default: worker3_output.json)")
    parser.add_argument("--include-new-states", action="store_true",
                        help="Include provisional states in the FE string")
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

    resolved, new_states, others = parse_worker2_output(data)
    logging.info("Records parsed: %d RESOLVED | %d NEW_STATE (provisional) | %d other",
                 len(resolved), len(new_states), len(others))

    if not resolved and not (args.include_new_states and new_states):
        logging.warning("No RESOLVED events found. Try --include-new-states.")

    all_for_alpha = resolved + (new_states if args.include_new_states else [])
    alphabet_map = assign_alphabets(all_for_alpha)
    logging.info("Alphabet: %d unique clusters", len(alphabet_map))

    result = build_expression(
        resolved_records=resolved, new_state_records=new_states,
        other_records=others, alphabet_map=alphabet_map,
        include_new_states=args.include_new_states)

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
        print("  Feedback loops (encoded as +2 on the initiating node's exponent):")
        for fb in result["feedback_loops"]:
            print(f"    {fb['trigger_cluster']}@{fb['later_interval']} \u2192 "
                  f"{fb['cluster']}@{fb['earlier_interval']}  ({', '.join(fb['evidence'])})")
        print()

    if result["human_review_flags"]:
        print("  \u26a0  Human review flags:")
        for flag in result["human_review_flags"]:
            print(f"    - {flag['check']}")
        print()

    print("  Excluded from FE:")
    print(f"    NEW_STATE (provisional, pending clinician review) : {len(result['new_states_created'])}")
    print(f"    UNMATCHED                                          : {len(result['unmatched_events'])}")
    print(f"    NOT_ADVERSE (neutral)                              : {len(result['not_adverse_events'])}")
    print(f"\n{sep}\n")

    output = {
        "patient_id": patient_id, "input_file": str(args.input),
        "worker2_stats": data.get("worker2_stats", {}),
        "include_new_states": args.include_new_states, **result,
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
<title>FiRE DFA — @@PID@@</title>
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
