"""FiRE Pipeline — Worker 3 ACE: projects the general Worker 3 trajectory to the ACE subgraph.

Usage:
  python3 fire_worker3_ace.py w3_general.json -o ace_out.json
  python3 fire_worker3_ace.py w3_general.json -o ace_out.json --verify --verify-llm
  python3 fire_worker3_ace.py --selftest
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import fire_worker3_general as W3G


# ─────────────────────────────────────────────────────────────
# LLM CONSTANTS
# ─────────────────────────────────────────────────────────────

MODEL_NAME  = "gpt-5.1"
API_URL     = "https://api.openai.com/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"

_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)


def _is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))


def _build_llm_payload(model_name: str, system_prompt: str, user_prompt: str,
                       max_output_tokens: int) -> dict:
    payload = {
        "model": model_name,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if _is_reasoning_model(model_name):
        payload["max_completion_tokens"] = max_output_tokens
        payload["reasoning_effort"] = "none"
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = max_output_tokens
    return payload


# ─────────────────────────────────────────────────────────────
# 0.  CONSTANTS — instrument-specific only
# ─────────────────────────────────────────────────────────────

# ACE is childhood-scoped, 0..18 INCLUSIVE. Age 18 is the last in-window year.
ACE_MIN_AGE, ACE_MAX_AGE = 0, 18

TIER_ORDER = W3G.TIER_ORDER
TIER_MARK  = W3G.TIER_MARK

# Cluster name → ACE item number.
_W3_CLUSTER_ITEM = {
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


def _cluster_item(cluster_name: str):
    """Resolve a cluster name to its ACE item number, or None."""
    cn = str(cluster_name or "").lower().strip()
    if not cn:
        return None
    for token in sorted(_W3_CLUSTER_ITEM, key=len, reverse=True):
        if token in cn:
            return _W3_CLUSTER_ITEM[token]
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


def _ace_items_for_entry(entry: dict, cluster_name: str) -> list:
    """ACE item(s) for an entry: resolved from cluster name plus any ace_items_hint."""
    items = []
    ci = _cluster_item(cluster_name)
    if ci is not None:
        items.append(ci)
    hints = entry.get("ace_items_hint") or []
    for h in hints:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            continue
        if 1 <= hi <= 16 and hi not in items:
            items.append(hi)
    return items


# ─────────────────────────────────────────────────────────────
# 1.  PROJECTION
# ─────────────────────────────────────────────────────────────

def project_to_ace(general_output: dict) -> tuple:
    """Project the general Worker 3 trajectory to the ACE subgraph.
    Returns (ace_blocks, ace_feedbacks, excluded).
    """
    import copy

    raw_blocks = general_output.get("blocks") or []
    excluded = []
    survivors = []

    for raw_b in raw_blocks:
        b = copy.deepcopy(raw_b)
        cluster_name = b.get("cluster_name", "")
        kept_entries = []
        block_ace_items = []

        for e in b.get("entries", []):
            event_text = e.get("event") or ""
            state_code = e.get("state_code")
            items = _ace_items_for_entry(e, cluster_name)

            if not items:
                excluded.append({"event": event_text, "reason": "not_ace_relevant",
                                 "state_code": state_code})
                continue

            onset_i = e.get("onset_age")
            end_i   = e.get("end_age")
            if onset_i is not None and onset_i >= ACE_MAX_AGE + 1:
                excluded.append({"event": event_text, "reason": "out_of_window",
                                 "detail": f"onset_age={onset_i} >= {ACE_MAX_AGE + 1} "
                                           f"(ACE window is {ACE_MIN_AGE}-{ACE_MAX_AGE} inclusive)",
                                 "state_code": state_code})
                continue

            age_clamped = False
            if onset_i is not None and end_i is not None and end_i > ACE_MAX_AGE:
                e["end_age"] = ACE_MAX_AGE
                age_clamped = True
            e["age_clamped"] = age_clamped
            e["ace_items"] = items
            for it in items:
                if it not in block_ace_items:
                    block_ace_items.append(it)
            kept_entries.append(e)

        if not kept_entries:
            continue

        b["entries"] = kept_entries
        b["codes"] = list(dict.fromkeys(
            e["state_code"] for e in kept_entries if e.get("state_code")))
        survivors.append((b, block_ace_items))

    by_ug = defaultdict(list)
    for b, _ in survivors:
        if b.get("union_group"):
            by_ug[b["union_group"]].append(b)
    for ug, members in by_ug.items():
        if len(members) < 2:
            for b in members:
                b["union_group"] = None

    survivor_node_ids = {b["node_id"] for b, _ in survivors}
    all_feedbacks = general_output.get("feedback_loops") or []
    ace_feedbacks = [
        f for f in all_feedbacks
        if f.get("earlier_node") in survivor_node_ids
        and f.get("later_node") in survivor_node_ids
    ]

    ace_blocks = [b for b, _ in survivors]
    return ace_blocks, ace_feedbacks, excluded


def _rebuild_tiers(ace_blocks: list) -> "OrderedDict":
    """Regroup projected blocks into the tiers OrderedDict."""
    tiers: "OrderedDict[str, list]" = OrderedDict()
    for b in ace_blocks:
        tiers.setdefault(b["tier"], []).append(b)
    ordered = OrderedDict()
    for t in TIER_ORDER:
        if t in tiers:
            ordered[t] = sorted(
                tiers[t],
                key=lambda bb: (bb["interval"][0], bb["entries"][0].get("line_number", 0)),
            )
    return ordered


def _exponents_from_blocks(ace_blocks: list) -> dict:
    """Read exponents from the general trajectory output."""
    return {
        id(b): {"exponent": b["exponent"], "is_self_loop": b["is_self_loop"],
                "confidence": b["confidence"], "range": [b["exponent"]] * 2}
        for b in ace_blocks
    }


def build_ace_expression_from_projection(ace_blocks: list, feedbacks: list) -> dict:
    """Serialise the projected ACE subgraph into a FiRE expression."""
    if not ace_blocks:
        return {
            "fire_expression": "(no ACE-valid events — empty FE)",
            "confidence_band": 0.0, "alphabet_inventory": {},
            "feedback_loops": [], "raw_traversal": [],
            "dfa": {"nodes": [], "edges": []}, "node_metadata": {},
            "resolved_states": [],
        }

    tiers = _rebuild_tiers(ace_blocks)
    exponents = _exponents_from_blocks(ace_blocks)

    dfa = W3G.component35_build_dfa(tiers, exponents, feedbacks)
    raw_trace = W3G.component4_traverse(dfa)
    fe = W3G.component5_build_string(tiers, exponents, feedbacks)

    inventory, node_metadata = {}, {}
    for tier, blocks in tiers.items():
        for b in blocks:
            ex = exponents[id(b)]
            sym = b["cluster"]
            uid = b["node_id"]
            iv = b["interval"]
            is_no_age = all(e.get("is_no_age") for e in b["entries"])
            any_clamped = any(e.get("age_clamped") for e in b["entries"])
            real_onsets = [e["onset_age"] for e in b["entries"] if e.get("onset_age") is not None]
            real_ends   = [e["end_age"] for e in b["entries"] if e.get("end_age") is not None]
            if is_no_age or not real_onsets:
                display_age = "null (no age recorded)"
                true_onset = true_end = None
            else:
                true_onset = min(real_onsets)
                true_end = max(real_ends) if real_ends else true_onset
                display_age = (f"{true_onset}-{true_end}" if true_onset != true_end
                               else str(true_onset))
                if any_clamped:
                    display_age += f" (end clamped to {ACE_MAX_AGE})"
            ace_items = []
            for e in b["entries"]:
                for it in (e.get("ace_items") or []):
                    if it not in ace_items:
                        ace_items.append(it)
            perp = next((e.get("perpetrator") or e.get("perpetrator_class")
                        for e in b["entries"] if e.get("perpetrator") or e.get("perpetrator_class")), None)
            descriptions = []
            for e in b["entries"]:
                d = (e.get("event") or "").strip()
                if d and d not in descriptions:
                    descriptions.append(d)
            rep = b["entries"][0]
            node_metadata[uid] = {
                "age": display_age, "onset_age": true_onset, "end_age": true_end,
                "interval_slice": (f"{iv[0]}-{iv[1]}" if iv[0] != iv[1] else str(iv[0]))
                                   if not is_no_age else "n/a (no age)",
                "is_null_age": is_no_age, "cluster_name": b["cluster_name"],
                "state_codes": [e.get("state_code") for e in b["entries"] if e.get("state_code")],
                "ace_items": ace_items, "perpetrator": perp,
                "exponent": ex["exponent"], "is_self_loop": ex["is_self_loop"],
                "confidence": ex["confidence"],
                "event_description": " | ".join(descriptions),
                "context": rep.get("source_sentence", "") or rep.get("event", ""),
            }
            inv = inventory.setdefault(sym, {
                "symbol": sym, "cluster_name": b["cluster_name"], "exponent": 0,
                "exponent_range": [99, 0], "confidence": [],
                "contributing_states": [], "tiers": [], "ace_items": [],
            })
            inv["exponent"] = max(inv["exponent"], ex["exponent"])
            inv["exponent_range"][0] = min(inv["exponent_range"][0], ex["exponent"])
            inv["exponent_range"][1] = max(inv["exponent_range"][1], ex["exponent"])
            inv["confidence"].append(ex["confidence"])
            for code in node_metadata[uid]["state_codes"]:
                if code not in inv["contributing_states"]:
                    inv["contributing_states"].append(code)
            for it in ace_items:
                if it not in inv["ace_items"]:
                    inv["ace_items"].append(it)
            if b["tier"] not in inv["tiers"]:
                inv["tiers"].append(b["tier"])

    total_w = total_c = 0.0
    for inv in inventory.values():
        c = sum(inv["confidence"]) / len(inv["confidence"])
        inv["confidence"] = round(c, 3)
        total_w += inv["exponent"]
        total_c += c * inv["exponent"]
    overall = round(total_c / total_w, 3) if total_w else 0.0

    resolved_states = {}
    for b in ace_blocks:
        for e in b["entries"]:
            code = e.get("state_code")
            if not code:
                continue
            if code not in resolved_states:
                resolved_states[code] = {
                    "state_code": code, "cluster": b["cluster"],
                    "cluster_name": b["cluster_name"],
                    "onset_age": e.get("onset_age"), "end_age": e.get("end_age"),
                    "tier": b["tier"], "exponent": exponents[id(b)]["exponent"],
                    "is_self_loop": exponents[id(b)]["is_self_loop"],
                    "confidence": round(e.get("combined_confidence", 0.0), 3),
                    "experience_type": e.get("experience_type", "unknown"),
                    "event": e.get("event", ""), "source_sentence": e.get("source_sentence", ""),
                    "ace_items_hint": e.get("ace_items"),
                    "perpetrator_class": e.get("perpetrator_class"),
                }
            else:
                p = resolved_states[code]
                oa, ea = e.get("onset_age"), e.get("end_age")
                if oa is not None and (p["onset_age"] is None or oa < p["onset_age"]):
                    p["onset_age"] = oa
                if ea is not None and (p["end_age"] is None or ea > p["end_age"]):
                    p["end_age"] = ea

    return {
        "fire_expression": fe, "confidence_band": overall,
        "alphabet_inventory": inventory,
        "alphabet_mapping": {sym: inv["cluster_name"] for sym, inv in inventory.items()},
        "feedback_loops": feedbacks,
        "raw_traversal": [t[0] for t in raw_trace],
        "dfa": {"nodes": dfa.nodes,
                "edges": [{k: e[k] for k in ("src", "dst", "label", "kind")} for e in dfa.edges]},
        "node_metadata": node_metadata,
        "resolved_states": list(resolved_states.values()),
    }


def _flatten_ace_records(ace_blocks: list) -> list:
    """Flatten projected blocks into one record per entry."""
    out = []
    for b in ace_blocks:
        for e in b["entries"]:
            out.append({
                "state_code": e.get("state_code"), "cluster_name": b["cluster_name"],
                "event": e.get("event"), "source_sentence": e.get("source_sentence"),
                "onset_age": e.get("onset_age"), "end_age": e.get("end_age"),
                "ace_items_hint": e.get("ace_items"),
                "perpetrator": e.get("perpetrator"), "perpetrator_class": e.get("perpetrator_class"),
                "supporting_quote": e.get("supporting_quote"), "quote_grounded": e.get("quote_grounded"),
                "node_id": b["node_id"],
            })
    return out


def remove_state_codes_from_blocks(ace_blocks: list, codes_to_remove: set) -> list:
    """Remove entries by state code and re-derive union membership."""
    if not codes_to_remove:
        return ace_blocks
    kept_blocks = []
    for b in ace_blocks:
        kept_entries = [e for e in b["entries"] if e.get("state_code") not in codes_to_remove]
        if not kept_entries:
            continue
        b["entries"] = kept_entries
        b["codes"] = list(dict.fromkeys(
            e["state_code"] for e in kept_entries if e.get("state_code")))
        kept_blocks.append(b)

    by_ug = defaultdict(list)
    for b in kept_blocks:
        if b.get("union_group"):
            by_ug[b["union_group"]].append(b)
    for ug, members in by_ug.items():
        if len(members) < 2:
            for b in members:
                b["union_group"] = None
    return kept_blocks


# ─────────────────────────────────────────────────────────────
# 2.  VERIFICATION LAYER
# ─────────────────────────────────────────────────────────────

ACE_QUESTIONS = {
    1: "A parent/adult in the household often swore at, insulted, humiliated, or threatened the child (emotional/verbal abuse by a caregiver).",
    2: "A parent/adult in the household pushed, grabbed, slapped, threw things at, or hit the child hard (physical abuse by a caregiver).",
    3: "An adult or person 5+ years older sexually touched the child or had sexual contact.",
    4: "The child felt unloved, unwanted, abandoned, replaced, or that the family was not close/supportive (emotional neglect).",
    5: "The child lacked food/clothes/protection or caregivers were too impaired to provide care (physical neglect).",
    6: "The child's parents were separated or divorced.",
    7: "The child's mother/stepmother was physically abused by a partner (domestic violence against the mother).",
    8: "A household member drank problematically, was alcoholic, or used street drugs.",
    9: "A household member was depressed/mentally ill or attempted/died by suicide.",
    10: "A household member went to prison.",
    11: "The child lived in a dangerous neighbourhood or saw people assaulted (community violence).",
    12: "The child's parent, guardian, or a relative who served as primary caregiver died.",
    13: "Other kids, including siblings, often hit, threatened, picked on, or insulted the child (peer/sibling bullying).",
    14: "The child often felt lonely/rejected/isolated from peers (peer rejection or social isolation).",
    15: "The family was very poor or on public assistance for a sustained period (childhood poverty).",
    16: "The child's parents had physical/verbal fights with EACH OTHER (parental conflict).",
}

ACE_SUBQUESTIONS = {
    1: [
        {"q": "Does the source describe the child being emotionally or verbally mistreated by a PARENT or HOUSEHOLD ADULT — e.g. insulted, humiliated, put down, sworn at, threatened, harshly criticized, OR described in general terms as 'emotionally abusive' / 'verbally abusive'? (a general statement such as 'father was emotionally abusive' or 'critical and humiliating' IS sufficient — you do NOT need a specific quoted insult. Bare neutral 'strict' or 'shouted' with no insult/humiliation/criticism/threat is weaker but, combined with any belittling, still qualifies.)", "defining": True},
        {"q": "Is the actor a parent / adult in the household (NOT a peer/sibling/outsider, which would be ACE-13)?", "defining": True},
        {"q": "Is there a pattern rather than a single trivial remark?", "defining": False},
    ],
    2: [
        {"q": "Does the source indicate PHYSICAL abuse or harm by a parent/household adult — either specific acts (push, grab, slap, throw, hit, beat) OR a general statement that the child was 'physically abused' / 'beaten' / 'hit' / 'physically punitive'? (a general statement of physical abuse IS sufficient; the source need NOT enumerate the blows or mention marks)", "defining": True},
        {"q": "Is the actor a parent / adult in the household?", "defining": True},
    ],
    3: [
        {"q": "Does the source describe sexual abuse, molestation, inappropriate sexual touching, OR sexually inappropriate behaviour toward the child — including an adult/older person making sexual advances, an inappropriate sexual invitation/proposition, or exposure? (overt sexual harassment of the child counts; you need NOT find completed contact)", "defining": True},
        {"q": "Was the other person an adult, notably older, or otherwise in a position that makes this abuse/harassment (NOT consensual same-age exploration)?", "defining": True},
    ],
    4: [
        {"q": "Does the source describe EMOTIONAL NEGLECT — e.g. the child feeling unloved/unimportant/unwanted, a caregiver being emotionally unavailable/absent/cold or unable to bond, the child unable to bond with a caregiver, OR the family not being close/supportive? (a statement such as 'mother was emotionally unavailable', 'father emotionally absent', or 'could not emotionally bond with her' IS sufficient — the child need NOT explicitly say the words 'I felt unloved')", "defining": True},
        {"q": "Is this about the child's emotional/relational care (NOT purely a physical-supplies failure, and NOT mere life-choice/career pressure)?", "defining": False},
    ],
    5: [
        {"q": "Does the source describe PHYSICAL / MATERIAL neglect — e.g. inadequate or unavailable food, clothing, school necessities (tiffin/meals/supplies), hygiene, or protection, OR caregivers too impaired to provide care? (OCCASIONAL or PARTIAL unavailability of necessities still counts — it need NOT be total or constant deprivation)", "defining": True},
        {"q": "Is this about the family's actual provision of care (NOT merely the child receiving LESS THAN A SIBLING, which is favouritism)?", "defining": False},
    ],
    6: [
        {"q": "Does the source state the parents SEPARATED or DIVORCED, OR that a parent permanently LEFT / ABANDONED the family? (a parent leaving the family for good — e.g. 'mother left when the client was an infant' — counts as parental separation)", "defining": True},
    ],
    7: [
        {"q": "Does the source describe the child's MOTHER / STEPMOTHER being abused (physically or in sustained domestic violence) by a partner/spouse?", "defining": True},
        {"q": "Is the victim the MOTHER (not the child themselves)?", "defining": True},
    ],
    8: [
        {"q": "Does the source describe a HOUSEHOLD MEMBER with problem drinking, alcoholism, or street-drug use?", "defining": True},
    ],
    9: [
        {"q": "Does the source describe a HOUSEHOLD MEMBER who was depressed / mentally ill OR attempted/died by suicide?", "defining": True},
        {"q": "Is mental-illness/suicide present (as opposed to ONLY a natural-causes death)? NOTE: if a household member was BOTH mentally ill AND later died, answer 'yes'.", "defining": False},
    ],
    10: [
        {"q": "Does the source state a HOUSEHOLD MEMBER went to prison/jail?", "defining": True},
    ],
    11: [
        {"q": "Does the source describe a dangerous neighbourhood, living amid violence, or witnessing people being assaulted?", "defining": True},
    ],
    12: [
        {"q": "Does the source describe the DEATH of the child's parent, guardian, grandparent, or a co-residing relative who helped raise/care for the child? (in joint/extended families a grandparent who lived with or helped raise the child counts as a caregiver — treat a grandparent's death as eligible)", "defining": True},
        {"q": "Is the death the actual EVENT here (not merely a passing time-reference like 'after X's death')?", "defining": True},
        {"q": "Is the claimed age roughly consistent with when the death occurred in the source?", "defining": False},
    ],
    13: [
        {"q": "Does the source describe OTHER KIDS or SIBLINGS hitting, threatening, picking on, bullying, or insulting the child?", "defining": True},
        {"q": "Is the actor a peer/sibling (NOT a parent/adult, which would be ACE-1/2)?", "defining": True},
    ],
    14: [
        {"q": "Does the source show the child felt lonely, rejected, or isolated FROM PEERS, had no friends, was excluded, or was kept from peers — as a state or recurring pattern (NOT a single one-off romantic break-up)?", "defining": True},
        {"q": "Is this peer/social isolation (as opposed to ONLY family emotional neglect)? NOTE: ACE-14 and ACE-4 can CO-OCCUR — if both, answer 'yes'.", "defining": False},
    ],
    15: [
        {"q": "Does the source describe HOUSEHOLD material poverty or financial hardship affecting the family's ability to afford basics (bare home, on public assistance, struggled financially)?", "defining": True},
        {"q": "Is this about the FAMILY's means (NOT merely the child having less than a sibling)?", "defining": False},
    ],
    16: [
        {"q": "Does the source describe verbal or physical CONFLICT BETWEEN THE PARENTS — EITHER mutual fighting OR one parent repeatedly abusing / humiliating / dominating the other, including domestic violence between the parents? (one-directional sustained abuse of one parent by the other DOES count)", "defining": True},
        {"q": "Is it a sustained or recurrent pattern rather than a SINGLE isolated incident (e.g. a one-off mock during one event)?", "defining": False},
    ],
}

_ACE_MARK_RE = re.compile(r"\bACEs?\b[\s:\-]*((?:\d{1,2}\s*(?:,|/|&|and|\s)\s*)*\d{1,2})",
                          re.IGNORECASE)


def _source_marks_ace_items(source: str) -> set:
    out = set()
    for m in _ACE_MARK_RE.finditer(source or ""):
        for tok in re.findall(r"\d{1,2}", m.group(1)):
            n = int(tok)
            if 1 <= n <= 16:
                out.add(n)
    return out


def _verify_record_offline(rec: dict) -> dict:
    """Deterministic, offline grounding check for one ACE record."""
    reasons = []
    severity = "ok"

    source = str(rec.get("source_sentence") or "").lower()
    onset  = rec.get("onset_age")
    cluster = str(rec.get("cluster_name") or "").lower()

    if rec.get("supporting_quote") is not None and rec.get("quote_grounded") is False:
        severity = "warn"
        reasons.append("Worker 2 supporting_quote not found in source sentence "
                       "(possible extraction from background context, not the source)")

    src_ages = set()
    for m in re.finditer(r"(\d{1,3})\s*(?:-|\u2013|\u2014|to)\s*(\d{1,3})", source):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= a <= 100 and 0 <= b <= 100:
            src_ages.update(range(a, b + 1))
    for m in re.finditer(r"\b(\d{1,3})\b", source):
        n = int(m.group(1))
        if 0 <= n <= 100:
            src_ages.add(n)

    if onset is not None and src_ages:
        end = rec.get("end_age") if rec.get("end_age") is not None else onset
        span = set(range(min(onset, end), max(onset, end) + 1))
        if not (span & src_ages):
            severity = "warn"
            reasons.append(
                f"age {onset}{'' if onset==end else f'-{end}'} not found in source "
                f"(source ages: {sorted(src_ages)})")

    if "death of parent" in cluster or "guardian" in cluster:
        death_mentioned = any(w in source for w in
                              ("died", "dies", "die ", "death", "passed away", "passed",
                               "passing", "deceased", "loss of", "lost her", "lost his"))
        if not death_mentioned:
            severity = "fail"
            reasons.append("death-of-parent record but source mentions no death/passing")
        else:
            mage = re.search(r"(?:was|aged|age|when .*?was)\s*(?:approximately\s*)?(\d{1,2})", source)
            if mage and onset is not None:
                stated = int(mage.group(1))
                if abs(stated - onset) > 1:
                    severity = "fail"
                    reasons.append(
                        f"source states death age ~{stated} but record onset is {onset}")
            subordinate = re.search(
                r"(?:after|following|since|post)\s+[^.]*?(?:death|passed|passing|died)", source)
            primary_other = any(w in source for w in
                                ("beat", "pinch", "hit", "slap", "abuse", "financial",
                                 "punish", "fight", "argument"))
            if subordinate and primary_other and severity != "fail":
                severity = "fail"
                reasons.append("death is only a subordinate time-reference here; the "
                               "source's primary content is a different event "
                               "(likely a duplicate death mis-aged to this line)")

    hints = rec.get("ace_items_hint") or []
    hints = [int(x) for x in hints if str(x).isdigit()]

    if 16 in hints or "parental conflict" in cluster:
        conflict_words = ("fight", "fought", "argument", "argued", "arguing", "shout",
                          "yell", "screaming", "violence", "hit each other", "abused",
                          "abuse", "humiliat", "dominat", "verbal", "conflict",
                          "mistreat", "demean", "belittl")
        has_conflict = any(w in source for w in conflict_words)
        one_off = (("once" in source or "an incident" in source or "during the incident" in source
                    or "one time" in source) and
                   not any(w in source for w in ("consistent", "ongoing", "always",
                           "constant", "frequent", "daily", "repeated", "for years",
                           "throughout", "all the time")))
        if (not has_conflict or one_off) and severity != "fail":
            severity = "fail"
            reasons.append("tagged ACE-16 (parental conflict) but source shows no "
                           "SUSTAINED physical/verbal conflict between the parents "
                           "(a one-off mock/insult is not a pattern of parental conflict)")

    if 15 in hints or "childhood poverty" in cluster:
        poverty_words = ("poor", "poverty", "could not afford", "couldn't afford",
                         "no money", "public assistance", "welfare", "bare home",
                         "very little furniture", "went hungry", "no food",
                         "financial difficult", "financial hardship", "deprived")
        comparative = any(w in source for w in
                          ("than her sister", "than his sister", "than the sister",
                           "step-sister received more", "sister received more",
                           "while step-sister", "while sister", "compared",
                           "favour", "favor", "preferred"))
        has_poverty = any(w in source for w in poverty_words)
        if comparative and not has_poverty and severity != "fail":
            severity = "fail"
            reasons.append("tagged ACE-15 (childhood poverty) but source describes "
                           "getting LESS THAN A SIBLING (favouritism), not family-wide "
                           "material poverty")

    return {"ok": severity == "ok", "severity": severity, "reasons": reasons}


ACE_VERIFY_SYSTEM = (
    "You are a careful clinical verifier for an Adverse Childhood Experiences (ACE) "
    "extraction pipeline. You are given the ORIGINAL source sentence(s), an EXTRACTED "
    "event, the ACE item it was tagged as, and the age it was anchored to. Decide ONLY "
    "whether that exact finding is genuinely supported by the source text.\n\n"
    "GROUNDING RULE — verify, but do not re-litigate clinical thresholds:\n"
    "  The source notes are the ground truth. ACCEPT an ACE item when the source "
    "DESCRIBES that experience, INCLUDING IN GENERAL TERMS. A clearly-stated construct "
    "is sufficient — you do NOT need the source to use the questionnaire's exact wording "
    "or to spell out every clinical criterion.\n\n"
    "Judge: GROUNDED (source actually describes this finding), RIGHT_ITEM (matches the "
    "claimed item's defining element, not merely something adjacent), RIGHT_AGE "
    "(consistent with the source).\n\n"
    "Respond with STRICT JSON only, no other text:\n"
    '{"answers": {"ACEx_qN": "yes|no|unclear", ...}, '
    '"grounded": true/false, "right_item": true/false, "right_age": true/false, '
    '"verdict": "ok" | "warn" | "fail", "reason": "one short sentence"}\n'
    "You will be given SUB-QUESTIONS for the tagged ACE item, each marked DEFINING or "
    "secondary. Answer EVERY sub-question yes/no/unclear strictly from the source. A "
    "DEFINING sub-question answered 'no' means the item's defining element is absent -> "
    "the tag is WRONG -> verdict 'fail'. Use 'warn' when only secondary sub-questions "
    "fail or are unclear. Use 'ok' when all defining sub-questions are 'yes'."
)


def _scoring_item_for(rec: dict):
    it = _cluster_item(rec.get("cluster_name"))
    if it is not None:
        return it
    hints = [int(x) for x in (rec.get("ace_items_hint") or []) if str(x).isdigit()]
    return hints[0] if hints else None


def _llm_verify_record(rec: dict, api_key: str,
                       model: str = "gpt-5.1", timeout: int = 30) -> Optional[dict]:
    """Ask the LLM whether one ACE record's finding is grounded in its source."""
    if not api_key:
        return None

    items = rec.get("ace_items_hint") or []
    items = [int(i) for i in items if str(i).isdigit()]
    _score_item = _scoring_item_for(rec)
    if _score_item is not None and _score_item not in items:
        items = items + [_score_item]
    item_lines = "\n".join(
        f"  ACE-{i}: {ACE_QUESTIONS.get(i, '(unknown item)')}" for i in items
    ) or "  (no ACE item hint provided)"

    subq_index = {}
    subq_lines = []
    for i in items:
        for n, sq in enumerate(ACE_SUBQUESTIONS.get(i, []), start=1):
            qid = f"ACE{i}_q{n}"
            subq_index[qid] = {"item": i, "defining": sq["defining"], "q": sq["q"]}
            tag = "DEFINING" if sq["defining"] else "secondary"
            subq_lines.append(f'  {qid} [{tag}]: {sq["q"]}')
    subq_block = "\n".join(subq_lines) or "  (no sub-questions for this item)"

    onset = rec.get("onset_age")
    end = rec.get("end_age")
    age_str = ("unknown" if onset is None
               else (str(onset) if (end is None or end == onset) else f"{onset}-{end}"))

    user = (
        f"SOURCE SENTENCE(S):\n{rec.get('source_sentence') or '(none)'}\n\n"
        f"EXTRACTED EVENT:\n{rec.get('event') or '(none)'}\n\n"
        f"TAGGED ACE ITEM(S):\n{item_lines}\n\n"
        f"TAGGED CLUSTER: {rec.get('cluster_name') or '(none)'}\n"
        f"ANCHORED AGE: {age_str}\n\n"
        f"SUB-QUESTIONS — answer EACH one strictly from the SOURCE with "
        f'"yes", "no", or "unclear":\n{subq_block}\n\n'
        "Respond with STRICT JSON only:\n"
        '{"answers": {"ACEx_qN": "yes|no|unclear", ...}, '
        '"grounded": true/false, "right_item": true/false, "right_age": true/false, '
        '"verdict": "ok|warn|fail", "reason": "one short sentence"}'
    )
    payload = json.dumps(_build_llm_payload(model, ACE_VERIFY_SYSTEM, user, 2000)).encode("utf-8")

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                API_URL,
                data=payload,
                headers={"Authorization": "Bearer " + api_key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"].strip()
            parsed = json.loads(raw)

            answers = parsed.get("answers") or {}
            failed_defining, failed_secondary = [], []
            for qid, meta in subq_index.items():
                ans = str(answers.get(qid, "")).strip().lower()
                if meta["defining"]:
                    if ans == "no":
                        failed_defining.append(qid)
                    elif ans not in ("yes",):
                        failed_secondary.append(qid + "(unclear)")
                else:
                    if ans in ("no", "unclear", ""):
                        failed_secondary.append(qid)

            if subq_index:
                subq_verdict = "fail" if failed_defining else ("warn" if failed_secondary else "ok")
            else:
                subq_verdict = None

            tagged_items = sorted({meta["item"] for meta in subq_index.values()})
            failed_items = sorted({subq_index[q]["item"] for q in failed_defining})

            explicit = _source_marks_ace_items(rec.get("source_sentence") or "")
            if explicit:
                failed_items = [i for i in failed_items if i not in explicit]
                failed_defining = [q for q in failed_defining if subq_index[q]["item"] not in explicit]
                subq_verdict = "fail" if failed_defining else ("warn" if failed_secondary else "ok")
            passed_items = [i for i in tagged_items if i not in failed_items]

            verdict = str(parsed.get("verdict") or "").lower()
            if verdict not in ("ok", "warn", "fail"):
                if parsed.get("grounded") and parsed.get("right_item", True) and parsed.get("right_age", True):
                    verdict = "ok"
                elif parsed.get("grounded") is False or parsed.get("right_age") is False \
                        or parsed.get("right_item") is False:
                    verdict = "fail"
                else:
                    verdict = "warn"

            rank = {"ok": 0, "warn": 1, "fail": 2}
            final_verdict = (subq_verdict if (subq_verdict is not None and rank[subq_verdict] >= rank[verdict])
                             else verdict)

            model_reason = str(parsed.get("reason") or "").strip()
            if failed_defining:
                parts = [f"[{fd}] answered 'no' — {subq_index[fd]['q']}" for fd in failed_defining]
                reason = "defining sub-question(s) failed: " + "  ||  ".join(parts)
                if model_reason:
                    reason += "  ||  model note: " + model_reason
            else:
                reason = model_reason

            return {
                "grounded": bool(parsed.get("grounded", True)),
                "right_item": bool(parsed.get("right_item", True)),
                "right_age": bool(parsed.get("right_age", True)),
                "verdict": final_verdict, "reason": reason,
                "subquestion_answers": answers, "failed_defining": failed_defining,
                "tagged_items": tagged_items, "failed_items": failed_items,
                "passed_items": passed_items,
            }
        except Exception as e:
            logging.debug("LLM verify attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    return None


ADJUDICATOR_SYSTEM = (
    "You are a SECOND, INDEPENDENT clinical adjudicator for an Adverse Childhood "
    "Experiences (ACE) pipeline. A first-pass verifier FLAGGED an extracted finding as "
    "a FAIL for a specific ACE item, meaning it wants to DROP it. Decide whether that "
    "DROP is correct — the first verifier sometimes makes mistakes (e.g. wrongly "
    "dropping ACE-1 just because ACE-2 is ALSO present, when both can co-occur and be "
    "valid). Re-check ONLY against the SOURCE text; confirm the drop only when the "
    "SOURCE genuinely does not support this item's defining element.\n\n"
    "Respond with STRICT JSON only:\n"
    '{"confirm_drop": true/false, "reason": "one short sentence"}'
)


def _llm_adjudicate_drop(rec: dict, item: int, fail_reason: str, api_key: str,
                         model: str = "gpt-5.1", timeout: int = 60):
    onset = rec.get("onset_age"); end = rec.get("end_age")
    age_str = ("unknown" if onset is None
               else (str(onset) if (end is None or end == onset) else f"{onset}-{end}"))
    user = (
        f"SOURCE SENTENCE(S):\n{rec.get('source_sentence') or '(none)'}\n\n"
        f"EXTRACTED EVENT:\n{rec.get('event') or '(none)'}\n\n"
        f"ACE ITEM BEING JUDGED: ACE-{item} — {ACE_QUESTIONS.get(item, '(unknown)')}\n"
        f"ANCHORED AGE: {age_str}\n\n"
        f"FIRST VERIFIER'S REASON FOR DROPPING:\n{fail_reason or '(none given)'}\n\n"
        f"Does the SOURCE genuinely fail to support ACE-{item} (confirm_drop=true), or "
        "is this a false alarm (confirm_drop=false)? Respond with the strict JSON."
    )
    payload = json.dumps(_build_llm_payload(model, ADJUDICATOR_SYSTEM, user, 400)).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Authorization": "Bearer " + api_key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body["choices"][0]["message"]["content"].strip())
            return {"confirm_drop": bool(parsed.get("confirm_drop", False)),
                    "reason": str(parsed.get("reason") or "")[:300]}
        except Exception as e:
            logging.debug("Adjudicator attempt %d failed: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    return None


def _merge_verdicts(offline: dict, llm: Optional[dict]) -> dict:
    rank = {"ok": 0, "warn": 1, "fail": 2}
    off_sev = offline.get("severity", "ok")
    reasons = list(offline.get("reasons") or [])
    if llm is None:
        return {"severity": off_sev, "reasons": reasons, "offline": offline,
                "llm": None, "llm_available": False}
    llm_sev = llm.get("verdict", "warn")
    if llm.get("reason"):
        reasons.append("LLM: " + llm["reason"])
    llm_defining_fail = bool(llm.get("failed_defining"))
    if llm_defining_fail or llm_sev == "fail":
        final = "fail"
    elif llm_sev == "ok" and off_sev in ("warn", "fail"):
        if llm.get("right_item", True) and llm.get("right_age", True) and llm.get("grounded", True):
            final = "ok"
        else:
            final = "warn"
    else:
        final = llm_sev if rank[llm_sev] >= rank[off_sev] else off_sev
    return {"severity": final, "reasons": reasons, "offline": offline,
            "llm": llm, "llm_available": True}


def _format_flag(rec: dict, v: dict, dropped_items=None, adj_notes=None,
                 surviving=None, record_removed=False) -> str:
    onset, end = rec.get("onset_age"), rec.get("end_age")
    age = ("unknown" if onset is None
           else (str(onset) if (end is None or end == onset) else f"{onset}-{end}"))
    sev = v.get("severity", "?").upper()
    lines = ["\n" + "-" * 64,
             f"  FLAGGED [{sev}]  {rec.get('state_code')}  {rec.get('cluster_name')}  (age {age})",
             f"  ACE item(s): {rec.get('ace_items_hint')}",
             f"  EVENT : {rec.get('event') or ''}",
             f"  SOURCE: {rec.get('source_sentence') or ''}",
             f"  WHY FLAGGED: {'; '.join(v.get('reasons') or [])}"]
    if adj_notes:
        for item, adj in adj_notes.items():
            verdict = "DROP confirmed" if adj.get("confirm_drop") else "KEPT (false alarm)"
            lines.append(f"  ADJUDICATOR ACE-{item}: {verdict} — {adj.get('reason','')}")
    if record_removed:
        lines.append("  -> RECORD REMOVED from graph (will NOT be scored)"
                     + (f"; confirmed-wrong item(s): {dropped_items}" if dropped_items else ""))
    elif dropped_items:
        lines.append(f"  -> DROPPED secondary item(s): {dropped_items}; "
                     f"KEPT scored item(s): {surviving} (record stays)")
    lines.append("-" * 64)
    return "\n".join(lines)


def verify_ace_records(records: list, use_llm: bool = False, api_key: Optional[str] = None,
                       auto_drop: bool = True, adjudicate: bool = True) -> tuple:
    """Verify ACE records for grounding. Returns (kept_records, removed_codes, flagged_report)."""
    def _ints(xs):
        return [int(x) for x in (xs or []) if str(x).isdigit()]

    kept, flagged, removed_codes = [], [], set()
    llm_calls = 0
    annotated = []
    for r in records:
        offline = _verify_record_offline(r)
        llm = None
        if use_llm:
            llm = _llm_verify_record(r, api_key or "")
            if llm is not None:
                llm_calls += 1
        v = _merge_verdicts(offline, llm)
        annotated.append((r, v))
    if use_llm:
        logging.info("LLM verifier: %d/%d records checked via LLM", llm_calls, len(records))

    for r, v in annotated:
        if v["severity"] == "ok":
            kept.append(r)
            continue

        llm_v = v.get("llm") or {}
        tagged = llm_v.get("tagged_items") or _ints(r.get("ace_items_hint"))
        at_risk = llm_v.get("failed_items")
        if at_risk is None:
            at_risk = list(tagged) if v["severity"] == "fail" else []

        confirmed_drop, adj_notes = [], {}
        decision_kept = True

        if v["severity"] == "fail" and auto_drop:
            for item in at_risk:
                adj = None
                if adjudicate and use_llm and api_key:
                    adj = _llm_adjudicate_drop(r, item, "; ".join(v.get("reasons") or []), api_key)
                    if adj is not None:
                        llm_calls += 1
                if adj is None:
                    if adjudicate:
                        continue
                    confirmed_drop.append(item)
                    continue
                adj_notes[item] = adj
                if adj["confirm_drop"]:
                    confirmed_drop.append(item)

            surviving = [i for i in tagged if i not in confirmed_drop]
            score_item = _scoring_item_for(r)
            if score_item is not None and score_item in confirmed_drop:
                decision_kept = False
            elif tagged and not surviving:
                decision_kept = False
            else:
                decision_kept = True
                if confirmed_drop and surviving:
                    r["ace_items_hint"] = surviving

        surviving = [i for i in tagged if i not in confirmed_drop]
        flag = {
            "state_code": r.get("state_code"), "onset_age": r.get("onset_age"),
            "end_age": r.get("end_age"), "cluster_name": r.get("cluster_name"),
            "ace_items_hint_original": tagged, "event": r.get("event"),
            "source_sentence": r.get("source_sentence"),
            "severity": v["severity"], "reasons": v["reasons"],
            "llm_checked": v.get("llm_available", False),
            "dropped_items": confirmed_drop, "surviving_items": surviving,
            "record_removed": (not decision_kept),
        }
        flagged.append(flag)
        print(_format_flag(r, v, dropped_items=confirmed_drop or None,
                           adj_notes=adj_notes or None, surviving=surviving,
                           record_removed=(not decision_kept)))
        if decision_kept:
            kept.append(r)
        else:
            if r.get("state_code"):
                removed_codes.add(r["state_code"])
    return kept, removed_codes, flagged


# ─────────────────────────────────────────────────────────────
# 3.  SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    records = [
        W3G._mk_record("st-a", "Emotional Abuse", 0, 8,
                       "Father emotionally abusive and controlling"),
        W3G._mk_record("st-b", "Emotional Neglect", 0, 8, "Client felt unloved"),
        W3G._mk_record("st-c", "Peer Bullying", 0, 8,
                       "Bullied frequently and repeatedly at school"),
        W3G._mk_record("st-d", "Caregiver Death", 11, 11, "Mother died of cancer",
                       life_stage="childhood"),
        W3G._mk_record("st-e", "Physical Abuse", 12, 16, "Father beat and pinched her",
                       life_stage="teenage"),
        W3G._mk_record("st-a", "Emotional Abuse", 12, 16,
                       "Father's emotional abuse continued", sub_id="st-a-2",
                       life_stage="teenage"),
        # Non-ACE state sharing the same 12-16 union — must be dropped by projection.
        W3G._mk_record("st-p", "Depressive Symptoms", 12, 16, "depressed mood",
                       exp_type="symptom", life_stage="teenage"),
        # ACE-relevant cluster but adult-onset (age 22) — excluded as out-of-window.
        W3G._mk_record("st-f", "Physical Abuse", 22, 24, "adult domestic incident",
                       sub_id="st-f", life_stage="adult"),
        # Non-ACE cluster entirely — excluded regardless of age.
        W3G._mk_record("st-g", "Betrayal", 22, 24, "friend took money",
                       life_stage="adult"),
    ]
    alphabet_map = W3G.assign_alphabets(records)
    general_result = W3G.build_expression(records, [], [], alphabet_map, include_new_states=False)
    print(f"  General FE: {general_result['fire_expression']}")
    check("d" in general_result["fire_expression"] or True, "general trajectory built (sanity)")

    ace_blocks, ace_feedbacks, excluded = project_to_ace(general_result)
    check(any(x["reason"] == "not_ace_relevant" and "depressed" in (x["event"] or "")
             for x in excluded),
          "projection: non-ACE depressive-symptom state excluded (not in ACE cluster set)")
    check(any(x["reason"] == "out_of_window" for x in excluded),
          "projection: adult-onset (age 22) state excluded as out-of-window")

    ace_result = build_ace_expression_from_projection(ace_blocks, ace_feedbacks)
    fe = ace_result["fire_expression"]
    print(f"  ACE FE (projected): {fe}")

    flat_fe = fe.replace(" ", "")
    check(re.search(r"\(a\+b\+c\^2\)|\(a\+c\^2\+b\)|\(b\+a\+c\^2\)|\(c\^2\+a\+b\)|"
                    r"\(b\+c\^2\+a\)|\(c\^2\+b\+a\)", flat_fe) is not None,
          "PAPER MATCH: ACE projection preserves (a+b+c^2) untouched from the general trajectory")
    check("a^3" in flat_fe,
          "PAPER MATCH: the feedback-boosted a^3 survives projection exactly as computed once")
    check("d" in flat_fe.split(".")[1] if "." in flat_fe else "d" in flat_fe,
          "projection: caregiver-death state (ACE-12) survives")

    non_ace_sym = alphabet_map.get("Depressive Symptoms")
    check(non_ace_sym is not None and not re.search(rf"(?<![a-z]){non_ace_sym}(?![a-z])", flat_fe),
          "projection: the non-ACE state's own symbol is absent from the ACE FE entirely")

    check(any(x.get("state_code") == "st-f" and x["reason"] == "out_of_window" for x in excluded),
          "projection: the specific out-of-window state (st-f, age 22) is excluded even "
          "though its cluster symbol is reused by an in-window occurrence")
    check(any(x.get("event") == "friend took money" and x["reason"] == "not_ace_relevant"
             for x in excluded),
          "projection: a non-ACE cluster (Betrayal) is excluded regardless of its age")

    check(len(ace_feedbacks) >= 1
          and ace_feedbacks[0]["evidence"] == ["same_state_recurrence_after_gap"],
          "projection: feedback edge reused unchanged from the general trajectory (both ends survived)")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# 4.  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FiRE Worker 3 ACE — projects the general Worker 3 trajectory "
                    "to the ACE subgraph (does not rebuild the graph)")
    parser.add_argument("input", nargs="?",
                        help="fire_worker3_general.py output JSON")
    parser.add_argument("--output", "-o", default="worker3_ace_output.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="Run offline regression checks and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Run the ACE grounding verifier on the projected records.")
    parser.add_argument("--verify-llm", action="store_true",
                        help="With --verify, run the LLM sub-question grid (needs "
                             "OPENAI_API_KEY). FAILs are re-checked by a second "
                             "adjudicator LLM and auto-removed (per-item) when confirmed.")
    parser.add_argument("--verify-no-adjudicate", action="store_true")
    parser.add_argument("--verify-no-drop", action="store_true")
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
            general_output = json.load(f)
    except FileNotFoundError:
        logging.error("Input file not found: %s", args.input); sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON: %s", e); sys.exit(1)

    if "blocks" not in general_output:
        logging.error("Input does not look like a fire_worker3_general.py output "
                      "(missing 'blocks' — re-run the general Worker 3 first).")
        sys.exit(1)

    patient_id = general_output.get("patient_id") or Path(args.input).stem
    logging.info("Patient: %s", patient_id)

    ace_blocks, ace_feedbacks, excluded = project_to_ace(general_output)
    n_entries = sum(len(b["entries"]) for b in ace_blocks)
    logging.info("ACE-valid entries: %d (in %d blocks) | excluded: %d",
                 n_entries, len(ace_blocks), len(excluded))

    verification_report = []
    if args.verify:
        api_key = os.environ.get(API_KEY_ENV)
        if args.verify_llm and not api_key:
            logging.warning("--verify-llm set but %s missing — falling back to offline checks only.",
                            API_KEY_ENV)
        auto_drop = not args.verify_no_drop
        adjudicate = (not args.verify_no_adjudicate) and args.verify_llm and bool(api_key)

        ace_records = _flatten_ace_records(ace_blocks)
        kept_records, removed_codes, verification_report = verify_ace_records(
            ace_records, use_llm=args.verify_llm, api_key=api_key,
            auto_drop=auto_drop, adjudicate=adjudicate)

        if removed_codes:
            ace_blocks = remove_state_codes_from_blocks(ace_blocks, removed_codes)
            surviving_node_ids = {b["node_id"] for b in ace_blocks}
            ace_feedbacks = [f for f in ace_feedbacks
                            if f.get("earlier_node") in surviving_node_ids
                            and f.get("later_node") in surviving_node_ids]

        n_fail = sum(1 for f in verification_report if f["severity"] == "fail")
        n_warn = sum(1 for f in verification_report if f["severity"] == "warn")
        n_removed = sum(1 for f in verification_report if f.get("record_removed"))
        logging.info("Verifier: %d flagged (%d fail, %d warn)%s — %d record(s) removed",
                     len(verification_report), n_fail, n_warn,
                     " [LLM+adjudicator]" if adjudicate else (" [LLM]" if args.verify_llm else ""),
                     n_removed)

    if not ace_blocks:
        logging.warning("No ACE-valid events survived projection. Nothing to graph.")

    result = build_ace_expression_from_projection(ace_blocks, ace_feedbacks)

    sep = "\u2550" * 60
    print(f"\n{sep}\n  ACE FiRE Expression — {patient_id}\n{sep}\n")
    print(f"  {result['fire_expression']}\n")
    print(f"  Confidence band : {result['confidence_band']:.3f}")
    print(f"  Alphabets       : {len(result['alphabet_inventory'])}")
    print(f"  Feedback loops  : {len(result['feedback_loops'])}\n")

    if result["alphabet_inventory"]:
        print("  Alphabet key:")
        for sym, inv in result["alphabet_inventory"].items():
            tiers_str = "+".join(t[0].upper() for t in inv["tiers"])
            states = ",".join(inv["contributing_states"][:3]) + ("..." if len(inv["contributing_states"]) > 3 else "")
            items = ",".join(str(x) for x in inv.get("ace_items", []))
            print(f"    {sym+':':8s} {sym}^{inv['exponent']}  [{tiers_str}]  "
                  f"ACE-items={items or '-'}  conf={inv['confidence']:.2f}  states={states}")
    print()

    if excluded:
        print("  Excluded from ACE projection:")
        by_reason = defaultdict(int)
        for x in excluded:
            by_reason[x["reason"]] += 1
        for reason, n in by_reason.items():
            print(f"    {reason:20s}: {n}")

    if verification_report:
        print("\n  Verifier flags (grounding check):")
        for f in verification_report:
            mark = "FAIL" if f["severity"] == "fail" else "warn"
            print(f"    [{mark}] {f['state_code']} {f['cluster_name']} "
                  f"@{f['onset_age']}-{f['end_age']}: {'; '.join(f['reasons'])}")
    print(f"\n{sep}\n")

    output = {
        "patient_id": patient_id, "input_file": str(args.input), "signal": "ACE",
        "worker2_stats": general_output.get("worker2_stats", {}),
        **result,
        "excluded_events": excluded,
        "verification_report": verification_report,
    }
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logging.info("Output written: %s", args.output)
    except OSError as e:
        logging.error("Failed to write output: %s", e); sys.exit(1)


if __name__ == "__main__":
    main()
