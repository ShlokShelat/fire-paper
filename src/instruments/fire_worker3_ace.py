"""FiRE Pipeline — Worker 3 (ACE): projects the general trajectory to ACE and
checks each projected item.

Paper concept: the instrument-specific half of stage (iv). It reads the
trajectory built by Worker 3 (general, fire_worker3_general.py) and keeps
the states whose clusters resolve to one of the 14 ACE items through the
cluster -> item table (ace14.py) and that fall inside the 0-18 window
(Supplementary Material, Section S1). The trajectory operators are not
re-implemented here: after projection, unions, self-loops and feedback are
recomputed with Worker 3 (general)'s own functions.

Item check (--verify, with --verify-llm for the model call). For each
projected item the model answers the item's defining sub-questions from
the source, and a second, independent model call must confirm any removal
(Supplementary Material, Section S1). Every main result in the paper was
produced with this check on, i.e. with --verify --verify-llm. Without an API
key the deterministic part of the check still runs, and a failing item is
kept and flagged drop_unconfirmed rather than removed.

Pipeline position: after Worker 3 (general), before Worker 4 (ACE).
Input: a Worker 3 (general) output JSON. Output: the ACE expression,
alphabet, resolved states, excluded events (with the reason), entries left
out for missing age evidence, and the item-check report.

Usage:
  python3 fire_worker3_ace.py worker3_output.json -o worker3_ace_output.json --verify --verify-llm
  python3 fire_worker3_ace.py worker3_output.json -o worker3_ace_output.json --verify   # offline checks only
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

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "algebra"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ace14                      # noqa: E402  (shared ACE-14 table, see ace14.py)
import fire_worker3_general as W3G  # noqa: E402


# ─────────────────────────────────────────────────────────────
# LLM CONSTANTS
# ─────────────────────────────────────────────────────────────

MODEL_NAME  = "gpt-5.1"
API_URL     = "https://api.openai.com/v1/chat/completions"
API_KEY_ENV = "OPENAI_API_KEY"

_REASONING_FAMILY_RE = re.compile(r'^(gpt-5|o1|o3|o4)', re.IGNORECASE)


def _is_reasoning_model(model_name: str) -> bool:
    """True if model_name is a reasoning-family model (GPT-5.x, o1/o3/o4)
    that requires the alternate payload shape built in `_build_llm_payload`."""
    return bool(_REASONING_FAMILY_RE.match(model_name or ""))


def _build_llm_payload(model_name: str, system_prompt: str, user_prompt: str,
                       max_output_tokens: int) -> dict:
    """Build a Chat Completions payload valid for both classic chat
    models and GPT-5-family reasoning models (which reject
    temperature/max_tokens in favour of max_completion_tokens)."""
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
ACE_MIN_AGE, ACE_MAX_AGE = ace14.MIN_AGE, ace14.MAX_AGE

TIER_ORDER = W3G.TIER_ORDER
TIER_MARK  = W3G.TIER_MARK

# The cluster -> item table lives in ace14.py, shared with Worker 4, so the
# projection rule and the scoring rule are one table, not two copies.
_W3_CLUSTER_ITEM = ace14.CLUSTER_ITEM


def _cluster_item(cluster_name: str):
    """Resolve a cluster name to its ACE-14 item number, or None."""
    return ace14.cluster_to_item(cluster_name)


# ─────────────────────────────────────────────────────────────
# 1.  PROJECTION
# ─────────────────────────────────────────────────────────────

def _block_is_no_age(b: dict) -> bool:
    """A block whose entries all lack age evidence (placed in the no-age tier)."""
    return bool(b.get("entries")) and all(e.get("is_no_age") for e in b["entries"])


def _demote_broken_unions(blocks: list) -> None:
    """A union group left with fewer than 2 members is no longer a union."""
    by_ug = defaultdict(list)
    for b in blocks:
        if b.get("union_group"):
            by_ug[b["union_group"]].append(b)
    for members in by_ug.values():
        if len(members) < 2:
            for b in members:
                b["union_group"] = None


def _merge_contiguous_slices(blocks: list) -> list:
    """Re-merge neighbouring interval slices that became identical once
    non-ACE members were dropped: same tier, touching intervals, and the
    same set of cluster symbols. Without this, an ACE state that was cut
    into two slices only because a non-ACE state started part-way through
    would print twice. Exponents and feedback are recomputed afterwards
    (`_tidy`), so only intervals, entries and codes are merged here."""
    slices = []                       # [(tier, interval, [blocks])] in order
    for b in blocks:
        key = (b["tier"], tuple(b["interval"]))
        if slices and slices[-1][0] == key:
            slices[-1][1].append(b)
        else:
            slices.append((key, [b]))

    merged = []
    for key, members in slices:
        if merged and not _block_is_no_age(members[0]):
            pkey, pmembers = merged[-1]
            same_syms = {m["cluster"] for m in members} == {m["cluster"] for m in pmembers}
            touching = key[1][0] == pkey[1][1] + 1
            if pkey[0] == key[0] and touching and same_syms \
                    and not _block_is_no_age(pmembers[0]):
                by_sym = {m["cluster"]: m for m in pmembers}
                for m in members:
                    tgt = by_sym[m["cluster"]]
                    tgt["interval"] = [tgt["interval"][0], m["interval"][1]]
                    seen = {e.get("event_uid") for e in tgt["entries"]}
                    tgt["entries"] += [e for e in m["entries"] if e.get("event_uid") not in seen]
                    tgt["codes"] = list(dict.fromkeys(tgt["codes"] + m["codes"]))
                new_key = (pkey[0], (pkey[1][0], key[1][1]))
                merged[-1] = (new_key, pmembers)
                continue
        merged.append((key, list(members)))
    return [b for _, ms in merged for b in ms]


def _tidy(blocks: list) -> tuple:
    """Re-derive structure from the entries that remain after projection or
    verification removed some: demote broken unions, re-merge identical
    neighbouring slices, then recompute each block's self-loop exponent and
    the feedback links with the general Worker 3's own functions, so the
    algebra stays implemented once and reflects only surviving evidence.
    Returns (blocks, feedbacks)."""
    _demote_broken_unions(blocks)
    blocks = _merge_contiguous_slices(blocks)
    for b in blocks:
        b["is_self_loop"] = W3G._self_loops(b["entries"])
        b["exponent"] = 2 if b["is_self_loop"] else 1
    feedbacks = W3G.detect_feedbacks(blocks)
    fb_count = defaultdict(int)
    for f in feedbacks:
        fb_count[f["later_node"]] += 1
    for b in blocks:
        b["feedback_count"] = fb_count[b["node_id"]]
    return blocks, feedbacks


def project_to_ace(general_output: dict) -> tuple:
    """Project the general Worker 3 trajectory to the ACE subgraph.

    Keeps an entry only if its cluster resolves to one of the 14 items
    (ace14.item_for_entry), and keeps a timeline slice only inside the
    0-18 window: a slice that starts after 18 is dropped, and a slice that
    straddles 18 is cut at 18, so no adult part of a span can appear in the
    ACE expression. Each entry is also checked on its own onset, since a
    slice can hold an entry that starts after 18 (an adult occurrence of a
    cluster can share a teenage-tier slice). An ACE entry with no age
    evidence cannot be shown to fall inside the window, so it is left out
    of the expression and the score until a psychotherapist confirms its
    age; it is listed in `age_unverified` so the item is visible, not lost.
    Returns (ace_blocks, ace_feedbacks, excluded, age_unverified).
    """
    import copy

    raw_blocks = general_output.get("blocks") or []
    excluded, age_unverified = [], []
    reported = set()

    def _exclude(e, reason, detail=None, item=None):
        key = (e.get("event_uid") or e.get("event"), reason)
        if key in reported:
            return
        reported.add(key)
        row = {"event": e.get("event") or "", "reason": reason,
               "state_code": e.get("state_code"), "event_uid": e.get("event_uid")}
        if detail:
            row["detail"] = detail
        if item is not None:
            # Kept so the notes-plus-form step (fire_reconcile_ace.py) can cite
            # an after-18 event as evidence that the item does not count.
            row.update({"ace_item": item, "source_sentence": e.get("source_sentence"),
                        "onset_age": e.get("onset_age"), "end_age": e.get("end_age")})
        excluded.append(row)

    survivors = []
    for raw_b in raw_blocks:
        b = copy.deepcopy(raw_b)
        cluster_name = b.get("cluster_name", "")
        no_age = _block_is_no_age(b)
        lo, hi = b["interval"]

        kept_entries = []
        for e in b.get("entries", []):
            item, source = ace14.item_for_entry(e, e.get("cluster_name") or cluster_name)
            if item is None:
                _exclude(e, "not_ace_relevant")
                continue
            if no_age:
                _exclude(e, "no_age_pending_review",
                         "no age evidence: left out of the ACE score until a "
                         "psychotherapist confirms the age")
                if e.get("event_uid") not in {x["event_uid"] for x in age_unverified}:
                    age_unverified.append({"event_uid": e.get("event_uid"),
                                           "state_code": e.get("state_code"),
                                           "cluster_name": e.get("cluster_name") or cluster_name,
                                           "event": e.get("event"),
                                           "source_sentence": e.get("source_sentence"),
                                           "ace_item": item})
                continue
            onset = e.get("onset_age")
            if lo > ACE_MAX_AGE or (onset is not None and onset > ACE_MAX_AGE):
                _exclude(e, "out_of_window",
                         f"onset {onset}, slice {lo}-{hi}: starts after {ACE_MAX_AGE} "
                         f"(ACE window is {ACE_MIN_AGE}-{ACE_MAX_AGE} inclusive)",
                         item=item)
                continue
            e["ace_item"] = item
            e["ace_item_source"] = source
            end = e.get("end_age")
            e["age_clamped"] = end is not None and end > ACE_MAX_AGE
            if e["age_clamped"]:
                e["original_end_age"] = end
                e["end_age"] = ACE_MAX_AGE
            kept_entries.append(e)

        if not kept_entries:
            continue
        if hi > ACE_MAX_AGE:
            b["interval"] = [lo, ACE_MAX_AGE]
            b["interval_clamped"] = True
        b["entries"] = kept_entries
        b["codes"] = list(dict.fromkeys(e["state_code"] for e in kept_entries
                                        if e.get("state_code")))
        survivors.append(b)

    ace_blocks, ace_feedbacks = _tidy(survivors)
    return ace_blocks, ace_feedbacks, excluded, age_unverified


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
                it = e.get("ace_item")
                if it is not None and it not in ace_items:
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
                    "ace_item": e.get("ace_item"),
                    "ace_item_source": e.get("ace_item_source"),
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
    """One record per projected sub-event (event_uid), however many
    interval slices it spans, so each is verified exactly once."""
    out, by_uid = [], {}
    for b in ace_blocks:
        for e in b["entries"]:
            uid = e.get("event_uid") or f"{e.get('state_code')}:{e.get('event')}"
            if uid in by_uid:
                by_uid[uid]["node_ids"].append(b["node_id"])
                continue
            # Verification judges the finding against the source, so it sees the
            # real span ("15 to 25"), not the end clamped to 18 for projection.
            rec = {
                "event_uid": uid,
                "state_code": e.get("state_code"), "cluster_name": b["cluster_name"],
                "event": e.get("event"), "source_sentence": e.get("source_sentence"),
                "onset_age": e.get("onset_age"),
                "end_age": e.get("original_end_age", e.get("end_age")),
                "ace_item": e.get("ace_item"), "ace_item_source": e.get("ace_item_source"),
                "perpetrator": e.get("perpetrator"), "perpetrator_class": e.get("perpetrator_class"),
                "supporting_quote": e.get("supporting_quote"), "quote_grounded": e.get("quote_grounded"),
                "node_ids": [b["node_id"]],
            }
            by_uid[uid] = rec
            out.append(rec)
    return out


def remove_entries_from_blocks(ace_blocks: list, feedbacks: list, uids_to_remove: set) -> tuple:
    """Remove specific sub-events (by event_uid) from every slice they
    occupy, then re-derive unions, slices, exponents and feedback links.
    Other sub-events that share the same state code are untouched: a wrong
    finding in one period must not delete a correct one in another."""
    if not uids_to_remove:
        return ace_blocks, feedbacks
    kept_blocks = []
    for b in ace_blocks:
        kept_entries = [e for e in b["entries"] if e.get("event_uid") not in uids_to_remove]
        if not kept_entries:
            continue
        b["entries"] = kept_entries
        b["codes"] = list(dict.fromkeys(
            e["state_code"] for e in kept_entries if e.get("state_code")))
        kept_blocks.append(b)
    return _tidy(kept_blocks)


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
    10: "The child lived in a dangerous neighbourhood or saw people assaulted (community violence).",
    11: "Other kids, including siblings, often hit, threatened, picked on, or insulted the child (peer/sibling bullying).",
    12: "The child often felt lonely/rejected/isolated from peers (peer rejection or social isolation).",
    13: "The family was very poor or on public assistance for a sustained period (childhood poverty).",
    14: "The child's parents had physical/verbal fights with EACH OTHER (parental conflict).",
}

ACE_SUBQUESTIONS = {
    1: [
        {"q": "Does the source describe the child being emotionally or verbally mistreated by a PARENT or HOUSEHOLD ADULT — e.g. insulted, humiliated, put down, sworn at, threatened, harshly criticized, OR described in general terms as 'emotionally abusive' / 'verbally abusive'? (a general statement such as 'father was emotionally abusive' or 'critical and humiliating' IS sufficient — you do NOT need a specific quoted insult. Bare neutral 'strict' or 'shouted' with no insult/humiliation/criticism/threat is weaker but, combined with any belittling, still qualifies.)", "defining": True},
        {"q": "Is the actor a parent / adult in the household (NOT a peer/sibling/outsider, which would be ACE-11)?", "defining": True},
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
        {"q": "Does the source describe a dangerous neighbourhood, living amid violence, or witnessing people being assaulted?", "defining": True},
    ],
    11: [
        {"q": "Does the source describe OTHER KIDS or SIBLINGS hitting, threatening, picking on, bullying, or insulting the child?", "defining": True},
        {"q": "Is the actor a peer/sibling (NOT a parent/adult, which would be ACE-1/2)?", "defining": True},
    ],
    12: [
        {"q": "Does the source show the child felt lonely, rejected, or isolated FROM PEERS, had no friends, was excluded, or was kept from peers — as a state or recurring pattern (NOT a single one-off romantic break-up)?", "defining": True},
        {"q": "Is this peer/social isolation (as opposed to ONLY family emotional neglect)? NOTE: ACE-12 and ACE-4 can CO-OCCUR — if both, answer 'yes'.", "defining": False},
    ],
    13: [
        {"q": "Does the source describe HOUSEHOLD material poverty or financial hardship affecting the family's ability to afford basics (bare home, on public assistance, struggled financially)?", "defining": True},
        {"q": "Is this about the FAMILY's means (NOT merely the child having less than a sibling)?", "defining": False},
    ],
    14: [
        {"q": "Does the source describe verbal or physical CONFLICT BETWEEN THE PARENTS — EITHER mutual fighting OR one parent repeatedly abusing / humiliating / dominating the other, including domestic violence between the parents? (one-directional sustained abuse of one parent by the other DOES count)", "defining": True},
        {"q": "Is it a sustained or recurrent pattern rather than a SINGLE isolated incident (e.g. a one-off mock during one event)?", "defining": False},
    ],
}

_ACE_MARK_RE = re.compile(r"\bACEs?\b[\s:\-]*((?:\d{1,2}\s*(?:,|/|&|and|\s)\s*)*\d{1,2})",
                          re.IGNORECASE)


def _source_marks_ace_items(source: str) -> set:
    """Extract ACE item numbers the source text explicitly self-labels
    with (e.g. "ACE 3, 7"), used as a cross-check signal during verification."""
    out = set()
    for m in _ACE_MARK_RE.finditer(source or ""):
        for tok in re.findall(r"\d{1,2}", m.group(1)):
            n = int(tok)
            if 1 <= n <= 14:
                out.add(n)
    return out


def _verify_record_offline(rec: dict) -> dict:
    """Deterministic, offline grounding check for one ACE record.

    Runs three independent checks and returns the worst outcome: (1) did
    Worker 2's supporting_quote actually appear in the source, (2) does
    the record's onset/end age appear anywhere in the source text, and
    (3) two ACE-item-specific false-positive guards — ACE-14 (parental
    conflict) requires a sustained pattern, not a one-off incident, and
    ACE-13 (poverty) must be family-wide hardship, not sibling
    favouritism. Returns {"ok", "severity", "reasons"}.
    """
    reasons = []
    severity = "ok"

    source = str(rec.get("source_sentence") or "").lower()
    onset  = rec.get("onset_age")

    # Check 1: was Worker 2's own supporting quote actually verifiable?
    if rec.get("supporting_quote") is not None and rec.get("quote_grounded") is False:
        severity = "warn"
        reasons.append("Worker 2 supporting_quote not found in source sentence "
                       "(possible extraction from background context, not the source)")

    # Check 2: does the assigned age appear anywhere in the source text?
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

    item = rec.get("ace_item")

    # Check 3a: ACE-14 requires a sustained conflict pattern, not a single incident.
    if item == 14 or "parental conflict" in str(rec.get("cluster_name") or "").lower():
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
            reasons.append("tagged ACE-14 (parental conflict) but source shows no "
                           "SUSTAINED physical/verbal conflict between the parents "
                           "(a one-off mock/insult is not a pattern of parental conflict)")

    # Check 3b: ACE-13 must be family-wide poverty, not sibling favouritism.
    if item == 13 or "childhood poverty" in str(rec.get("cluster_name") or "").lower():
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
            reasons.append("tagged ACE-13 (childhood poverty) but source describes "
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
    """The single ACE-14 item this record projects to (set at projection)."""
    if rec.get("ace_item") is not None:
        return rec["ace_item"]
    return _cluster_item(rec.get("cluster_name"))


def _llm_verify_record(rec: dict, api_key: str,
                       model: str = "gpt-5.1", timeout: int = 30) -> Optional[dict]:
    """Ask the LLM whether one ACE record's finding is grounded in its source."""
    if not api_key:
        return None

    _score_item = _scoring_item_for(rec)
    items = [_score_item] if _score_item is not None else []
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
    """Ask a second, independent model call to confirm or overturn a
    proposed drop of one ACE item, so a single verifier mistake (e.g.
    conflating co-occurring items) doesn't unilaterally remove a valid
    finding. Returns {"confirm_drop", "reason"} or None on failure."""
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
    """Combine the offline deterministic verdict with the LLM verifier's
    (when available) into one severity: the more severe of the two. An LLM
    "ok" may clear an offline *warn*, but never an offline *fail*: a fail
    from either source is a candidate drop that only the adjudicator can
    overturn (Supplementary Material, Section S1)."""
    rank = {"ok": 0, "warn": 1, "fail": 2}
    off_sev = offline.get("severity", "ok")
    reasons = list(offline.get("reasons") or [])
    if llm is None:
        return {"severity": off_sev, "reasons": reasons, "offline": offline,
                "llm": None, "llm_available": False}
    llm_sev = llm.get("verdict", "warn")
    if llm.get("failed_defining"):
        llm_sev = "fail"
    if llm.get("reason"):
        reasons.append("LLM: " + llm["reason"])
    if llm_sev == "ok" and off_sev == "warn" and llm.get("grounded", True) \
            and llm.get("right_item", True) and llm.get("right_age", True):
        final = "ok"
    else:
        final = llm_sev if rank[llm_sev] >= rank[off_sev] else off_sev
    return {"severity": final, "reasons": reasons, "offline": offline,
            "llm": llm, "llm_available": True}


def _format_flag(rec: dict, v: dict, decision: str, adj: Optional[dict] = None) -> str:
    """Render a human-readable audit block for one flagged ACE record,
    for the --verify report and CLI output."""
    onset, end = rec.get("onset_age"), rec.get("end_age")
    age = ("unknown" if onset is None
           else (str(onset) if (end is None or end == onset) else f"{onset}-{end}"))
    sev = v.get("severity", "?").upper()
    lines = ["\n" + "-" * 64,
             f"  FLAGGED [{sev}]  {rec.get('state_code')}  {rec.get('cluster_name')}  (age {age})",
             f"  ACE item: {rec.get('ace_item')}",
             f"  EVENT : {rec.get('event') or ''}",
             f"  SOURCE: {rec.get('source_sentence') or ''}",
             f"  WHY FLAGGED: {'; '.join(v.get('reasons') or [])}"]
    if adj is not None:
        verdict = "DROP confirmed" if adj.get("confirm_drop") else "KEPT (false alarm)"
        lines.append(f"  ADJUDICATOR: {verdict} — {adj.get('reason', '')}")
    lines.append(f"  -> {decision}")
    lines.append("-" * 64)
    return "\n".join(lines)


def verify_ace_records(records: list, use_llm: bool = False, api_key: Optional[str] = None,
                       auto_drop: bool = True, adjudicate: bool = True,
                       drop_without_adjudicator: bool = False) -> tuple:
    """Verify every projected ACE record and remove the ones confirmed wrong.

    Input: one record per projected sub-event (`_flatten_ace_records`).
    A record whose verdict is "fail" (from the offline deterministic checks
    or from the LLM verifier) is a candidate drop. Only a confirmed drop
    removes it (Supplementary Material, Section S1):
      - with the adjudicator on, a second, independent model call must
        confirm the drop; a disputed drop keeps the record, and a failed
        adjudicator call keeps it too (a failed call is never a verdict);
      - with no adjudicator available (offline run, or no API key), the
        record is kept and flagged `drop_unconfirmed` for review;
      - only an explicit opt-out of adjudication (`drop_without_adjudicator`,
        CLI --verify-no-adjudicate) lets the failing check's verdict stand.
    Removal is per sub-event (event_uid), never per state code.
    Output: (kept_records, removed_event_uids, flagged_report).
    """
    kept, flagged, removed_uids = [], [], set()
    llm_calls = 0
    annotated = []
    for r in records:
        offline = _verify_record_offline(r)
        llm = None
        if use_llm:
            llm = _llm_verify_record(r, api_key or "")
            if llm is not None:
                llm_calls += 1
        annotated.append((r, _merge_verdicts(offline, llm)))
    if use_llm:
        logging.info("LLM verifier: %d/%d records checked via LLM", llm_calls, len(records))

    for r, v in annotated:
        if v["severity"] == "ok":
            kept.append(r)
            continue

        adj, removed = None, False
        if v["severity"] == "warn" or not auto_drop:
            decision = "KEPT (warning only)" if v["severity"] == "warn" else "KEPT (--verify-no-drop)"
        elif adjudicate and use_llm and api_key:
            adj = _llm_adjudicate_drop(r, r.get("ace_item"),
                                       "; ".join(v.get("reasons") or []), api_key)
            if adj is None:
                decision = "KEPT (adjudicator call failed — not treated as a verdict)"
            elif adj["confirm_drop"]:
                removed, decision = True, "RECORD REMOVED (drop confirmed by adjudicator)"
            else:
                decision = "KEPT (adjudicator disputed the drop)"
        elif drop_without_adjudicator:
            basis = ("LLM verifier" if (v.get("llm") or {}).get("verdict") == "fail"
                     or (v.get("llm") or {}).get("failed_defining")
                     else "offline deterministic check")
            removed, decision = True, f"RECORD REMOVED ({basis}; adjudication switched off)"
        else:
            decision = ("KEPT — drop_unconfirmed: failed a check but no adjudicator "
                        "was available to confirm the drop; review recommended")

        flagged.append({
            "event_uid": r.get("event_uid"), "state_code": r.get("state_code"),
            "onset_age": r.get("onset_age"), "end_age": r.get("end_age"),
            "cluster_name": r.get("cluster_name"), "ace_item": r.get("ace_item"),
            "event": r.get("event"), "source_sentence": r.get("source_sentence"),
            "severity": v["severity"], "reasons": v["reasons"],
            "llm_checked": v.get("llm_available", False),
            "adjudicator": adj, "decision": decision, "record_removed": removed,
            "drop_unconfirmed": (v["severity"] == "fail" and auto_drop and not removed
                                 and adj is None),
        })
        print(_format_flag(r, v, decision, adj))
        if removed:
            removed_uids.add(r.get("event_uid"))
        else:
            kept.append(r)
    return kept, removed_uids, flagged


# ─────────────────────────────────────────────────────────────
# 3.  SELF-TEST
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    """Offline regression guard: projects the paper's worked example and
    checks the exact ACE expression, then exercises the window cut at 18,
    the ignored-hint guard, per-record removal, and payload construction."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    def general(records):
        return W3G.build_expression(records, [], [], W3G.assign_alphabets(records))

    def project(records):
        g = general(records)
        blocks, fbs, excl, unverified = project_to_ace(g)
        return g, blocks, fbs, excl, unverified, build_ace_expression_from_projection(blocks, fbs)

    # Worked example (main paper, Section V): E_ACE = (a + b + c^2) . (e + a^3)
    g, blocks, fbs, excl, _, res = project(W3G.worked_example_records())
    print(f"  General FE: {g['fire_expression']}")
    print(f"  ACE FE    : {res['fire_expression']}")
    check(res["fire_expression"].replace(" ", "") == "[C](a+b+c^2).(e+a^3)",
          "PAPER MATCH: ACE projection of the worked example is exactly (a+b+c^2) . (e+a^3)")
    check(any(x["reason"] == "not_ace_relevant" and x["state_code"] == "q-p" for x in excl),
          "depressive symptoms (outcome, not adversity) excluded at projection")
    check(any(x["reason"] == "not_ace_relevant" and x["state_code"] == "q-d" for x in excl),
          "caregiver death has no ACE-14 item and is excluded at projection")
    check(all(x["reason"] != "not_ace_relevant" or x["state_code"] not in ("q-a", "q-b", "q-c", "q-e")
              for x in excl), "the four ACE states all survive projection")
    check(len(fbs) == 1 and fbs[0]["cluster"] == "a",
          "the emotional-abuse feedback edge survives projection (both ends in window)")
    items = sorted({e["ace_item"] for b in blocks for e in b["entries"]})
    check(items == [1, 2, 4, 11], "projected items are exactly {1, 2, 4, 11}")

    # A span that straddles 18 is cut at 18; a span starting after 18 is dropped.
    r_win = [
        W3G._mk_record("w1", "Physical Abuse", 15, 25, "Father hit client", sub_id="w1"),
        W3G._mk_record("w2", "Emotional Abuse", 22, 24, "Partner insulted client", sub_id="w2",
                       life_stage="adult"),
    ]
    _, b_win, _, x_win, _, r_win_res = project(r_win)
    check(all(b["interval"][1] <= ACE_MAX_AGE for b in b_win),
          "no projected slice extends past 18 (straddling span cut at 18)")
    check("[A]" not in r_win_res["fire_expression"],
          "no adult-tier node appears in the ACE expression")
    check(any(x["state_code"] == "w2" and x["reason"] == "out_of_window" for x in x_win),
          "an ACE cluster whose span starts after 18 is excluded as out-of-window")

    # `ace_items_hint` uses a different numbering and is ignored; explicit ACE-14 hints are honoured.
    leg = W3G._mk_record("h1", "Death of Father", 9, 9, "Father died", sub_id="h1")
    leg["ace_items_hint"] = [12]          # different numbering, ignored
    new = W3G._mk_record("h2", "Unnamed Construct", 6, 7, "Peers excluded client", sub_id="h2")
    new["ace14_items_hint"] = [12]        # ACE-14 numbering: 12 = peer rejection
    _, b_h, _, x_h, _, _ = project([leg, new])
    proj = {e["state_code"]: (e["ace_item"], e["ace_item_source"]) for b in b_h for e in b["entries"]}
    check("h1" not in proj and any(x["state_code"] == "h1" for x in x_h),
          "an ignored ace_items_hint never projects a non-ACE cluster")
    check(proj.get("h2") == (12, "hint"),
          "an explicit ace14_items_hint resolves an unmapped cluster")

    # Per-record removal: a wrong finding in one period must not delete
    # a correct finding of the same state code in another period.
    pov_ok = W3G._mk_record("p1", "Childhood Poverty", 4, 6,
                            "Family could not afford food", sub_id="p1-a",
                            source="Age 4-6: the family could not afford food and was very poor.")
    pov_bad = W3G._mk_record("p1", "Childhood Poverty", 10, 11,
                             "Client received fewer gifts than her sister", sub_id="p1-b",
                             source="Age 10-11: received fewer gifts than her sister; "
                                    "step-sister received more.")
    _, b_p, f_p, _, _, _ = project([pov_ok, pov_bad])
    recs = _flatten_ace_records(b_p)
    check(len(recs) == 2, "each projected sub-event is verified exactly once")
    _, removed0, report0 = verify_ace_records(recs, use_llm=False, adjudicate=False)
    check(not removed0 and [f["event_uid"] for f in report0 if f["drop_unconfirmed"]] == ["p1-b"],
          "with no adjudicator available, a failed check keeps the record and flags "
          "it drop_unconfirmed (only a confirmed drop removes an item)")
    _, removed, report = verify_ace_records(recs, use_llm=False, adjudicate=False,
                                            drop_without_adjudicator=True)
    check(removed == {"p1-b"}, "with adjudication explicitly off, the ACE-13 favouritism "
                               "check removes only the sibling-comparison sub-event")
    b_p2, _ = remove_entries_from_blocks(b_p, f_p, removed)
    left = [e["event_uid"] for b in b_p2 for e in b["entries"]]
    check(left == ["p1-a"], "the correct poverty finding with the same state code is kept")

    # A failed adjudicator call keeps the record (never a negative verdict).
    import unittest.mock as _mock
    with _mock.patch(__name__ + "._llm_adjudicate_drop", return_value=None), \
         _mock.patch(__name__ + "._llm_verify_record", return_value=None):
        _, removed_adj, _ = verify_ace_records(recs, use_llm=True, api_key="x", adjudicate=True)
    check(not removed_adj, "with the adjudicator on, a failed adjudicator call keeps the record")
    with _mock.patch(__name__ + "._llm_adjudicate_drop",
                     return_value={"confirm_drop": True, "reason": "test"}), \
         _mock.patch(__name__ + "._llm_verify_record", return_value=None):
        _, removed_conf, _ = verify_ace_records(recs, use_llm=True, api_key="x", adjudicate=True)
    check(removed_conf == {"p1-b"},
          "an offline fail confirmed by the adjudicator is removed even when the LLM is on")

    # Removing the anchoring occurrence keeps a genuine later reactivation.
    r_fb = [
        W3G._mk_record("a1", "Emotional Abuse", 0, 1, "Father insulted client", sub_id="a1-x"),
        W3G._mk_record("a1", "Emotional Abuse", 4, 5, "Father insulted client again",
                       sub_id="a1-y"),
        W3G._mk_record("a1", "Emotional Abuse", 8, 9, "Father insulted client once more",
                       sub_id="a1-z"),
    ]
    _, b_fb, f_fb, _, _, _ = project(r_fb)
    b_fb2, f_fb2 = remove_entries_from_blocks(b_fb, f_fb, {"a1-x"})
    fe_fb = build_ace_expression_from_projection(b_fb2, f_fb2)["fire_expression"]
    check(fe_fb.replace(" ", "") == "[C]a.a^3",
          "after removing the first occurrence, feedback is recomputed: a . a^3")

    # No-age ACE entries are left out of the score, pending age review.
    r_na = [W3G._mk_record("n1", "Physical Abuse", None, None, "Father hit client", sub_id="n1"),
            W3G._mk_record("n2", "Peer Bullying", 7, 8, "Peers bullied client", sub_id="n2")]
    _, b_na, _, x_na, unv_na, res_na = project(r_na)
    check(all(e["state_code"] != "n1" for b in b_na for e in b["entries"])
          and any(x["state_code"] == "n1" and x["reason"] == "no_age_pending_review" for x in x_na)
          and [u["event_uid"] for u in unv_na] == ["n1"] and "[N]" not in res_na["fire_expression"],
          "an ACE entry with no age is left out of the score and listed for age review")

    # An adult occurrence sharing a teenage-tier slice is still out of window.
    r_19 = [W3G._mk_record("t1", "Physical Abuse", 15, 18, "Father hit client", sub_id="t1"),
            W3G._mk_record("t2", "Physical Abuse", 19, 19, "Partner hit client", sub_id="t2")]
    _, b_19, _, x_19, _, _ = project(r_19)
    check(all(e["state_code"] != "t2" for b in b_19 for e in b["entries"])
          and any(x["state_code"] == "t2" and x["reason"] == "out_of_window" for x in x_19),
          "an entry starting at 19 is excluded even inside a slice that starts at 15")

    # A span cut at 18 carries end_age 18 to the scorer (duration check).
    _, b_cl, _, _, _, res_cl = project([W3G._mk_record("c1", "Childhood Poverty", 14, 25,
                                                       "Family was very poor", sub_id="c1")])
    st = {r["state_code"]: r for r in res_cl["resolved_states"]}
    check(st["c1"]["end_age"] == ACE_MAX_AGE,
          "a span straddling 18 reaches the scorer with end_age clamped to 18")

    # Sub-event ids are unique even when Worker 2 repeats a sub_id.
    w2 = {"events": [{"status": "COMPOUND", "line_number": 1, "source_sentence": "x",
                      "sub_events": [
                          {"sub_id": "u_a", "status": "MAPPED", "state_code": "s1",
                           "cluster_name": "Peer Bullying", "event": "bullied", "onset_age": 8},
                          {"sub_id": "u_a", "status": "MAPPED", "state_code": "s2",
                           "cluster_name": "Physical Abuse", "event": "hit", "onset_age": 8}]}]}
    res_w2, _, _ = W3G.parse_worker2_output(w2)
    check(len({r["event_uid"] for r in res_w2}) == 2,
          "two sub-events sharing a Worker 2 sub_id still get distinct ids")

    p = _build_llm_payload("gpt-5.1", "s", "u", 100)
    check("temperature" not in p and p.get("reasoning_effort") == "none"
          and p.get("max_completion_tokens") == 100, "GPT-5 payload shape")
    p = _build_llm_payload("gpt-4.1", "s", "u", 100)
    check(p.get("temperature") == 0.0 and p.get("max_tokens") == 100, "classic payload shape")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# 4.  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: load a general Worker 3 output file, project it
    to the ACE subgraph, optionally run grounding verification (removing
    records confirmed wrong), rebuild the ACE expression from whatever
    survives, and write the result to --output."""
    parser = argparse.ArgumentParser(
        description="FiRE Worker 3 (ACE): project the general trajectory to ACE items "
                    "in ages 0-18 and check each item. The paper's runs used "
                    "--verify --verify-llm.")
    parser.add_argument("input", nargs="?",
                        help="Worker 3 (all instruments) output (worker3_output.json)")
    parser.add_argument("--output", "-o", default="worker3_ace_output.json",
                        help="Output path (default: worker3_ace_output.json)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument("--selftest", action="store_true",
                        help="Run offline regression checks and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Run the ACE grounding verifier on the projected records.")
    parser.add_argument("--verify-llm", action="store_true",
                        help="With --verify, also run the LLM sub-question grid (needs "
                             "OPENAI_API_KEY). Every FAIL is re-checked by a second "
                             "adjudicator call and removed only when it confirms.")
    parser.add_argument("--verify-no-adjudicate", action="store_true",
                        help="Let a failed check remove a record without the second "
                             "adjudicator call. Default: only an adjudicator-confirmed "
                             "drop removes a record; otherwise it is kept and flagged.")
    parser.add_argument("--verify-no-drop", action="store_true",
                        help="With --verify, report failing items but never remove one.")
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

    ace_blocks, ace_feedbacks, excluded, age_unverified = project_to_ace(general_output)
    n_entries = sum(len(b["entries"]) for b in ace_blocks)
    logging.info("ACE-valid entries: %d (in %d blocks) | excluded: %d",
                 n_entries, len(ace_blocks), len(excluded))

    verification_report = []
    if args.verify:
        api_key = os.environ.get(API_KEY_ENV)
        if args.verify_llm and not api_key:
            logging.warning("--verify-llm set but %s missing — falling back to offline checks only.",
                            API_KEY_ENV)
        use_llm = args.verify_llm and bool(api_key)
        auto_drop = not args.verify_no_drop
        adjudicate = (not args.verify_no_adjudicate) and use_llm
        drop_without_adjudicator = args.verify_no_adjudicate

        ace_records = _flatten_ace_records(ace_blocks)
        _, removed_uids, verification_report = verify_ace_records(
            ace_records, use_llm=use_llm, api_key=api_key,
            auto_drop=auto_drop, adjudicate=adjudicate,
            drop_without_adjudicator=drop_without_adjudicator)
        ace_blocks, ace_feedbacks = remove_entries_from_blocks(
            ace_blocks, ace_feedbacks, removed_uids)

        n_fail = sum(1 for f in verification_report if f["severity"] == "fail")
        n_warn = sum(1 for f in verification_report if f["severity"] == "warn")
        logging.info("Verifier: %d flagged (%d fail, %d warn)%s — %d record(s) removed",
                     len(verification_report), n_fail, n_warn,
                     " [LLM+adjudicator]" if adjudicate else (" [LLM]" if use_llm else " [offline]"),
                     len(removed_uids))

    if not ace_blocks:
        logging.warning("No ACE-valid events survived projection. Nothing to graph.")

    result = build_ace_expression_from_projection(ace_blocks, ace_feedbacks)

    sep = "═" * 60
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
    if age_unverified:
        print(f"  ! {len(age_unverified)} ACE entr{'y' if len(age_unverified) == 1 else 'ies'} "
              f"with no age evidence left out of the score, pending psychotherapist age review:")
        for x in age_unverified:
            print(f"      ACE-{x['ace_item']}  {x['state_code']}  {x['event']}")

    if verification_report:
        print("\n  Verifier flags (grounding check):")
        for f in verification_report:
            mark = "FAIL" if f["severity"] == "fail" else "warn"
            print(f"    [{mark}] {f['state_code']} {f['cluster_name']} "
                  f"@{f['onset_age']}-{f['end_age']}: {f['decision']}")
    print(f"\n{sep}\n")

    output = {
        "patient_id": patient_id, "input_file": str(args.input), "signal": "ACE",
        "instrument": "ACE-14",
        "worker2_stats": general_output.get("worker2_stats", {}),
        **result,
        "excluded_events": excluded,
        "age_unverified": age_unverified,
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
