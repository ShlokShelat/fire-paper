#!/usr/bin/env python3
"""
Trim a full state-space JSON to what the knowledge-matched prompt uses.

The knowledge-matched baseline (fire_llm_knowledge_baseline.py) inserts a
state-space file into its prompt as given. Supplementary Material, Section S4
lists what that prompt contains: the clusters and their constructs; each
state with its description, allowed actors and roles, age window and example
phrasings; the 69-entry cluster-to-item table; and the rules for each item.
A working state-space file holds more than that (review notes, provisional
metadata, where in a note a state was seen, the quote that created it), and
none of it may reach the prompt. This script keeps only:

  clusters          cluster_id, name, construct, item
  states            state_id, cluster_id, name, description, experience_type,
                    allowed_actors, age_window, example_phrasings, ace_eligible,
                    ace_eligibility_reason, counts_with, also_counts_as
  cluster_to_item   the 69-entry table; must equal src/instruments/ace14.py
                    (inserted from there when the input has none)
  non_ace_clusters  kept as given, if present
  item_rules        the per-item verification rules, kept as given, if present

Every other key is dropped. Inside kept cluster and state fields, any nested
key that names a note, source, location, line, patient, review, quote,
evidence or provenance is dropped as well; inside the item rules and the
non-ACE list, nested review, location, patient and provenance keys are. Two input layouts are accepted:

  flat     {"clusters": [...] or {...}, "states": [...] or {...}, ...}
  nested   {"domains": {d: {"clusters": {c: {"cluster_name", "cluster_definition",
            "states": [{"code", "name", "description", ...}]}}}}}
           (the layout Worker 2 reads and writes). A nested cluster's
           cluster_definition becomes its construct, a state's code its
           state_id, and a cluster with no item is resolved through the
           cluster-to-item table ("none" if it has no item).

Usage:
  python3 trim_state_space.py full_state_space.json -o state_space_for_prompt.json
  python3 trim_state_space.py --selftest

Offline; makes no model call. The output is the file to pass to
fire_llm_knowledge_baseline.py --state-space.
"""

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "instruments"))
import ace14  # noqa: E402  (the canonical 69-entry table; data only)

CLUSTER_FIELDS = ("cluster_id", "name", "construct", "item")
STATE_FIELDS = ("state_id", "cluster_id", "name", "description", "experience_type",
                "allowed_actors", "age_window", "example_phrasings", "ace_eligible",
                "ace_eligibility_reason", "counts_with", "also_counts_as")

# Accepted spellings of the top-level sections, mapped to the output key.
_TABLE_KEYS = ("cluster_to_item", "cluster_to_item_table", "cluster_item_table",
               "cluster_item_map")
_NON_ACE_KEYS = ("non_ace_clusters", "non_ace")
_RULE_KEYS = ("item_rules", "item_verification_rules", "verification_rules",
              "per_item_rules", "rules")

# Aliases inside a cluster or state record, mapped to the kept field name.
_CLUSTER_ALIASES = {"cluster_name": "name", "cluster_definition": "construct",
                    "definition": "construct", "ace_item": "item", "code": "cluster_id",
                    "id": "cluster_id", "key": "cluster_id"}
_STATE_ALIASES = {"code": "state_id", "id": "state_id", "state_code": "state_id",
                  "state_name": "name"}

# Nested keys that point back into notes or review records.
_PRIVATE_KEY_RE = re.compile(
    r"note|source|location|line|patient|review|quote|evidence|provenance|reviewer|"
    r"individual|session|unit_id|sub_id", re.IGNORECASE)


# Narrower pattern for the item rules and non-ACE list, whose own fields may
# legitimately be called "evidence" or "source".
_REVIEW_KEY_RE = re.compile(
    r"review|location|patient|provenance|note_ref|source_note|line_ref|individual",
    re.IGNORECASE)


def _scrub(value, pattern=_PRIVATE_KEY_RE):
    """Copy a kept value, dropping nested keys that match `pattern`."""
    if isinstance(value, dict):
        return {k: _scrub(v, pattern) for k, v in value.items()
                if not pattern.search(str(k))}
    if isinstance(value, list):
        return [_scrub(v, pattern) for v in value]
    return value


def _pick(record: dict, fields, aliases) -> dict:
    """Keep only `fields` from one record, accepting the given aliases."""
    out = {}
    for k, v in record.items():
        name = k if k in fields else aliases.get(k)
        if name in fields and name not in out:
            out[name] = _scrub(v)
    return {f: out[f] for f in fields if f in out}


def _records(section, id_field: str) -> list:
    """A list of records from a list, or from a dict keyed by id."""
    if isinstance(section, list):
        return [r for r in section if isinstance(r, dict)]
    if isinstance(section, dict):
        return [dict({id_field: k}, **r) if id_field not in r else dict(r)
                for k, r in section.items() if isinstance(r, dict)]
    return []


def _table_from(section) -> dict:
    """Normalise a cluster-to-item table to {lower-case entry: item}."""
    table = {}
    if isinstance(section, dict):
        for k, v in section.items():
            if isinstance(v, list):                   # {item: [entries]}
                for entry in v:
                    table[str(entry).strip().lower()] = int(k)
            else:                                     # {entry: item}
                table[str(k).strip().lower()] = int(v)
    elif isinstance(section, list):                   # [{"cluster": .., "item": ..}]
        for r in section:
            if isinstance(r, dict):
                name = r.get("cluster") or r.get("name") or r.get("entry")
                table[str(name).strip().lower()] = int(r.get("item"))
    return table


def _first(data: dict, keys):
    return next((data[k] for k in keys if k in data), None)


def trim_state_space(full: dict) -> dict:
    """Return the prompt-ready state space (see the module docstring)."""
    if not isinstance(full, dict):
        raise ValueError("state space must be a JSON object")

    clusters, states = [], []
    if "domains" in full:
        for dom in (full.get("domains") or {}).values():
            for ckey, clu in ((dom or {}).get("clusters") or {}).items():
                c = _pick(dict({"cluster_id": ckey}, **clu), CLUSTER_FIELDS, _CLUSTER_ALIASES)
                clusters.append(c)
                for st in clu.get("states") or []:
                    s = _pick(st, STATE_FIELDS, _STATE_ALIASES)
                    s.setdefault("cluster_id", ckey)
                    states.append({f: s[f] for f in STATE_FIELDS if f in s})
    clusters += [_pick(r, CLUSTER_FIELDS, _CLUSTER_ALIASES)
                 for r in _records(full.get("clusters"), "cluster_id")]
    states += [_pick(r, STATE_FIELDS, _STATE_ALIASES)
               for r in _records(full.get("states"), "state_id")]

    for c in clusters:
        if "item" not in c:
            item = ace14.cluster_to_item(c.get("name"))
            c["item"] = item if item is not None else "none"
        clusters[clusters.index(c)] = {f: c[f] for f in CLUSTER_FIELDS if f in c}

    table_in = _first(full, _TABLE_KEYS)
    if table_in is not None:
        table = _table_from(table_in)
        if table != ace14.CLUSTER_ITEM:
            diff = sorted(set(table.items()) ^ set(ace14.CLUSTER_ITEM.items()))
            raise ValueError("the cluster-to-item table in the input differs from "
                             f"src/instruments/ace14.py: {diff[:10]}")
    out = {"clusters": clusters, "states": states,
           "cluster_to_item": dict(ace14.CLUSTER_ITEM)}
    non_ace = _first(full, _NON_ACE_KEYS)
    if non_ace is not None:
        out["non_ace_clusters"] = _scrub(non_ace, _REVIEW_KEY_RE)
    rules = _first(full, _RULE_KEYS)
    if rules is not None:
        out["item_rules"] = _scrub(rules, _REVIEW_KEY_RE)
    return out


def dropped_keys(full: dict) -> list:
    """Top-level keys of the input that the trimmed file does not carry."""
    kept = {"domains", "clusters", "states"} | set(_TABLE_KEYS) | set(_NON_ACE_KEYS) \
        | set(_RULE_KEYS)
    return sorted(k for k in full if k not in kept)


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────
def selftest() -> bool:
    """Offline checks on synthetic state spaces in both layouts."""
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    flat = {
        "version": "dev-3",
        "review_log": [{"reviewer": "P1", "note": "approved"}],
        "clusters": [
            {"cluster_id": "c1", "name": "Emotional Abuse", "construct": "caregiver insults",
             "item": 1, "review_notes": "ok", "created_from": "patient_07 line 12"},
            {"cluster_id": "c2", "name": "Betrayal", "construct": "trust broken", "item": "none"},
        ],
        "states": [
            {"state_id": "s1", "cluster_id": "c1", "name": "Verbal humiliation",
             "description": "a parent repeatedly humiliates the child",
             "experience_type": "pattern",
             "allowed_actors": [{"actor": "parent", "role": "perpetrator",
                                 "source_note": "patient_03"}],
             "age_window": {"min": 0, "max": 18},
             "example_phrasings": ["father called her useless"],
             "ace_eligible": True, "ace_eligibility_reason": "caregiver act in childhood",
             "counts_with": [], "also_counts_as": [],
             "note_locations": ["patient_03:L14"], "review_note": "merged with s9",
             "supporting_quote": "called her useless", "status": "permanent"},
        ],
        "cluster_to_item": {str(k): v for k, v in ace14.CLUSTER_ITEM.items()},
        "non_ace_clusters": ["Betrayal", "Caregiver death"],
        "item_rules": {"1": {"rule": "actor must be a parent or household adult",
                             "evidence": "an act by that adult", "reviewer": "P2"}},
    }
    t = trim_state_space(flat)
    check(set(t) == {"clusters", "states", "cluster_to_item", "non_ace_clusters", "item_rules"},
          "only the five prompt sections are kept")
    check(list(t["states"][0]) == list(STATE_FIELDS),
          "a state keeps exactly the listed fields, in order")
    check(set(t["clusters"][0]) == set(CLUSTER_FIELDS), "a cluster keeps id, name, construct, item")
    text = json.dumps(t)
    check(all(w not in text for w in ("patient_", "review", "note_locations",
                                      "supporting_quote", "merged with")),
          "review notes, note locations and quotes are dropped, also when nested")
    check(t["states"][0]["allowed_actors"] == [{"actor": "parent", "role": "perpetrator"}],
          "allowed actors and roles survive with their nested note reference removed")
    check(t["cluster_to_item"] == ace14.CLUSTER_ITEM and len(t["cluster_to_item"]) == 69,
          "the 69-entry table equals ace14.py")
    check(dropped_keys(flat) == ["review_log", "version"], "dropped top-level keys are reported")
    check(t["item_rules"] == {"1": {"rule": "actor must be a parent or household adult",
                                    "evidence": "an act by that adult"}},
          "item rules keep their own fields and lose the reviewer")

    bad = dict(flat, cluster_to_item=dict(flat["cluster_to_item"], bullying=12))
    try:
        trim_state_space(bad)
        check(False, "a table that differs from ace14.py is refused")
    except ValueError:
        check(True, "a table that differs from ace14.py is refused")

    grouped = dict(flat, cluster_to_item={})
    for k, v in ace14.CLUSTER_ITEM.items():
        grouped["cluster_to_item"].setdefault(str(v), []).append(k)
    check(trim_state_space(grouped)["cluster_to_item"] == ace14.CLUSTER_ITEM,
          "a table grouped by item is accepted")

    nested = {"domains": {"family": {"domain_name": "Family", "clusters": {
        "emotional_abuse__1": {"cluster_name": "Emotional Abuse",
                               "cluster_definition": "caregiver insults", "states": [
            {"code": "dyn-001", "name": "Humiliation", "description": "d",
             "experience_type": "event", "state_status": "provisional",
             "provisional_for_patients": ["p07"],
             "provisional_source": {"event_text": "x", "supporting_quote": "y"},
             "review": {"decision": None}}]},
        "betrayal__2": {"cluster_name": "Betrayal", "states": []}}}}}
    tn = trim_state_space(nested)
    check(tn["states"] == [{"state_id": "dyn-001", "cluster_id": "emotional_abuse__1",
                            "name": "Humiliation", "description": "d",
                            "experience_type": "event"}],
          "nested layout: a state keeps its code as state_id and drops provisional metadata")
    check([c["item"] for c in tn["clusters"]] == [1, "none"]
          and tn["clusters"][0]["construct"] == "caregiver insults",
          "nested layout: cluster item resolved through the table, definition kept as construct")
    check("cluster_to_item" in tn and len(tn["cluster_to_item"]) == 69,
          "the table is inserted when the input has none")
    check(trim_state_space(t) == t, "trimming is idempotent")

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Trim a full state-space JSON to the fields the knowledge-matched "
                    "prompt uses (Supplementary Material, Section S4).")
    ap.add_argument("input", nargs="?", help="Full state-space JSON")
    ap.add_argument("--output", "-o", help="Where to write the trimmed JSON")
    ap.add_argument("--selftest", action="store_true", help="Run offline checks and exit.")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.input or not args.output:
        ap.error("input and --output are required (unless using --selftest)")

    with open(args.input, encoding="utf-8") as f:
        full = json.load(f)
    trimmed = trim_state_space(full)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"clusters: {len(trimmed['clusters'])}  states: {len(trimmed['states'])}  "
          f"table entries: {len(trimmed['cluster_to_item'])}  "
          f"non-ACE clusters: {'yes' if 'non_ace_clusters' in trimmed else 'none given'}  "
          f"item rules: {'yes' if 'item_rules' in trimmed else 'none given'}")
    if dropped_keys(full):
        print(f"dropped top-level keys: {', '.join(dropped_keys(full))}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
