#!/usr/bin/env python3
"""
Knowledge-matched baseline: the structure-guided prompt plus FiRE's state space.

Paper concept: the knowledge-matched rows of the main paper's Table II
(Supplementary Material, Section S4). The prompt is the structure-guided
prompt (fire_llm_structured_baseline.py) word for word, with two changes:
Step 2 maps each event to a state, its cluster, and that cluster's item, and
FiRE's state space is inserted before the glossary. The model therefore has
FiRE's steps and FiRE's knowledge (clusters and constructs; states with
description, allowed actors and roles, age window and example phrasings;
the 69-entry cluster-to-item table; the rules for each item), and does
every step itself in one call, with no separate verification.

The state-space file is inserted as given (JSON is re-printed with indent 1,
anything else is inserted as plain text). Trim a working state-space file
first, so no review notes or note references reach the prompt:

    python3 trim_state_space.py full_state_space.json -o state_space_for_prompt.json

Usage (all options of the structured script apply, plus --state-space):

    python3 fire_llm_knowledge_baseline.py notes.txt --model gpt-5.1 \\
        --state-space state_space_for_prompt.json -o out.json
    python3 fire_llm_knowledge_baseline.py notes.txt --form form.json --model gpt-5.1 \\
        --state-space state_space_for_prompt.json -o out.json
    python3 fire_llm_knowledge_baseline.py notes.txt --state-space S.json --print-prompt
    python3 fire_llm_knowledge_baseline.py --selftest

The path may also be given in the FIRE_STATE_SPACE environment variable.
Model-call settings and API keys are those of the structured script.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fire_llm_structured_baseline as sg  # noqa: E402


def _take_arg(flag: str):
    """Remove `flag VALUE` from sys.argv and return VALUE (or None)."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        sys.exit(f"error: {flag} needs a path")
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


def load_state_space(path: str) -> str:
    """The state-space file as prompt text (JSON re-printed with indent 1)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return json.dumps(json.loads(text), indent=1, ensure_ascii=False)
    except ValueError:
        return text.strip()


_OLD_STEP2 = """STEP 2 — MAP EACH EVENT
Assign each event to the one ACE item whose defining element it describes, or to "none"."""

_NEW_STEP2 = """STEP 2 — MAP EACH EVENT
Map each event to one state in the STATE SPACE below, then to that state's cluster, then to the ACE item the cluster-to-item table gives for that cluster. If no state fits, map the event to "none". Use only the states, clusters and actors the state space defines."""

_STATE_SPACE_HEADER = "STATE SPACE (states, clusters, cluster-to-item table, allowed actors):\n\n"


def install(state_space_text: str):
    """Patch the structure-guided prompt: new Step 2, state space after the
    procedure. Everything else (system prompt, Steps 1 and 3-6, glossary,
    output schema) is unchanged."""
    assert _OLD_STEP2 in sg.PROCEDURE, "structure-guided Step 2 text changed"
    sg.PROCEDURE = sg.PROCEDURE.replace(_OLD_STEP2, _NEW_STEP2)
    glossary = sg._glossary_block
    sg._glossary_block = lambda: (_STATE_SPACE_HEADER + state_space_text + "\n\n" + glossary())
    # Tag outputs so they never overwrite the structure-guided files.
    sg.OUTPUT_TAG = "ace_knowledge"
    sg.SCORING_PREFIX = "llm_knowledge"


def selftest() -> bool:
    """Offline checks: the structure-guided self-test still passes, and after
    install() the prompt carries the new Step 2 and the trimmed state space
    while every other part of the structure-guided prompt is unchanged."""
    import tempfile
    from collections import OrderedDict
    import trim_state_space as trim

    print("structure-guided prompt (before install):")
    ok = sg.selftest()

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    print("\nknowledge-matched prompt:")
    form = OrderedDict((n, 1 if n in (1, 4) else 0) for n in range(1, 15))
    before_d, before_r = sg.build_user_prompt("NOTES"), sg.build_reconcile_prompt("NOTES", form)
    saved = (sg.PROCEDURE, sg._glossary_block)

    full = {"clusters": [{"cluster_id": "c1", "name": "Emotional Abuse",
                          "construct": "caregiver insults", "item": 1,
                          "review_notes": "PRIVATE-REVIEW"}],
            "states": [{"state_id": "s1", "cluster_id": "c1", "name": "Humiliation",
                        "description": "a parent humiliates the child",
                        "allowed_actors": ["parent"], "age_window": [0, 18],
                        "example_phrasings": ["called her useless"],
                        "note_locations": ["PRIVATE-LOCATION"]}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ss.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trim.trim_state_space(full), f)
        text = load_state_space(path)
    try:
        install(text)
        d, r = sg.build_user_prompt("NOTES"), sg.build_reconcile_prompt("NOTES", form)
        check(_NEW_STEP2 in d and _NEW_STEP2 in r and _OLD_STEP2 not in d,
              "Step 2 maps event -> state -> cluster -> item in both prompts")
        check(_STATE_SPACE_HEADER + text in d and _STATE_SPACE_HEADER + text in r,
              "the state space is inserted in both prompts")
        check(d.index("STATE SPACE") < d.index("INDIAN CLINICAL CONTEXT"),
              "the state space comes before the glossary")
        check('"emotional abuse": 1' in text and text.count('": 14') == 9,
              "the 69-entry cluster-to-item table is in the prompt")
        check("PRIVATE" not in d, "review notes and note locations do not reach the prompt")
        restored = d.replace(_NEW_STEP2, _OLD_STEP2).replace(_STATE_SPACE_HEADER + text + "\n\n", "")
        check(restored == before_d and
              r.replace(_NEW_STEP2, _OLD_STEP2).replace(_STATE_SPACE_HEADER + text + "\n\n", "")
              == before_r,
              "nothing else in the structure-guided prompts changes")
    finally:
        sg.PROCEDURE, sg._glossary_block = saved

    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    path = _take_arg("--state-space") or os.environ.get("FIRE_STATE_SPACE")
    if "-h" in sys.argv or "--help" in sys.argv:
        print("Knowledge-matched baseline: the structured script's options below, plus\n"
              "  --state-space PATH   state-space file inserted into the prompt "
              "(required; or FIRE_STATE_SPACE)\n")
    elif not path:
        sys.exit("error: --state-space PATH (or FIRE_STATE_SPACE) is required")
    if path:
        install(load_state_space(path))
    if "--batch" in sys.argv and "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", "results_knowledge"]   # keep apart from structured outputs
    sg.main()
