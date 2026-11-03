#!/usr/bin/env python3
"""
FiRE ablation runs (Supplementary Material, Section S9).

Each run removes one part of FiRE and keeps everything else fixed. The
extraction model's output is produced once, by the main run's Stages 2-3,
and reused: this script rebuilds every extraction from that cached
validated_events.json (plus stage1_output.json for the section context)
and never calls the extraction model, so all runs share the same extracted
events and differ by design only in the part removed.

A removed part is replaced by a neutral stand-in so the next stage still
receives its usual input. The stand-ins are installed by patching the
pipeline's own functions for the duration of a run (unittest.mock); no
pipeline code is copied.

  --no-construct-check    Stages 2-3 check_construct_correctness always passes
                          (its similarity score is still computed, because
                          Stage 3 also uses it to calibrate confidence).
  --no-actor-role-check   Stages 2-3 check_actor_role always passes.
  --no-age-check          Stages 2-3 check_age_window always passes, and the
                          0-18 window at ACE projection (Worker 3 ACE) is
                          removed. Entries with no age evidence are still
                          held for review, as in the main run.
  --no-cited-span-check   The cited-span (groundedness) check of Stages 2-3,
                          full_hallucination_check, always passes, and so
                          does Stage 4's grounding recheck
                          (_check_semantic_grounding). Stage 4's entity and
                          timing rechecks stay on.
  --no-checks             All four of the above together.
  --no-state-space        Worker 2 still splits each event (same splitting
                          call), but each piece is mapped straight to an ACE
                          item by the model instead of to a state: the
                          semantic search over the state space is replaced
                          by the 14 items plus "none", and the matching
                          (Judge) call by a call that picks one of them. The
                          number of calls is the same, one splitting call per
                          event and one mapping call per piece. The chosen
                          item becomes the piece's cluster, named so that the
                          cluster-to-item table resolves it to that item. All
                          checks stay on.
  --model-scoring         The model decides each cluster's item from that
                          cluster's verified states (one call per cluster),
                          instead of the cluster-to-item table, at projection
                          (Worker 3 ACE) and at scoring (Worker 4). The 0-18
                          window and the item checks stay on.
  --no-algebra            Offline, no model call: score the distinct items
                          among the verified, in-window states of a
                          Worker 3 (ACE) output and compare with Worker 4's
                          score from the parsed expression. They must agree.

Usage (one switch per run; --cached-extraction is the main run's Stages 2-3
output, --state-space the state space the main run used, which is copied
into --out-dir so the original is never modified):

  export OPENAI_API_KEY=...
  python3 src/ablation/fire_ablation.py --no-age-check \\
      --stage1 stage1_output.json --cached-extraction validated_events.json \\
      --state-space my_state_space.json --out-dir ablation/no_age_check
  python3 src/ablation/fire_ablation.py --no-algebra --w3-ace worker3_ace_output.json
  python3 src/ablation/fire_ablation.py --selftest

A run writes, in --out-dir, the same files a pipeline run writes
(validated_events.json, validated_events_s4.json, validated_events_s5.json,
final_mapped_events.json, worker3_output.json, worker3_ace_output.json,
ace_scorecard.json) and ablation_manifest.json, which records the run, the
patched functions and the input hashes. The ablation results (Section S9) were
judged from such outputs by the same double-blind process as the main
results; nothing here computes accuracy.
"""

import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import types
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parent.parent
for _d in ("extraction", "verification", "state_space", "algebra", "instruments"):
    if str(SRC / _d) not in sys.path:
        sys.path.insert(0, str(SRC / _d))

import ace14                                   # noqa: E402
import fire_stage2_3_single_model as S23       # noqa: E402  (needs aiohttp)
import fire_stage4 as S4                       # noqa: E402
import fire_stage5 as S5                       # noqa: E402
import fire_worker2 as W2                      # noqa: E402
import fire_worker3_general as W3G             # noqa: E402
import fire_worker3_ace as W3A                 # noqa: E402
import fire_worker4_ace as W4                  # noqa: E402


CHECK_SWITCHES = ("no_construct_check", "no_actor_role_check", "no_age_check",
                  "no_cited_span_check")
RUNS = CHECK_SWITCHES + ("no_checks", "no_state_space", "model_scoring")

# The 0-18 window is removed by moving its upper bound out of reach.
_NO_WINDOW_MAX_AGE = 10 ** 6


# ─────────────────────────────────────────────────────────────
# 1.  NEUTRAL STAND-INS FOR THE FOUR CHECKS
# ─────────────────────────────────────────────────────────────

def _construct_always_passes(original):
    def check_construct_correctness(extraction):
        _passed, _flags, self_score = original(extraction)
        return True, [], self_score
    return check_construct_correctness


def _actor_role_always_passes(extraction):
    return True, []


def _age_window_always_passes(extraction, unit):
    return True, []


def _cited_span_always_passes(event, source, context_sentences, experience_type="event"):
    return []


def _stage4_grounding_always_passes(event, source_clean, context, exp_type):
    return None


def check_switches(run: str) -> tuple:
    """The individual check switches a run turns on."""
    return CHECK_SWITCHES if run == "no_checks" else tuple(
        s for s in CHECK_SWITCHES if s == run)


@contextlib.contextmanager
def check_patches(run: str):
    """Install the neutral stand-ins for `run`; restore everything on exit.
    Yields the list of patched names, for the manifest."""
    patched = []
    with contextlib.ExitStack() as stack:
        def patch(module, name, value):
            stack.enter_context(mock.patch.object(module, name, value))
            patched.append(f"{module.__name__}.{name}")

        switches = check_switches(run)
        if "no_construct_check" in switches:
            patch(S23, "check_construct_correctness",
                  _construct_always_passes(S23.check_construct_correctness))
        if "no_actor_role_check" in switches:
            patch(S23, "check_actor_role", _actor_role_always_passes)
        if "no_age_check" in switches:
            patch(S23, "check_age_window", _age_window_always_passes)
            patch(W3A, "ACE_MAX_AGE", _NO_WINDOW_MAX_AGE)
        if "no_cited_span_check" in switches:
            patch(S23, "full_hallucination_check", _cited_span_always_passes)
            patch(S4, "_check_semantic_grounding", _stage4_grounding_always_passes)
        yield patched


# ─────────────────────────────────────────────────────────────
# 2.  NO STATE SPACE: map each piece straight to an ACE item
# ─────────────────────────────────────────────────────────────

NONE_CODE = "direct-ace-none"
NONE_CLUSTER = "No ACE item (direct mapping)"


def representative_cluster(item: int) -> str:
    """The first table entry for `item`, title-cased; the table resolves it
    back to `item`, so projection and scoring need no change."""
    return next(k for k, v in ace14.CLUSTER_ITEM.items() if v == item).title()


def direct_options() -> list:
    """The 14 items and "none", in the shape Worker 2's search returns.
    "none" comes first so the same-cluster guard can never pick an item."""
    opts = [{"code": NONE_CODE, "name": "none",
             "description": "The experience describes no ACE item (for example a "
                            "symptom or outcome, or an experience no item asks about).",
             "cluster_name": NONE_CLUSTER, "score": 1.0, "state_status": "permanent"}]
    for n in range(1, ace14.N_ITEMS + 1):
        opts.append({"code": f"direct-ace-{n}", "name": f"ACE-{n}",
                     "description": W3A.ACE_QUESTIONS[n],
                     "cluster_name": representative_cluster(n), "score": 1.0,
                     "state_status": "permanent"})
    return opts


# Replaces Worker 2's MATCH_JUDGE_PROMPT; same placeholders, same reply shape.
DIRECT_ITEM_PROMPT = """You map one clinical experience, taken from consultation notes, straight to one item of the 14-item Adverse Childhood Experiences (ACE) questionnaire.

EXPERIENCE:
  Name:        {proposed_name}
  Type:        {proposed_type}
  Description: {proposed_desc}

OPTIONS — the 14 ACE items, and "none":
{candidates_json}

Choose the one option whose defining element the experience describes. Choose "none" when it describes no ACE item, for example a symptom or outcome the individual developed, or an experience none of the items asks about.

OUTPUT EXACTLY THIS JSON — no other text:
{{
  "map": true,
  "code": "the code of the chosen option",
  "reasoning": "One sentence."
}}"""


class DirectItemSpace:
    """Stands in for Worker 2's StateSpaceTree: no state space is loaded or
    searched, and every piece is offered the 14 items and "none"."""

    def __init__(self, filepath: str = ""):
        self.filepath = filepath
        self.db = {"domains": {}}
        self.index_lock = asyncio.Lock()
        self.creation_lock = asyncio.Lock()

    async def global_search(self, text, top_k=None, patient_id=None):
        return direct_options()

    async def dedup_sweep(self, threshold):
        return {}

    async def add_new_state(self, cluster_name, cluster_def, state_name, state_desc,
                            lifecycle_tier="Lifespan", experience_type="unknown",
                            patient_id=None, source=None):
        # Reached only if the model answers map=false; the piece is then
        # recorded as mapping to no item, and no state is created.
        return {"code": NONE_CODE, "cluster_name": NONE_CLUSTER,
                "description": state_desc, "state_status": "permanent"}


@contextlib.contextmanager
def no_state_space_patches():
    with mock.patch.object(W2, "StateSpaceTree", DirectItemSpace), \
         mock.patch.object(W2, "MATCH_JUDGE_PROMPT", DIRECT_ITEM_PROMPT):
        yield ["fire_worker2.StateSpaceTree", "fire_worker2.MATCH_JUDGE_PROMPT"]


# ─────────────────────────────────────────────────────────────
# 3.  MODEL SCORING: the model decides each cluster's item
# ─────────────────────────────────────────────────────────────

MODEL_SCORING_SYSTEM = (
    "You are a clinical expert scoring the 14-item Adverse Childhood Experiences "
    "(ACE) questionnaire. You are given one clinical cluster and the verified "
    "states an individual's notes were mapped to in that cluster. Decide which ACE "
    "item, if any, this cluster's evidence counts toward.\n\n"
    'Respond with STRICT JSON only: {"item": <1-14 or "none">, "reason": "one short sentence"}'
)


def _norm(name) -> str:
    return str(name or "").strip().lower()


def cluster_evidence(w3_general: dict) -> dict:
    """{cluster name: [verified state records]} from a Worker 3 (general) output."""
    out = defaultdict(list)
    for b in w3_general.get("blocks") or []:
        for e in b.get("entries") or []:
            name = e.get("cluster_name") or b.get("cluster_name")
            rec = {k: e.get(k) for k in ("state_code", "event", "source_sentence",
                                         "onset_age", "end_age", "experience_type")}
            if rec not in out[name]:
                out[name].append(rec)
    return dict(out)


def model_scoring_prompt(cluster_name: str, states: list) -> str:
    items = "\n".join(f"  ACE-{n}: {q}" for n, q in W3A.ACE_QUESTIONS.items())
    lines = []
    for s in states:
        onset, end = s.get("onset_age"), s.get("end_age")
        age = "unknown" if onset is None else (str(onset) if end in (None, onset) else f"{onset}-{end}")
        lines.append(f"  - state {s.get('state_code')} (age {age}, {s.get('experience_type')}): "
                     f"{s.get('event')}\n    source: {s.get('source_sentence')}")
    return (f"ACE ITEMS:\n{items}\n\nCLUSTER: {cluster_name}\n\nVERIFIED STATES:\n"
            + "\n".join(lines) + "\n\nWhich ACE item does this cluster count toward? "
            'Answer "none" if it counts toward no item. Respond with the strict JSON.')


def _parse_item(value):
    try:
        n = int(str(value).strip().upper().replace("ACE-", "").replace("ACE", ""))
    except ValueError:
        return None
    return n if 1 <= n <= ace14.N_ITEMS else None


def ask_model_for_item(cluster_name: str, states: list, api_key: str,
                       timeout: int = 60):
    """One GPT-5.1 call (Worker 3's payload settings, output limit 2000).
    Returns (item or None, reason, ok)."""
    payload = json.dumps(W3A._build_llm_payload(
        W3A.MODEL_NAME, MODEL_SCORING_SYSTEM, model_scoring_prompt(cluster_name, states),
        2000)).encode("utf-8")
    for attempt in range(3):
        try:
            req = urllib.request.Request(W3A.API_URL, data=payload, headers={
                "Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body["choices"][0]["message"]["content"].strip())
            return _parse_item(parsed.get("item")), str(parsed.get("reason") or ""), True
        except Exception as e:                       # noqa: BLE001
            logging.debug("model-scoring call %d failed: %s", attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    return None, "call failed", False


def model_item_decisions(w3_general: dict, decide) -> tuple:
    """Ask `decide(cluster_name, states)` once per cluster.
    Returns ({normalised cluster name: item or None}, report rows)."""
    decisions, report = {}, []
    for name, states in cluster_evidence(w3_general).items():
        item, reason, ok = decide(name, states)
        decisions[_norm(name)] = item
        report.append({"cluster_name": name, "item": item, "reason": reason,
                       "call_ok": ok, "n_states": len(states)})
    return decisions, report


def model_item_table(decisions: dict):
    """A stand-in for the ace14 module in which every cluster's item is the
    model's decision; the table and the non-ACE list are not consulted."""
    def cluster_to_item(cluster_name):
        return decisions.get(_norm(cluster_name))

    def item_for_entry(entry, cluster_name):
        item = cluster_to_item(cluster_name)
        return (item, "model") if item is not None else (None, None)

    return types.SimpleNamespace(
        N_ITEMS=ace14.N_ITEMS, MIN_AGE=ace14.MIN_AGE, MAX_AGE=ace14.MAX_AGE,
        CLUSTER_ITEM=ace14.CLUSTER_ITEM, NON_ACE_CLUSTERS=ace14.NON_ACE_CLUSTERS,
        cluster_to_item=cluster_to_item, item_for_entry=item_for_entry,
        is_non_ace_cluster=lambda cluster_name: False, hint_items=ace14.hint_items)


@contextlib.contextmanager
def model_scoring_patches(decisions: dict):
    table = model_item_table(decisions)
    with mock.patch.object(W3A, "ace14", table), mock.patch.object(W4, "ace14", table):
        yield ["fire_worker3_ace.ace14", "fire_worker4_ace.ace14"]


# ─────────────────────────────────────────────────────────────
# 4.  NO ALGEBRA: distinct items of the verified, in-window states
# ─────────────────────────────────────────────────────────────

def score_without_algebra(w3_ace: dict) -> tuple:
    """Distinct items among the resolved states of a Worker 3 (ACE) output,
    which are exactly the verified, in-window states that survived
    projection and the item check. Returns (score, sorted items)."""
    items = sorted({int(s["ace_item"]) for s in (w3_ace.get("resolved_states") or [])
                    if s.get("ace_item") is not None})
    return len(items), items


def compare_with_algebra(w3_ace: dict, scorecard: dict = None) -> dict:
    """Score both ways; `scorecard` defaults to Worker 4 run in-process."""
    algebra = scorecard or W4.score_from_expression(w3_ace)
    score, items = score_without_algebra(w3_ace)
    return {"algebra_score": algebra["canonical_ace_score"],
            "algebra_items": algebra["items_present"],
            "no_algebra_score": score, "no_algebra_items": items,
            "equal": (score == algebra["canonical_ace_score"]
                      and items == algebra["items_present"])}


# ─────────────────────────────────────────────────────────────
# 5.  REUSING THE CACHED EXTRACTION
# ─────────────────────────────────────────────────────────────

def cached_results(cached: dict, units: list) -> dict:
    """Rebuild every unit's extraction result from a Stages 2-3 output, so
    Stage 3 can be rerun without calling the model. Units the main run has
    no event for are left out, as the main run left them out."""
    by_unit = defaultdict(list)
    for ev in cached.get("events") or []:
        by_unit[ev.get("unit_id")].append(ev)
    known = {u.unit_id for u in units}
    missing = sorted(set(by_unit) - known)
    if missing:
        raise ValueError(f"{len(missing)} cached unit id(s) do not occur in the Stage 1 "
                         f"grouping (wrong stage1 file or patient id?): {missing[:5]}")

    results = {}
    for unit in units:
        evs = by_unit.get(unit.unit_id)
        if not evs:
            continue
        failed = [e for e in evs if e.get("status") == "EXTRACTION_FAILED"]
        if failed:
            flags = failed[0].get("verification_flags") or []
            status = next((f.split(":", 1)[1].strip() for f in flags
                           if f.startswith("MODEL_STATUS")), "API_ERROR")
            results[unit.unit_id] = S23.UnitLLMResult(unit.unit_id, S23.MODEL_NAME, [], status)
            continue
        extractions = [S23.LLMExtraction(
            source_sentence=e.get("source_sentence") or "",
            event_present=True,
            event=e.get("event") or "",
            experience_type=e.get("experience_type") or "unknown",
            persons_involved=e.get("persons_involved") or [],
            life_stage=e.get("life_stage") or "unknown",
            frequency=e.get("frequency") or "unknown",
            confidence="CACHED",
            null_reason=None,
            requires_new_state=bool(e.get("requires_new_state")),
            new_state_description=e.get("new_state_description"),
        ) for e in evs if e.get("status") != "NO_EVENT"]
        results[unit.unit_id] = S23.UnitLLMResult(unit.unit_id, S23.MODEL_NAME,
                                                  extractions, "SUCCESS")
    return results


def rerun_stage3(units: list, results: dict) -> list:
    """Stage 3 on the cached extractions, in the main run's unit order."""
    validated = []
    for unit in units:
        res = results.get(unit.unit_id)
        if res is not None:
            validated.extend(S23.verify_unit(unit, res))
    return validated


# ─────────────────────────────────────────────────────────────
# 6.  ONE ABLATION RUN
# ─────────────────────────────────────────────────────────────

def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _call_main(fn, argv: list):
    with mock.patch.object(sys, "argv", argv):
        fn()


def run_ablation(run: str, stage1_path: str, cached_path: str, out_dir: str,
                 state_space: str = None, patient_id: str = None,
                 concurrency: int = 3, api_key: str = None) -> dict:
    """Run the pipeline from the cached extraction with `run` removed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = {name: str(out / f) for name, f in (
        ("s3", "validated_events.json"), ("s4", "validated_events_s4.json"),
        ("s5", "validated_events_s5.json"), ("w2", "final_mapped_events.json"),
        ("w3", "worker3_output.json"), ("w3a", "worker3_ace_output.json"),
        ("w4", "ace_scorecard.json"), ("ss", "state_space.json"))}

    stage1 = json.loads(Path(stage1_path).read_text(encoding="utf-8"))
    cached = json.loads(Path(cached_path).read_text(encoding="utf-8"))
    patient_id = patient_id or cached.get("patient_id") or "patient"
    units = S23.group_sentences(stage1, patient_id)
    results = cached_results(cached, units)

    manifest = {
        "run": run, "patient_id": patient_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"stage1": {"path": str(stage1_path), "sha256": _sha(stage1_path)},
                   "cached_extraction": {"path": str(cached_path), "sha256": _sha(cached_path)}},
        "extraction_model_called": False, "patched": [],
    }
    if run != "no_state_space":
        if not state_space:
            raise ValueError("--state-space is required for this run")
        shutil.copyfile(state_space, p["ss"])
        manifest["inputs"]["state_space"] = {"path": str(state_space),
                                             "sha256": _sha(state_space),
                                             "copied_to": p["ss"]}

    with contextlib.ExitStack() as stack:
        if api_key is not None:
            stack.enter_context(mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}))
        manifest["patched"] += stack.enter_context(check_patches(run))
        if run == "no_state_space":
            manifest["patched"] += stack.enter_context(no_state_space_patches())

        # Stage 3 on the cached extraction, then Stages 4 and 5.
        validated = rerun_stage3(units, results)
        S23.atomic_write_json(p["s3"], {
            "patient_id": patient_id, "pipeline_stage": "worker1_stage2_3_single_model",
            "model": S23.MODEL_NAME, "verification_method": "four_axis_deterministic",
            "ablation_run": run, "reused_extraction_from": str(cached_path),
            "manifest": S23.build_run_manifest(),
            "statistics": {"status_counts": dict(Counter(v.status for v in validated))},
            "events": [asdict(v) for v in validated],
        })
        _call_main(S4.main, ["fire_stage4.py", p["s3"], "--stage1", stage1_path, "-o", p["s4"]])
        _call_main(S5.main, ["fire_stage5.py", p["s4"], "--stage1", stage1_path, "-o", p["s5"]])

        # Worker 2 (state space, or the 14 items with --no-state-space).
        ns = argparse.Namespace(
            input=p["s5"], state_space=p["ss"], fresh_start=False, stage1=stage1_path,
            output=p["w2"], concurrency=concurrency, resume=False, keep_resume=False,
            dry_run=False, verbose=False, no_dedup_sweep=False, dedup_threshold=0.82)
        asyncio.run(W2.main_loop(ns))

        _call_main(W3G.main, ["fire_worker3_general.py", p["w2"], "-o", p["w3"]])

        if run == "model_scoring":
            w3 = json.loads(Path(p["w3"]).read_text(encoding="utf-8"))
            key = os.environ.get(W3A.API_KEY_ENV, "")
            decisions, report = model_item_decisions(
                w3, lambda name, states: ask_model_for_item(name, states, key))
            manifest["model_item_decisions"] = report
            manifest["patched"] += stack.enter_context(model_scoring_patches(decisions))

        _call_main(W3A.main, ["fire_worker3_ace.py", p["w3"], "-o", p["w3a"],
                              "--verify", "--verify-llm"])
        _call_main(W4.main, ["fire_worker4_ace.py", p["w3a"], "-o", p["w4"]])

    score = json.loads(Path(p["w4"]).read_text(encoding="utf-8"))
    manifest["result"] = {"ace_score": score["canonical_ace_score"],
                          "items_present": score["items_present"]}
    manifest["outputs"] = p
    Path(out / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ─────────────────────────────────────────────────────────────
# 7.  SELF-TEST (offline, synthetic inputs, no model call)
# ─────────────────────────────────────────────────────────────

def selftest() -> bool:
    ok = True

    def check(cond, name):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # Keep the similarity checks offline and deterministic: lexical
    # backends, no embedding or entailment model is loaded.
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(S23, "_similarity_backend", "jaccard"))
    stack.enter_context(mock.patch.object(S23, "_nli_available", False))
    stack.enter_context(mock.patch.object(S4, "_sim_backend", "containment"))
    # The construct check's similarity is stubbed (entailed only when the event
    # restates its source), so the fixture does not depend on a lexical overlap.
    _orig_sim = S23.compute_semantic_similarity

    def _stub_similarity(premise, hypothesis):
        same = premise.strip(" .").lower() == hypothesis.strip(" .").lower()
        return (0.95 if same else 0.10), "transformers"
    stack.enter_context(mock.patch.object(S23, "compute_semantic_similarity", _stub_similarity))
    stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
    os.environ.pop("OPENAI_API_KEY", None)

    unit = S23.ProcessingUnit(
        unit_id="t_L1_0001", line_number=1, subsection="Childhood", age_context="25-27",
        source_type="NARRATIVE", stage1_confidence="HIGH",
        sentences=["Mother beat the children."], sentence_objects=[],
        context_before=[], context_after=[])

    def ext(event, source, persons, stage="childhood", exp="event"):
        return S23.LLMExtraction(source_sentence=source, event_present=True, event=event,
                                 experience_type=exp, persons_involved=persons,
                                 life_stage=stage, frequency="single", confidence="HIGH",
                                 null_reason=None)

    # Fails all four checks: not entailed, father vs mother as actor, age 25-27
    # against childhood, and "father"/"hit" not anchored in the source.
    bad = ext("Father hit the client with a belt", "Mother beat the children.",
              ["Father", "Mother"])
    axes = ("construct_check_passed", "actor_role_check_passed",
            "age_window_check_passed", "groundedness_check_passed")
    base = S23.verify_extraction(unit, bad)
    check(not any(getattr(base, a) for a in axes) and base.status == "DISCARD",
          "fixture: the synthetic event fails all four checks and is discarded")

    for run, axis in zip(CHECK_SWITCHES, axes):
        with check_patches(run):
            v = S23.verify_extraction(unit, bad)
        flipped = [a for a in axes if getattr(v, a) != getattr(base, a)]
        check(flipped == [axis], f"--{run.replace('_', '-')}: only the {axis.split('_check')[0]} "
                                 f"check is neutralised (now always passes)")
    with check_patches("no_checks"):
        v = S23.verify_extraction(unit, bad)
    check(all(getattr(v, a) for a in axes) and v.status == "CONFIRMED" and not v.verification_flags,
          "--no-checks: all four checks pass, so the event is kept as CONFIRMED")
    after = S23.verify_extraction(unit, bad)
    check(not any(getattr(after, a) for a in axes), "the original checks are restored after a run")

    good = ext("Mother beat the children", "Mother beat the children.", ["Mother", "Children"],
               stage="adult")
    with check_patches("no_construct_check"):
        g = S23.verify_extraction(unit, good)
    check(g.self_groundedness_score == S23.verify_extraction(unit, good).self_groundedness_score,
          "--no-construct-check keeps the similarity score Stage 3 uses for confidence")

    # Stage 4's grounding recheck.
    s4_ev = {"status": "CONFIRMED", "event": "Client won a national chess trophy",
             "source_sentence": "Father beat mother during childhood.",
             "experience_type": "event", "persons_involved": [], "life_stage": "unknown",
             "confidence": 0.95, "subsection": "", "line_number": 1}
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        e0 = S4.stage4_verify([dict(s4_ev)])[0]
        with check_patches("no_cited_span_check"):
            e1 = S4.stage4_verify([dict(s4_ev)])[0]
        with check_patches("no_construct_check"):
            e2 = S4.stage4_verify([dict(s4_ev)])[0]
    check(e0["status"] == "TENTATIVE" and any("SEMANTIC" in f for f in e0["stage4_flags"]),
          "fixture: Stage 4's grounding recheck lowers an ungrounded event to TENTATIVE")
    check(e1["status"] == "CONFIRMED" and not e1["stage4_flags"],
          "--no-cited-span-check: Stage 4's grounding recheck always passes")
    check(e2["status"] == "TENTATIVE", "other switches leave Stage 4's grounding recheck on")

    # The 0-18 window at projection.
    recs = [W3G._mk_record("w1", "Emotional Abuse", 22, 24, "Partner insulted client",
                           sub_id="w1", life_stage="adult"),
            W3G._mk_record("w2", "Physical Abuse", 15, 25, "Father hit client", sub_id="w2")]
    g = W3G.build_expression(recs, [], [], W3G.assign_alphabets(recs))
    b0, *_ = W3A.project_to_ace(g)
    with check_patches("no_age_check"):
        b1, *_ = W3A.project_to_ace(g)
    kept0 = {e["state_code"] for b in b0 for e in b["entries"]}
    kept1 = {e["state_code"] for b in b1 for e in b["entries"]}
    check(kept0 == {"w2"} and max(b["interval"][1] for b in b0) == 18,
          "fixture: projection drops an adult span and cuts a straddling one at 18")
    check(kept1 == {"w1", "w2"} and max(b["interval"][1] for b in b1) == 25,
          "--no-age-check: the 0-18 window at projection is removed")
    with check_patches("no_cited_span_check"):
        b2, *_ = W3A.project_to_ace(g)
    check({e["state_code"] for b in b2 for e in b["entries"]} == kept0,
          "other switches leave the projection window on")

    # No state space: Worker 2's splitting call is kept, the state space is
    # never loaded, and each piece is mapped straight to an item.
    calls = []

    async def fake_call_llm(session, api_key, system_prompt, user_prompt, counter,
                            max_retries=3, max_output_tokens=4000):
        calls.append((system_prompt[:20], user_prompt[:40], max_output_tokens))
        if system_prompt == W2.ORACLE_PROMPT:
            return {"decomposition_reasoning": "two domains", "sub_events": [
                {"status": "ADVERSE", "experience_type": "event",
                 "text": "Client [Childhood] was bullied by classmates",
                 "supporting_quote": "bullied by classmates", "onset_age": 9, "end_age": 11,
                 "ideal_cluster_name": "Peer Bullying", "ideal_state_name": "Bullying",
                 "ideal_state_description": "peers repeatedly bully the child"},
                {"status": "ADVERSE", "experience_type": "symptom",
                 "text": "Client [Childhood] felt low", "supporting_quote": "felt low",
                 "onset_age": 9, "end_age": 11, "ideal_cluster_name": "Low Mood",
                 "ideal_state_name": "Low mood", "ideal_state_description": "low mood"}]}
        assert user_prompt.startswith("You map one clinical experience"), "judge prompt not replaced"
        code = "direct-ace-11" if "peers repeatedly bully" in user_prompt else NONE_CODE
        return {"map": True, "code": code, "reasoning": "test"}

    with __import__("tempfile").TemporaryDirectory() as tmp:
        inp = Path(tmp) / "s5.json"
        inp.write_text(json.dumps({"patient_id": "t", "events": [{
            "unit_id": "t_L1_0001", "status": "CONFIRMED", "life_stage": "childhood",
            "event": "Bullied by classmates and felt low",
            "source_sentence": "Age 9-11: bullied by classmates and felt low."}]}))
        ns = argparse.Namespace(input=str(inp), state_space=str(Path(tmp) / "absent.json"),
                                fresh_start=False, stage1=None, output=str(Path(tmp) / "w2.json"),
                                concurrency=1, resume=False, keep_resume=False, dry_run=False,
                                verbose=False, no_dedup_sweep=False, dedup_threshold=0.82)
        with no_state_space_patches(), mock.patch.object(W2, "call_llm", fake_call_llm), \
                mock.patch.dict(os.environ, {"OPENAI_API_KEY": "selftest-not-a-key"}), \
                mock.patch.object(W2, "SentenceTransformer", None):
            logging.disable(logging.CRITICAL)
            try:
                asyncio.run(W2.main_loop(ns))
            finally:
                logging.disable(logging.NOTSET)
        w2 = json.loads((Path(tmp) / "w2.json").read_text())
        subs = w2["events"][0]["sub_events"]
    check(len(calls) == 3 and calls[0][0] == W2.ORACLE_PROMPT[:20]
          and all(c[2] == 300 for c in calls[1:]),
          "--no-state-space: one splitting call per event and one mapping call per piece "
          "(the Judge's slot and output limit)")
    check([(s["status"], s["state_code"]) for s in subs]
          == [("MAPPED", "direct-ace-11"), ("MAPPED", NONE_CODE)],
          "--no-state-space: each piece is mapped straight to an item (or none), no state space loaded")
    check([ace14.cluster_to_item(s["cluster_name"]) for s in subs] == [11, None]
          and all(ace14.cluster_to_item(representative_cluster(n)) == n for n in range(1, 15)),
          "--no-state-space: the chosen item resolves through the unchanged table at projection")
    check(W2.StateSpaceTree is not DirectItemSpace and W2.MATCH_JUDGE_PROMPT != DIRECT_ITEM_PROMPT,
          "Worker 2 is restored after the run")

    # Model scoring: items come from the model's decision, not the table.
    w3 = W3G.build_expression(W3G.worked_example_records(), [], [],
                              W3G.assign_alphabets(W3G.worked_example_records()))
    asked = []

    def fake_decide(name, states):
        asked.append(name)
        table = {"emotional abuse": 1, "felt unloved": 4, "physical abuse": 2,
                 "peer bullying": 12, "caregiver death": 9}
        return table.get(_norm(name)), "test", True

    decisions, report = model_item_decisions(w3, fake_decide)
    with model_scoring_patches(decisions):
        blocks, fbs, _, _ = W3A.project_to_ace(w3)
        res = W3A.build_ace_expression_from_projection(blocks, fbs)
        sc = W4.score_from_expression(res)
    check(sorted(asked) == sorted(cluster_evidence(w3)) and len(asked) == 9,
          "--model-scoring: one model decision per cluster of verified states")
    check(sc["items_present"] == [1, 2, 4, 9, 12],
          "--model-scoring: the score follows the model's items (bullying -> 12, "
          "caregiver death -> 9), not the table")
    check(all(e["ace_item_source"] == "model" for b in blocks for e in b["entries"])
          and max(b["interval"][1] for b in blocks) <= 18,
          "--model-scoring: projection takes the item from the model; the 0-18 window stays on")
    check(W3A.ace14 is ace14 and W4.ace14 is ace14,
          "the cluster-to-item table is restored after the run")

    # No algebra: the worked example scores the same both ways.
    blocks, fbs, _, _ = W3A.project_to_ace(w3)
    w3a = W3A.build_ace_expression_from_projection(blocks, fbs)
    cmp_ = compare_with_algebra(w3a)
    check(cmp_["no_algebra_score"] == 4 and cmp_["no_algebra_items"] == [1, 2, 4, 11],
          "--no-algebra: the worked example scores ACE = 4, items {1, 2, 4, 11}")
    check(cmp_["equal"] and w3a["fire_expression"].replace(" ", "") == "[C](a+b+c^2).(e+a^3)",
          "--no-algebra equals the score parsed from (a + b + c^2) . (e + a^3)")

    # Reusing the cached extraction reproduces Stage 3 exactly.
    stage1 = [{"sentence": "Childhood: Mother beat the children.", "line_number": 1,
               "confidence_tag": "HIGH", "source_type": "NARRATIVE", "age_context": None,
               "temporal_markers": [], "hedging_flags": [], "clinical_flags": [],
               "exclusion_reason": None, "is_direct_quote": False}]
    units = S23.group_sentences(stage1, "t")
    if units:
        u = units[0]
        orig = S23.verify_unit(u, S23.UnitLLMResult(u.unit_id, S23.MODEL_NAME, [bad, good], "SUCCESS"))
        cache = {"patient_id": "t", "events": [asdict(v) for v in orig]}
        again = rerun_stage3(units, cached_results(cache, units))
        check([asdict(v) for v in again] == cache["events"],
              "the cached extraction reproduces Stage 3's output exactly when nothing is removed")
    else:
        check(False, "fixture: Stage 1 grouping produced a unit")

    # A whole run, end to end, with every model call faked: the cached
    # extraction goes through Stages 3-5, Worker 2 (no state space), Worker 3
    # and Worker 4 by their own entry points, and nothing calls a model.
    async def fake_w2(session, api_key, system_prompt, user_prompt, counter,
                      max_retries=3, max_output_tokens=4000):
        if system_prompt == W2.ORACLE_PROMPT:
            return {"decomposition_reasoning": "one", "sub_events": [{
                "status": "ADVERSE", "experience_type": "event",
                "text": "Client [Childhood] was beaten by father",
                "supporting_quote": "Father beat the client", "onset_age": 8, "end_age": 8,
                "ideal_cluster_name": "Physical Abuse", "ideal_state_name": "Beaten",
                "ideal_state_description": "a parent beats the child"}]}
        return {"map": True, "code": "direct-ace-2", "reasoning": "test"}

    st1 = [{"sentence": "Age 8: Father beat the client.", "line_number": 1,
            "confidence_tag": "HIGH", "source_type": "NARRATIVE", "age_context": "8",
            "temporal_markers": [], "hedging_flags": [], "clinical_flags": [],
            "exclusion_reason": None, "is_direct_quote": False}]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, \
            contextlib.redirect_stdout(open(os.devnull, "w")), \
            mock.patch.object(W2, "call_llm", fake_w2), \
            mock.patch.object(S5, "_load_model", lambda: None), \
            mock.patch.object(S5, "rescue_call", lambda *a, **k: None), \
            mock.patch.object(W3A, "_llm_verify_record", lambda *a, **k: None), \
            mock.patch.object(W3A, "_llm_adjudicate_drop", lambda *a, **k: None), \
            mock.patch.object(W2, "SentenceTransformer", None):
        logging.disable(logging.CRITICAL)
        try:
            u = S23.group_sentences(st1, "t")[0]
            ev = ext("Father beat the client", "Father beat the client", ["Father", "Client"])
            cache_path, st1_path = Path(tmp) / "cache.json", Path(tmp) / "stage1.json"
            cache_path.write_text(json.dumps({"patient_id": "t", "events": [
                asdict(v) for v in S23.verify_unit(u, S23.UnitLLMResult(
                    u.unit_id, S23.MODEL_NAME, [ev], "SUCCESS"))]}))
            st1_path.write_text(json.dumps(st1))
            man = run_ablation("no_state_space", str(st1_path), str(cache_path),
                               str(Path(tmp) / "out"), api_key="selftest-not-a-key")
            files = sorted(f.name for f in (Path(tmp) / "out").iterdir())
        finally:
            logging.disable(logging.NOTSET)
    check(man["result"] == {"ace_score": 1, "items_present": [2]}
          and not man["extraction_model_called"]
          and {"validated_events.json", "validated_events_s4.json", "validated_events_s5.json",
               "final_mapped_events.json", "worker3_output.json", "worker3_ace_output.json",
               "ace_scorecard.json", "ablation_manifest.json"} <= set(files),
          "a full run from the cached extraction writes every stage's output and a manifest")

    stack.close()
    check(S23.compute_semantic_similarity is _orig_sim, "fixture stubs are removed")
    print("\nSELF-TEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


# ─────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FiRE ablation runs (Supplementary Material, Section S9). One switch per run.")
    g = ap.add_mutually_exclusive_group()
    helps = {
        "no_construct_check": "Stages 2-3 construct check always passes.",
        "no_actor_role_check": "Stages 2-3 actor and role check always passes.",
        "no_age_check": "Stages 2-3 age check always passes; no 0-18 window at projection.",
        "no_cited_span_check": "Stages 2-3 cited-span check and Stage 4 grounding recheck "
                               "always pass.",
        "no_checks": "All four checks always pass.",
        "no_state_space": "Worker 2 maps each piece straight to an ACE item (same calls).",
        "model_scoring": "The model decides each cluster's item instead of the table.",
    }
    for run in RUNS:
        g.add_argument("--" + run.replace("_", "-"), dest="run", action="store_const",
                       const=run, help=helps[run])
    g.add_argument("--no-algebra", dest="run", action="store_const", const="no_algebra",
                   help="Offline: compare distinct-item scoring with Worker 4's score.")
    g.add_argument("--selftest", dest="run", action="store_const", const="selftest",
                   help="Run offline checks and exit.")
    ap.add_argument("--stage1", help="stage1_output.json of the main run")
    ap.add_argument("--cached-extraction",
                    help="The main run's Stages 2-3 output (validated_events.json)")
    ap.add_argument("--state-space", help="The state space the main run used (copied, not modified)")
    ap.add_argument("--out-dir", help="Folder for this run's outputs")
    ap.add_argument("--patient-id", default=None,
                    help="Defaults to the patient_id in the cached extraction")
    ap.add_argument("--concurrency", type=int, default=3, help="Worker 2 concurrency")
    ap.add_argument("--w3-ace", help="--no-algebra: a Worker 3 (ACE) output JSON")
    ap.add_argument("--scorecard", help="--no-algebra: the matching Worker 4 output "
                                        "(default: score in-process)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if args.run == "selftest":
        sys.exit(0 if selftest() else 1)
    if args.run is None:
        ap.error("choose one switch, e.g. --no-age-check, --no-algebra or --selftest")

    if args.run == "no_algebra":
        if not args.w3_ace:
            ap.error("--no-algebra needs --w3-ace")
        w3a = json.loads(Path(args.w3_ace).read_text(encoding="utf-8"))
        card = json.loads(Path(args.scorecard).read_text(encoding="utf-8")) if args.scorecard else None
        res = compare_with_algebra(w3a, card)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res["equal"] else 1)

    missing = [f for f in ("stage1", "cached_extraction", "out_dir") if not getattr(args, f)]
    if args.run != "no_state_space" and not args.state_space:
        missing.append("state_space")
    if missing:
        ap.error("missing: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("error: OPENAI_API_KEY is required (Stage 5, Worker 2 and Worker 3 ACE call GPT-5.1)")

    manifest = run_ablation(args.run, args.stage1, args.cached_extraction, args.out_dir,
                            state_space=args.state_space, patient_id=args.patient_id,
                            concurrency=args.concurrency)
    print(f"\n[{args.run}] ACE = {manifest['result']['ace_score']}  "
          f"items {manifest['result']['items_present']}  -> {args.out_dir}")


if __name__ == "__main__":
    main()
