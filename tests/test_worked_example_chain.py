"""Offline regression tests: the paper's worked example through the real CLIs,
every script's --selftest, the --no-algebra ablation, and the prompt files.
tests/test_scripts.py covers the end-to-end runner and the evaluation scripts.

Runs Worker 3 general -> Worker 3 ACE -> Worker 4 ACE as separate processes on
data/worked_example/worker2_output.json (Worker 2-shaped: shared source
sentences, per-piece quotes), exactly as a pipeline run chains them by file,
and checks the paper's numbers:

    E_ACE = (a + b + c^2) . (e + a^3),  ACE = 4,  items {1, 2, 4, 11}

Nothing here calls a model: every API key is removed from the environment of
each subprocess. Scripts that import aiohttp (Stages 2-3, Worker 2, the
ablation and the prompt export) are skipped, with a message, if it is not
installed.

Usage:
    python3 tests/test_worked_example_chain.py      # or: python3 -m pytest tests/
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
W3G = SRC / "algebra" / "fire_worker3_general.py"
W3A = SRC / "instruments" / "fire_worker3_ace.py"
W4 = SRC / "instruments" / "fire_worker4_ace.py"
RECON = SRC / "instruments" / "fire_reconcile_ace.py"
ABLATION = SRC / "ablation" / "fire_ablation.py"
EXPORT_PROMPTS = REPO / "prompts" / "export_prompts.py"
WORKED_INPUT = REPO / "data" / "worked_example" / "worker2_output.json"

SCRIPTS = REPO / "scripts"

# Self-tests that need only the standard library (Stage 1 uses spaCy when it
# is installed and a stand-in sentence splitter otherwise).
SELFTESTS = [
    SRC / "preprocessing" / "fire_preprocessor.py",
    SRC / "extraction" / "fire_stage1.py",
    W3G, W3A, W4, RECON,
    SRC / "verification" / "fire_stage4.py",
    SRC / "verification" / "fire_stage5.py",
    SRC / "baselines" / "fire_llm_baseline.py",
    SRC / "baselines" / "fire_llm_baseline_self_form_included.py",
    SRC / "baselines" / "fire_llm_structured_baseline.py",
    SRC / "baselines" / "fire_llm_knowledge_baseline.py",
    SRC / "baselines" / "trim_state_space.py",
    SCRIPTS / "run_fire.py",
    SCRIPTS / "evaluate_judgements.py",
    SCRIPTS / "make_tables.py",
]
# Self-tests whose module imports aiohttp.
SELFTESTS_AIOHTTP = [
    SRC / "extraction" / "fire_stage2_3_single_model.py",
    SRC / "state_space" / "fire_worker2.py",
    ABLATION,
]
HAVE_AIOHTTP = importlib.util.find_spec("aiohttp") is not None
_FAILED_CHECK = re.compile(r"^\s*\[FAIL\]", re.MULTILINE)

_KEY_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
             "XAI_API_KEY", "GROK_API_KEY", "FIRE_STATE_SPACE")


class Skipped(Exception):
    pass


def _skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise Skipped(reason)


def _offline_env() -> dict:
    """Environment with every API key removed, so no step can reach a model."""
    env = {k: v for k, v in os.environ.items() if k not in _KEY_VARS}
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _run(*args, cwd, ok_codes=(0,)) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=_offline_env(),
                          capture_output=True, text=True)
    assert proc.returncode in ok_codes, (
        f"{Path(str(args[0])).name} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}")
    return proc


def run_chain(workdir: Path) -> tuple:
    """Run the three CLIs in sequence; return their output paths and parsed JSON."""
    w3, ace, score = workdir / "w3.json", workdir / "ace.json", workdir / "score.json"
    _run(W3G, WORKED_INPUT, "-o", w3, cwd=workdir)
    _run(W3A, w3, "-o", ace, "--verify", cwd=workdir)
    _run(W4, ace, "-o", score, cwd=workdir)
    paths = (w3, ace, score)
    return paths, tuple(json.loads(p.read_text(encoding="utf-8")) for p in paths)


def test_worked_example_chain():
    with tempfile.TemporaryDirectory() as tmp:
        _, (w3, ace, score) = run_chain(Path(tmp))

    assert "blocks" in w3 and w3["blocks"], "Worker 3 general must hand blocks to Worker 3 ACE"
    assert w3["fire_expression"].replace(" ", "") == \
        "[C](a+b+c^2).d.(e+a^3+f).[A]g.g^3.(h+i^2)", w3["fire_expression"]

    assert ace["instrument"] == "ACE-14"
    assert ace["fire_expression"].replace(" ", "") == "[C](a+b+c^2).(e+a^3)", \
        ace["fire_expression"]
    assert not ace["age_unverified"]
    assert not any(f["record_removed"] for f in ace["verification_report"])

    assert score["canonical_ace_score"] == 4
    assert score["items_present"] == [1, 2, 4, 11]
    assert score["high_risk_threshold"] == 4 and score["crosses_high_risk_threshold"] is True
    assert score["exponent_weights_reference"]["a"] == 4      # a + a^3
    assert score["exponent_weights_reference"]["c"] == 2      # c^2
    assert not score["unmapped_symbols"] and not score["mapping_conflicts"]


def test_reconcile_worked_example():
    """Notes-plus-form step on the worked example. The form says yes to
    items 2 and 6 only. FiRE's evidence adds 1, 4 and 11 (changed from the
    form), confirms 2, and the notes are silent on 6 (the excerpt has no
    parental separation), so the form's yes on 6 is kept and sent for review."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (_, ace_path, score_path), _ = run_chain(tmp)
        form = {str(i): {"score": 1 if i in (2, 6) else 0} for i in range(1, 15)}
        form_path, out_path = tmp / "form.json", tmp / "reconciled.json"
        form_path.write_text(json.dumps(form), encoding="utf-8")
        _run(RECON, ace_path, score_path, "--form", form_path, "-o", out_path, cwd=tmp)
        out = json.loads(out_path.read_text(encoding="utf-8"))
    items = {r["ace_item"]: r for r in out["items"]}
    assert out["reconciled_ace_score"] == 5 and out["form_ace_score"] == 2
    assert out["items_changed"] == [1, 4, 11]
    assert items[2]["final_answer"] == 1 and not items[2]["changed"]
    assert items[2]["source"] == "FiRE evidence" and items[2]["evidence"]
    assert items[6]["final_answer"] == 1 and items[6]["source"] == "self-filled form"
    assert items[6]["needs_review"] and not items[6]["evidence"]


def test_selftests():
    with tempfile.TemporaryDirectory() as tmp:
        for script in SELFTESTS:
            proc = _run(script, "--selftest", cwd=tmp)
            assert "ALL PASS" in proc.stdout and not _FAILED_CHECK.search(proc.stdout), \
                f"{script.name}:\n{proc.stdout[-3000:]}"


def test_selftests_needing_aiohttp():
    if not HAVE_AIOHTTP:
        _skip("aiohttp not installed (pip install -r requirements.txt)")
    with tempfile.TemporaryDirectory() as tmp:
        for script in SELFTESTS_AIOHTTP:
            proc = _run(script, "--selftest", cwd=tmp)
            assert "ALL PASS" in proc.stdout and not _FAILED_CHECK.search(proc.stdout), \
                f"{script.name}:\n{proc.stdout[-3000:]}"


def test_no_algebra_matches_worker4():
    if not HAVE_AIOHTTP:
        _skip("aiohttp not installed (pip install -r requirements.txt)")
    with tempfile.TemporaryDirectory() as tmp:
        (_, ace, score), _ = run_chain(Path(tmp))
        proc = _run(ABLATION, "--no-algebra", "--w3-ace", ace, "--scorecard", score, cwd=tmp)
    res = json.loads(proc.stdout)
    assert res["equal"] and res["no_algebra_score"] == 4 and res["no_algebra_items"] == [1, 2, 4, 11]


def test_prompt_files_are_verbatim():
    if not HAVE_AIOHTTP:
        _skip("aiohttp not installed (pip install -r requirements.txt)")
    with tempfile.TemporaryDirectory() as tmp:
        _run(EXPORT_PROMPTS, "--check", cwd=tmp)


if __name__ == "__main__":
    failures = 0
    for name, fn in (("worked example chain", test_worked_example_chain),
                     ("notes-plus-form step", test_reconcile_worked_example),
                     ("offline self-tests", test_selftests),
                     ("offline self-tests needing aiohttp", test_selftests_needing_aiohttp),
                     ("--no-algebra equals Worker 4", test_no_algebra_matches_worker4),
                     ("prompt files match the code", test_prompt_files_are_verbatim)):
        try:
            fn()
            print(f"[PASS] {name}")
        except Skipped as e:
            print(f"[SKIP] {name}: {e}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}\n{e}")
    sys.exit(1 if failures else 0)
