"""Offline tests of the scripts in scripts/: the end-to-end runner and the
evaluation and table scripts.

The runner is exercised on the paper's worked example from Worker 3 onward
(the steps before it call GPT-5.1), for one individual and in batch mode,
and its refusal to start model steps without a key is checked. The
evaluation scripts are run on a synthetic judgements file whose counts are
the published counts, and must reproduce Table II.

Nothing here calls a model: every API key is removed from the environment
of each subprocess.

Usage:
    python3 tests/test_scripts.py      # or: python3 -m pytest tests/
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_worked_example_chain import (REPO, WORKED_INPUT, Skipped,  # noqa: E402
                                       _offline_env, _run)

SCRIPTS = REPO / "scripts"
RUN_FIRE = SCRIPTS / "run_fire.py"
EVALUATE = SCRIPTS / "evaluate_judgements.py"
TABLES = SCRIPTS / "make_tables.py"
NOTES = REPO / "data" / "worked_example" / "input.txt"


def _form(path: Path) -> Path:
    path.write_text(json.dumps({str(i): {"score": 1 if i in (2, 6) else 0}
                                for i in range(1, 15)}), encoding="utf-8")
    return path


def test_runner_worked_example_from_worker3():
    """Worker 3 -> Worker 3 (ACE) -> Worker 4 -> notes-plus-form step, chained
    by the runner, gives the paper's score and the expected reconciliation."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wd = tmp / "worked_example"
        wd.mkdir()
        shutil.copyfile(WORKED_INPUT, wd / "final_mapped_events.json")
        _run(RUN_FIRE, "--patient-id", "worked_example", "--workdir", wd, "--from", "worker3",
             "--offline-item-check", "--form", _form(tmp / "form.json"), cwd=tmp)
        score = json.loads((wd / "ace_scorecard.json").read_text(encoding="utf-8"))
        ace = json.loads((wd / "worker3_ace_output.json").read_text(encoding="utf-8"))
        rec = json.loads((wd / "ace_reconciled.json").read_text(encoding="utf-8"))
        log = json.loads((wd / "run_log.json").read_text(encoding="utf-8"))
    assert ace["fire_expression"].replace(" ", "") == "[C](a+b+c^2).(e+a^3)"
    assert score["canonical_ace_score"] == 4 and score["items_present"] == [1, 2, 4, 11]
    assert rec["reconciled_ace_score"] == 5 and rec["items_changed"] == [1, 4, 11]
    assert [s["step"] for s in log["steps"]] == ["worker3", "worker3_ace", "worker4", "reconcile"]
    assert all(s["returncode"] == 0 for s in log["steps"])


def test_runner_batch_and_skip():
    """Batch mode finds patient_NN.txt and patient_NN_ace_form.json, runs each
    individual in its own folder, and skips finished individuals."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data, out = tmp / "notes", tmp / "runs"
        data.mkdir()
        for pid in ("patient_01", "patient_02"):
            shutil.copyfile(NOTES, data / f"{pid}.txt")
            _form(data / f"{pid}_ace_form.json")
            (out / pid).mkdir(parents=True)
            shutil.copyfile(WORKED_INPUT, out / pid / "final_mapped_events.json")
        args = (RUN_FIRE, "--batch", data, "--with-forms", "--out-dir", out,
                "--from", "worker3", "--offline-item-check")
        _run(*args, cwd=tmp)
        scores = [json.loads((out / p / "ace_reconciled.json").read_text(encoding="utf-8"))
                  ["reconciled_ace_score"] for p in ("patient_01", "patient_02")]
        again = _run(*args, cwd=tmp)
    assert scores == [5, 5]
    assert again.stdout.count("skip (exists)") == 2


def test_runner_refuses_model_steps_without_key():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(RUN_FIRE, NOTES, "--state-space", Path(tmp) / "ss.json",
                    "--workdir", Path(tmp) / "w", cwd=tmp, ok_codes=(1,))
        assert "OPENAI_API_KEY is not set" in proc.stderr
        assert not (Path(tmp) / "w").exists() and not (Path(tmp) / "ss.json").exists()


def test_runner_preprocess_step():
    """The first step runs offline and writes the flat text Stage 1 reads."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp) / "w"
        _run(RUN_FIRE, NOTES, "--workdir", wd, "--to", "preprocess", cwd=tmp)
        flat = (wd / "input_flat.txt").read_text(encoding="utf-8")
        pre = json.loads((wd / "input_preprocessed.json").read_text(encoding="utf-8"))
    assert "father emotionally abusive" in flat and pre["validation_ok"]


def test_runner_print_commands_runs_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        proc = _run(RUN_FIRE, NOTES, "--state-space", "ss.json", "--form", "f.json",
                    "--workdir", Path(tmp) / "w", "--print-commands", cwd=tmp)
        assert not (Path(tmp) / "w").exists()
    lines = [l for l in proc.stdout.splitlines() if not l.startswith("#")]
    assert len(lines) == 10 and "--verify --verify-llm" in lines[7]


def test_tables_from_synthetic_judgements():
    """A judgements CSV with the published counts reproduces Table II exactly,
    through both scripts' command lines."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _run(TABLES, "--example-csv", tmp / "j.csv", cwd=tmp)
        _run(EVALUATE, tmp / "j.csv", "--json", tmp / "m.json", cwd=tmp)
        metrics = {(m["system"], m["condition"]): m
                   for m in json.loads((tmp / "m.json").read_text(encoding="utf-8"))}
        table = _run(TABLES, tmp / "j.csv", "--table", "main", cwd=tmp).stdout
        published = json.loads((REPO / "data" / "paper_results" / "published_results.json")
                               .read_text(encoding="utf-8"))
    assert r"\textbf{FiRE} & \textbf{94.46} & \textbf{95.54} & \textbf{100.00} & \textbf{100.00}" \
        in table
    for entry in published["systems"]:
        for cond in ("notes", "notes+form"):
            m = metrics[(entry["system"], cond)]
            assert abs(m["accuracy"] - entry[cond]["accuracy"]) < 0.005, (entry["system"], cond)
            assert abs(m["grounded"] - entry[cond]["grounded"]) < 0.005, (entry["system"], cond)


def test_evaluate_rejects_incomplete_judgements():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.csv"
        path.write_text("system,condition,individual,item,error_mode\nFiRE,notes,I01,1,\n",
                        encoding="utf-8")
        proc = _run(EVALUATE, path, cwd=tmp, ok_codes=(1,))
    assert "1 of 14 items judged" in proc.stderr


def test_parameters_match_supplement_s8():
    """The fixed thresholds and output limits of Supplementary Table
    "Model-call parameters and fixed thresholds" (Section S8)."""
    import importlib
    import inspect
    from test_worked_example_chain import HAVE_AIOHTTP, SRC
    for d in ("extraction", "verification", "state_space", "algebra", "instruments", "baselines"):
        if str(SRC / d) not in sys.path:
            sys.path.insert(0, str(SRC / d))
    S4 = importlib.import_module("fire_stage4")
    S5 = importlib.import_module("fire_stage5")
    W3G = importlib.import_module("fire_worker3_general")
    W4 = importlib.import_module("fire_worker4_ace")
    ace14 = importlib.import_module("ace14")
    B0 = importlib.import_module("fire_llm_baseline")
    B1 = importlib.import_module("fire_llm_baseline_self_form_included")
    assert B1.MAX_OUTPUT_TOKENS == 16000      # read before the structured script raises it
    SG = importlib.import_module("fire_llm_structured_baseline")

    assert (S4.COSINE_THRESHOLD, S4.COSINE_THRESHOLD_PATTERN) == (0.60, 0.55)
    assert (S5.RESCUE_SIM_THRESHOLD, S5.RESCUE_JAC_THRESHOLD) == (0.88, 0.60)
    assert S5._build_rescue_payload("gpt-5.1", "u")["max_completion_tokens"] == 16
    assert "confidence = 0.80" in inspect.getsource(W3G.parse_worker2_output)
    assert W4.HIGH_RISK_THRESHOLD == 4 and (ace14.MIN_AGE, ace14.MAX_AGE) == (0, 18)
    assert len(ace14.CLUSTER_ITEM) == 69
    assert B0.MAX_OUTPUT_TOKENS == 8000 and SG.base.MAX_OUTPUT_TOKENS == 32000
    if not HAVE_AIOHTTP:
        return
    S23 = importlib.import_module("fire_stage2_3_single_model")
    W2 = importlib.import_module("fire_worker2")
    assert (S23.COSINE_THRESHOLD, S23.JACCARD_THRESHOLD) == (0.82, 0.30)
    assert (S23.NLI_COSINE_LOWER, S23.NLI_COSINE_UPPER) == (0.50, 0.82)
    assert (S23.NLI_JACCARD_LOWER, S23.NLI_JACCARD_UPPER, S23.NLI_ENTAILMENT_MIN) == (0.10, 0.30, 0.55)
    assert S23.build_payload("gpt-5.1", "s", "u")["max_completion_tokens"] == 2000
    assert (W2.CANDIDATE_MIN_SCORE, W2.CLUSTER_MATCH_SCORE, W2.SAME_CLUSTER_AUTO_MAP) == (0.30, 0.80, 0.82)
    assert inspect.signature(W2.call_llm).parameters["max_output_tokens"].default == 4000
    assert "max_output_tokens=300" in inspect.getsource(W2.process_event)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"[PASS] {name}")
        except Skipped as e:
            print(f"[SKIP] {name}: {e}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {name}\n{e}")
    sys.exit(1 if failures else 0)
