# FiRE: Auditable Clinical Scoring from Consultation Notes with a Neurosymbolic Concept Bottleneck

Code, prompts, evaluation scripts and a worked example for the paper by
Shlok Shelat, Shrey Salvi, Souvik Roy, Manas Gaur, Gunjan Trivedi, Riri
Trivedi, and Amit Sheth (IEEE Journal of Biomedical and Health Informatics,
under review).

FiRE (Finite-automata inspired Reasoning Engine) scores a clinical instrument,
here the 14-item Adverse Childhood Experiences (ACE) questionnaire, from
psychotherapists' consultation notes. Instead of asking a language model for a
score, it extracts events from the notes, checks each one against its source
text (construct, actor and role, age, and cited span), maps it to a shared
state space of clinical states and clusters, and assembles the verified states
into a symbolic trajectory with four operations (concatenation, union,
self-loop, feedback). The ACE score is computed from that trajectory by a fixed
projection and a fixed cluster-to-item table, so every point of the score
traces back to a span of the notes and can be inspected and corrected. GPT-5.1
is the only language model FiRE calls.

The consultation notes, forms and reference scores cannot be released (see
[Data availability and ethics](#data-availability-and-ethics)). This repository
therefore contains everything needed to run FiRE and the baselines on new
notes, to recompute every reported metric from expert judgements, and to check
the paper's worked example offline; it does not contain the study data.

**Contents:**
[Quick start](#quick-start) ·
[Results](#results) ·
[Pipeline](#pipeline) ·
[Running FiRE](#running-fire) ·
[Baselines](#running-the-baselines) ·
[Ablation](#running-the-ablation) ·
[Evaluation and tables](#evaluation-and-tables) ·
[Tests](#tests-and-self-tests) ·
[Repository structure](#repository-structure) ·
[Citation](#citation)

## Quick start

Everything in this section is offline: no API key, no model download.

```
git clone https://github.com/ShlokShelat/fire-paper && cd fire-paper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest

make check            # tests, every script's --selftest, prompt files current
make worked-example   # the paper's worked example: ACE = 4, items [1, 2, 4, 11]
make tables           # the paper's tables, from the published counts
```

Without `make`, the same checks are
`python3 -m pytest -q tests`, `python3 <script> --selftest` for each script,
and `python3 prompts/export_prompts.py --check`. The algebra, instrument,
baseline and evaluation code need only the standard library, so the test
files also run before `pip install` (`python3 tests/test_worked_example_chain.py`
and `python3 tests/test_scripts.py`); the tests of the scripts that import
`aiohttp` (Stages 2-3, Worker 2, the ablation, the prompt check) are then
skipped.

The worked example (`data/worked_example/`) is the main paper's excerpt in the
shape Worker 2 writes, so the rest of the pipeline runs without a model.
`make worked-example` runs Worker 3 (all instruments), Worker 3 (ACE) and
Worker 4 on it and prints the ACE expression `[C] (a+b+c^2) . (e+a^3)` and a
score of 4 with items [1, 2, 4, 11], as in the paper (Fig. 3).

## Results

Main paper Table II: 560 item decisions per system and condition (40
held-out individuals, 14 items each). Acc.: agreement with the
psychotherapists' consensus reference; Grd.: share of decisions with no E1,
E3, E4 or E6 error; Sens. and Spec.: share of present items found and of
absent items correctly left out; MAE: mean absolute error of the total
score; >= 4: individuals on the correct side of ACE >= 4 (of 40). All in %
except MAE and >= 4. SG: structure-guided; KM: knowledge-matched.

| Model | Notes: Acc. | Grd. | Sens. | Spec. | MAE | >= 4 | + form: Acc. | Grd. | Sens. | Spec. | MAE | >= 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FiRE** | **94.5** | **100.0** | **87.3** | **98.8** | **0.7** | **36** | **95.5** | **100.0** | **90.8** | **99.1** | **0.6** | **37** |
| GPT-5.1 | 69.6 | 80.2 | 92.5 | 55.6 | 3.5 | 31 | 57.1 | 69.1 | 84.2 | 36.9 | 4.1 | 30 |
| Grok-4.20 | 65.7 | 82.9 | 77.0 | 58.8 | 2.8 | 31 | 56.3 | 71.1 | 75.0 | 42.2 | 3.3 | 31 |
| Gemini 3.1 Pro | 67.9 | 82.0 | 82.2 | 59.1 | 2.9 | 31 | 54.6 | 70.4 | 73.8 | 40.3 | 3.4 | 30 |
| GPT-5.1, SG | 62.5 | 86.4 | 49.3 | 70.6 | 2.0 | 30 | 53.0 | 75.0 | 50.4 | 55.0 | 2.4 | 29 |
| Grok-4.20, SG | 60.5 | 88.2 | 41.8 | 72.0 | 2.2 | 29 | 54.1 | 76.1 | 48.3 | 58.4 | 2.4 | 29 |
| Gemini 3.1 Pro, SG | 63.9 | 87.5 | 51.2 | 71.8 | 2.0 | 30 | 53.6 | 75.4 | 49.6 | 56.6 | 2.3 | 30 |
| GPT-5.1, KM | 74.5 | 88.4 | 74.2 | 74.6 | 1.8 | 32 | 64.5 | 77.7 | 69.2 | 60.9 | 2.2 | 31 |
| Grok-4.20, KM | 72.3 | 89.1 | 69.0 | 74.4 | 1.8 | 32 | 63.8 | 78.0 | 65.4 | 62.5 | 2.2 | 31 |
| Gemini 3.1 Pro, KM | 75.2 | 88.9 | 75.6 | 74.9 | 1.8 | 32 | 65.0 | 77.9 | 70.4 | 60.9 | 2.2 | 32 |

Error modes (main paper, Section VI): E1 over-inference, E2 construct
under-recognition, E3 referent and relational misattribution, E4 surface
keyword matching, E5 age-window violation, E6 reconciliation instability.
Error counts by mode, the error columns of Table II:

| Model | Condition | E1 | E2 | E3 | E4 | E5 | E6 | Tot. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **FiRE** | Notes | 0 | 27 | 0 | 0 | 4 | 0 | **31** |
| **FiRE** | Notes + form | 0 | 22 | 0 | 0 | 3 | 0 | **25** |
| GPT-5.1 | Notes | 35 | 16 | 47 | 29 | 43 | 0 | 170 |
| GPT-5.1 | Notes + form | 38 | 18 | 47 | 31 | 49 | 57 | 240 |
| Grok-4.20 | Notes | 32 | 49 | 44 | 20 | 47 | 0 | 192 |
| Grok-4.20 | Notes + form | 34 | 40 | 46 | 26 | 43 | 56 | 245 |
| Gemini 3.1 Pro | Notes | 32 | 38 | 43 | 26 | 41 | 0 | 180 |
| Gemini 3.1 Pro | Notes + form | 33 | 44 | 49 | 29 | 44 | 55 | 254 |
| GPT-5.1, SG | Notes | 23 | 108 | 28 | 25 | 26 | 0 | 210 |
| GPT-5.1, SG | Notes + form | 25 | 98 | 31 | 25 | 25 | 59 | 263 |
| Grok-4.20, SG | Notes | 21 | 124 | 26 | 19 | 31 | 0 | 221 |
| Grok-4.20, SG | Notes + form | 22 | 102 | 30 | 20 | 21 | 62 | 257 |
| Gemini 3.1 Pro, SG | Notes | 22 | 104 | 25 | 23 | 28 | 0 | 202 |
| Gemini 3.1 Pro, SG | Notes + form | 24 | 100 | 33 | 21 | 22 | 60 | 260 |
| GPT-5.1, KM | Notes | 22 | 55 | 23 | 20 | 23 | 0 | 143 |
| GPT-5.1, KM | Notes + form | 22 | 54 | 26 | 21 | 20 | 56 | 199 |
| Grok-4.20, KM | Notes | 20 | 66 | 22 | 19 | 28 | 0 | 155 |
| Grok-4.20, KM | Notes + form | 20 | 61 | 24 | 16 | 19 | 63 | 203 |
| Gemini 3.1 Pro, KM | Notes | 21 | 52 | 24 | 17 | 25 | 0 | 139 |
| Gemini 3.1 Pro, KM | Notes + form | 23 | 51 | 27 | 18 | 21 | 56 | 196 |

Further runs (Supplementary Section S8 and Table SIV):

- **Three runs** of every system with the same settings and judging (the main
  run is one of them): FiRE 94.3 ± 0.4% accuracy with notes alone and
  95.4 ± 0.3% with the form, 100% grounding in every run; at most
  74.3 ± 1.6% and 65.3 ± 1.4% for any baseline (mean ± sample s.d.).
- **Reasoning on** (GPT-5.1 at reasoning effort "high"): FiRE 94.6% and
  95.7%; the GPT-5.1 baselines reach at most 76.1% (knowledge-matched, notes).
- **Compute-matched** GPT-5.1 (each item asked 11 times, majority answer,
  154 calls per individual against FiRE's 156): 72.14% with notes alone and
  60.00% with the form; grounding 82.50% and 70.89%.
- **Non-LLM baselines** (Table SVII): keyword rules and a trained classifier
  reach at most 69.1% with notes alone; the form left unchanged scores 81.79%.

Ablation over the same 560 decisions with GPT-5.1 (Supplementary Table SVIII,
Section S9). Grd.: share of decisions with no E1, E3, E4 or E6 error.

Notes only:

| Run | E1 | E2 | E3 | E4 | E5 | E6 | Tot. | Acc. (%) | Grd. (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FiRE | 0 | 27 | 0 | 0 | 4 | 0 | 31 | 94.46 | 100.00 |
| Construct off | 4 | 26 | 0 | 1 | 4 | 0 | 35 | 93.75 | 99.11 |
| Actor and role off | 0 | 26 | 4 | 0 | 4 | 0 | 34 | 93.93 | 99.29 |
| Age off | 0 | 25 | 0 | 0 | 19 | 0 | 44 | 92.14 | 100.00 |
| Cited span off | 6 | 21 | 4 | 5 | 5 | 0 | 41 | 92.68 | 97.32 |
| All four off | 13 | 17 | 10 | 10 | 21 | 0 | 71 | 87.32 | 94.11 |
| No state space | 8 | 47 | 8 | 9 | 7 | 0 | 79 | 85.89 | 95.54 |
| Model scores | 3 | 30 | 2 | 4 | 6 | 0 | 45 | 91.96 | 98.39 |

Notes + form:

| Run | E1 | E2 | E3 | E4 | E5 | E6 | Tot. | Acc. (%) | Grd. (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FiRE | 0 | 22 | 0 | 0 | 3 | 0 | 25 | 95.54 | 100.00 |
| Construct off | 3 | 23 | 0 | 1 | 3 | 1 | 31 | 94.46 | 99.11 |
| Actor and role off | 0 | 22 | 4 | 0 | 3 | 1 | 30 | 94.64 | 99.11 |
| Age off | 0 | 21 | 0 | 0 | 15 | 0 | 36 | 93.57 | 100.00 |
| Cited span off | 5 | 18 | 4 | 5 | 4 | 5 | 41 | 92.68 | 96.61 |
| All four off | 11 | 15 | 10 | 9 | 17 | 13 | 75 | 86.61 | 92.32 |
| No state space | 8 | 43 | 9 | 9 | 5 | 8 | 82 | 85.36 | 93.93 |
| Model scores | 3 | 25 | 3 | 4 | 4 | 5 | 44 | 92.14 | 97.32 |

How these numbers were produced: the two psychotherapists who wrote the notes
judged every system's output item by item, double-blind, with disagreements
settled by senior psychotherapists and the result checked by an auditor (Supplementary
Material, Section S5), and the metrics follow from those judgements.
`scripts/evaluate_judgements.py` and `scripts/make_tables.py` compute the
metrics and render the tables from a per-item judgements file (see
[Evaluation and tables](#evaluation-and-tables)). The judgements themselves
cannot be released; the aggregate counts are in
`data/paper_results/published_results.json`, and `make_tables.py --selftest`
checks that every accuracy, grounding and error total above, and the
intervals and McNemar bounds of Table SIV, follows from those counts;
`clinical_measures.py --selftest` checks that the notes-only sensitivity and
specificity follow from them too.

## Pipeline

The paper describes four stages: (i) preprocessing the notes into events,
(ii) mapping each event to the shared state space, (iii) verification by four
checks, and (iv) building the trajectory, projecting it, and scoring. The
implementation (Supplementary Material, Section S1):

| Step | What it does | Stage | Script |
|---|---|---|---|
| Preprocessor | Rule-based normalisation to flat text, no LLM | (i) | `src/preprocessing/fire_preprocessor.py` |
| Stage 1 | Tags lines; marks commentary as low confidence | (i) | `src/extraction/fire_stage1.py` |
| Stages 2-3 | GPT-5.1 extracts events; code runs the four checks | (i), (iii) | `src/extraction/fire_stage2_3_single_model.py` |
| Stage 4 | Rechecks entities, grounding and timing | (iii) | `src/verification/fire_stage4.py` |
| Stage 5 | Rescues discards; removes duplicates | (iii) | `src/verification/fire_stage5.py` |
| Worker 2 | Splits events; maps each to a state | (ii) | `src/state_space/fire_worker2.py` |
| Worker 3 (all) | Builds the trajectory and its operations | (iv) | `src/algebra/fire_worker3_general.py` |
| Worker 3 (ACE) | Projects to ACE, ages 0 to 18; checks each item and its conditions | (iv) | `src/instruments/fire_worker3_ace.py` |
| Worker 4 | Parses the expression; counts distinct items | (iv) | `src/instruments/fire_worker4_ace.py` |
| Form step | Notes + form only: decides each item from FiRE's evidence and the form (no model call) | (iv) | `src/instruments/fire_reconcile_ace.py` |

The Preprocessor and Stages 1 to 5 together are Worker 1. The four checks run
on each extracted event before Worker 2 maps it to a state; at that point the
age check tests the event's age against the text, and the 0 to 18 window is
applied at ACE projection.

```
 consultation notes
        |
 Preprocessor ......... flat, self-contained lines                (no LLM)
        |
 Stage 1 .............. tag lines; commentary = low confidence   (no LLM)
        |
 Stages 2-3 ........... extract events (GPT-5.1), then four checks in code:
        |               construct | actor and role | age | cited span
 Stage 4 .............. recheck entities, grounding, timing       (no LLM)
        |
 Stage 5 .............. rescue a discard on a yes, or keep it unverified if the call fails;
        |               remove duplicates within one note section
 Worker 2 ............. split each event (GPT-5.1); search the whole state
        |               space; match (GPT-5.1) or create a provisional state
 Worker 3 (all) ....... trajectory: concatenation, union, self-loop, feedback
        |
 Worker 3 (ACE) ....... keep ACE clusters in ages 0-18; item check (GPT-5.1);
        |               hold an item whose frequency, duration or age gap is not stated
        |
 Worker 4 ............. symbol -> cluster -> item; ACE = number of distinct items
        |
 Form step ............ notes + form only: change a form answer only on evidence
```

FiRE holds or flags its output in these places (Section S1): a new state is
used only for the individual whose event created it until a psychotherapist
approves it; a Worker 3 (ACE) removal needs a second model call to confirm it;
an event failing one Stages 2-3 check, an event the Stage 4 audit marks
uncertain, and an event restored by the Stage 5 rescue are kept but flagged
(a discard whose text reads as therapist commentary is never sent to the
rescue call); a failed Worker 2 matching call leaves the event for review
instead of guessing; and an item is held for review, not counted, when the
notes do not state a condition its wording requires or the event's age.
The item conditions (Supplementary Table SI): ACE-1, 2, 4, 5, 7, 11 and 12 must
have happened often, and ACE-14 all the time, so the line must say so
("often", "repeatedly"), describe an ongoing pattern, or cover a year or
more; ACE-2 also counts one event that caused marks or injury, and ACE-7
counts "sometimes" for kicking, biting or hitting and one threat with a weapon
or a long beating; ACE-3 needs the other person to be an adult or at least
five years older; ACE-10 and ACE-13 need two or more years. The code finds a
condition in the record's age span, its self-loop or the wording of its line,
and with `--verify-llm` GPT-5.1 also answers one narrow condition question per
item in the item-check call (`prompts/ace_verification_prompts.txt`). If Worker 2's splitting call fails, the event is logged as an error and not mapped. Worker 3 joins neighbouring states by union when their ages overlap, and also when the note says they happened at the same time (for example "meanwhile"); union never changes an ACE score. The score card gives the score and the items behind it, with no risk level or clinical interpretation.

The cluster-to-item table (69 entries, Supplementary Material, Section S2) is
in one place, `src/instruments/ace14.py`, which Worker 3 (ACE), Worker 4 and
the knowledge-matched baseline's state-space trimmer all import.

## Running FiRE

### Requirements

Python 3.10 or later; the paper's runs used Python 3.13.12, torch 2.11.0 and
numpy 2.4.4 on a CPU-only laptop (Supplementary Material, Section S7).
`pip install -r requirements.txt` installs the four third-party packages the
pipeline imports: `aiohttp`, `numpy`, `sentence-transformers` (which installs
`torch`) and `spacy`. The two small local models (`all-MiniLM-L6-v2` and
`cross-encoder/nli-MiniLM2-L6-H768`) are downloaded from the Hugging Face Hub
on first use. All language-model inference is remote, through hosted APIs.

API keys are read from environment variables (Stages 2-3 also accept
`--api-key`); no script stores or prints one.

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | every FiRE model call (GPT-5.1: Stages 2-3, Stage 5, Worker 2, Worker 3 ACE), the ablation runs, and the GPT-5.1 baselines |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini 3.1 Pro baselines |
| `XAI_API_KEY` (or `GROK_API_KEY`) | Grok-4.20 baselines |
| `FIRE_STATE_SPACE` | optional: state-space path for the knowledge-matched baseline, instead of `--state-space` |

Model-call settings follow Supplementary Material, Section S7: every model runs
at temperature 0. GPT-5.1 accepts temperature 0 only with reasoning off, so it
is sent reasoning effort "none" and temperature 0, inside FiRE and as a
baseline alike; Gemini 3.1 Pro and Grok-4.20 reason at their default settings.
`--reasoning-effort high` (on `scripts/run_fire.py` and every baseline script,
or `FIRE_REASONING_EFFORT=high` in the environment) gives the reasoning-on runs
of Section S8: GPT-5.1 at effort "high", no temperature, and every FiRE output
limit raised to 8000 tokens. Baseline outputs of these runs are named
`*.gpt-5.1.reasoning-high.*.json` and record `reasoning_effort`, and FiRE
refuses to reuse a working folder made with the other setting, so the two
sets of runs never overwrite each other. Output limits in the main runs:
2000 tokens for extraction and the Worker 3 item check, 4000 and 300 for
Worker 2's splitting and matching calls, 400 for the removal confirmation, 16
for the rescue call.

### End to end, one command

`scripts/run_fire.py` chains the scripts below with the options the paper's
runs used, in a working folder per individual, and writes `run_log.json`
(commands, exit codes, durations):

```
export OPENAI_API_KEY=...
# one individual, notes only
python3 scripts/run_fire.py notes/patient_01.txt \
    --state-space runs/state_space.json --workdir runs/patient_01
# the same individual, notes plus the self-filled form: reuse the notes-only
# run above and add only the form step (no new model calls)
python3 scripts/run_fire.py notes/patient_01.txt --form notes/patient_01_ace_form.json \
    --workdir runs/patient_01 --from reconcile
# every patient_NN.txt in a folder, one at a time, sharing one state space
python3 scripts/run_fire.py --batch notes/ [--with-forms] \
    --state-space runs/state_space.json --out-dir runs
```

`--print-commands` prints the exact commands without running anything;
`--from STEP` and `--to STEP` run part of the chain. The state-space file is
created empty if it does not exist (see [State space](#state-space)). The
self-filled form is JSON such as `{"1": 1, "2": 0, ..., "14": 0}` or
`{"1": {"score": 1}, ...}`; only the yes or no answer is read.

### Step by step

Each script writes the file the next one reads; this is what `run_fire.py`
runs. From an individual's working folder:

```
export OPENAI_API_KEY=...
export FIRE_SRC=/path/to/fire-paper/src
mkdir -p runs && echo '{"domains": {}}' > runs/state_space.json   # once; shared by all individuals
mkdir -p runs/patient01 && cp /path/to/patient01.txt runs/patient01/ && cd runs/patient01

# Preprocessor -> patient01_preprocessed.json, patient01_flat.txt
python3 "$FIRE_SRC/preprocessing/fire_preprocessor.py" patient01.txt
# Stage 1 -> stage1_output.json
python3 "$FIRE_SRC/extraction/fire_stage1.py" patient01_flat.txt -o stage1_output.json
# Stages 2-3 -> validated_events.json (+ validated_events_audit.json)
python3 "$FIRE_SRC/extraction/fire_stage2_3_single_model.py" stage1_output.json \
    --patient-id patient01 --output validated_events.json
# Stage 4
python3 "$FIRE_SRC/verification/fire_stage4.py" validated_events.json \
    --stage1 stage1_output.json -o validated_events_s4.json
# Stage 5
python3 "$FIRE_SRC/verification/fire_stage5.py" validated_events_s4.json \
    --stage1 stage1_output.json -o validated_events_s5.json
# Worker 2 (adds new provisional states to the shared state space)
python3 "$FIRE_SRC/state_space/fire_worker2.py" validated_events_s5.json \
    --stage1 stage1_output.json --state-space ../state_space.json \
    --output final_mapped_events.json
# Worker 3 (all instruments) -> also worker3_output.html
python3 "$FIRE_SRC/algebra/fire_worker3_general.py" final_mapped_events.json -o worker3_output.json
# Worker 3 (ACE), with the item check as in the paper
python3 "$FIRE_SRC/instruments/fire_worker3_ace.py" worker3_output.json \
    -o worker3_ace_output.json --verify --verify-llm
# Worker 4
python3 "$FIRE_SRC/instruments/fire_worker4_ace.py" worker3_ace_output.json -o ace_scorecard.json
# Notes + form condition only: reconcile with the self-filled form (no API call)
python3 "$FIRE_SRC/instruments/fire_reconcile_ace.py" worker3_ace_output.json ace_scorecard.json \
    --form form.json -o ace_reconciled.json
```

Every script prints its options with `--help`. Useful ones:
`fire_stage2_3_single_model.py stage1_output.json --dry-run` prints the
extraction prompts without calling the API, and `fire_worker2.py --dry-run`
prints the splitting prompts. Do not pass `--fresh-start` to Worker 2 unless
you mean it: it deletes the state-space file. Worker 3 (ACE) with `--verify`
but without `--verify-llm` runs only the deterministic part of the item check
(no API call); the paper's runs used `--verify --verify-llm`.

`worker3_output.html` is an interactive view of the trajectory. When opened, it
loads the vis-network library from cdnjs.cloudflare.com; place
`vis-network-bundle.js` next to `fire_worker3_general.py` to inline it and
make the page work offline.

The Preprocessor and Stage 1 were written for the partner clinic's note
layout: `SECTION n:` headings, numbered subsections such as `2.1 Father`, a
Section 3 timeline with age markers such as `~6-8` (on their own line or
starting a row), and `Key:` lines. Other layouts run, but the line tagging and
section grouping that Stages 2-3 depend on may be poorer.

### The notes-plus-form step

In the notes-plus-form condition FiRE does not read the form during the
pipeline. It runs on the notes alone, then `fire_reconcile_ace.py` decides each
of the 14 items from FiRE's finished output and the form's yes/no answer (the
frequency is ignored), with no model call (Supplementary Section S1):

1. FiRE counted the item on verified evidence: **yes**, citing that evidence.
2. FiRE held the item (a required condition or the age is not stated): the
   form's answer stays, flagged.
3. FiRE's only evidence for the item failed a check whose removal the second
   GPT-5.1 call did not confirm: the form's answer stays, flagged.
4. FiRE has evidence the item does not count (the event starts after 18, or
   the second GPT-5.1 call confirmed a removal): **no**, citing that evidence.
5. The notes are silent: the form's answer stays, flagged.

Only 1 and 4 can change a form answer, and every change cites FiRE's evidence
from the notes, so an unsupported change (E6) can arise only from wrong
evidence. The output lists, per item, the final answer, its source, the
evidence, whether it changed and why, and whether a psychotherapist must
review it, plus the reconciled score.

### State space

The paper's state space was built from development data: notes from the same
clinic and process, from individuals disjoint from the 40 evaluation
individuals (Supplementary Material, Section S3). It is not released.
`src/state_space/fire_state_space_tree.json` documents the schema only.
Starting from `{"domains": {}}`, every state is created by Worker 2 as
provisional. A provisional state is used at once for the individual whose
event created it and hidden from every other individual until a
psychotherapist approves it. After each run, Worker 2 merges approved states
of one cluster whose embeddings have cosine 0.82 or more (`--no-dedup-sweep`
turns this off; the paper's runs kept it on). Provisional states are never
part of this merge, so no individual's new state is merged with another
state. To review provisional states:

```
python3 "$FIRE_SRC/state_space/fire_worker2.py" --state-space ../state_space.json --export-review review.json
# set "decision": "approve" or "reject" (and optionally "reviewer", "note") on each entry, then
python3 "$FIRE_SRC/state_space/fire_worker2.py" --state-space ../state_space.json --apply-review review.json \
    --patch-events final_mapped_events.json final_mapped_events_reviewed.json
```

Approval makes a state permanent and visible to every individual; rejection
removes it, and `--patch-events` flags every piece that mapped to it for
reprocessing. In the paper, the 14 provisional states created across the 40
evaluation individuals were reviewed and approved as one batch after mapping
and before any scoring (Section S3).

## Running the baselines

Each baseline is one model call per individual, from scripts that share no
code with the pipeline (Supplementary Material, Section S4). `--model` is one
of `gpt-5.1`, `gemini-3.1-pro`, `grok-4.20-0309-reasoning`.

```
# Plain (direct scoring), notes only
python3 src/baselines/fire_llm_baseline.py notes.txt --model gpt-5.1 -o plain.json
# Plain, notes + form (reconciliation)
python3 src/baselines/fire_llm_baseline_self_form_included.py notes.txt --form form.json \
    --model gpt-5.1 -o plain_form.json
# Structure-guided (add --form form.json for the notes + form condition)
python3 src/baselines/fire_llm_structured_baseline.py notes.txt --model gpt-5.1 -o structured.json
# Knowledge-matched: trim the state space once, then run (add --form for notes + form)
python3 src/baselines/trim_state_space.py state_space.json -o state_space_for_prompt.json
python3 src/baselines/fire_llm_knowledge_baseline.py notes.txt --model gpt-5.1 \
    --state-space state_space_for_prompt.json -o knowledge.json
```

`--print-prompt` prints the exact prompt without calling the API. The
structure-guided and knowledge-matched scripts also take
`--batch DATA_DIR [--with-forms] --out-dir DIR`, which runs every
`patient_NN.txt` in a folder (with `patient_NN_ace_form.json`).
`trim_state_space.py` keeps only what the knowledge-matched prompt uses
(clusters with construct and item; states with their description, actors and
roles, age window, example phrasings and ACE eligibility; the 69-entry table,
which must equal `ace14.py`; non-ACE clusters; per-item rules) and drops review
notes and every reference to where a state was seen in a note. The paper's
knowledge-matched prompt used the development state space, whose states carry
all the fields listed in Section S3, together with per-item rules; neither is
released. Worker 2 here writes a shorter state (code, name, description, life
stage, experience type, status), so a state space built with it gives a
shorter prompt.

The output limits are 8000 tokens for `fire_llm_baseline.py` (plain, notes
only), 16000 for `fire_llm_baseline_self_form_included.py` (plain, notes plus
form), and 32000 for the structure-guided and knowledge-matched scripts, whose
replies are much longer (Supplementary Section S7). The plain baselines also
accept `--strip-ace-annotations`, which removes note lines that already state
an ACE score or item list; it is off by default.

### Compute-matched baseline

`src/baselines/fire_llm_compute_matched.py` asks GPT-5.1 about each item
separately, 11 times (14 x 11 = 154 calls per individual, about FiRE's 156),
with the full notes and one item in the plain prompt's wording; with `--form`,
the plain reconciliation wording with the individual's answer to that item.
Reasoning is off and no temperature is set, so answers can differ. The final
answer is the most common one; the cited sentence comes from the first call
that gave it. Every answer is kept, with the majority of the first 1, 3, 5,
7, 9 and 11 answers (`answers_by_k`), and the run resumes after an
interruption (Supplementary Section S8):

```
python3 src/baselines/fire_llm_compute_matched.py notes.txt -o cm.json
python3 src/baselines/fire_llm_compute_matched.py notes.txt --form form.json -o cm_form.json
```

### Reasoning-on runs

The same scripts with `--reasoning-effort high` (Supplementary Section S8):

```
python3 scripts/run_fire.py --batch notes/ --state-space runs_reasoning/state_space.json \
    --out-dir runs_reasoning --reasoning-effort high
python3 src/baselines/fire_llm_baseline.py notes.txt --model gpt-5.1 --reasoning-effort high -o plain_r.json
```

### Non-LLM baselines

`src/baselines/nonllm/` holds the keyword baseline (published ACE terms with
NegEx, the design of Bejan et al.), a TF-IDF and logistic-regression classifier
trained leave-one-individual-out, and two floors: each item's most common
answer among the other individuals, and the form left unchanged
(Supplementary Section S4, Table SVII). Its README lists the term sources,
the commands and the known limitations.

### Repeated runs

The paper's stability figures come from three runs of every system with the
same settings and judging (Section S8). Run each system again into a new
folder; for FiRE, start every run from a copy of the same reviewed state space,
since Worker 2 adds provisional states to the file it is given.

## Running the ablation

`src/ablation/fire_ablation.py` implements the runs of Supplementary Material,
Section S9. It reuses the main run's Stages 2-3 output, so the Stage 2
extraction call is never repeated and every run shares the same extracted
events. Later GPT-5.1 calls (Stage 5, Worker 2, the Worker 3 item check) are
re-run. A removed part is replaced by a neutral stand-in (a removed check
always passes), installed by patching the pipeline's own functions for that
run. One switch per run:

| Switch | What is removed |
|---|---|
| `--no-construct-check` | Stages 2-3 construct check |
| `--no-actor-role-check` | Stages 2-3 actor and role check |
| `--no-age-check` | Stages 2-3 age check, and the 0-18 window at ACE projection |
| `--no-cited-span-check` | Stages 2-3 cited-span check and Stage 4's grounding recheck |
| `--no-checks` | all four checks |
| `--no-state-space` | Worker 2 maps each piece straight to an ACE item (same number of calls; checks on) |
| `--model-scoring` | the model decides each cluster's item from its verified states, instead of the table |
| `--no-algebra` | offline: distinct items of the verified, in-window states, compared with Worker 4 |

From a folder holding one individual's main-run files (as in `runs/patient01`):

```
python3 "$FIRE_SRC/ablation/fire_ablation.py" --no-age-check \
    --stage1 stage1_output.json --cached-extraction validated_events.json \
    --state-space ../state_space.json --out-dir ablation_no_age_check
python3 "$FIRE_SRC/ablation/fire_ablation.py" --no-algebra \
    --w3-ace worker3_ace_output.json --scorecard ace_scorecard.json
```

The state space is copied into `--out-dir` and never modified. A run writes
every stage's output plus `ablation_manifest.json` (switch, patched functions,
input hashes, resulting score). `--no-algebra` exits with status 1 if the two
scores differ.

## Evaluation and tables

Accuracy and grounding come from expert judgements, not from a script: in the
paper, the reference team judged each of the 560 decisions per system and
condition, blind to the system, recording the error mode of any wrong or
ungrounded decision (Supplementary Material, Section S5). The scripts below
prepare the blind review, then recompute every reported number offline:

```
# 1. one plain, unlabelled, shuffled sheet for the reviewers (and a separate key)
python3 scripts/blind_review.py export --seed 20260501 --sheet sheet.csv --key key.csv \
    --add "FiRE" notes "runs/*/" --add "FiRE" notes+form "runs/*/" \
    --add "GPT-5.1" notes "out/*.gpt-5.1.ace.json"
# 2. after judging: back to a judgements CSV
python3 scripts/blind_review.py import sheet_filled.csv key.csv --reference ref.json -o judgements.csv
# 3. accuracy, grounding, intervals, McNemar; the paper's tables
python3 scripts/evaluate_judgements.py judgements.csv [--json metrics.json]
python3 scripts/make_tables.py judgements.csv --table main      # Table II, accuracy and grounding
python3 scripts/make_tables.py judgements.csv --table errors    # Table II, error columns
python3 scripts/make_tables.py judgements.csv --table stats     # Table SIV, main run
python3 scripts/make_tables.py ablation.csv   --table ablation  # Table SVIII
# 4. sensitivity, specificity, MAE and the ACE >= 4 cutoff (Table II); per item (Tables SV, SVI)
python3 scripts/collect_answers.py --reference ref.json --system FiRE --condition notes \
    runs/*/ace_scorecard.json -o answers.csv
python3 scripts/collect_answers.py --reference ref.json --system "GPT-5.1" --condition notes \
    out/*.gpt-5.1.ace.json -o answers.csv --append
python3 scripts/clinical_measures.py answers.csv --per-item
# 5. three runs: mean ± sample s.d. of accuracy and grounding (Table SIV)
python3 scripts/aggregate_runs.py judgements_run1.csv judgements_run2.csv judgements_run3.csv
# 6. entries left for psychotherapist review (Section S5)
python3 scripts/review_burden.py runs/*/
```

The reference JSON has one entry per individual, item 1 first:
`{"P01": {"notes": "10010000100101", "notes+form": "...", "form": "..."}}`.

The judgements file is a CSV with one row per item decision:

| Column | Content |
|---|---|
| `system` | e.g. `FiRE`, `GPT-5.1`, `GPT-5.1, structure-guided` (for the ablation table, the run name) |
| `condition` | `notes` or `notes+form` |
| `individual` | coded identifier |
| `item` | 1 to 14 |
| `error_mode` | empty if judged correct and grounded, otherwise one of `E1` to `E6` |
| `system_answer`, `reference_answer` | optional, 0 or 1; a row whose answers differ must carry an error mode |

Definitions, as in the paper: accuracy is the share of decisions with no error
mode (in the paper every judged error carries exactly one mode, so the
error total of Table II is the number of decisions that do not match the
reference); grounding is the share with no E1, E3, E4 or E6 error; intervals
are Wilson 95% intervals over the 560 decisions, plus the clustered extreme of
Section S8 (one effective decision per individual); the McNemar statistic
against FiRE is bounded below from the error counts, as in Table SIV, and
computed exactly when both systems' judgements cover the same decisions. The
scripts check that each system has 14 judged items per individual.

`python3 scripts/make_tables.py --published` renders the tables from the
published counts, and `--example-csv FILE` writes a synthetic judgements file
with those counts to show the format (it is not the study's data).

## Tests and self-tests

Every pipeline, baseline, ablation and evaluation script has an offline
`--selftest` that uses synthetic inputs and makes no network call, and two
test files run the command-line tools as separate processes:

```
python3 -m pytest -q tests                       # or run each test file directly
python3 src/ablation/fire_ablation.py --selftest # any script: --selftest
python3 prompts/export_prompts.py --check        # prompts/*.txt match the code
make check                                       # all of the above
```

Every script in `scripts/`, the compute-matched baseline and the non-LLM
baselines also have a `--selftest`, run by `make check`.
`tests/test_worked_example_chain.py` runs the worked example through Worker 3
(all) -> Worker 3 (ACE) -> Worker 4 and checks the paper's expression and
score, runs the notes-plus-form step on it with a test form, runs every
`--selftest`, checks that `--no-algebra` equals Worker 4's score, and checks
that the prompt files are current. `tests/test_scripts.py` runs
`scripts/run_fire.py` on the worked example (one individual and batch mode),
checks that it refuses to start model steps without a key, and checks that
the evaluation scripts reproduce Table II from a judgements file with the
published counts, and checks the model-call settings of Sections S7 and S8. Every API key is removed from the environment of each test
process. Tests for scripts that import `aiohttp` are skipped if it is not
installed. The ablation self-test checks that each switch neutralises exactly
its check and nothing else, that the originals are restored afterwards, and
that `--no-algebra` gives ACE = 4 with items {1, 2, 4, 11} on the worked
example. The Stage 1 self-test uses spaCy if it is installed and a simple
sentence splitter otherwise.

## Prompts

`prompts/*.txt` hold every prompt the pipeline, the baselines and the
ablation send, verbatim, written by `prompts/export_prompts.py` from the code
itself (`--check` verifies them). The only change from the paper's runs: the
extraction prompt named the partner clinic, which now reads `[partner clinic]`.
`prompts/indian_cultural_glossary.txt` is an input, not generated: the
structure-guided and knowledge-matched prompts include its table.

The examples inside the prompts and the code are de-identified paraphrases
of development notes, never of the 40 evaluation notes. Two labels are role
names inside the prompts, not claims about the method: the "Oracle" splits an
event into pieces and the "Matching Judge" decides whether a piece matches an
existing state. The splitting prompt still carries the header "may be
incomplete for A_ONLY/B_ONLY", left from an earlier design with two
extraction models. The paper's runs used GPT-5.1 as the only extractor, so
that case never arises; the text is kept so the prompt stays exactly as run.

## The worked example

The main paper's excerpt (`data/worked_example/input.txt`) describes emotional
abuse by the father, feeling unloved and frequent bullying at ages 0 to 8, the
mother's death at 11, physical abuse with continuing emotional abuse and
depressed mood at 12 to 16, and adult events (a friend's betrayal, a spouse's
affair, divorce, low mood and insomnia). A frontier model scoring directly
reports 7 items; FiRE maps each event to a state (Table I), drops the
depressive symptoms (an outcome), the mother's death (no ACE item) and the
adult events at projection, and serialises the ACE subgraph as
`(a + b + c^2) . (e + a^3)`: union of three co-occurring constructs, a
self-loop for the frequent bullying, and feedback for the returning emotional
abuse. The distinct items are {1, 2, 4, 11}, so ACE = 4; the exponents record
recurrence and do not change the count. `data/worked_example/README.md`
describes the files.

## Repository structure

```
fire-paper/
├── README.md                 this file
├── REPRODUCING.md            what each reported result needs, step by step
├── CITATION.cff  LICENSE  Makefile  pyproject.toml  requirements.txt
├── src/
│   ├── preprocessing/fire_preprocessor.py        Preprocessor
│   ├── extraction/fire_stage1.py                 Stage 1
│   ├── extraction/fire_stage2_3_single_model.py  Stages 2-3
│   ├── verification/fire_stage4.py               Stage 4
│   ├── verification/fire_stage5.py               Stage 5
│   ├── state_space/fire_worker2.py               Worker 2
│   ├── state_space/fire_state_space_tree.json    state-space schema (no content)
│   ├── algebra/fire_worker3_general.py           Worker 3 (all instruments)
│   ├── instruments/ace14.py                      ACE items, window, 69-entry table
│   ├── instruments/fire_worker3_ace.py           Worker 3 (ACE)
│   ├── instruments/fire_worker4_ace.py           Worker 4 (ACE)
│   ├── instruments/fire_reconcile_ace.py         notes + form step
│   ├── baselines/fire_llm_baseline.py                   plain, notes only
│   ├── baselines/fire_llm_baseline_self_form_included.py plain, notes + form
│   ├── baselines/fire_llm_structured_baseline.py        structure-guided
│   ├── baselines/fire_llm_knowledge_baseline.py         knowledge-matched
│   ├── baselines/trim_state_space.py                    state space -> prompt file
│   ├── baselines/fire_llm_compute_matched.py            compute-matched (Section S8)
│   ├── baselines/nonllm/                                non-LLM baselines (Section S4)
│   └── ablation/fire_ablation.py                 ablation runs (Section S9)
├── scripts/
│   ├── run_fire.py               the pipeline end to end, one individual or a folder
│   ├── evaluate_judgements.py    accuracy, grounding, intervals, McNemar from judgements
│   ├── make_tables.py            Table II, Table SIV (main run) and Table SVIII
│   ├── blind_review.py           blind, shuffled review sheets and their import
│   ├── collect_answers.py        every system's answers next to the reference
│   ├── clinical_measures.py      sensitivity, specificity, MAE, ACE >= 4; per item
│   ├── aggregate_runs.py         mean ± sample s.d. over repeated runs
│   └── review_burden.py          entries left for psychotherapist review
├── prompts/                  every prompt, verbatim, generated from the code
│   ├── export_prompts.py     regenerates the .txt files (--check verifies them)
│   ├── extraction_system_prompt.txt   oracle_and_judge_prompts.txt
│   ├── rescue_prompt.txt              ace_verification_prompts.txt
│   ├── baseline_prompts.txt           ablation_prompts.txt
│   └── indian_cultural_glossary.txt   glossary read by two baselines
├── data/
│   ├── worked_example/       the paper's worked example and expected output
│   └── paper_results/published_results.json   published counts and metrics
└── tests/
    ├── test_worked_example_chain.py
    └── test_scripts.py
```

## Data availability and ethics

The consultation notes, self-filled forms, reference scores and expert
judgements cannot be released, as required by the approved protocol and the
participants' consent; the worked example is the only note text here. The
evaluation notes contained no ACE labels, so no system saw the reference in
its input. The code can read an explicit "ACE n" label, and the baselines
offer `--strip-ace-annotations`; both were written for development notes and
had no effect in the reported runs. The
study was approved by the ACEAS-Independent Ethics Committee, Ahmedabad, India
(CDSCO Reg. No. ECR/281/Indt/GJ/2017/RR-25; OHRP Reg. No. IRB00011046;
Protocol WS/Apr/2026, Version 2.0, approved 23 May 2026). Written informed
consent was obtained from every participant, and every system, including the
commercial models, received only coded, de-identified records (Supplementary
Material, Section S6).

FiRE is assistive decision support for psychotherapists, not a diagnostic
authority; its output is meant to be checked and corrected by a clinician.
Total API cost and token counts were not recorded for the paper's runs, so
none is reported (Section S7); the running estimates the scripts print use
placeholder rates.

## Citation

```bibtex
@article{shelat2026fire,
  title   = {{FiRE}: Auditable Clinical Scoring from Consultation Notes with a
             Neurosymbolic Concept Bottleneck},
  author  = {Shelat, Shlok and Salvi, Shrey and Roy, Souvik and Gaur, Manas and
             Trivedi, Gunjan and Trivedi, Riri and Sheth, Amit},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

`CITATION.cff` holds the same reference in machine-readable form.

## License

MIT; see `LICENSE`. The license covers the code and prompts in this
repository, not any clinical data.
