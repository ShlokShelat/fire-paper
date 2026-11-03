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
score of 4 with items [1, 2, 4, 11], as in the paper (Figs. 4 and 5).

## Results

Accuracy and grounding over 560 item decisions (40 held-out individuals, 14
items each), main paper Table II. Accuracy is agreement with the
psychotherapists' consensus reference; a decision is ungrounded when it carries
an E1, E3, E4 or E6 error.

| Model | Accuracy, notes (%) | Accuracy, + form (%) | Grounded, notes (%) | Grounded, + form (%) |
|---|---:|---:|---:|---:|
| **FiRE** | **94.46** | **95.54** | **100.00** | **100.00** |
| GPT-5.1 | 69.64 | 57.14 | 80.18 | 69.11 |
| Grok-4.20 | 65.71 | 56.25 | 82.86 | 71.07 |
| Gemini 3.1 Pro | 67.86 | 54.64 | 81.96 | 70.36 |
| GPT-5.1, structure-guided | 62.50 | 53.04 | 86.43 | 75.00 |
| Grok-4.20, structure-guided | 60.54 | 54.11 | 88.21 | 76.07 |
| Gemini 3.1 Pro, structure-guided | 63.93 | 53.57 | 87.50 | 75.36 |
| GPT-5.1, knowledge-matched | 74.46 | 64.46 | 88.39 | 77.68 |
| Grok-4.20, knowledge-matched | 72.32 | 63.75 | 89.11 | 78.04 |
| Gemini 3.1 Pro, knowledge-matched | 75.18 | 65.00 | 88.93 | 77.86 |

Error modes (main paper, Section VI): E1 over-inference, E2 construct
under-recognition, E3 referent and relational misattribution, E4 surface
keyword matching, E5 age-window violation, E6 reconciliation instability.
Error counts by mode, main paper Table III (SG, structure-guided; KM,
knowledge-matched):

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

Ablation over the same 560 decisions with GPT-5.1 (Supplementary Material,
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
checks that every derived number above, and the intervals and McNemar bounds
of Supplementary Section S8, follows from those counts.

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
| Worker 3 (ACE) | Projects to ACE, ages 0 to 18; verifies items | (iv) | `src/instruments/fire_worker3_ace.py` |
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
 Worker 3 (ACE) ....... keep ACE clusters in ages 0-18; item check (GPT-5.1)
        |
 Worker 4 ............. symbol -> cluster -> item; ACE = number of distinct items
        |
 Form step ............ notes + form only: change a form answer only on evidence
```

FiRE holds or flags its output in nine places (Section S1): a new state is used
only for the individual whose event created it until a psychotherapist
approves it; a Worker 3 (ACE) removal needs a second model call to confirm it;
an ACE item with no age evidence is held out of the score until a
psychotherapist confirms the age; an event failing one Stages 2-3 check, an
event lowered by the Stage 4 audit, and an event restored by the Stage 5
rescue are kept but flagged (a discard whose text reads as therapist
commentary is never sent to the rescue call); a failed Worker 2 matching call leaves the event
for review instead of guessing; and Worker 3's structural checks and Worker 4's unconfirmed duration on items 10 and 13 are flagged without changing the score. If Worker 2's splitting call fails, the event is logged as an error and not mapped. Worker 3 joins neighbouring states by union when their ages overlap, and also when the note says they happened at the same time (for example "meanwhile"); union never changes an ACE score. The score card gives the score and the items behind it, with no risk level or clinical interpretation.

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

Model-call settings follow Supplementary Material, Section S7: GPT-5.1 is sent no
temperature, so the API defaults apply, with reasoning effort "none" for FiRE
(set explicitly) and for the GPT-5.1 baselines (the API default), so the
same-model comparison uses identical settings; Gemini 3.1 Pro and Grok-4.20
receive temperature 0. Output limits:
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
frequency is ignored), with no model call:

1. FiRE found the item, with verified evidence: **yes**, with that evidence.
2. FiRE has evidence the item does not count (the event starts after 18, or the
   item check confirmed the notes do not support it): **no**, with that evidence.
3. The notes are silent: the form's answer is kept, with no evidence, and
   marked for review as taken from the form.

In 1 and 2 the answer changes when it differs from the form. Evidence FiRE found
but could not confirm (an event with no age, an unconfirmed drop, an unconfirmed
two-year duration) is kept and the item is marked for review; an event with no
age does not make the answer yes. Every change therefore cites a quote from the
notes. The output lists, per item, the final answer, its source, the evidence,
whether it changed and why, and whether a psychotherapist must review it, plus
the reconciled score (Supplementary Section S1).

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
condition, recording the error mode of any wrong or ungrounded decision
(Supplementary Material, Section S5). Given such judgements, the metrics and
tables are recomputed offline:

```
python3 scripts/evaluate_judgements.py judgements.csv [--json metrics.json]
python3 scripts/make_tables.py judgements.csv --table main      # Table II (LaTeX rows)
python3 scripts/make_tables.py judgements.csv --table errors    # Table III
python3 scripts/make_tables.py judgements.csv --table stats     # Table SV
python3 scripts/make_tables.py ablation.csv   --table ablation  # ablation table
python3 scripts/make_tables.py judgements.csv --format markdown
```

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
total of Table III is the number of decisions that do not match the
reference); grounding is the share with no E1, E3, E4 or E6 error; intervals
are Wilson 95% intervals over the 560 decisions, plus the clustered extreme of
Section S8 (one effective decision per individual); the McNemar statistic
against FiRE is bounded below from the error counts, as in Table SV, and
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

`tests/test_worked_example_chain.py` runs the worked example through Worker 3
(all) -> Worker 3 (ACE) -> Worker 4 and checks the paper's expression and
score, runs the notes-plus-form step on it with a test form, runs every
`--selftest`, checks that `--no-algebra` equals Worker 4's score, and checks
that the prompt files are current. `tests/test_scripts.py` runs
`scripts/run_fire.py` on the worked example (one individual and batch mode),
checks that it refuses to start model steps without a key, and checks that
the evaluation scripts reproduce Table II from a judgements file with the
published counts. Every API key is removed from the environment of each test
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
depressed mood at 12 to 16, and adult events (a friend's betrayal, a husband's
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
│   └── ablation/fire_ablation.py                 ablation runs (Section S9)
├── scripts/
│   ├── run_fire.py               the pipeline end to end, one individual or a folder
│   ├── evaluate_judgements.py    accuracy, grounding, intervals, McNemar from judgements
│   └── make_tables.py            Tables II, III, SV and the ablation table
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
