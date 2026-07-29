# FiRE: Finite-automata inspired Reasoning Engine

Supporting repository for "FiRE: Auditable Clinical Scoring from Consultation
Notes with a Neurosymbolic Concept Bottleneck" (AAAI 2026, AI for Social
Impact track).

Anonymized mirror for double-blind review:
https://anonymous.4open.science/r/fire-paper-76E4/README.md

## Overview

FiRE turns a clinical consultation note into an auditable, symbolic
Adverse Childhood Experiences (ACE) score. It extracts discrete
clinical events from the note (**Event Extraction**), checks each one
against four independent evidence criteria (**Verification**), maps
every verified event onto a shared, growable taxonomy of clinical
states (**State Space Mapping**), assembles the patient's full symbolic
timeline as a "FiRE Expression" — an algebra of union, concatenation,
self-loop, and feedback over that state space (**Expression
Construction**) — and finally restricts and scores that expression
against a specific instrument, here the 14-item India-adapted ACE
questionnaire (**Projection** and **Scoring**). Every step from
extraction onward requires its output to be traceable back to a
verbatim span of the source note. See "Relation to the paper" below for
exactly which file implements which of these terms.

## What this repository contains

- **`src/`** — the actual pipeline implementation: ten stages, from
  rule-based note preprocessing through ACE scoring, plus the three
  frontier-model baselines.
- **`prompts/`** — the system prompts used at every LLM-driven stage,
  extracted and checked word-for-word against the original source,
  including two corrections applied after an initial draft was found to
  have silently truncated part of the Judge prompt's matching rules (see
  `technical_appendix.pdf` for that correction's own account).
- **`data/worked_example/`** — the main paper's Section 4 worked example:
  input excerpt, expected event-to-state mapping, expected ACE expression,
  and the two source figures, usable as a regression check against any
  implementation of the trajectory algebra.

**There is no `eval/` folder, and none is planned.** The results reported
in the main paper's Table 1 and Table 2 (accuracy, groundedness, and the
six-mode error-taxonomy breakdown) were produced by a human team counting
directly over the pipeline's per-patient outputs, not by an aggregation
script; see `technical_appendix.pdf`, Section 8 (Annotator and Consensus
Protocol), for that process. There is accordingly no "reproduce the
tables" code to include, and a stub claiming otherwise would misrepresent
how the paper's numbers were actually produced.

**What has been independently verified, not just asserted:**
- The ACE-14 item list and the 69-entry cluster-to-item mapping table
  (`technical_appendix.pdf`, Section 2) were checked programmatically to
  be byte-for-byte identical between the two pipeline files that each
  depend on them.
- The ten-stage pipeline architecture, the human-in-the-loop touchpoints,
  the state-space provisional/permanent gate mechanism and its real
  counts (14 provisional states created across the 40 evaluation
  patients, all reviewed and resolved before scoring), and the
  annotator/consensus protocol (17 clinically qualified reviewers per
  patient) are documented in `technical_appendix.pdf` and were checked
  against source material, not summarized from memory.

## Repository structure

```
fire-paper/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── preprocessing/       Rule-based note normalization (no LLM)
│   ├── extraction/          Stage 1 (sentence tagging) and Stage 2/3
│   │                        (extraction + four-axis verification)
│   ├── verification/        Stage 4 (second-pass audit) and Stage 5
│   │                        (rescue + deduplication)
│   ├── state_space/         Worker 2: Oracle/Judge matching, provisional/
│   │                        permanent clinician gate
│   ├── algebra/             Worker 3 (general): the trajectory algebra,
│   │                        implemented once, instrument-independent
│   ├── instruments/         Worker 3 (ACE) projection, Worker 4 (ACE)
│   │                        scoring
│   └── baselines/           Direct-scoring and reconciliation baseline
│                            harnesses for the three frontier models
├── data/
│   └── worked_example/      The paper's Section 4 worked example: input
│                            excerpt, expected mapping, expected ACE
│                            expression, and the two source figures
└── prompts/
    ├── extraction_system_prompt.txt
    ├── indian_cultural_glossary.txt
    ├── oracle_and_judge_prompts.txt
    ├── rescue_prompt.txt
    ├── ace_verification_prompts.txt
    └── baseline_prompts.txt
```

## Relation to the paper

The paper describes four pipeline stages: (i) preprocessing into
events, (ii) mapping to the shared state space, (iii) a four-axis
verification step, and (iv) trajectory assembly plus instrument
projection/scoring. This repository implements that pipeline as:

| Paper stage | Paper terminology | Implementation |
|---|---|---|
| (i) | Preprocessing / Event Extraction | `src/preprocessing/fire_preprocessor.py`, `src/extraction/fire_stage1.py` (both rule-based) + Stage 2 of `src/extraction/fire_stage2_3_single_model.py` (LLM-driven) |
| (iii) | Verification | Stage 3 of `src/extraction/fire_stage2_3_single_model.py` — the four-axis check (construct, actor/role, age window, groundedness) |
| (ii) | State Space Mapping | `src/state_space/fire_worker2.py` (Oracle → Judge → Genesis) |
| (iv), part 1 | Expression Construction | `src/algebra/fire_worker3_general.py` |
| (iv), part 2 | Projection | `src/instruments/fire_worker3_ace.py` |
| (iv), part 3 | Scoring | `src/instruments/fire_worker4_ace.py` |

`src/verification/fire_stage4.py` and `src/verification/fire_stage5.py`
are additional passes this repository runs between Verification and
State Space Mapping (a second evidence audit, and a rescue +
deduplication step). Neither is one of the four stages the paper
describes; each file's own module docstring explains in detail what it
does and why it is not presented as part of stage (iii). The two
`src/baselines/` scripts implement no pipeline stage at all — they are
the single-LLM-call comparison points ("direct scoring" and
"reconciliation") the paper's ablation measures FiRE against.

## Setup

**Requirements:** Python 3.10 or later (the state-space matcher in
`src/state_space/fire_worker2.py` uses the `X | None` union type-hint
syntax introduced in 3.10) and an API key for the LLM-driven stages
(see below).

```
pip install -r requirements.txt
```

`requirements.txt` installs four packages — `aiohttp`, `spacy`,
`sentence-transformers`, and `numpy` — each with a comment explaining
exactly where it is used, verified directly against every `import` in
`src/` (via Python's `ast` module, not just a text search). All
model API calls in this pipeline are hand-built HTTP requests (via
`aiohttp` or the standard-library `urllib.request`), not a vendor SDK,
so no `openai`/`requests`/`google-generativeai` client package is
required; `requirements.txt` explains this and keeps those names
commented out for anyone who wants to swap in an official SDK instead.

API keys are read from environment variables, not hardcoded; this was
confirmed directly against the source in the security pass described
below. The exact environment variable name each script expects was not
catalogued here in one place; check the constant near the top of each
`src/` file (`API_KEY_ENV`, `RESCUE_API_KEY_ENV`, and a per-model
`key_env` list in the baseline scripts) for the name it actually reads.

## Running the pipeline

There is no training step anywhere in this repository — FiRE prompts
frozen, off-the-shelf LLMs (see "Setup" for the model) and never
fine-tunes or updates any model weights. Every stage below is either
rule-based/deterministic code or a prompting call against a hosted model.

Every script under `src/` is runnable directly and prints a `Usage:`
line (or supports `--help` via `argparse`) if invoked without
arguments. The stages chain together by filename — each stage's
default output is the next stage's expected input. The commands below
use bash/POSIX syntax (macOS, Linux, WSL, Git Bash); on native Windows
`cmd.exe` use `set OPENAI_API_KEY=...` instead of `export`, or
`$env:OPENAI_API_KEY = "..."` in PowerShell — everything after that line
is the same `python3 ...` command on any platform:

```
# 1. Preprocessing (rule-based, no LLM, no API key needed)
python3 src/preprocessing/fire_preprocessor.py notes.txt
#   -> notes_preprocessed.json, notes_flat.txt

# 2. Stage 1: sentence tagging (rule-based, no LLM)
python3 src/extraction/fire_stage1.py notes_flat.txt
#   -> stage1_output.json

# 3. Stage 2+3: LLM event extraction + four-axis verification
export OPENAI_API_KEY=...
python3 src/extraction/fire_stage2_3_single_model.py stage1_output.json \
    --patient-id patient01 --output validated_events.json

# 4. Stage 4 (optional second-pass audit — see "Relation to the paper")
python3 src/verification/fire_stage4.py validated_events.json \
    --stage1 stage1_output.json --output validated_events_s4.json

# 5. Stage 5 (optional rescue + deduplication — likewise supplementary)
python3 src/verification/fire_stage5.py validated_events_s4.json \
    --stage1 stage1_output.json --output validated_events_s5.json

# 6. Worker 2: state-space mapping (Oracle / Judge / Genesis)
# --fresh-start is used here deliberately — see the note below the block.
python3 src/state_space/fire_worker2.py validated_events_s5.json \
    --stage1 stage1_output.json \
    --state-space my_state_space.json --fresh-start \
    --output final_mapped_events.json

# 7. Worker 3 (general): FiRE Expression construction
python3 src/algebra/fire_worker3_general.py final_mapped_events.json \
    --output worker3_output.json

# 8. Worker 3 (ACE): projection to the 14-item ACE subgraph
python3 src/instruments/fire_worker3_ace.py worker3_output.json \
    -o worker3_ace_output.json

# 9. Worker 4 (ACE): final 0-14 ACE score card
python3 src/instruments/fire_worker4_ace.py worker3_ace_output.json \
    -o ace_scorecard.json
```

**On the state-space file (Step 6):** `src/state_space/fire_state_space_tree.json`
is a schema placeholder, not the actual shared taxonomy built from the
paper's 40-patient evaluation set (that dataset is private — see "Data
availability" below), and its one example entry uses the literal
strings `"string"` as placeholder names/descriptions. Passing
`--fresh-start` (as the command above does) discards that placeholder
and starts Worker 2 from an empty `{"domains": {}}` database instead of
embedding the literal placeholder entry into the semantic search index.
Every state your own run of the pipeline discovers will be created as a
new `provisional` state from scratch — it will not reuse the paper's
actual cluster/state codes or their cluster-to-ACE-item mapping,
which is a property of the private taxonomy, not of this schema file.
The 14-item ACE mapping logic itself (`CLUSTER_ACE_ITEM` in
`fire_worker3_ace.py` / `fire_worker4_ace.py`) is fully included and
reproducible independent of this — only the discovered clinical
*states* (as opposed to the fixed ACE *items* they can project to) come
from the private dataset.

Steps 3, 5, and 6 need the environment variable named in that script's
`API_KEY_ENV` / `RESCUE_API_KEY_ENV` constant (`OPENAI_API_KEY` for the
default `gpt-5.1` model everywhere in this pipeline). Step 4 calls no
LLM at all — it is pure cosine-similarity/keyword code and needs no API
key. Steps 1, 2, 7, 8, and 9 likewise need no API key.

Steps 3–6 additionally use the `sentence-transformers` package, which
downloads its ~90MB embedding model from the Hugging Face Hub on first
use (see `requirements.txt`). For step 6 (Worker 2) this package is a
hard requirement — it is imported unconditionally and there is no
fallback. Steps 3, 4, and 5 use it only as an optional quality
enhancement and automatically fall back to a lexical (Jaccard)
similarity check if it is unavailable, so they still run without it,
just with a slightly less precise similarity signal.

Nine of the eleven scripts under `src/` (everything except the two
rule-based Stage 1 and preprocessing scripts) also accept a `--selftest`
flag that runs offline regression checks against synthetic data and
exits — useful for confirming the pipeline logic works in your
environment before spending any API budget, e.g.:

```
python3 src/extraction/fire_stage2_3_single_model.py --selftest
python3 src/state_space/fire_worker2.py --selftest
python3 src/algebra/fire_worker3_general.py --selftest
```

The `data/worked_example/` files are a smaller, independent regression
check against the later Worker 2–4 stages specifically (see
`data/worked_example/README.md`); they are a small excerpt already in
event form, not raw clinical notes, and are not meant to be run through
the Stage 1 / preprocessing front end.

**Baselines** run standalone, directly against a raw notes file, with no
other pipeline stage involved:

```
python3 src/baselines/fire_llm_baseline.py notes.txt --model gpt-5.1
python3 src/baselines/fire_llm_baseline_self_form_included.py notes.txt \
    --form patient_form.json --model gpt-5.1 -o result.json
```

### Understanding the outputs

- `validated_events.json` / `..._s4.json` / `..._s5.json` — one record
  per candidate clinical event, each carrying its four-axis verification
  result (`construct_check_passed`, `actor_role_check_passed`,
  `age_window_check_passed`, `groundedness_check_passed`) and an overall
  `status` (`CONFIRMED` / `TENTATIVE` / `REVIEW_REQUIRED` / `DISCARD`).
- `final_mapped_events.json` — the same events with each sub-event
  resolved to a `state_code` in the shared taxonomy; newly created codes
  are `provisional` and require clinician review before counting as
  `permanent` (see `worker2_stats.provisional_states_pending_review`).
- `worker3_output.json` / `worker3_ace_output.json` — the patient's
  `fire_expression` string (the symbolic algebra described in the
  paper) plus the DFA it was built from and a `human_review_flags` list
  of anything the automatic checks could not confirm.
- `ace_scorecard.json` — the final per-item (1–14) presence table, the
  canonical 0–14 `canonical_ace_score`, its risk band, and any clinical
  interaction/feedback flags.

Two stages also write a second file alongside their main output:
`validated_events_audit.json` (Step 3) records per-unit token usage,
latency, and status for every LLM call, and both Step 3 and Step 6 write
a `.resume`/`.resume.json` checkpoint (only used if you pass `--resume`
on a re-run; Worker 2 deletes its own checkpoint on a normal completed
run unless you pass `--keep-resume`, Step 3's is not auto-deleted).

All output filenames above are the scripts' own defaults, written
relative to your current working directory (not next to the input
file) unless you pass an absolute path via `--output`/`-o`. No script
writes a persistent log file — every script logs to the console only
(via Python's `logging` module); redirect it yourself (e.g.
`... 2>&1 | tee run.log`) if you want a saved log.

## Security note

Before this repository was made public, the source under `src/` was
audited for three categories of risk, each checked with targeted searches
rather than a single skim:

1. **Identifying information** (real names, institutional email domains,
   the clinical partner site's name) — none found in `src/`.
2. **Hardcoded credentials** (API keys, secrets, tokens, passwords as
   string literals) — none found; every `api_key`/`key` variable traces
   to `os.environ.get(...)` or an equivalent environment-variable read.
3. **Local file paths that would leak a system username** — none found.

If you add new files to `src/`, re-run an equivalent check before pushing;
none of the above is enforced automatically by anything in this
repository.

## A note on the clinical partner site's name

The extraction system prompt (`prompts/extraction_system_prompt.txt`)
originally named the specific clinical partner site providing the
consultation notes. That name has been replaced with
`[the partner clinical site]` throughout this repository, as a
precaution for double-blind review; this is a presentational substitution
only and changes no instruction given to any model. Whether this
redaction is necessary, or whether the main paper already discloses the
site by name, is a decision pending confirmation; see
`technical_appendix.pdf` for the current state of that decision.

## Data availability

The 40-patient clinical evaluation dataset is not, and will not be, made
publicly available, in accordance with the ethical requirements of
working with real patient consultation notes; see `technical_appendix.pdf`,
Section 8 (Annotator and Consensus Protocol), for how the reference
standard was constructed. The full Ethics and Data Handling section
covering consent and anonymization procedure in detail is still pending
from the author team as of this writing.

## Reproducibility

Running the stages under "Running the pipeline" above reproduces the
paper's described mechanism end-to-end on notes you supply; it does not
by itself regenerate the main paper's Table 1/Table 2 numbers, since
those were produced by human annotation over the pipeline's outputs,
not a scripted aggregation (see "What this repository contains" above).

See `technical_appendix.pdf` for the full reproducibility documentation:
the complete ten-stage pipeline architecture (Section 1), the verbatim
prompts used at every LLM-driven stage (Sections 4 and 5), the
state-space construction and its real counts (Section 3), wall-clock
latency and compute environment (Section 9), and the AAAI-26
Reproducibility Checklist itself (Section 11), which states plainly, item
by item, what is and is not yet confirmed about this repository's
reproducibility.

### Determinism and randomness

No script in this repository imports Python's `random` module or seeds
any RNG — there is no numeric seed to set, because there is no sampling,
shuffling, or other stochastic step in the code itself. The only source
of run-to-run variation is the LLM API calls, which every script
configures for minimum variation: `temperature: 0` for classic chat
models, and `reasoning_effort: "none"` for the GPT-5-family reasoning
models that reject `temperature` outright (see e.g.
`_build_llm_payload`/`_build_openai_payload` in each script). This
minimizes but does not mathematically guarantee bit-identical output
across runs or across time — provider-side inference is not guaranteed
deterministic even at `temperature: 0`, and hosted models can be updated
by the provider independent of this repository. Every script's
`--selftest` flag, by contrast, is fully deterministic and offline: it
runs against fixed synthetic inputs with no network or LLM call
involved, so it is the one check in this repository with no
reproducibility caveat at all.

### Runtime and hardware

No GPU is required. `sentence-transformers` (used in Steps 3–6; see
"Running the pipeline") pulls in `torch`, but no code in this repository
requests a CUDA device, and the paper's reported evaluation ran
CPU-only. Concurrency is configurable and affects wall-clock time and
API rate-limit exposure: Step 3 defaults to 5 concurrent units
(`--concurrency`), Worker 2 (Step 6) defaults to 3. Beyond that, this
repository does not itself state expected wall-clock runtime, peak
memory, or the exact compute environment used for the paper's reported
numbers — see `technical_appendix.pdf`, Section 9, for those figures;
they are not duplicated here to avoid two sources of truth drifting
apart.

## License

MIT (see `LICENSE`), a common choice for academic code releases and
sufficient for the AAAI reproducibility checklist's requirement of "a
license that allows free usage for research purposes." Two things to
confirm before camera-ready release, not blocking for double-blind review:
this is the intended license (not yet confirmed by the authors), and the
placeholder copyright line ("The FiRE paper authors") should be replaced
with real author names once the repository is de-anonymized. The clinical
evaluation dataset itself is explicitly not covered by this license and is
not included in this repository regardless; see "Data availability" above.
