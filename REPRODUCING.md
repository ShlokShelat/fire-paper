# Reproducing the results

This guide lists, for each reported result, what it needs and the commands
that produce it. Section numbers S1 to S9 refer to the Supplementary
Material.

## What can be reproduced from this repository alone

| Result | Reproducible here? |
|---|---|
| The worked example (main paper, Sections II and V; Figs. 4 and 5; Table I) | Yes, offline: `make worked-example` |
| Every derived number in Tables II, III, SV and the ablation table, from the published error counts | Yes, offline: `python3 scripts/make_tables.py --selftest` |
| Running FiRE, the baselines and the ablation on new notes | Yes, with API keys (GPT-5.1; Gemini 3.1 Pro and Grok-4.20 for their baselines) |
| The paper's numbers themselves | No: they need the 40 individuals' notes, forms, the development state space and the expert judgements, none of which can be released (S6) |

The rest of this guide describes the full procedure, for a site with its own
notes and its own reference team.

## Environment

- Python 3.10 or later; the paper's runs used Python 3.13.12, torch 2.11.0 and
  numpy 2.4.4 on an Apple M1 laptop without a GPU (S7). `pip install -r
  requirements.txt`.
- Models: `gpt-5.1` (FiRE and the GPT-5.1 baselines), `gemini-3.1-pro`
  (API model `gemini-3.1-pro-preview`) and `grok-4.20-0309-reasoning`.
  Hosted models change over time; a later model version may shift both FiRE's
  and the baselines' results (S8).
- Every result in the paper comes from a single run. Each individual was
  processed once by FiRE and once by each baseline in each condition, one
  individual at a time, and nothing was averaged over repeats (S7).
- Check the installation offline first: `make check`.

## Parameters (Supplementary Table "Model-call parameters and fixed thresholds")

None of these was tuned by a sweep; all were fixed on development data before
evaluation (S7). `tests/test_scripts.py` checks that the code still holds the
values below.

| Where | Value | In the code |
|---|---|---|
| FiRE model calls | GPT-5.1, no temperature, reasoning effort "none" | `build_payload` / `_build_llm_payload` / `_build_rescue_payload` in each stage |
| Baseline model calls | GPT-5.1 without temperature, default reasoning effort; Gemini and Grok temperature 0 | `src/baselines/fire_llm_baseline*.py` |
| Output limits | extraction 2000; Worker 3 item check 2000; removal confirmation 400; Worker 2 splitting 4000, Judge 300; rescue 16 | `fire_stage2_3_single_model.build_payload`, `fire_worker3_ace._llm_verify_record` / `_llm_adjudicate_drop`, `fire_worker2.call_llm` and its Judge call, `fire_stage5._build_rescue_payload` |
| Baseline output limits | plain, notes only 8000; plain, notes + form 16000; structure-guided and knowledge-matched 32000 | `MAX_OUTPUT_TOKENS` in the baseline scripts |
| Stages 2-3 construct check | cosine >= 0.82 (Jaccard >= 0.30 without embeddings); in 0.50-0.82 (Jaccard 0.10-0.30) pass if entailment >= 0.55 | `COSINE_THRESHOLD`, `JACCARD_THRESHOLD`, `NLI_*` in `fire_stage2_3_single_model.py` |
| Stage 4 grounding recheck | cosine >= 0.60 (0.55 for patterns); confidence -0.20 / -0.10 / -0.05 per failed check of high / medium / low severity | `COSINE_THRESHOLD`, `COSINE_THRESHOLD_PATTERN`, `stage4_verify` in `fire_stage4.py` |
| Stage 5 deduplication | cosine >= 0.88 (Jaccard >= 0.60 without embeddings) | `RESCUE_SIM_THRESHOLD`, `RESCUE_JAC_THRESHOLD` in `fire_stage5.py` |
| Worker 2 | candidates >= 0.30; new cluster name matches an old one at 0.80; same-cluster candidate >= 0.82 mapped without the Judge; 70% of a quote's words in the source | `CANDIDATE_MIN_SCORE`, `CLUSTER_MATCH_SCORE`, `SAME_CLUSTER_AUTO_MAP`, `_quote_grounded` in `fire_worker2.py` |
| Worker 3 | confidence 0.80 for a newly created state | `parse_worker2_output` in `fire_worker3_general.py` |
| ACE window | ages 0 to 18 inclusive | `MIN_AGE`, `MAX_AGE` in `ace14.py` |

## Main results (Tables II and III)

### 1. State space

FiRE maps events onto a state space built from development data: notes from
the same process and site, from individuals who are not in the evaluation set
(S3). Build it by running the pipeline (step 2) on the development notes with
a fresh state-space file, reviewing every provisional state, and then fixing
the file before any evaluation individual is processed. Starting the
evaluation from an empty space also works, but then every state is new and
needs review.

### 2. FiRE, notes only

Run every evaluation individual through the pipeline, sharing one state-space
file, one individual at a time:

```
export OPENAI_API_KEY=...
python3 scripts/run_fire.py --batch notes/ --state-space runs/state_space.json \
    --out-dir runs --to worker2
```

Then review the provisional states created during mapping, before any
scoring, as the paper did (S3: the 14 states created across the 40
individuals were approved as one batch after mapping and before scoring, and
no accuracy figure influenced the approvals):

```
python3 src/state_space/fire_worker2.py --state-space runs/state_space.json --export-review review.json
# mark each entry "approve" or "reject", then:
python3 src/state_space/fire_worker2.py --state-space runs/state_space.json --apply-review review.json
```

If a state is rejected, rerun Worker 2 for the individuals whose pieces mapped
to it (`--patch-events` lists them). Then score:

```
python3 scripts/run_fire.py --batch notes/ --state-space runs/state_space.json \
    --out-dir runs --from worker3
```

Each `runs/patient_NN/ace_scorecard.json` holds the 14 item decisions, with
the evidence for each.

### 3. FiRE, notes plus form

FiRE does not rerun for this condition: the form step reads FiRE's finished
output and the form (S1). For each individual, with the form as
`notes/patient_NN_ace_form.json`:

```
python3 scripts/run_fire.py --batch notes/ --with-forms --out-dir runs \
    --state-space runs/state_space.json --from reconcile
```

`runs/patient_NN/ace_reconciled.json` holds the 14 reconciled decisions.

### 4. Baselines

Nine baseline configurations (three models by three prompts) in each
condition, one call per individual (S4):

```
for m in gpt-5.1 gemini-3.1-pro grok-4.20-0309-reasoning; do
  for f in notes/patient_*[0-9].txt; do
    b=$(basename "$f" .txt)
    python3 src/baselines/fire_llm_baseline.py "$f" --model $m -o "results_plain/$b.$m.ace.json"
    python3 src/baselines/fire_llm_baseline_self_form_included.py "$f" \
        --form "notes/${b}_ace_form.json" --model $m -o "results_plain/$b.$m.ace_reconciled.json"
  done
  python3 src/baselines/fire_llm_structured_baseline.py --batch notes/ --model $m --out-dir results_structured
  python3 src/baselines/fire_llm_structured_baseline.py --batch notes/ --with-forms --model $m --out-dir results_structured
  python3 src/baselines/fire_llm_knowledge_baseline.py --batch notes/ --model $m \
      --state-space state_space_for_prompt.json --out-dir results_knowledge
  python3 src/baselines/fire_llm_knowledge_baseline.py --batch notes/ --with-forms --model $m \
      --state-space state_space_for_prompt.json --out-dir results_knowledge
done
```

(`mkdir -p results_plain` first; `state_space_for_prompt.json` is the fixed
state space trimmed by `src/baselines/trim_state_space.py`.)

### 5. Expert judgement

The paper's metrics come from a double-blind expert evaluation (S5): the
reference is finished before any output is seen; outputs are stripped of any
label; two note-takers review every output item by item and agree on one
judgement per item, including the error mode (E1 to E6) of any wrong or
ungrounded item; disagreements go to two senior psychotherapists, and an
independent auditor checks the finished evaluation. In the notes-plus-form
condition the reference starts from the individual's own answers and changes
an item only on explicit evidence in the notes.

Record the judgements as a CSV with one row per item decision (format in the
README and in `scripts/evaluate_judgements.py`).

### 6. Metrics and tables

```
python3 scripts/evaluate_judgements.py judgements.csv --json metrics.json
python3 scripts/make_tables.py judgements.csv --table main     # Table II
python3 scripts/make_tables.py judgements.csv --table errors   # Table III
python3 scripts/make_tables.py judgements.csv --table stats    # Table SV
```

## Ablation (Section S9)

The ablation reuses each individual's main-run Stages 2-3 output, so the
extraction call is never repeated (S9). For every individual and every
switch:

```
cd runs/patient_NN
for run in no-construct-check no-actor-role-check no-age-check no-cited-span-check \
           no-checks no-state-space model-scoring; do
  python3 ../../src/ablation/fire_ablation.py --$run \
      --stage1 stage1_output.json --cached-extraction validated_events.json \
      --state-space ../state_space.json --out-dir ablation/$run
done
python3 ../../src/ablation/fire_ablation.py --no-algebra \
    --w3-ace worker3_ace_output.json --scorecard ace_scorecard.json
```

As in the main run, the notes-plus-form rows come from the form step applied
to each ablation run's output (`fire_reconcile_ace.py ablation/<run>/worker3_ace_output.json
ablation/<run>/ace_scorecard.json --form ...`). Removing the algebra needs no
run: `--no-algebra` checks that distinct-item scoring gives the same
decisions. The outputs are judged by the same double-blind process, with the
run name in the `system` column, and rendered with
`python3 scripts/make_tables.py ablation_judgements.csv --table ablation`.

## Offline checks of the published numbers

```
python3 scripts/make_tables.py --selftest
```

checks, from `data/paper_results/published_results.json`, that each total in
Table III is the sum of its modes; that every accuracy and grounding value in
Table II and the ablation table follows from the counts; and that every
Wilson interval, McNemar lower bound and the clustered extreme in Section S8
follow from them as well.
