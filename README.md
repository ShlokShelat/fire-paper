# FiRE: Finite-automata inspired Reasoning Engine

Supporting repository for "FiRE: Auditable Clinical Scoring from Consultation
Notes with a Neurosymbolic Concept Bottleneck" (AAAI 2026, AI for Social
Impact track).

Anonymized mirror for double-blind review:
https://anonymous.4open.science/r/fire-paper-76E4/README.md

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
- The ACE-16 item list and the 84-entry cluster-to-item mapping table
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
- One open item, stated plainly rather than glossed over: the source code
  in `src/` passed through a separate anonymization and security pass
  before publication (see "Security note" below), and this appendix's own
  factual claims were verified against the source as it existed *before*
  that pass, not re-diffed against the published files afterward. The two
  are expected to match, since that pass was redaction-only by design, but
  this hasn't been independently re-confirmed.

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

## Setup

```
pip install -r requirements.txt
```

API keys are read from environment variables, not hardcoded, this was
confirmed directly against the source in the security pass described
below. The exact environment variable name each script expects was not
catalogued here in one place; check the constant near the top of each
`src/` file (`API_KEY_ENV`, `RESCUE_API_KEY_ENV`, and a per-model
`key_env` list in the baseline scripts) for the name it actually reads.
See `requirements.txt` for per-package confidence notes, several
dependencies were inferred from code patterns rather than confirmed
against a working install.

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

See `technical_appendix.pdf` for the full reproducibility documentation:
the complete ten-stage pipeline architecture (Section 1), the verbatim
prompts used at every LLM-driven stage (Sections 4 and 5), the
state-space construction and its real counts (Section 3), wall-clock
latency and compute environment (Section 9), and the AAAI-26
Reproducibility Checklist itself (Section 11), which states plainly, item
by item, what is and is not yet confirmed about this repository's
reproducibility.

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
