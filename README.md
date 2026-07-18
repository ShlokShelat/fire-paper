# FiRE: Finite-automata inspired Reasoning Engine

Supporting repository for "FiRE: Auditable Clinical Scoring from Consultation
Notes with a Neurosymbolic Concept Bottleneck" (AAAI 2026, AI for Social
Impact track).

## What this repository is

`src/` now contains the actual pipeline implementation. `prompts/` contains
the system prompts used at every LLM-driven stage, extracted and checked
word-for-word against the original source, including two corrections
applied after an initial draft was found to have silently truncated part of
the Judge prompt's matching rules (see `technical_appendix.pdf`, and the
appendix's own account of catching and fixing this). `data/worked_example/`
contains the main paper's Section 4 worked example: input excerpt, expected
mapping, expected ACE expression, and the source figures.

**There is no `eval/` folder, and none is planned.** The results reported in
the main paper's Table 1 and Table 2 (accuracy, groundedness, and the
six-mode error-taxonomy breakdown) were produced by human counting over the
pipeline's per-patient outputs, not by an aggregation script; see
`technical_appendix.pdf` for the annotator and consensus protocol this
connects to. There is accordingly no "reproduce the tables" code to include
here, and a stub claiming otherwise would misrepresent how the paper's
numbers were actually produced.

**What is real and verified here:**
- The ACE-16 item list and the 84-entry cluster-to-item mapping table,
  reproduced in `technical_appendix.pdf` Section 2, were verified
  programmatically to be byte-for-byte identical between the two pipeline
  files that each depend on them.
- The ten-stage pipeline architecture, the human-in-the-loop touchpoints,
  the state-space provisional/permanent gate mechanism and its real counts
  (14 provisional states created across 40 evaluation patients, all
  reviewed and resolved before scoring), and the annotator/consensus
  protocol (17 clinically qualified reviewers per patient) are all
  documented in `technical_appendix.pdf` and were checked against source
  material, not summarized from memory.

**Status of `src/`:** populated with the actual pipeline implementation.

## Repository structure

```
fire-paper/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── preprocessing/       Stage: rule-based note normalization (no LLM)
│   ├── extraction/          Stage 1 (sentence tagging) and Stage 2/3
│   │                        (extraction + four-axis verification)
│   ├── verification/        Stage 4 (second-pass audit) and Stage 5
│   │                        (rescue + deduplication)
│   ├── state_space/         Worker 2: Oracle/Judge matching, provisional/
│   │                        permanent clinician gate
│   ├── algebra/              Worker 3 (general): the trajectory algebra,
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
publicly available, in accordance with the ethical requirements of working
with real patient consultation notes; see `technical_appendix.pdf`,
Section 8 (Annotator and Consensus Protocol) for how the reference standard
was constructed, and the (forthcoming) Ethics and Data Handling section for
the full data-handling and anonymization procedure once available.

## Reproducibility

See `technical_appendix.pdf` for the full reproducibility documentation:
the complete ten-stage pipeline architecture (Section 1), the verbatim
prompts used at every LLM-driven stage (Sections 4 and 5), the state-space
construction and its real counts (Section 3), wall-clock latency and
compute environment (Section 9), and the AAAI-26 Reproducibility Checklist
itself (final section of the appendix).

## License

See `LICENSE`. Code and prompts in this repository are intended for
research use; the license file specifies the exact terms.
