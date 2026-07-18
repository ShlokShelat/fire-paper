"""
Component: Worker 2 (State Mapping)
Status: PLACEHOLDER -- not the original source. See README.md.

Function: decomposes each verified event into atomic sub-events (the
Oracle), searches the full shared state space for a semantic match, and
either maps to an existing state or creates a new state tagged
"provisional" pending clinician review (the Judge). Provisional states are
scoped to the triggering patient only, invisible to every other patient's
matching search until reviewed via an explicit export/approve-or-reject
workflow.

The actual Oracle and Judge system prompts ARE reproduced in full and
verified, including the complete clinical-term/lay-wording equivalence
list and same-cluster contrast examples: see
prompts/oracle_and_judge_prompts.txt.

Documented in full at: technical_appendix.pdf, Section 3 ("State Space
Construction"), and Section 8 ("Annotator and Consensus Protocol") for how
this connects to reference-standard construction.

Real counts from the 40-patient evaluation (14 provisional states created,
all reviewed and resolved before scoring, none rejected) are in Section
3.4, "Counts from the 40-patient evaluation."
"""
raise NotImplementedError(
    "This is a documentation stub, not the original pipeline code. "
    "The verified Oracle and Judge prompts are in "
    "prompts/oracle_and_judge_prompts.txt."
)
