"""
Script: reproduce_tables.py
Status: PLACEHOLDER -- not the original source. See README.md.

Intended function: regenerate Table 1 and Table 2 of the main paper
(accuracy, groundedness, and the six-mode error-taxonomy breakdown, for
FiRE and all three baselines, in both the notes-only and notes+form
conditions) from logged per-patient model outputs.

The verified numbers this script would need to reproduce are documented at:
technical_appendix.pdf, Section 6 ("Extended Error Taxonomy Examples") for
worked error examples, and the main paper's Section 6 (Experiments) for the
full Table 1 / Table 2 figures themselves (FiRE 640/640 in both conditions;
frontier baselines 70.0-73.4% notes-only, 60.3-62.5% notes+form; see the
main paper for exact per-model breakdowns).

This script requires the actual per-patient logged outputs to run, which
are not included in this scaffold (see data/worked_example/README.md and
the main README's note on why the full 40-patient dataset is not publicly
released).
"""
raise NotImplementedError(
    "This is a documentation stub. Populate with the actual per-patient "
    "evaluation logs and scoring logic to regenerate the paper's tables."
)
