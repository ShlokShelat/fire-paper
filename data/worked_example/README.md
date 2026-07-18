# Worked example

The anonymized consultation-note excerpt used as the main paper's Section 4
worked example, reproduced as a small, checkable demo.

## Files

- `input.txt` -- the six-event excerpt (with ages), verbatim from the main
  paper.
- `expected_mapping.json` -- the event-to-state-to-ACE-item table, verbatim
  from the main paper's own table for this example.
- `expected_output.txt` -- the expected ACE-projected symbolic expression
  `(a + b + c^2) . d . (e + a^3)`, with the reasoning for each exponent, plus
  a descriptive (not independently re-verified) account of the full
  trajectory's structure before ACE projection. Read the confidence note
  inside this file before relying on the full-trajectory portion for a
  strict regression test; the ACE-projected expression is the
  well-verified part.
- `figures.tex` -- the two TikZ figures (full trajectory, ACE subgraph)
  reproduced verbatim from the main paper's LaTeX source.

## How to use this

If you populate `src/algebra/fire_worker3_general.py` and
`src/instruments/fire_worker3_ace.py` with the real pipeline
implementation, running `input.txt` through extraction, mapping, and the
trajectory algebra should reproduce `expected_mapping.json`'s event-to-state
mapping and `expected_output.txt`'s ACE-projected expression exactly. This
is the same check the original pipeline's own self-test suite uses (see
technical_appendix.pdf, Section 7, "Regression-tested corrections"): if a
future change to the union, self-loop, or feedback logic breaks this exact
worked example, that's a signal to look closely before trusting the change.
