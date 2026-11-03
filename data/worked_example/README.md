# Worked example

The note excerpt the main paper uses as its worked example, with the
expected output, as a small offline regression check. It is the only
note text in this repository; the evaluation notes cannot be released.

## Files

- `input.txt`: the six-line excerpt with ages, verbatim from the main paper.
- `expected_mapping.json`: the event, state, cluster and item of each event,
  from the main paper's Table I (event labels slightly fuller).
- `worker2_output.json`: the same excerpt in the shape `fire_worker2.py`
  writes (one event per note line; every piece carries that line's source
  sentence, its own supporting quote, ages, state code and cluster).
  Written by hand from the paper, not produced by a model, so the later
  stages can be checked without an API key.
- `expected_output.txt`: the expected full trajectory and the ACE expression
  `(a + b + c^2) . (e + a^3)`, ACE = 4, items {1, 2, 4, 11}, with the reason
  for each exponent.
- `figures.tex`: the paper's Figs. 4 and 5 (full trajectory, ACE subgraph),
  taken from its LaTeX source.

## Running it

```
python3 src/algebra/fire_worker3_general.py data/worked_example/worker2_output.json -o worker3_output.json
python3 src/instruments/fire_worker3_ace.py worker3_output.json -o worker3_ace_output.json --verify
python3 src/instruments/fire_worker4_ace.py worker3_ace_output.json -o ace_scorecard.json
```

or `make worked-example`, which runs the same chain through
`scripts/run_fire.py`; with the assertions, `python3
tests/test_worked_example_chain.py` (also runs under pytest). `--verify` without `--verify-llm` runs the item
check's deterministic part only, so no API key is needed. The front end
(preprocessing, extraction, verification and Worker 2) calls a model and is
not exercised here; `worker2_output.json` stands in for its output.

The implementation assigns symbols in order of first onset, so its letters
for the non-ACE clusters differ from the paper's (see `expected_output.txt`);
the ACE expression and score are identical.
