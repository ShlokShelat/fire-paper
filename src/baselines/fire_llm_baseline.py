"""
Baseline: Direct scoring (notes only)
Status: PLACEHOLDER -- not the original source. See README.md.

Function: gives a single frontier model the raw consultation notes and asks
it to score all 16 ACE items directly, with no pipeline, no state space, no
multi-stage verification. Models used: GPT-5.1, Gemini 3.1 Pro,
grok-4.20-0309-reasoning. Same-model comparison against FiRE's own
extraction model (GPT-5.1) is the paper's central ablation; see
technical_appendix.pdf, Section 7.3 ("Same-model discipline").

Deliberately does NOT include an explicit age-window instruction; see
technical_appendix.pdf, Section 5.3, for why that omission is intentional
and load-bearing for the fairness of the E5 (age-window violation)
comparison.

The actual system prompt and full user-prompt/schema construction ARE
reproduced in full and verified: see prompts/baseline_prompts.txt.
"""
raise NotImplementedError(
    "This is a documentation stub, not the original pipeline code. "
    "The verified prompts are in prompts/baseline_prompts.txt."
)
