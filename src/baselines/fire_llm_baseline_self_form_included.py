"""
Baseline: Reconciliation (notes + patient self-report)
Status: PLACEHOLDER -- not the original source. See README.md.

Function: gives a single frontier model the patient's own self-filled ACE
form alongside the consultation notes, and asks it to revise the
self-report using the notes, changing an answer only where the notes give
evidence for a different one. Every changed item requires a stated
revision_reason; the harness never trusts the model's own account of what
it changed, it independently compares the model's final value against the
self-report value the harness itself loaded, and flags any changed item
with an empty revision_reason. This is the direct, executable diagnostic
for the paper's E6 (reconciliation instability) error mode.

The actual system prompt and full user-prompt/schema construction ARE
reproduced in full and verified: see prompts/baseline_prompts.txt.
"""
raise NotImplementedError(
    "This is a documentation stub, not the original pipeline code. "
    "The verified prompts are in prompts/baseline_prompts.txt."
)
