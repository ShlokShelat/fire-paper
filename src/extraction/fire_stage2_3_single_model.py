"""
Stage: Stage 2/3 (Extraction + Four-Axis Verification)
Status: PLACEHOLDER -- not the original source. See README.md.

Function: a single LLM (GPT-5.1) extracts candidate clinical events grouped
by section; a deterministic engine then checks each against four axes
(construct, actor/role, age window, groundedness).

The actual system prompt used at this stage IS reproduced in full and
verified: see prompts/extraction_system_prompt.txt, and
technical_appendix.pdf, Section 4 ("Extraction and Verification Prompts"),
subsection "Extraction prompt (Stage 2/3)".

Documented in full at: technical_appendix.pdf, Section 1, Table 1,
"Stage 2/3" row, and Section 9 ("Same-model discipline") for why GPT-5.1
specifically is used here and nowhere else is a different model substituted.
"""
raise NotImplementedError(
    "This is a documentation stub, not the original pipeline code. "
    "The verified system prompt for this stage is in "
    "prompts/extraction_system_prompt.txt."
)
