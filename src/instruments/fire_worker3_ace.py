"""
Component: Worker 3, ACE (Instrument Projection)
Status: PLACEHOLDER -- not the original source. See README.md.

Function: projects the full trajectory (built once by fire_worker3_general)
to the ACE-relevant subgraph (cluster membership and the 0-18 age window),
re-derives union structure among survivors only, and runs ACE-specific
grounding verification against each item's defining clinical elements.

Uses the deterministic 84-entry cluster-to-item table documented at
technical_appendix.pdf, Section 2.2 ("Cluster-to-item resolution"),
verified programmatically byte-for-byte identical to the copy in
fire_worker4_ace.py.

The actual ACE-verification and adjudicator system prompts ARE reproduced
in full and verified: see prompts/ace_verification_prompts.txt.
"""
raise NotImplementedError(
    "This is a documentation stub, not the original pipeline code. "
    "The verified verification prompts are in "
    "prompts/ace_verification_prompts.txt."
)
