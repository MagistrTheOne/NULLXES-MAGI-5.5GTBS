# Responsibility

Owns MAGI tokenizer architecture, experiment matrix, artifact naming, and tokenizer evaluation reports.

This module does not train tokenizer artifacts in the architecture phase. It defines the interface and future outputs for:

- BPE / Unigram / byte-hybrid candidates;
- vocab size 131072;
- RU/EN/code/math/profanity/code-switch holdouts;
- fertility and fragmentation gates;
- tokenizer artifact paths consumed by shard builders.

Authoritative spec: `docs/TOKENIZER_ARCHITECTURE_SPEC_v0.1.md`.
