# Architectural Decision Records

ADRs record durable BitNet-rs decisions. Use them when a lane needs a stable
architecture, proof, or policy choice that should survive individual PRs.

ADRs do not own active work state or generated dashboards. Active execution
state belongs in campaign-local `active.toml`; generated dashboards are derived
from campaign manifests and events.

## Current ADRs

- ADR-0001: [Configuration layering and clamp location](./0001-configuration-layering.md)
- ADR-0002: [GPU Backend Strategy](./0002-gpu-backend-strategy.md)
- ADR-0003: [GPU HAL Disposition](./0003-gpu-hal-disposition.md)
- BITNET-ADR-0004: [9950X3D + RTX 5070 Ti CUDA Product Bench](./BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
- BITNET-ADR-0005: [Proof Families Are Not Interchangeable](./BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
- BITNET-ADR-0006: [PR Closure Creates Backlog Unless Disposed](./BITNET-ADR-0006-pr-closure-creates-backlog.md)
- BITNET-ADR-0007: [A770 Diagnostics Are Lineage](./BITNET-ADR-0007-a770-diagnostics-are-lineage.md)
- BITNET-ADR-0008: [Self-Hosted-Only CI With No GitHub-Hosted Fallback](./BITNET-ADR-0008-self-hosted-only-ci-no-hosted-fallback.md) *(superseded)*
- BITNET-ADR-0011: [Lean Opt-In GitHub-Hosted Rust Fallback](./BITNET-ADR-0011-lean-opt-in-github-hosted-fallback.md)
- BITNET-ADR-0010: [Compute Dispatch Architecture](./BITNET-ADR-0010-compute-dispatch-architecture.md)

## Source-Of-Truth Role

| Layer | Owns |
| --- | --- |
| Proposal | Why the effort exists |
| Spec | What must be true |
| ADR | What decision was made and why it is durable |
| Plan | PR order and proof commands |
| Campaign `active.toml` | Current executable work |
| Handoff | Operator transfer context and closeout notes |
| Policy TOML | Enforceable ledger |
| Receipt or artifact | Evidence |

For BitNet proof work, ADRs should keep claim boundaries explicit. For example,
an ADR may decide that answer-ready model artifacts must precede backend answer
claims, or that dense SLM proof is first-class but must not be treated as
BitNet I2_S or QK256 proof.

## Template for new ADRs

Copy as `docs/adr/NNNN-title.md` or use a BitNet-prefixed filename when a
cross-lane proof decision needs a stable identifier:

```md
# ADR-NNNN: Title
- **Status:** Proposed | Accepted | Superseded by NNNN | Rejected
- **Date:** YYYY-MM-DD
- **Linked proposal/spec:** <paths>
- **Context:** <problem / forces>
- **Decision:** <what we chose and why>
- **Consequences:** <positive/negative trade-offs>
- **Claim boundary:** <what this decision does not prove>
- **Alternatives considered:** <brief bullets>
- **How to revert:** <what to change back if needed>
```
