# GPU HAL Disposition Campaign

Campaign ID: `gpu-hal-disposition`

Status: active

## Objective

Land durable governance artifacts (Proposal, ADR, Spec, Plan, design doc)
that fix the disposition of `crates/bitnet-gpu-hal/` and stop the crate
from being re-litigated by future agents. Docs-only; no runtime changes.

## End State

- ADR-0003 is `Accepted` and linked from `docs/adr/README.md`.
- BITNET-SPEC-GPU-HAL-REFERENCE-LAYER is `accepted`.
- `docs/reference/gpu-hal-design.md` is the canonical reference for the
  crate.
- This campaign is `complete` with all work items `merged`.
- Issue [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)
  is closed with a pointer to ADR-0003.

## Hard Constraints

- Do not change runtime code.
- Do not add or remove the crate from the workspace.
- Do not promote, demote, or create any support-tier claim.
- Do not change CI lane routing or policy ledgers.
- Do not wire any consumer onto `bitnet-gpu-hal`.
- A future change to the disposition MUST arrive as a superseding ADR.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| GH-DISP-001 | ready | Contract rails: ADR-0003 + BITNET-PROP-0019 + spec |
| GH-DISP-002 | ready | Design reference doc (`docs/reference/gpu-hal-design.md`) |
| GH-DISP-003 | ready | Closeout: README link, campaign complete, close #1639 |

## Linked Artifacts

- Proposal: [BITNET-PROP-0019](../../../docs/proposals/BITNET-PROP-0019-gpu-hal-disposition.md)
- ADR: [ADR-0003](../../../docs/adr/0003-gpu-hal-disposition.md)
- Spec: [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../../../docs/specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md)
- Plan: [plans/gpu-hal/implementation-plan.md](../../../plans/gpu-hal/implementation-plan.md)
- Issue: [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)

## Non-Claims

- This campaign does not claim `bitnet-gpu-hal` is a validated GPU path.
- This campaign does not claim the HAL is the future dispatch layer.
- This campaign does not claim any capability the real path lacks is
  supplied by the HAL.
- This campaign records a disposition; it does not endorse the crate's
  scope drift as correct.
