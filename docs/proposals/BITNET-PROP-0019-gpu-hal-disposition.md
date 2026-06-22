# BITNET-PROP-0019: GPU HAL Disposition

Status: proposed
Owner: gpu-hal
Type: proposal
Created: 2026-06-21

Linked specs:
- [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md)

Linked ADRs:
- [ADR-0003 GPU HAL disposition](../adr/0003-gpu-hal-disposition.md)

Linked plan:
- [plans/gpu-hal/implementation-plan.md](../../plans/gpu-hal/implementation-plan.md)

## Problem

`crates/bitnet-gpu-hal/` is the second-largest crate in the workspace
(148 source files, 188,392 LOC, ~15,282 functions, ~3,000 tests). It
landed on 2026-02-28 as the **upper abstraction layer** of a two-layer
multi-backend GPU plan documented in
`docs/reference/dual-backend-roadmap.md` (origin 2025-11-03). The lower
layer is `bitnet-kernels` (origin 2025-08-01, already load-bearing and
consumed by inference/cli/server/wasm/quantization/receipts at the time).
gpu-hal was intended to sit *above* the per-backend kernels, exposing a
unified `KernelDispatcher` / `GpuBackend` trait surface across 8 backends
with backend-agnostic memory pools, multi-device scheduler, and model
sharding.

The lower layer (`bitnet-kernels`) kept growing because it was already
wired and load-bearing — `neon_speculative_decoding`, `dispatch_planner`,
the `avx2_*_parity` tests, etc. all landed in the days after the same
2026-02-28 burst. The upper layer (gpu-hal) landed its CPU reference/mock
phase but its **integration phase** (roadmap Phase 9 — real backends
consuming the HAL traits) never started. No wiring PR, no campaign, no
handoff scoped Phase 9, so gpu-hal has sat with zero inbound dependents
since landing.

Because the integration phase never started, and because the crate
predates the repo's current contract conventions (the first campaign
`active.toml` landed 2026-05-05, `AGENTS.md` 2026-05-10, the first
`BITNET-PROP` 2026-05-12, `SPEC_SYSTEM.md` 2026-05-20 — all ~2.5 months
*after* gpu-hal landed), the crate today has no Proposal, no Spec, no
campaign owner, and no ADR recording its role. The result is that
repeated investigation (issue
[#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639))
re-flags it as dead code, orphan, or missed opportunity, because no
durable artifact records what it is and what should happen with it.

**This is not framed as a governance failure.** The contract conventions
the crate lacks did not exist when it landed. The accurate framing is: a
two-layer plan where the lower layer is wired and the upper layer's
integration phase was never started, and the crate now needs to be
brought inside the contract system that post-dates it.

The full layered-architecture diagram, the backend-mock detail, and the
timeline live in the canonical design reference
[`docs/reference/gpu-hal-design.md`](../reference/gpu-hal-design.md) and
[ADR-0003](../adr/0003-gpu-hal-disposition.md); this proposal links to
them rather than duplicating them.

## Users and affected surfaces

- **Contributors and agents** reading the workspace tree and asking "what is
  this 188K-LOC crate for, and should I touch it?"
- **CI**, which compiles the crate on every default workspace build and
  lints it via the no-panic baseline (it contributes ~4,668 unwrap/expect
  sites to the grandfathered inventory).
- **Future GPU-backend work**, which needs a single recorded decision about
  whether the upper HAL layer's integration phase is still wanted, and if
  so when; or whether the lower layer's own dispatch abstractions own that
  role going forward. See also
  [BITNET-ADR-0010](../adr/BITNET-ADR-0010-compute-dispatch-architecture.md)
  for the broader 6-surface dispatch inventory.
- **Issue #1639** and any future agent that re-encounters the crate, which
  need a durable record to read instead of re-investigating from scratch.

## Why now

Three forces converge:

1. The crate has been re-investigated at least three times
   ([#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639))
   with contradictory conclusions (delete wholesale -> intentional Phase 8
   reference -> superseded artifact -> prototype corpus). Each pass costs
   real effort. A durable decision stops the cycle.
2. The repo is actively investing in its contract system (BITNET-PROP /
   BITNET-SPEC / BITNET-ADR / campaign manifests). Leaving the largest
   crate outside that system is the exact kind of gap the system exists to
   close.
3. The crate is still being mechanically modified by source->swarm syncs
   (the 2026-05-20 sync touched 16 test files), so the question is not
   going away on its own.

## Desired outcome

A durable, contract-recorded disposition for `bitnet-gpu-hal` that:

- Records what the crate is and how it was produced, honestly.
- Records why it has no dependents and why that is not a regression.
- Records the relationship between the HAL and the real GPU path.
- Stops the crate from being re-flagged as dead code or re-investigated
  from scratch by future agents.
- Does not require touching runtime code (this proposal is docs-only).

## Success criteria

- An accepted ADR records the disposition decision and is linked from
  `docs/adr/README.md`.
- An accepted spec defines what the reference layer's contract is and is not.
- A campaign owns the documentation work and tracks it to closeout.
- `docs/reference/gpu-hal-design.md` exists and is the canonical reference
  a contributor or agent reads to understand the crate.
- The campaign closes out with no stale active work items.
- Future agents reading AGENTS.md + the ADR + the design doc can answer
  "what is gpu-hal and should I touch it?" without re-running the
  investigation in [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639).

## Proposed shape

A docs-only lane (`gpu-hal-disposition`) that lands:

1. **ADR-0003** — the durable decision: `bitnet-gpu-hal` is retained as-is
   as a read-only educational/reference crate. The real GPU dispatch surface
   continues to be owned by `bitnet-kernels` (`DispatchBackend`) and
   `bitnet-opencl` (`backend_dispatcher`). The HAL is not slated for
   deletion, not slated for active integration, and not a candidate for new
   feature work without a superseding ADR.
2. **BITNET-SPEC-GPU-HAL-REFERENCE-LAYER** — the contract describing what
   the reference layer is, what its public surface is (`HalError`, `HalResult`,
   the trait glossary in `hal_traits.rs`), and the explicit non-claims
   (the backend mocks do not compute; nothing depends on it; it is not a
   parity oracle).
3. **`docs/reference/gpu-hal-design.md`** — the human-readable design
   reference: origin, the parity table showing the real path already has
   every capability, the 26 orphan files, the misleading headers, and the
   forward guidance.
4. **A `gpu-hal-disposition` campaign** that owns the three documentation
   PRs and closes out when they merge.

No code changes. No runtime changes. No support-tier changes. No policy
changes. The crate continues to compile in CI exactly as it does today.

## Alternatives considered

- **Backend trait unification (Play C).** Harden `hal_traits` into a real
  backend trait, migrate `bitnet-kernels`/`opencl`/`inference` onto it over
  30-50 PRs. Rejected for this proposal: it is a real strategic option but
  it is a multi-quarter commitment, not a disposition. If pursued later it
  must arrive as its own proposal and superseding ADR. Recording the
  retain-as-is decision now does not foreclose it; it stops the crate being
  ambiguous in the meantime.
- **Triage + extract the salvageable core.** Strip to ~15-20K LOC of
  coherent core (hal_traits + tensor_serde + model_cache + matmul reference)
  and delete ~170K LOC of drift/superseded code. Rejected for this proposal:
  owner guidance is retain as-is. The spec records which modules are
  coherent-core vs drift so a future extraction proposal has the inventory.
- **Delete wholesale.** Rejected: owner guidance is "don't delete it,
  document and investigate it."

## Risks

- **Continued maintenance cost.** The crate continues to compile and lint in
  CI, and continues to contribute to the no-panic baseline. This is accepted
  (see ADR-0003 consequences).
- **Sync churn.** Source->swarm syncs will continue to touch the crate. The
  ADR records that such touches are mechanical and do not change the
  disposition.
- **Future re-litigation.** Mitigated by the ADR + design doc + spec; a
  future agent that wants to revisit must write a superseding ADR, not
  re-investigate from scratch.
- **Misread as endorsement of the scope drift.** The design doc and spec
  explicitly call out the drift modules (`semantic_search`, `rate_limiter`,
  `api_gateway`, etc.) as out-of-scope for a HAL and not candidates for
  extension.

## Non-goals

- No runtime code changes.
- No deletion of crate code.
- No support-tier promotion or demotion.
- No change to the real GPU dispatch path.
- No change to CI lane routing.
- No commitment to backend trait unification (that would be a separate
  proposal and superseding ADR).
- No orphan-file triage or cosmetic cleanup in this proposal — those are
  recorded as open investigation items in the design doc for a future
  campaign if pursued, but are explicitly out of scope for the
  disposition decision.

## Specs required

- [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md)

## Decisions required

- [ADR-0003 GPU HAL disposition](../adr/0003-gpu-hal-disposition.md)

## Evidence strategy

Proof for this lane is docs-rails validation:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check gpu-hal-disposition
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

No runtime proof is claimed or required. The disposition is a governance
artifact, not a behavioral claim.

## Exit criteria

- ADR-0003 is `Accepted` and linked from `docs/adr/README.md`.
- BITNET-SPEC-GPU-HAL-REFERENCE-LAYER is `accepted` and linked from
  `docs/specs/README.md` if one exists.
- `docs/reference/gpu-hal-design.md` is the canonical reference.
- The `gpu-hal-disposition` campaign reaches `complete` with all work items
  `merged`/`done`.
- Issue [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)
  closes with a pointer to the ADR.
