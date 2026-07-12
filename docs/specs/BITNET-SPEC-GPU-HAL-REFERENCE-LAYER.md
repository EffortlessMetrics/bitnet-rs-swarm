# BITNET-SPEC-GPU-HAL-REFERENCE-LAYER: GPU HAL Reference Layer Contract

Status: proposed
Owner: gpu-hal
Created: 2026-06-21

Linked proposal: [BITNET-PROP-0019](../proposals/BITNET-PROP-0019-gpu-hal-disposition.md)
Linked ADRs: [ADR-0003](../adr/0003-gpu-hal-disposition.md)
Linked plan: [plans/gpu-hal/implementation-plan.md](../../plans/gpu-hal/implementation-plan.md)
Support-tier impact: no support promotion; reference-layer contract only
Policy impact: no policy exception

## Context

`crates/bitnet-gpu-hal/` is the **upper abstraction layer of a two-layer
multi-backend GPU plan** (`docs/reference/dual-backend-roadmap.md`, origin
2025-11-03). The lower layer is `bitnet-kernels` (origin 2025-08-01),
which is load-bearing and consumed by 9 crates today. gpu-hal landed on
2026-02-28 (Copilot co-authored, in a 504-commit burst alongside
`bitnet-kernels/src/cuda/`, `bitnet-opencl`, ADR-0002, and ~200 other
PRs). Its CPU reference/mock phase landed complete; its integration phase
(roadmap Phase 9 — real backends consuming the HAL traits) never started.
It has no inbound dependency edges.

The crate **predates the repo's current contract conventions**: the first
campaign `active.toml` landed 2026-05-05, `AGENTS.md` 2026-05-10, the
first `BITNET-PROP` 2026-05-12, `SPEC_SYSTEM.md` 2026-05-20 — all ~2.5
months after gpu-hal. So the absence of a proposal/campaign at landing
time is a temporal artifact, not a governance failure. This spec is part
of the work that brings the crate inside the contract system.

Because the lower layer was already wired and load-bearing, each new
capability got implemented in `bitnet-kernels` directly rather than routed
through the upper HAL. Today there is duplication across the two layers,
not because one "won" but because both advanced independently and only the
lower layer had live consumers. This spec records the boundary so the
upper layer's status is unambiguous.

This spec is a **disposition contract**, not a behavioral requirements
contract. It defines what the crate is, what it claims, and what it must not
be treated as. It does not require any runtime behavior from the crate beyond
what it already does (compile and self-test).

## Scope

- `crates/bitnet-gpu-hal/` (the workspace member crate).
- Its public API surface as declared in `src/lib.rs`.
- Its role relative to the real GPU dispatch path.

Out of scope:
- The real GPU path (`bitnet-kernels`, `bitnet-opencl`, etc.) — owned by
  their own contracts.
- Any future backend-trait-unification effort — that would require a
  superseding ADR and its own spec.
- Runtime behavioral requirements on the HAL — it is a reference crate, not
  a production path.

## Requirements

### REQ-001 — Stable public API surface

The crate's public API surface SHALL be exactly the items re-exported from
`src/lib.rs`. As of this spec the surface is:

- `pub use hal_traits::{HalError, HalResult};`
- The 113 `pub mod` declarations exposing module namespaces for reference
  reading.

The surface MUST NOT be reduced (modules removed from the `pub mod` list)
without a superseding ADR, because reducing it changes the reference
contract for anyone reading the crate.

Rationale: locking the surface prevents accidental API drift while keeping
the disposition stable.

### REQ-002 — No new inbound dependencies

No other workspace crate SHALL add `bitnet-gpu-hal` to its `[dependencies]`
without a superseding ADR. The crate is a read-only reference, not a
consumed abstraction.

Rationale: the disposition (ADR-0003) is "retained, not integrated." Wiring
a consumer would reverse that and must be a deliberate decisioned change.

### REQ-003 — Backend mocks are non-computing references

The backend modules (`cuda_backend`, `vulkan_compute`, `opencl_backend`,
`metal_backend`, `rocm_backend`, `level_zero_backend`, `webgpu_backend`)
are **API-shape references**, not numerical compute implementations.
`CUDAKernel::launch()` records `launch_count`; it does not execute a kernel.
This is the documented and accepted behavior, not a defect.

These modules MUST NOT be represented (in docs, comments, READMEs, or
claims) as executing GPU work. Any change that makes a backend mock compute
for real is a superseding-ADR-class change.

Rationale: prevents the mocks from being mistaken for a parity oracle or a
validated execution path.

### REQ-004 — Trait glossary is the reference value

The `hal_traits.rs` trait family (`GpuDevice`, `GpuBuffer`, `GpuKernel`,
`GpuQueue`, `GpuProgram`, `GpuEvent`, `GpuContext`, `GpuBackend`,
`GpuMemoryAllocator`, plus `HalError`, `MemoryType`,
`ComputeCapabilities`, `ProgramSource`) is the primary retained value of
the crate. It SHOULD be treated as a readable glossary of GPU HAL concepts.

The glossary is not load-bearing: no production code path depends on these
traits. The glossary MAY inform future design (e.g., a trait-unification
proposal), but until such a proposal is accepted it is reference material.

### REQ-005 — Drift modules are frozen, not extended

The modules identified as out-of-scope for a GPU HAL
(`semantic_search`, `rate_limiter`/`rate_limiting`, `safety_guardrails`,
`instruction_tuning`, `prompt_template`, `api_gateway`, `api_server`,
`docker_ci`, `model_pruning`, `openai_compat`, `tokenizer_detokenizer`)
are **frozen**. They MUST NOT receive new feature work. Mechanical sync
touch-ups and lint fixes are permitted; new functions or modules in these
areas are not.

Rationale: these are Copilot scope-drift, not HAL concerns. Freezing them
stops the drift from accumulating further while respecting the retain-as-is
disposition.

### REQ-006 — Compilation is the only proof claimed

The crate's validation SHALL be:

```bash
cargo check -p bitnet-gpu-hal --no-default-features
cargo test  -p bitnet-gpu-hal --no-default-features
```

i.e. "it compiles and its own self-tests pass." No stronger claim (parity,
conformance, performance, hardware validation) is made or implied by the
crate's continued presence in the workspace.

## Non-goals

- Defining requirements for the real GPU path.
- Promoting the HAL to a support tier.
- Requiring cleanup of the 26 orphan files, the 75 misleading headers, or
  the drift modules. These are recorded as open investigation items in
  `docs/reference/gpu-hal-design.md`; a future campaign may pursue them,
  but the disposition spec does not require it.
- Committing to backend trait unification.

## Acceptance examples

| Condition | Expected |
|---|---|
| `cargo check -p bitnet-gpu-hal` on main | Compiles |
| `grep -rl bitnet-gpu-hal crates/*/Cargo.toml \| grep -v bitnet-gpu-hal/Cargo.toml` | Empty (no consumers) |
| `git log --oneline -- crates/bitnet-gpu-hal/src/semantic_search.rs` since ADR-0003 date | No feature additions (mechanical sync only) |
| A contributor reads `docs/reference/gpu-hal-design.md` | Can answer "what is this crate and should I touch it?" without re-investigating |

## Failure modes

- A future sync or PR adds `bitnet-gpu-hal` as a dependency of another
  crate. This reverses the disposition and must be caught at review with a
  request for a superseding ADR.
- A future PR extends a drift module with new feature work. This violates
  REQ-005 and must be rejected or redirected.
- A future doc/README claims the HAL is a validated or integrated GPU path.
  This violates REQ-003 and must be corrected.

## Compatibility

No migration required. The crate's behavior is unchanged. The spec
formalizes the existing state.

## Implementation ownership

- Crate: `crates/bitnet-gpu-hal/`
- Disposition owner: the `gpu-hal-disposition` campaign
- Real GPU path owner: `bitnet-kernels`/`bitnet-opencl` maintainers

## Evidence mapping

| Requirement | Proof | Evidence class |
|---|---|---|
| REQ-001 (stable surface) | `cargo check -p bitnet-gpu-hal` continues to compile; `lib.rs` re-exports unchanged | Compile |
| REQ-002 (no consumers) | `grep -rl bitnet-gpu-hal crates/*/Cargo.toml \| grep -v bitnet-gpu-hal/Cargo.toml` is empty | Policy |
| REQ-003 (mocks don't compute) | `cuda_backend.rs` `launch()` body audit | Code inspection |
| REQ-004 (glossary value) | `hal_traits.rs` is readable and documented | Documentation |
| REQ-005 (drift frozen) | No feature commits to drift modules post-ADR | Policy |
| REQ-006 (compile-only proof) | `cargo test -p bitnet-gpu-hal` passes | Unit |

## Support-tier impact

None. The crate is not promoted to any support tier. It remains
undocumented at the support-tier level (intentionally not "experimental,"
because that would imply a contract that could change — the contract is
fixed by this spec).

## Claim boundary

This spec does **not** establish:

- That the HAL is a validated GPU execution path.
- That the HAL is the future dispatch layer.
- That the HAL's modules are correct or production-fit.
- That any capability the real path lacks is supplied by the HAL.

This spec **does** establish:

- The crate's disposition is contractually fixed as "retained read-only
  reference" until a superseding ADR.
- The public surface, non-computing mocks, and frozen drift modules are
  the enforceable boundary.
