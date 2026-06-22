# ADR-0003: GPU HAL Disposition
- **Status:** Proposed
- **Date:** 2026-06-21
- **Linked proposal/spec:** [BITNET-PROP-0019](../proposals/BITNET-PROP-0019-gpu-hal-disposition.md), [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md)
- **Supersedes:** none
- **Superseded by:** none

## Context

`crates/bitnet-gpu-hal/` is the second-largest crate in the workspace
(148 source files, 188,392 LOC, ~15,282 functions, ~3,000 self-contained
tests). It landed on 2026-02-28 as the **upper abstraction layer** of a
two-layer multi-backend GPU plan documented in
`docs/reference/dual-backend-roadmap.md` (origin 2025-11-03, ~3 months
before the crate landed).

The two layers and their roles:

- **Lower layer — `bitnet-kernels` (origin 2025-08-01).** The kernel
  implementations, CPU reference, and per-backend dispatch (CUDA, OpenCL,
  NEON, AVX2). Already load-bearing at the time gpu-hal landed, consumed by
  9 crates today (inference, cli, server, wasm, quantization, receipts,
  qk256-dispatch, device-config-core, kernels-self). Owns its own dispatch
  abstraction (`DispatchBackend` enum at `src/dispatch_planner.rs`) and its
  own parity infrastructure (`gpu_quantization_parity.rs`, which predates
  gpu-hal entirely, since 2025-09-01).
- **Upper layer — `bitnet-gpu-hal` (origin 2026-02-28).** A unified
  abstraction meant to sit *above* the per-backend kernels, exposing one
  `KernelDispatcher` / `GpuBackend` trait surface across 8 backends
  (CPU/CUDA/OpenCL/Vulkan/Metal/ROCm/WebGPU/Level-Zero) with
  backend-agnostic memory pools, multi-device scheduler, and model sharding.
  Phase 10 of the roadmap. The backend modules are API-shape CPU mocks
  (`CUDAKernel::launch()` body is `self.launch_count += 1`), which is the
  stated Phase 10 "CPU mock/reference" deliverable.

The lower layer kept growing because it was already wired and load-bearing
— `neon_speculative_decoding`, `dispatch_planner`, the `avx2_*_parity`
tests all landed in the days after the same 2026-02-28 burst. The upper
layer landed its reference/mock phase, but its **integration phase**
(roadmap Phase 9 — real backends consuming the HAL traits) never started.
No wiring PR, no campaign, no handoff scoped Phase 9. gpu-hal has sat with
zero inbound dependents since landing.

**Temporal note on the contract gap.** The crate predates the repo's
current contract conventions. The first campaign `active.toml` landed
2026-05-05; `AGENTS.md` 2026-05-10; the first `BITNET-PROP` 2026-05-12;
`SPEC_SYSTEM.md` 2026-05-20 — all ~2.5 months *after* gpu-hal landed. So
the absence of a campaign, proposal, or spec at landing time is a temporal
artifact, not a governance failure. Only ADRs pre-existed (since
2025-08-16), and ADR-0002 was written the same day as the burst but covers
the feature-flag strategy, not the layered HAL architecture. This ADR
(ADR-0003) is the first to record the HAL's role and disposition.

The crate has been re-investigated multiple times
([#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639))
with contradictory conclusions (delete / intentional-phase-8 / superseded
/ two-layer-upper). Each pass costs real effort because no durable
artifact records the disposition.

## Decision

**`bitnet-gpu-hal` is retained as-is as a read-only educational/reference
crate.** Specifically:

1. **Retained, not deleted.** The crate stays in the workspace `members`
   list and continues to compile and lint in CI exactly as it does today.
2. **Not actively integrated.** No work is slated to wire the HAL into
   `bitnet-kernels`, `bitnet-opencl`, `bitnet-inference`, or any consumer.
   The real GPU dispatch surface continues to be owned by
   `bitnet-kernels` (`DispatchBackend` enum) and `bitnet-opencl`
   (`backend_dispatcher`).
3. **Not a candidate for new feature work** without a superseding ADR.
   The drift modules (`semantic_search`, `rate_limiter`, `api_gateway`,
   `instruction_tuning`, `prompt_template`, `docker_ci`, etc.) are
   explicitly out-of-scope for a GPU HAL and must not be extended.
4. **Reference value.** The trait glossary in `hal_traits.rs`
   (`GpuDevice`, `GpuBuffer`, `GpuKernel`, `GpuQueue`, `GpuProgram`,
   `GpuEvent`, `GpuContext`, `GpuBackend`, `GpuMemoryAllocator`) is a
   clean, documented reference for what a GPU HAL surface can look like,
   and is the primary retained value. New contributors may read it as a
   glossary; it is not a load-bearing production abstraction.
5. **Disposition is durable.** Reopening the question requires writing a
   superseding ADR (e.g., ADR-NNNN proposing backend trait unification or
   wholesale extraction/deletion), not re-investigating from scratch.

## Consequences

- *Positive:* The crate's status is no longer ambiguous. Future agents and
  contributors have one canonical answer (this ADR + the linked design doc)
  instead of re-deriving it. Re-investigation cycles stop.
- *Positive:* The strategic option of backend trait unification remains
  open — it would arrive as a separate proposal and superseding ADR. This
  ADR does not foreclose it.
- *Positive:* The crate's clean `hal_traits` reference remains available
  for onboarding and future design work.
- *Negative:* The crate continues to compile and lint in CI, contributing
  ~4,668 unwrap/expect sites to the no-panic baseline and ~150 blanket
  `#[allow]` suppressions. This is accepted as the cost of retention.
- *Negative:* Source->swarm syncs will continue to mechanically touch the
  crate (as the 2026-05-20 sync did). Such touches do not change the
  disposition.
- *Negative:* The 26 undeclared orphan files and 75 misleading "Module
  stub — implementation pending merge from feature branch" headers remain.
  These are recorded as open investigation items in
  `docs/reference/gpu-hal-design.md` for a future campaign if pursued;
  they are not blockers for this disposition.

## Claim boundary

This decision does **not** prove:

- That the HAL is a validated GPU execution path. It is not. The backend
  mocks do not compute.
- That the HAL is the intended future dispatch layer. That would require a
  separate strategic decision and superseding ADR.
- That the HAL's modules are correct or fit for purpose as production
  reference implementations. Several duplicate work the real path already
  does more rigorously.
- That any capability the real path lacks is supplied by the HAL. The
  parity analysis shows the real path has every capability the HAL
  targeted.

This decision **does** establish:

- The crate's disposition is "retained read-only reference," recorded
  durably so it is not re-litigated.
- New integration or feature work on the HAL requires a superseding ADR.

## Alternatives considered

- **Backend trait unification (integrate the HAL as the real dispatch
  layer).** Credible and high-value long-term, but it is a multi-quarter
  strategic commitment requiring trait hardening (zero-copy buffers,
  explicit queue on `launch`, async execution model) and migration of
  three parallel backend abstractions. It must arrive as its own proposal
  and superseding ADR, not as part of a disposition decision. This ADR
  leaves it open.
- **Triage + extract the salvageable core.** Strip the crate to its
  ~15-20K LOC coherent core (`hal_traits`, `tensor_serde`, `model_cache`,
  real matmul reference) and delete ~170K LOC of drift. Owner guidance is
  to retain as-is; the spec records the coherent-core inventory so a future
  extraction proposal has the map.
- **Delete wholesale.** Rejected per owner guidance ("don't delete it,
  document and investigate it").

## Reversibility

High. Because no runtime code depends on the crate, the disposition can be
reversed at any time by a superseding ADR:

- To pursue trait unification: write a proposal + superseding ADR + spec +
  plan, then begin the multi-quarter migration.
- To extract the core: write a proposal + superseding ADR, then run the
  triage campaign recorded in the design doc.
- To delete: write a superseding ADR and a deletion PR.

The cost of reversal is the cost of writing the new artifacts plus the
chosen code work — there is no runtime entanglement to untangle.

## Related contracts

- [BITNET-PROP-0019 GPU HAL Disposition](../proposals/BITNET-PROP-0019-gpu-hal-disposition.md)
- [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md)
- [ADR-0002 GPU Backend Strategy](./0002-gpu-backend-strategy.md) (the
  additive `gpu` feature-flag decision; does not mention the HAL)
- [`docs/reference/gpu-hal-design.md`](../reference/gpu-hal-design.md) (the
  canonical design reference)
