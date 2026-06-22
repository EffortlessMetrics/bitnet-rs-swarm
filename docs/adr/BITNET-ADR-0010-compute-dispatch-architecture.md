# BITNET-ADR-0010: Compute Dispatch Architecture
- **Status:** Proposed
- **Date:** 2026-06-21
- **Linked proposal/spec:** (none yet — this ADR records an architecture
  inventory and target; a proposal to execute unification would supersede)
- **Related:** [ADR-0003 GPU HAL disposition](./0003-gpu-hal-disposition.md),
  [BITNET-ADR-0005 Proof families are not interchangeable](./BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
- **Supersedes:** none
- **Superseded by:** none

## Context

BitNet-rs has **five** compute-dispatch surfaces today. They overlap in
scope, do not compose, and conflate three concerns that are genuinely
orthogonal. This ADR inventories them, decomposes them into the three axes,
records the target architecture the repo intends to converge toward, and
defines the forcing functions that would trigger execution. It does not
commit to a refactor timeline.

### The five surfaces (verified against the source)

| # | Surface | Location | Ops | Weight formats | Providers | Wired to inference? |
|---|---|---|---|---|---|---|
| A | `KernelProvider` trait | `crates/bitnet-kernels/src/lib.rs:166` | `matmul_i2s`, `quantize` (2 ops) | `I2S`, `TL1`, `TL2` (BitNet only — see `crates/bitnet-common/src/types.rs:7`) | CPU×4 (`Fallback`, `Avx2`, `Avx512`, `Neon`), `Cuda`, `OpenCl` (in-kernels), `Npu`, `Rocm`, `Ffi`, `DebugLayer<P>` | **Yes** — `crates/bitnet-inference/src/layers/quantized_linear.rs` calls `&dyn KernelProvider` at 5 sites |
| B | `BackendProvider` + `BackendDispatcher` | `crates/bitnet-opencl/src/backend_{registry,dispatcher}.rs` | 8 (`MatMul`, `Quantize`, `Dequantize`, `Softmax`, `LayerNorm`, `Attention`, `RoPE`, `Sampling`) | (OpenCL-internal) | OpenCL (A770) | **No** — siloed inside the OpenCL crate |
| C | `bitnet-qk256-dispatch` | `crates/bitnet-qk256-dispatch/src/lib.rs` | QK256 linear | QK256 (BitNet variant) | `#[cfg(cuda)]` / `#[cfg(opencl)]` / fallback branches | **Yes** — via cfg routing |
| D | `GpuBackend` HAL trait family | `crates/bitnet-gpu-hal/src/hal_traits.rs` | full device/buffer/kernel/queue/program/event/context | n/a (HAL plumbing) | CPU mocks only (`launch_count += 1`) | **No** — zero consumers; disposition in [ADR-0003](./0003-gpu-hal-disposition.md) |
| E | `DenseLinear` (dense SLM path) | `crates/bitnet-inference/src/dense_forward.rs:56` | linear, attention, FFN (FP32) | FP32 only (Q8_0/Q4_K/Q4_0/Q5_0/Q2_K/Q3_K/Q5_K/Q6_K/Q8_K/F16/BF16 routed here as dequantized FP32 via string matching at `crates/bitnet-inference/src/engine.rs:485-495`) | **CPU scalar, f64-accumulated triple loop** | **YES** — the working Qwen/Phi/Gemma path |

#### The decisive detail

`QuantizationType` (`crates/bitnet-common/src/types.rs:7`) has exactly three
variants: `I2S`, `TL1`, `TL2` — **all BitNet formats**. None of the GGUF
quant formats (`Q8_0`, `Q4_K`, `F16`, `BF16`, etc.) are in the enum. The
inference engine recognizes 12 GGUF quant format names as **strings** at
`engine.rs:485-495` and routes them into Surface E (`DenseLinear`), where
dequantized FP32 weights get multiplied in a scalar f64-accumulated loop
with no SIMD, no GPU, and no connection to `KernelProvider`.

**The path producing validated coherent dense SLM (Qwen) answers cannot use
any of the AVX2/AVX-512/NEON/CUDA/OpenCL work** — it bypasses all of
Surfaces A–D. This is the central architectural gap.

### Why each surface exists (the history)

- **Surface A (`KernelProvider`)** — origin 2025-08-01 with `bitnet-kernels`.
  Narrow by design: it answers "which kernel runs my BitNet linear layer."
  Runtime-polymorphic, priority-ordered. Load-bearing.
- **Surface B (`BackendProvider`)** — origin 2026-02-28 in the same burst
  as gpu-hal. More sophisticated operation model (8 ops with capability
  advertisement and 4 dispatch strategies), but lives inside the OpenCL
  crate and never reached inference.
- **Surface C (`qk256-dispatch`)** — a dedicated router for the QK256
  BitNet variant. Grows by cfg-branch accretion: each new backend adds a
  branch.
- **Surface D (`GpuBackend`)** — origin 2026-02-28. The most ambitious
  (full HAL) but mock-only and zero-consumer. Disposition recorded in
  ADR-0003.
- **Surface E (`DenseLinear`)** — origin of the dense SLM path. Predates
  the others in spirit but is structurally the simplest: a pure-Rust FP32
  linear layer. Load-bearing for the only models with validated coherent
  answers today.

## Decision

**Record the architecture and target; defer execution until a forcing
function fires.** Specifically:

1. **The five surfaces are inventoried and cross-referenced** here and in
   `docs/reference/compute-dispatch-architecture.md`. Future agents read
   this ADR instead of re-deriving the fragmentation.
2. **The target is operator × weight-format × provider separation** (see
   "Target architecture" below). The repo *intends* to converge toward it.
3. **No refactor is committed in this ADR.** Execution requires a separate
   proposal + superseding ADR triggered by one of the forcing functions in
   §"Forcing functions."
4. **Per-surface dispositions** are recorded (see table below) so the
   current state is unambiguous and each surface knows its role while the
   target is deferred.
5. **Two cheap wins are recommended** (not committed) as low-risk slices
   that move code toward the target axis-separation without a refactor.

## Target architecture

Three orthogonal axes. Today each surface conflates at least two of them.
The target separates them.

| Axis | Question | DRY (shared) or SRP (separate)? |
|---|---|---|
| **Operator** | What is computed (Linear, Attention, FFN, RMSNorm, RoPE, Sampling, Dequantize) | **DRY** — every transformer has these, BitNet and dense alike |
| **Weight format** | How weights are laid out (I2S, TL1, TL2, QK256, FP32, Q8_0, Q4_K, F16, BF16, ...) | **SRP** — different math, different layouts; fusing bodies would be fake DRY |
| **Provider/Device** | Where it runs (CPU-scalar, AVX2, AVX-512, NEON, CUDA, OpenCL, Metal, Vulkan, wgpu, NPU) | **DRY** — orthogonal to what is computed |

```
                     ┌──────────────────────────────────────┐
                     │  Model composition layer              │
                     │  (Qwen, BitNet-2B, Llama, Phi ...)    │
                     │  composes operators, owns topology    │
                     └─────────────────┬─────────────────────┘
                                       │ calls operator traits
                     ┌─────────────────▼─────────────────────┐
                     │  Operator trait layer  ← DRY axis     │
                     │  Linear, Attention, FFN, RMSNorm,     │
                     │  RoPE, Sampling, Dequantize           │
                     │  one trait per operator, generic over │
                     │  weight format and provider           │
                     └─────────────────┬─────────────────────┘
                                       │ dispatched through
              ┌────────────────────────┼─────────────────────────┐
              │                        │                          │
   ┌──────────▼─────────┐  ┌───────────▼──────────┐  ┌────────────▼───────────┐
   │ Weight-format impls│  │ Provider/Device      │  │ (future) HAL plumbing  │
   │ ← SRP axis         │  │ registry             │  │ gpu-hal GpuBuffer/...  │
   │ i2s, TL1, TL2,     │  │ ← DRY axis           │  │ only when a feature    │
   │ QK256, FP32, Q8_0, │  │ CPU-scalar, AVX2,    │  │ needs GPU-resident     │
   │ Q4_K, F16, BF16    │  │ AVX-512, NEON, CUDA, │  │ buffers. Reference     │
   │ each its own module│  │ OpenCL, NPU, ...     │  │ until then (ADR-0003). │
   └────────────────────┘  └──────────────────────┘  └────────────────────────┘
```

**Principle:** operators are DRY (one Linear trait), weight formats are SRP
(one module per format), providers are DRY (one registry), HAL plumbing is
deferred (reference until a forcing function). A Linear forward becomes
"operator(Linear) × weight(format) × provider(device)" instead of today's
5-way hardcoding.

## Per-surface dispositions (current state, while target is deferred)

| Surface | Disposition | Rationale |
|---|---|---|
| A `KernelProvider` | **Keep, load-bearing.** Documented here as the operator×provider axis for BitNet weight formats. | Works, wired, narrow but correct for its scope. |
| B `BackendProvider` | **Keep, siloed.** Its 8-op `Operation` enum informs the target operator layer. Not merged now. | Real dispatch logic; moving it would be risky; informs the target. |
| C `qk256-dispatch` | **Keep, document as debt.** Each new backend adds a cfg branch; this is the unification trigger. | Works; cfg-accretion is the measurable refactor signal. |
| D `GpuBackend` (gpu-hal) | **Reference, per ADR-0003.** Candidate for the future HAL-plumbing axis. Not deleted, not integrated. | Predates conventions; trait needs hardening; no forcing function today. |
| E `DenseLinear` | **Keep, priority target when funded.** The cross-model gap — load-bearing for SLM answers but cannot inherit any SIMD/GPU work. | Currently CPU-scalar-only; provider-ization is the highest-value unification work. |

## Forcing functions (what would trigger execution)

A proposal to execute unification should be opened when **any** of these
fires:

1. **A new backend that would add an Nth cfg branch in Surface C.** Around
   backend #5–6, the cfg-accretion cost exceeds the unification cost.
   Triggers: widen Surface A to absorb Surface B's operation model, move
   OpenCL dispatch into A impls, collapse C's cfg branches. (~15–25 PRs,
   medium risk.)
2. **A feature needing GPU-resident buffers** (on-device KV cache, fused
   kernels, real multi-GPU tensor parallel). Triggers: harden Surface D's
   `GpuBackend` trait (zero-copy buffers, explicit queue on `launch`,
   async) and migrate. (~40–60 PRs, high risk, multi-quarter.)
3. **A dense-path performance need** (e.g., Qwen inference too slow on the
   scalar path for a target device). Triggers: provider-ize Surface E so
   it can dispatch through AVX2/NEON/CUDA like Surface A. (~5–10 PRs,
   medium risk.)

Today the repo is near the threshold on (1) and (3). No forcing function
has fired on (2). Until one fires, the dispositions above hold.

## Recommended cheap wins (not committed; would be separate proposals)

Two slices that move code toward the target axis-separation without a
refactor, each independently mergeable in 1–3 PRs:

1. **Add the missing weight formats to `QuantizationType`.** Today the
   enum is BitNet-only. Add `FP32`, `Q8_0`, `Q4_K`, `F16`, `BF16`, etc.
   This makes the weight-format axis *explicit* in the type system instead
   of hidden in string matching at `engine.rs:485-495`. SRP-correct: each
   format stays its own module, but becomes first-class.
2. **Extract `HalError` into `bitnet-common`.** The one gpu-hal artifact
   with clean cross-cutting value. Gives all five surfaces a shared GPU
   error taxonomy. Zero risk.

A third, slightly larger slice:

3. **Give Surface E (`DenseLinear`) a provider seam.** Add an optional
   `provider` parameter (no-op for now) so a future migration is a trait
   change, not a surgery.

## Consequences

- *Positive:* The dispatch fragmentation is recorded durably. Future agents
  and contributors have one canonical answer (this ADR + the design doc)
  instead of re-deriving it. The "is gpu-hal dead" / "should we unify"
  cycle stops.
- *Positive:* The strategic option of unification is preserved without
  being forced. Each forcing function has a defined response.
- *Positive:* The cross-model gap (Surface E cannot use SIMD/GPU) is
  recorded as the priority target, so it isn't lost.
- *Negative:* The fragmentation persists until a forcing function fires.
  Each new backend still adds a cfg branch in Surface C. The dense SLM
  path stays scalar. This is accepted as the cost of deferral.
- *Negative:* Recording the target without executing it can look like
  indecision. Mitigated by the explicit forcing-function triggers.

## Claim boundary

This ADR does **not** prove:

- That unification will happen on any timeline.
- That any specific surface's design is correct.
- That the dense SLM path will be accelerated.
- That gpu-hal will ever be integrated.

This ADR **does** establish:

- The five surfaces are inventoried and their relationships recorded.
- The three-axis target is the intended convergence direction.
- Forcing functions and their triggered responses are defined.
- Per-surface dispositions are fixed until a superseding ADR.

## Alternatives considered

- **Execute Option 1 (widen `KernelProvider`) now.** Rejected: no forcing
  function has fired. The cost (~15–25 PRs, medium risk to the hot
  inference path) is not justified while the current surfaces work.
- **Execute Option 2 (adopt gpu-hal's HAL) now.** Rejected: gpu-hal's
  trait needs hardening before real adoption, no feature needs
  GPU-resident buffers, and the cost (~40–60 PRs, high risk) is not
  justified. See ADR-0003 for the gpu-hal disposition.
- **Do nothing, don't record.** Rejected: the fragmentation has already
  caused multiple re-investigations (issue #1639 and its predecessors).
  Recording it is cheap and stops the cycle.

## Reversibility

High. This ADR records state and intent; it changes no code. Any of the
target convergences can be pursued via a superseding ADR + proposal.
Reversing *this* ADR means writing a new ADR that records a different
target (or no target).

## Related contracts

- [ADR-0003 GPU HAL disposition](./0003-gpu-hal-disposition.md) — Surface D's
  specific disposition.
- [BITNET-ADR-0005 Proof families are not interchangeable](./BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
  — the principle that BitNet vs dense vs CUDA proof must not be conflated.
  This ADR's operator×format×provider separation is the architectural
  expression of that proof principle.
- `docs/reference/compute-dispatch-architecture.md` — the canonical
  human-readable design reference that elaborates this ADR.
- `docs/reference/gpu-hal-design.md` — Surface D's detailed reference,
  including a "where gpu-hal fits" section pointing here.
