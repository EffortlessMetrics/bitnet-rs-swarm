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

BitNet-rs has **six** compute-dispatch surfaces today, spread across five
architectural layers that reuse the same vocabulary — *backend, dispatch,
provider, device* — without clearly distinguishing scope. Several surfaces
duplicate the same control-plane types. This ADR inventories the layers,
decomposes them, records the target architecture the repo intends to
converge toward, and defines the forcing functions that would trigger
execution. It does not commit to a refactor timeline.

### The six surfaces across five layers (verified against the source)

| # | Surface | Layer | Location | Role | Wired to inference? |
|---|---|---|---|---|---|
| A | `KernelProvider` trait | Format-specific execution | `crates/bitnet-kernels/src/lib.rs:166` | Executes 2 ops (`matmul_i2s`, `quantize`) for BitNet formats (`I2S`/`TL1`/`TL2`). 7+ impls: CPU×4, CUDA, OpenCL, NPU, ROCm, FFI, `DebugLayer<P>` | **Yes** — `quantized_linear.rs` at 5 sites |
| B | `BackendProvider` + `BackendDispatcher` | **Selection policy (no execution)** | `crates/bitnet-opencl/src/backend_{registry,dispatcher}.rs` | Reports name/status/capabilities/priority. **Has no execute method** — pure selection policy + decision recorder | **No** — siloed inside OpenCL crate |
| C | `bitnet-qk256-dispatch` | Format-specific execution | `crates/bitnet-qk256-dispatch/src/lib.rs` | QK256 linear via `#[cfg(cuda)]`/`#[cfg(opencl)]`/fallback branches | **Yes** — via cfg |
| D | `GpuBackend` HAL trait family | Accelerator-resource prototype | `crates/bitnet-gpu-hal/src/hal_traits.rs` | Device/buffer/kernel/queue/program/event/context. CPU mocks only | **No** — zero consumers; disposition in [ADR-0003](./0003-gpu-hal-disposition.md) |
| E | `DenseLinear` (dense SLM path) | Format-specific execution | `crates/bitnet-inference/src/dense_forward.rs:56` | FP32 linear/attention/FFN via f64 scalar loop | **YES — Qwen/Phi/Gemma path** |
| F | `inference::Backend` trait | Full-model executor | `crates/bitnet-inference/src/backends.rs:20` | Runs a complete model forward. **GPU/NPU paths are shape-preserving mocks** (`backends.rs:360` "just create a mock GPU tensor", `backends.rs:193` "shape-preserving tensor fallback") | **Yes** — model-level boundary |

#### Correction to the earlier "5 surfaces" model

A prior version of this analysis counted 5 surfaces and mischaracterized
Surface B. Three corrections based on direct code verification:

1. **Surface F (`inference::Backend`) was missed entirely.** The
   full-model executor trait's GPU and NPU paths are **simulated** — they
   create shape-preserving mock tensors instead of transferring to a real
   accelerator (`backends.rs:193,360`). This is a real surface that must
   be included: a new HAL wired underneath a full-model abstraction that
   still obscures whether real compute occurred would be a hollow victory.
2. **Surface B (OpenCL `BackendProvider`) is policy, not execution.** Its
   trait has only `name`/`status`/`capabilities`/`priority_score` —
   **no `execute`/`matmul`/`compute` method**. The `BackendDispatcher`
   has `backends_for`/`is_supported`/`record`/`strategy`. Calling it a
   "parallel dispatch trait" (as the earlier draft did) overstates it; it
   is a **selection-policy prototype**.
3. **A dense-Q8 sidecar selector is evolving** (eager-F32 vs packed-Q8
   paths, candidate availability, payload identity) — narrow, proof-bound,
   tied to a real model path. This is an example of useful architecture
   evolving outside the gpu-hal corpus and outside `KernelProvider`.

#### The decisive detail

`QuantizationType` (`crates/bitnet-common/src/types.rs:7`) is doc-commented
**"Quantization types supported by BitNet"** and has exactly three
variants: `I2S`, `TL1`, `TL2` — all BitNet formats by design. The inference
engine recognizes 12 GGUF quant format names (`q8_0`, `q4_k`, ...) as
**strings** at `engine.rs:485-495` and routes them into Surface E
(`DenseLinear`), where dequantized FP32 weights get multiplied in a scalar
f64-accumulated loop with no SIMD, no GPU, and no connection to
`KernelProvider`.

**The path producing validated coherent dense SLM (Qwen) answers cannot use
any of the AVX2/AVX-512/NEON/CUDA/OpenCL work** — it bypasses all of
Surfaces A–D. This is the central architectural gap, and it is the
cross-model constraint the target must satisfy.

> **Correction note:** an earlier draft of this ADR recommended "add
> FP32/Q8_0/Q4_K/F16/BF16 to `QuantizationType`" as a cheap win. That is
> **wrong** — `QuantizationType` is explicitly BitNet-scoped by its doc
> comment and design. Conflating it with GGUF/dense formats would corrupt a
> narrow type. The target introduces separate concepts (`OperatorKind`,
> `ElementType`, `WeightEncoding`, `TensorLayout`, `ProviderId`,
> `DeviceId`, `KernelId`, `ExecutionSignature`) so the router has one DRY
> vocabulary without fusing BitNet-specific and format-generic concerns.

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

Five planes, each owning a distinct concern. Today the six surfaces
conflate them. The target separates them.

| Plane | Owns | DRY or SRP? |
|---|---|---|
| **Product/control** | RequestedRoute, profile, support policy, strictness | DRY (one route identity) |
| **Compute contract** | Operator + tensor/weight format + shape + dtype + policy; exact provider capabilities and selection | DRY vocabulary (OperatorKind, ElementType, WeightEncoding, TensorLayout, ProviderId, DeviceId, KernelId, ExecutionSignature) |
| **Data** | Format-specific executors: I2S/TL1/TL2 \| QK256 \| dense Q8/Q4 \| FP16/BF16/FP32 across CPU \| CUDA \| OpenCL \| Metal \| wgpu \| ROCm \| NPU | **SRP** per format; DRY at provider identity |
| **Accelerator-resource (optional)** | Context, buffers, queues, programs, kernels, events, copies | DRY when a forcing function fires; deferred otherwise |
| **Proof** | Provider/kernel identity, fallback, transfers, residency, timing, parity, support tier, claim boundary | DRY (receipt fragments) |

```
┌─────────────────────────────────────────────────────────────┐
│ Product/control plane                                       │
│ RequestedRoute, profile, support policy, strictness         │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Compute contract plane                                      │
│ Operator + tensor/weight format + shape + dtype + policy    │
│ Exact provider capabilities and selection                   │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Data plane                                                  │
│ Format-specific executors                                   │
│ I2S/TL1/TL2 | QK256 | dense Q8/Q4 | FP16/BF16/FP32          │
│ CPU | CUDA | OpenCL | Metal | wgpu | ROCm | NPU             │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Optional accelerator-resource plane                         │
│ Context, buffers, queues, programs, kernels, events, copies │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Proof plane                                                 │
│ Provider/kernel identity, fallback, transfers, residency,   │
│ timing, parity, support tier, and claim boundary             │
└─────────────────────────────────────────────────────────────┘
```

**DRY responsibilities** (one shared definition): operator identity,
provider identity, device identity, capability signatures, selection
outcomes, fallback semantics, kernel identity, transfer/residency
accounting, receipt fragments, support/proof status.

**SRP responsibilities** (remain modular): I2S/TL1/TL2 math, QK256
packing/scaling, GGUF Q8/Q4 layouts, dense FP16/BF16 paths, CUDA/OpenCL/
Metal resources and kernels, model composition, tokenization, server.

**Important type correction:** do **not** add FP32/Q8_0/Q4_K/F16/BF16 to
the current `QuantizationType`. That enum is explicitly BitNet-scoped
(`"Quantization types supported by BitNet"`, variants I2S/TL1/TL2). The
target introduces separate vocabulary (`WeightEncoding`, `ElementType`,
`OperatorKind`, etc.) so the router has one DRY vocabulary without fusing
BitNet-specific and format-generic concerns. Example:

```text
operator        = linear
weight_encoding = bitnet_qk256   | gguf_q8_0 | dense_fp32
activation_type = f32
provider        = cuda           | cpu_avx2  | opencl_a770
kernel          = cuda_qk256_gemv_v2 | dense_q8_sidecar_avx2
```

## Per-surface dispositions (current state, while target is deferred)

| Surface | Layer | Disposition | Rationale |
|---|---|---|---|
| A `KernelProvider` | Format exec | **Keep, load-bearing.** The operator×provider axis for BitNet weight formats. | Works, wired, narrow but correct for its scope. |
| B `BackendProvider` | Selection policy | **Keep, label honestly.** It is a selection-policy prototype, not an executor. Its `Operation` enum informs the target compute-contract plane. Has no execute method. | Real selection logic; over-characterizing it as a "parallel dispatch trait" is incorrect. |
| C `qk256-dispatch` | Format exec | **Keep, document as debt.** Each new backend adds a cfg branch; this is the unification trigger. | Works; cfg-accretion is the measurable refactor signal. |
| D `GpuBackend` (gpu-hal) | Accelerator-resource prototype | **Reference corpus, per ADR-0003.** Adoption-by-extraction only; not wholesale-integrated, not frozen forever. Candidate for the accelerator-resource plane when a forcing function fires. | Prototype trait needs v2 hardening before real adoption; no forcing function today. |
| E `DenseLinear` | Format exec | **Keep, priority target when funded.** The cross-model gap — load-bearing for SLM answers but cannot inherit any SIMD/GPU work. | Currently CPU-scalar-only; provider-ization is the highest-value unification work. |
| F `inference::Backend` | Full-model executor | **Keep, label the mock paths honestly.** GPU/NPU paths create shape-preserving mock tensors (`backends.rs:360`). A convergence pilot must prove real compute occurred, not just that the trait is wired. | Legitimate full-model boundary; but the mock paths must not be hidden under a new HAL. |

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

Slices that move code toward the target plane-separation without a
refactor, each independently mergeable in 1–3 PRs:

1. **Generated gpu-hal module inventory (`xtask`).** Replaces manually
   maintained counts (156 files, 26 undeclared, 75 stale headers) with
   generated evidence: declared/undeclared source files, public items,
   internal module edges, inbound workspace deps, external runtime deps,
   test files by evidence class, mock/reference headers. This is the
   highest-value tooling deliverable — it prevents another
   interpretation-by-grep cycle.
2. **Define the shared compute-contract vocabulary** (`OperatorKind`,
   `WeightEncoding`, `ElementType`, `ProviderId`, `DeviceId`, `KernelId`,
   `ExecutionSignature`) in `bitnet-common` as new types — **without**
   touching the BitNet-scoped `QuantizationType`. The router can then use
   one DRY vocabulary across all six surfaces without fusing BitNet and
   format-generic concerns.
3. **Extract `HalError` concepts into `bitnet-common`, split by concern.**
   Runtime/device errors separate from execution/validation errors. Gives
   all surfaces a shared taxonomy. Zero risk to existing routes.
4. **Give Surface E (`DenseLinear`) a provider seam.** Add an optional
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
