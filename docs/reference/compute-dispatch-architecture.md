# Compute Dispatch Architecture — Reference

> **Authority:** [BITNET-ADR-0010](../adr/BITNET-ADR-0010-compute-dispatch-architecture.md).
>
> **Purpose:** Canonical human-readable reference for how compute dispatch
> is structured in BitNet-rs today, why it fragmented, and where it intends
> to converge. If you are adding a backend, a weight format, a model, or an
> operator — read this first.

## TL;DR

There are **six** compute-dispatch surfaces across five architectural
layers in the repo. The layers reuse the same vocabulary — *backend,
dispatch, provider, device* — without clearly distinguishing scope, and
several surfaces duplicate the same control-plane types. The repo intends
to converge toward a five-plane separation (control / contract / data /
accelerator-resource / proof), but no refactor is committed until a
forcing function fires (a new backend that adds cfg-branches, a feature
needing GPU-resident buffers, or a dense-path perf need).

The single biggest gap: **the dense SLM path (Qwen/Phi/Gemma) — the one
producing validated coherent answers — is CPU-scalar-only and cannot use
any of the AVX2/NEON/CUDA/OpenCL work.**

The second-biggest gap: **the full-model `inference::Backend` trait's GPU
and NPU paths are shape-preserving mocks** — a convergence pilot must prove
real compute occurred, not just that a trait is wired.

## The six surfaces across five layers today

| # | Surface | Layer | Where | Role | Wired to inference? |
|---|---|---|---|---|---|
| A | `KernelProvider` | Format exec | `bitnet-kernels/src/lib.rs:166` | 2 ops (matmul_i2s, quantize), BitNet formats (I2S/TL1/TL2), 7+ impls | **Yes** — `quantized_linear.rs` at 5 sites |
| B | `BackendProvider` | **Selection policy (no execution)** | `bitnet-opencl/src/backend_*.rs` | Reports name/status/capabilities/priority. **No execute method** — pure selection + decision recording | **No** — siloed |
| C | `qk256-dispatch` | Format exec | dedicated crate | QK256 linear via cfg branches | **Yes** — via cfg |
| D | `GpuBackend` HAL | Accelerator-resource prototype | `bitnet-gpu-hal/src/hal_traits.rs` | Device/buffer/kernel/queue/context. CPU mocks only | **No** — zero consumers |
| E | `DenseLinear` | Format exec | `bitnet-inference/src/dense_forward.rs:56` | FP32 linear/attention/FFN, f64 scalar loop | **YES — Qwen/Phi/Gemma** |
| F | `inference::Backend` | Full-model executor | `bitnet-inference/src/backends.rs:20` | Runs complete model forward. **GPU/NPU paths are shape-preserving mocks** (`backends.rs:360`) | **Yes** — model boundary |

### Surface A — `KernelProvider` (the load-bearing BitNet dispatch)

```rust
// crates/bitnet-kernels/src/lib.rs:166
pub trait KernelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn matmul_i2s(&self, a:&[i8], b:&[u8], c:&mut [f32], m:usize, n:usize, k:usize) -> Result<()>;
    fn quantize(&self, input:&[f32], output:&mut [u8], scales:&mut [f32], qtype:QuantizationType) -> Result<()>;
}
```

Narrow — just the BitNet quantized-linear op pair. Runtime-polymorphic via
`Vec<Box<dyn KernelProvider>>` in `KernelManager` with priority-ordered
selection (GPU first, SIMD CPU next, FFI last). Seven-plus real
implementations. **This is the trait `bitnet-inference` actually calls.**

Strength: working, wired, correct for its narrow scope.
Limitation: only 2 ops, only BitNet weight formats, slice-based memory
model (no GPU buffer type).

### Surface B — `BackendProvider` (the OpenCL silo)

```rust
// crates/bitnet-opencl/src/backend_registry.rs:10
pub trait BackendProvider: Send + Sync {
    fn name(&self) -> &str;
    fn status(&self) -> BackendStatus;        // Available | Unavailable | Degraded
    fn capabilities(&self) -> Vec<Operation>;  // 8 ops
    fn priority_score(&self) -> u32;
}
// crates/bitnet-opencl/src/backend_dispatcher.rs:16
pub enum Operation { MatMul, Quantize, Dequantize, Softmax, LayerNorm, Attention, RoPE, Sampling }
pub enum DispatchStrategy { Priority, RoundRobin, LoadBased, SpecificBackend(String) }
```

More sophisticated *operation vocabulary* than A (8 ops with capability
advertisement and 4 dispatch strategies), but **it is selection policy,
not execution.** The trait has only `name`/`status`/`capabilities`/
`priority_score` — no `execute`/`matmul`/`compute` method. The
`BackendDispatcher` has `backends_for`/`is_supported`/`record`/`strategy`.
It chooses a backend and records the decision; it does not submit the
selected operation. Calling it a "parallel dispatch trait" (as an earlier
draft of this doc did) overstates it.

Strength: the `Operation` enum (8 ops) informs the target compute-contract
plane's operator vocabulary.
Limitation: siloed, metadata-only — not an executor.

### Surface C — `qk256-dispatch` (the cfg-branch router)

```rust
// crates/bitnet-qk256-dispatch/src/lib.rs
#[cfg(feature = "cuda")]     // CUDA QK256 kernel path
#[cfg(feature = "opencl")]   // A770 OpenCL QK256 path
// fallback by strict-mode policy
```

Not a trait — a `match`/`cfg` router dedicated to the QK256 BitNet variant.
Grows by accretion: each new backend adds a branch. This is the most
visible symptom of the fragmentation — it's where the cfg-branch cost
becomes measurable.

### Surface D — `GpuBackend` HAL (the reference layer)

Full HAL trait family (`GpuDevice`, `GpuBuffer`, `GpuKernel`, `GpuQueue`,
`GpuProgram`, `GpuEvent`, `GpuContext`, `GpuBackend`, `GpuMemoryAllocator`)
in `bitnet-gpu-hal/src/hal_traits.rs`. The most ambitious surface — but
mock-only (`CUDAKernel::launch()` body is `self.launch_count += 1`) and
zero-consumer. Disposition recorded in [ADR-0003](../adr/0003-gpu-hal-disposition.md):
retained as reference, candidate for the future HAL-plumbing axis.

Limitation for real adoption: `GpuBuffer::write(&[u8])` / `read() -> Vec<u8>`
has no zero-copy path; `launch()` has no queue/stream parameter; sync not
async. Designed around mock ergonomics, not real-backend constraints.

### Surface E — `DenseLinear` (the cross-model gap)

```rust
// crates/bitnet-inference/src/dense_forward.rs:56
pub struct DenseLinear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}
// forward_into: f64-accumulated triple loop, CPU scalar
```

The dense SLM path. Predates the others, structurally simplest. **This is
the path producing validated coherent Qwen/Phi/Gemma answers today.** It
is CPU-scalar-only with f64 accumulation and zero connection to
`KernelProvider`.

The kicker: `QuantizationType` (`bitnet-common/src/types.rs:7`) has 3
variants — `I2S`, `TL1`, `TL2`, all BitNet. None of the GGUF quant
formats are in it. The engine recognizes 12 GGUF quant format names
(`q8_0`, `q4_k`, `q5_0`, `q2_k`, ...) as **strings** at `engine.rs:485-495`
and routes them into `DenseLinear`, where dequantized FP32 weights get
multiplied in the scalar loop. **So the working SLM path cannot inherit any
SIMD/GPU work.**

> **Type-system note:** `QuantizationType` is doc-commented *"Quantization
> types supported by BitNet"* and is BitNet-scoped by design (variants
> I2S/TL1/TL2). Do **not** add FP32/Q8_0/Q4_K/F16/BF16 to it as a quick
> fix — that would corrupt a narrow type. The target introduces separate
> vocabulary (`OperatorKind`, `WeightEncoding`, `ElementType`,
> `ProviderId`, `DeviceId`, `KernelId`, `ExecutionSignature`) so the router
> has one DRY vocabulary without fusing BitNet-specific and format-generic
> concerns.

### Surface F — `inference::Backend` (the full-model executor with mock GPU)

```rust
// crates/bitnet-inference/src/backends.rs:20
pub trait Backend: Send + Sync { ... }
// backends.rs:360 — GPU path
// For now, just create a mock GPU tensor
// backends.rs:193 — NPU path
// "NPU tensor transfer unavailable, using shape-preserving tensor fallback"
```

The full-model executor trait — a legitimate model-level boundary. But its
GPU and NPU paths are **simulated**: they create shape-preserving mock
tensors instead of transferring to a real accelerator. A convergence pilot
that wires a new HAL underneath this trait without verifying real compute
occurred would be a hollow victory. This surface must be included in any
dispatch-architecture analysis.

Strength: correct layer (full-model forward); the CPU/CUDA-via-KernelProvider
paths are real.
Limitation: GPU/NPU paths hide whether real compute happened.

## The five planes (the target decomposition)

The six surfaces conflate concerns across five planes. Properly separated:

| Plane | Owns | DRY or SRP? |
|---|---|---|
| **Product/control** | RequestedRoute, profile, support policy, strictness | DRY (one route identity) |
| **Compute contract** | Operator + tensor/weight format + shape + dtype + policy; exact provider capabilities and selection | DRY vocabulary (OperatorKind, WeightEncoding, ElementType, ProviderId, DeviceId, KernelId, ExecutionSignature) |
| **Data** | Format-specific executors across providers | **SRP** per format; DRY at provider identity |
| **Accelerator-resource (optional)** | Context, buffers, queues, programs, kernels, events, copies | DRY when a forcing function fires; deferred otherwise |
| **Proof** | Provider/kernel identity, fallback, transfers, residency, timing, parity, support tier, claim boundary | DRY (receipt fragments) |

**DRY responsibilities** (one shared definition): operator identity,
provider identity, device identity, capability signatures, selection
outcomes, fallback semantics, kernel identity, transfer/residency
accounting, receipt fragments, support/proof status.

**SRP responsibilities** (remain modular): I2S/TL1/TL2 math, QK256
packing/scaling, GGUF Q8/Q4 layouts, dense FP16/BF16 paths, CUDA/OpenCL/
Metal resources and kernels, model composition, tokenization, server.

**No fake unification.** If two things aren't actually related (e.g.,
gpu-hal's HAL plumbing vs BitNet's GEMV provider), they stay separate.

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

Today's surfaces map onto these planes imperfectly: `KernelProvider` (A)
is data-plane exec for BitNet formats; `BackendProvider` (B) is a
selection-policy prototype for the compute-contract plane but has no
execution; `qk256-dispatch` (C) is data-plane exec for QK256; `GpuBackend`
(D) is an accelerator-resource prototype; `DenseLinear` (E) is data-plane
exec for dense FP32; `inference::Backend` (F) is the full-model executor
whose GPU/NPU paths simulate the accelerator-resource plane.

## DRY and SRP — applied

**DRY where genuinely the same:**
- The *operator* concept: every transformer has Linear/Attention/FFN/etc.
- The *linear forward shape*: activations × weights → activations, regardless of weight format.
- The *provider selection* logic: priority/availability/capability.

**SRP where genuinely different:**
- The *weight-format algorithms*: i2s GEMV ≠ QK256 GEMV ≠ FP32 GEMM ≠ Q8_0 dequantize+GEMM.
- The *model composition*: Qwen vs BitNet-2B vs Llama compose operators differently.
- The *HAL plumbing* (gpu-hal's `GpuBuffer`/`GpuQueue`): device-level resource management, only relevant if you need GPU-resident buffers (BitNet GEMV doesn't today).

**No fake unification.** If two things aren't actually related (e.g.,
gpu-hal's HAL plumbing vs BitNet's GEMV provider), they stay separate.

## When does unification trigger?

Three forcing functions. Any one justifies a unification proposal +
superseding ADR:

| Trigger | Implies | Unification worth it? |
|---|---|---|
| Adding backend #6+ (Metal real kernels, Vulkan real) | another Surface-A impl + another Surface-C cfg branch | **Yes** — Option 1 (widen `KernelProvider`) around backend #5–6 |
| Need GPU-resident KV cache or fused kernels | buffer abstraction beyond slices | **Yes** — Option 2 (adopt hardened gpu-hal HAL) |
| Dense-path perf need (Qwen too slow on scalar) | provider-ize Surface E | **Yes** — provider-ize `DenseLinear` |
| Just adding more CPU SIMD variants | none — fits Surface A as-is | **No** — defer |
| Status quo (BitNet GEMV on host buffers) | none | **No** — defer |

The repo is near the threshold on (1) and (3). No forcing function has
fired on (2).

## Recommended cheap wins (low-risk slices toward the target)

Each independently mergeable in 1–3 PRs. None commits to the full
unification.

1. **Generated gpu-hal module inventory (`xtask`).** Replaces manually
   maintained counts (156 files, 26 undeclared, 75 stale headers) with
   generated evidence. Highest-value tooling deliverable — prevents another
   interpretation-by-grep cycle.
2. **Define the shared compute-contract vocabulary** (`OperatorKind`,
   `WeightEncoding`, `ElementType`, `ProviderId`, `DeviceId`, `KernelId`,
   `ExecutionSignature`) in `bitnet-common` as new types — **without**
   touching the BitNet-scoped `QuantizationType`. The router can then use
   one DRY vocabulary across all six surfaces without fusing BitNet and
   format-generic concerns.
3. **Extract `HalError` concepts into `bitnet-common`, split by concern.**
   Runtime/device errors separate from execution/validation errors. Gives
   all six surfaces a shared taxonomy. Zero risk to existing routes.
4. **Give Surface E (`DenseLinear`) a provider seam.** Add an optional
   `provider` parameter (no-op for now) so a future migration is a trait
   change, not a surgery.

## Where gpu-hal fits (Surface D)

gpu-hal's `GpuBackend` trait family is the candidate for the **future
accelerator-resource plane** in the target diagram. It is retained as a
prototype corpus per [ADR-0003](../adr/0003-gpu-hal-disposition.md) — adoption by verified
extraction or adapter, not wholesale integration and not permanent
freezing. It is activated only when forcing function (2) fires (a feature
needing GPU-resident buffers). Until then:

- Its `hal_traits.rs` is a readable design input for a future HAL v2 (the
  current trait has real deficiencies — duplicated launch/submit,
  byte-only buffers, no dtype/shape/layout, raw-slice map — that must be
  fixed before real backends can implement it; see ADR-0003 §"hal_traits
  v2 issues").
- Its backend mocks are non-computing references.
- Its drift modules (`semantic_search`, `rate_limiter`, etc.) are frozen.
- Useful pieces (error taxonomy, CPU numerical references, profiling) may
  be extracted one at a time with evidence.
- New integration requires a superseding ADR.

See [`docs/reference/gpu-hal-design.md`](./gpu-hal-design.md) for the full
gpu-hal reference.

## Provenance

Every claim reproducible from the repo at `3a43daa3b`:

```bash
# Surface A trait + impls
sed -n '166,185p' crates/bitnet-kernels/src/lib.rs
grep -rn 'impl.*KernelProvider for' crates/ --include='*.rs'

# Surface B trait
sed -n '1,30p' crates/bitnet-opencl/src/backend_registry.rs
sed -n '1,60p' crates/bitnet-opencl/src/backend_dispatcher.rs

# Surface C cfg router
head -20 crates/bitnet-qk256-dispatch/src/lib.rs

# Surface D mocks-don't-compute
sed -n '638p' crates/bitnet-gpu-hal/src/cuda_backend.rs  # self.launch_count += 1

# Surface E + the gap
sed -n '56,125p' crates/bitnet-inference/src/dense_forward.rs
sed -n '7,14p'   crates/bitnet-common/src/types.rs         # 3 BitNet-only variants
sed -n '485,495p' crates/bitnet-inference/src/engine.rs    # string-match routing
```
