# Compute Dispatch Architecture — Reference

> **Authority:** [BITNET-ADR-0010](../adr/BITNET-ADR-0010-compute-dispatch-architecture.md).
>
> **Purpose:** Canonical human-readable reference for how compute dispatch
> is structured in BitNet-rs today, why it fragmented, and where it intends
> to converge. If you are adding a backend, a weight format, a model, or an
> operator — read this first.

## TL;DR

There are **five** compute-dispatch surfaces in the repo. They overlap, do
not compose, and conflate three concerns that are genuinely orthogonal:
**operator** (what's computed), **weight format** (how weights are laid
out), and **provider/device** (where it runs). The repo intends to
converge toward separating these axes, but no refactor is committed until a
forcing function fires (a new backend that adds cfg-branches, a feature
needing GPU-resident buffers, or a dense-path perf need).

The single biggest gap: **the dense SLM path (Qwen/Phi/Gemma) — the one
producing validated coherent answers — is CPU-scalar-only and cannot use
any of the AVX2/NEON/CUDA/OpenCL work.**

## The five surfaces today

| # | Surface | Where | Ops | Weight formats | Providers | Wired to inference? |
|---|---|---|---|---|---|---|
| A | `KernelProvider` | `bitnet-kernels/src/lib.rs:166` | matmul_i2s, quantize | I2S, TL1, TL2 (BitNet only) | CPU×4, CUDA, OpenCL, NPU, ROCm, FFI, DebugLayer | **Yes** — `quantized_linear.rs` at 5 sites |
| B | `BackendProvider` | `bitnet-opencl/src/backend_*.rs` | 8 ops | (OpenCL-internal) | OpenCL (A770) | **No** — siloed |
| C | `qk256-dispatch` | dedicated crate | QK256 linear | QK256 | cfg-branched | **Yes** — via cfg |
| D | `GpuBackend` HAL | `bitnet-gpu-hal/src/hal_traits.rs` | full HAL | n/a | CPU mocks only | **No** — zero consumers |
| E | `DenseLinear` | `bitnet-inference/src/dense_forward.rs:56` | linear, attention, FFN | FP32 only (12 GGUF quants routed here as FP32) | **CPU scalar, f64 loop** | **YES — Qwen/Phi/Gemma** |

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

More sophisticated than A: 8-operation capability advertisement, 4 dispatch
strategies, a registry pattern. But it lives inside the OpenCL crate and
inference never calls it. A parallel evolution that never bridged.

Strength: richer operation model that informs the target operator layer.
Limitation: siloed; duplicate effort.

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

## The three axes (the target decomposition)

Every surface conflates at least two of these. Properly separated:

| Axis | DRY or SRP? | Why |
|---|---|---|
| **Operator** (Linear, Attention, FFN, RMSNorm, RoPE, Sampling, Dequantize) | **DRY** | Every transformer has these, BitNet and dense alike |
| **Weight format** (I2S, TL1, TL2, QK256, FP32, Q8_0, Q4_K, F16, BF16) | **SRP** | Different math, different layouts; fusing bodies would be fake DRY |
| **Provider/device** (scalar, AVX2, AVX-512, NEON, CUDA, OpenCL, Metal, Vulkan, wgpu, NPU) | **DRY** | "Where does this run" is orthogonal to "what's computed" |

Today: `KernelProvider` bakes (operator=Linear, format=i2s, provider=any).
`DenseLinear` bakes (operator=Linear, format=FP32, provider=CPU-scalar).
`qk256-dispatch` bakes cfg-branches on provider. The target: **one operator
trait family × one module per weight format × one provider registry**.

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

1. **Add the missing weight formats to `QuantizationType`.** Today the
   enum is BitNet-only (`I2S`/`TL1`/`TL2`). Add `FP32`, `Q8_0`, `Q4_K`,
   `F16`, `BF16`, etc. Makes the weight-format axis explicit in the type
   system instead of hidden in string matching at `engine.rs:485-495`.
   SRP-correct: each format stays its own module, but becomes first-class.
2. **Extract `HalError` into `bitnet-common`.** Gives all five surfaces a
   shared GPU error taxonomy. Zero risk.
3. **Give Surface E (`DenseLinear`) a provider seam.** Add an optional
   `provider` parameter (no-op for now) so a future migration is a trait
   change, not a surgery.

## Where gpu-hal fits (Surface D)

gpu-hal's `GpuBackend` trait family is the candidate for the **future
HAL-plumbing axis** in the target diagram — the bottom-right box. It is
retained as a read-only reference per [ADR-0003](./gpu-hal-design.md) and
activated only when forcing function (2) fires (a feature needing
GPU-resident buffers). Until then:

- Its `hal_traits.rs` is a readable glossary of GPU HAL concepts.
- Its backend mocks are non-computing references (by design — Phase 10 CPU
  mock deliverable).
- Its drift modules (`semantic_search`, `rate_limiter`, etc.) are frozen.
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
