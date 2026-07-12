# `bitnet-gpu-hal` — Design Reference

> **Authority:** [ADR-0003](../adr/0003-gpu-hal-disposition.md) (disposition),
> [BITNET-PROP-0019](../proposals/BITNET-PROP-0019-gpu-hal-disposition.md),
> [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER](../specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md).
>
> **Purpose:** This is the canonical human-readable reference for
> `crates/bitnet-gpu-hal/`. If you are an agent or contributor wondering
> "what is this 188K-LOC crate, and should I touch it?" — read this, then
> the ADR. Do not re-investigate from scratch.

## TL;DR

`bitnet-gpu-hal` is the **upper abstraction layer of a two-layer
multi-backend GPU plan** (`docs/reference/dual-backend-roadmap.md`,
origin 2025-11-03). The lower layer is `bitnet-kernels` (origin
2025-08-01), which is load-bearing and consumed by 9 crates. gpu-hal
landed on 2026-02-28 with its CPU reference/mock phase complete, but its
**integration phase** (real backends consuming the HAL traits) never
started. Because the lower layer was already wired and load-bearing, each
new capability got implemented in `bitnet-kernels` directly rather than
routed through the upper HAL — so today there is duplication, not because
one path "won" but because both layers advanced independently and only the
lower layer had live consumers.

The disposition (ADR-0003) is: **retain as a prototype corpus; adoption
by verified extraction or adapter, not wholesale integration and not
permanent freezing.** The crate predates the repo's current contract
conventions (which landed 2026-05-05 onward, ~2.5 months later), so the
absence of a proposal/campaign at landing time is a temporal artifact,
not a governance failure. The roadmap's phase numbering is internally
inconsistent and should not be treated as authoritative (see ADR-0003
§Context). Useful pieces may be extracted one at a time with evidence;
reopening the overall disposition requires a superseding ADR.

> **Note on the phase numbering:** an earlier version of this doc called
> gpu-hal the "Phase 8 reference layer awaiting Phase 9." The roadmap is
> internally inconsistent (it places gpu-hal under Phase 10 while listing
> Phase 9 as still-planned afterward), and at least one cited PR mapping
> is wrong (PR #1165, claimed as the "HAL abstraction" landing, is a
> closed unmerged rustdoc PR). The phase documents are historical intent,
> not a trustworthy current architecture contract. See ADR-0003 and
> BITNET-ADR-0010 for the verified current state.

## How it got here — and what it actually is

gpu-hal is **not** a stray or abandoned experiment. It is the **upper
abstraction layer of a two-layer multi-backend GPU plan** documented in
`docs/reference/dual-backend-roadmap.md` (origin 2025-11-03, ~3 months
before the crate landed).

The two layers:

- **Lower layer — `bitnet-kernels` (origin 2025-08-01).** The kernel
  implementations + CPU reference + per-backend dispatch. Already
  load-bearing when gpu-hal landed; consumed by 9 crates today (inference,
  cli, server, wasm, quantization, receipts, qk256-dispatch,
  device-config-core, kernels-self). Owns its own dispatch abstraction
  (`DispatchBackend` enum, `src/dispatch_planner.rs`) and its own parity
  infrastructure (`gpu_quantization_parity.rs`, since 2025-09-01 — predates
  gpu-hal entirely).
- **Upper layer — `bitnet-gpu-hal` (origin 2026-02-28).** A unified
  abstraction meant to sit *above* the per-backend kernels: one
  `KernelDispatcher` / `GpuBackend` trait surface across 8 backends, with
  backend-agnostic memory pools, multi-device scheduler, model sharding.
  The roadmap places this in its Phase 10 (though the phase numbering is
  inconsistent — see ADR-0003). Backend modules are API-shape CPU mocks
  (`CUDAKernel::launch()` body is `self.launch_count += 1`) — not a
  numerical compute path.

### Timeline

| When | What |
|---|---|
| 2025-08-01 | `bitnet-kernels` lower layer established (origin commit) |
| 2025-09-01 | `gpu_quantization_parity.rs` — kernels' own GPU/CPU parity infra |
| 2025-11-03 | `dual-backend-roadmap.md` introduces the two-layer plan |
| **2026-02-28** | **The 504-commit burst.** gpu-hal upper layer lands alongside `bitnet-kernels/src/cuda/`, `bitnet-opencl`, ADR-0002 (feature-flag strategy), and ~200 other PRs. Copilot co-authored. |
| 2026-03-02 → 2026-03-04 | Lower layer keeps growing in the days after: `dispatch_planner`, `neon_speculative_decoding`, `neon_model_sharding`, `avx2_*_parity_tests` |
| 2026-03-06 → 2026-05-05 | gpu-hal goes quiet (~2 months zero substantive commits) |
| 2026-05-05 onward | **The contract conventions land** — first campaign `active.toml`, `AGENTS.md` (05-10), first `BITNET-PROP` (05-12), `SPEC_SYSTEM.md` (05-20) |
| 2026-05-20 | Source→swarm sync mechanically touches 16 gpu-hal test files |
| 2026-05-21 | Last gpu-hal commit (`c46060531`, trivial lowercase precompute) |
| 2026-06-21 | This design doc + ADR-0003 bring gpu-hal inside the contract system |

### Why it has no dependents — the integration phase never started

The roadmap's Phase 9 (real backends consuming the HAL traits) never
started. No wiring PR, no campaign, no handoff scoped it. The lower layer
(kernels) kept growing because it was already wired and load-bearing, so
each new capability (speculative decoding, sharding, dispatch) got
implemented in kernels directly rather than routing through the upper HAL
layer. That's why there is duplication today — not because one path "won"
but because both layers of the plan advanced independently and only the
lower layer had live consumers.

### Important: the contract gap is a temporal artifact, not a failure

The crate predates the conventions it now lacks. First campaign `active.toml`
(2026-05-05), `AGENTS.md` (2026-05-10), first `BITNET-PROP` (2026-05-12),
`SPEC_SYSTEM.md` (2026-05-20) all landed ~2.5 months *after* gpu-hal. So
"no proposal / no campaign owner at landing" is anachronistic. The accurate
statement: the contract system post-dates the crate, and this disposition
(ADR-0003) is what brings the crate inside it.

### Origin attribution

148 commits, all co-authored by Copilot (attributed to
`Steven Zimmerman, CPA`), in the same 2026-02-28 burst that landed
`bitnet-kernels/src/cuda/`, `bitnet-opencl`, and ~200 other PRs. The
Copilot `.copilot/notes/` from 2026-03-01 are Apple-Silicon-team session
notes and do not mention gpu-hal specifically — but they document the same
orchestration campaign (parallel agent dispatch, worktrees, ~740 branches)
that produced the whole burst.

## The parity table (why the tooling potential is limited)

Over the ~4 months the crate sat quiet, the real GPU path independently
grew every capability the HAL targeted — generally more rigorously.

| Capability | `bitnet-gpu-hal` | Real path (`bitnet-kernels`) | Verdict |
|---|---|---|---|
| CPU reference matmul | `matmul_kernels.rs` (real O(n³)) | `src/cpu/cache_matmul.rs`, `matrix_ops.rs`, `quantized_matmul.rs` | Real path has it |
| SIMD-vs-scalar parity | (mocks don't compute) | `tests/avx2_matmul_cache_parity_tests.rs` @ `MATMUL_ABS_TOL=1e-4` + 8 more `avx2_*_parity_tests.rs` | Real path has it, rigorously |
| GPU-vs-CPU parity | (mocks don't compute) | `tests/gpu_quantization_parity.rs` (gated `feature="gpu"`) | Real path has it |
| Speculative decoding | HAL module (CPU ref) | `src/cpu/neon_speculative_decoding.rs` (real NEON, 341+ LOC) | Real path has it, better |
| Model sharding | `shard_planner.rs` | `src/cpu/neon_model_sharding.rs` | Real path has it |
| Tensor/pipeline parallelism | `multi_device.rs`, `tensor_parallel_v2` (orphan) | `src/cpu/tensor_parallel.rs`, `pipeline_parallel.rs`, `simd_tensor_parallel.rs`, `neon_tensor_parallel.rs` | Real path has it, 4 variants |
| Autotuner | `kernel_autotuner.rs` (mock) | `src/kernel_select.rs`, `kernel_selection.rs` | Real path has it |
| Dispatch abstraction | `KernelDispatcher` (claimed) | `src/dispatch_planner.rs` `DispatchBackend` + `bitnet-opencl/backend_dispatcher.rs` | Real path has two |

**Reading this table correctly.** This is not "the real path won and
gpu-hal lost." Both layers of the same plan advanced independently after
the 2026-02-28 burst. The lower layer (`bitnet-kernels`) had live
consumers, so each capability got implemented there directly; the upper
layer (gpu-hal) had its CPU reference/mock version but no integration
phase to route through. The practical consequence: the HAL does not offer
a capability the lower layer lacks, and the lower layer's versions are
generally more rigorous (real parity tolerances, real NEON/SIMD paths).
So the most-hyped tooling play — "use the HAL's CPU mocks as a golden
parity oracle for GPU kernels" — is both redundant (the lower layer
already has parity tests) and not buildable as-is (the HAL's mocks don't
compute; see next section).

### Why the backend mocks are not a parity oracle

The single most important content fact: the backend mocks do not execute
kernels. `CUDAKernel::launch()` body (`cuda_backend.rs:638`):

```rust
self.validate(device)?;
stream.record_work();
self.launch_count += 1;
Ok(())
```

`vulkan_compute.rs:707` pushes a `RecordedCommand` into a `Vec` that nothing
interprets. So the most-hyped play (CPU golden reference for GPU-kernel
parity) is not buildable on the HAL as-is — there is no numerical output to
compare against. The real CPU math that does exist (`matmul_kernels.rs`) is
not wired to any `.cl`/`.cu` source loader.

## What is coherent and valuable (the retained core)

The genuinely valuable, retained part of the crate is small:

- **`hal_traits.rs`** (1,633 LOC) — 8 traits (`GpuDevice`, `GpuBuffer`,
  `GpuKernel`, `GpuQueue`, `GpuProgram`, `GpuEvent`, `GpuContext`,
  `GpuBackend`, `GpuMemoryAllocator`) plus `HalError` (12 variants),
  `MemoryType`, `ComputeCapabilities`, `ProgramSource`. Clean, documented,
  readable glossary of GPU HAL concepts. **This is the primary retained
  value.**
- **`tensor_serde.rs`** (2,580 LOC) — real binary/SafeTensors/NumPy/GGUF/JSON
  ser/de with hand-rolled SHA-256.
- **`model_cache.rs`** (one of the orphan files) — real LRU/LFU/FIFO disk
  cache, but currently undeclared in `lib.rs` so not compiled.
- **`matmul_kernels.rs`** — real CPU reference matmul, though the real path
  has equivalent or better.

These are referenced in the spec as the coherent core so any future
extraction proposal has the inventory.

## Known issues (recorded, not fixed by this disposition)

These are real findings worth knowing, but are explicitly **out of scope**
for the disposition lane. A future campaign may pursue them.

1. **26 undeclared orphan files** — present in `src/` but not in `lib.rs`'s
   `pub mod` list, so never compiled: `async_exec`, `batch_engine`,
   `config_validator`, `model_cache`, `model_lifecycle`, `openai_compat`,
   `performance_monitor`, `prompt_template`, `rate_limiting`,
   `safety_guardrails`, `sliding_window`, `tokenizer_detokenizer`,
   `shard_planner`, `runtime_config`, `power`, `health`, `graceful_shutdown`,
   `graph_executor`, `inference_profiler`, `mmap_loader`, `format_converter`,
   `data_pipeline`, `rope_variants`, `config_reload`, `attention_analyzer`,
   `tensor_parallel_v2`.
2. **75 misleading headers** — files carry the verbatim fictional header
   `//! Module stub - implementation pending merge from feature branch`
   despite holding 150-200 real functions each. No such feature branch
   exists in git history.
3. **Internally disconnected** — only 3 of 148 files reference a sibling via
   `use crate::` (all just `use crate::HalError;`). The roadmap's
   `KernelDispatcher` is described as routing across modules but the modules
   don't connect.
4. **Scope drift** — modules unrelated to a GPU HAL: `semantic_search`,
   `rate_limiter`, `safety_guardrails`, `instruction_tuning`,
   `prompt_template`, `api_gateway`, `api_server`, `docker_ci`,
   `model_pruning`, `openai_compat`, `tokenizer_detokenizer`. Per
   [BITNET-SPEC-GPU-HAL-REFERENCE-LAYER REQ-005](../specs/BITNET-SPEC-GPU-HAL-REFERENCE-LAYER.md),
   these are frozen and must not be extended.
5. **Maintenance cost** — ~4,668 unwrap/expect sites in the no-panic
   baseline; ~150 blanket `#[allow]` suppressions. Accepted per ADR-0003
   consequences as the cost of retention.

## Forward guidance

### What you may do without a new ADR

- Read the crate as a reference, especially `hal_traits.rs`.
- Mechanical sync touch-ups and lint fixes that come from source->swarm syncs.
- Point new contributors at this doc and the ADR.
- Close [#1639](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1639)
  with a pointer to ADR-0003.

### What requires a superseding ADR

- Adding `bitnet-gpu-hal` as a dependency of any other crate.
- Extending any drift module with new feature work.
- Changing the public API surface in `lib.rs`.
- Pursuing backend trait unification (the "Play C" strategic option).
- Extracting the salvageable core into a smaller crate.
- Deleting the crate.

### Strategic options left open by this disposition

- **Backend trait unification.** Harden `hal_traits` for real backends
  (zero-copy buffers, explicit queue on `launch`, async model), then migrate
  `bitnet-kernels`/`opencl`/`inference` onto it over 30-50 PRs. Real
  multi-quarter commitment. Must arrive as its own proposal + superseding
  ADR.
- **Triage + extract.** Strip to the ~15-20K LOC coherent core and delete
  ~170K LOC of drift. Must arrive as its own proposal + superseding ADR.
  The coherent-core inventory above is the starting map.

## Provenance

Every architectural claim here is reproducible from the repo at commit
`3a43daa3b`. Key commands:

```bash
# Scale
find crates/bitnet-gpu-hal/src -name '*.rs' | wc -l        # 148
wc -l crates/bitnet-gpu-hal/src/*.rs | tail -1             # 188392 total

# Orphan status
grep -l bitnet-gpu-hal crates/*/Cargo.toml | grep -v bitnet-gpu-hal   # (empty)

# Mocks don't compute
sed -n '638p' crates/bitnet-gpu-hal/src/cuda_backend.rs    # self.launch_count += 1

# Real path parity
ls crates/bitnet-kernels/tests/ | grep parity              # 12 files
ls crates/bitnet-kernels/src/cpu/ | grep -E 'speculat|shard|parallel'

# Governance gap
grep -rn 'gpu.hal\|bitnet-gpu-hal' docs/adr/               # (empty before ADR-0003)
git log --reverse --format='%h %ad %s' --date=short -- crates/bitnet-gpu-hal/ | head -3
```

## Change history

- 2026-06-21: Created as part of the `gpu-hal-disposition` lane
  (BITNET-PROP-0019, ADR-0003, BITNET-SPEC-GPU-HAL-REFERENCE-LAYER).
