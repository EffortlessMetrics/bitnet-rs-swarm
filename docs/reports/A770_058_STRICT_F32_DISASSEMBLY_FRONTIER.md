# A770-058 Strict F32 Disassembly Frontier

## Scope

A770-058 is a diagnostic-only follow-on to A770-057. A770-057 captured
selected-device Intel Arc A770 OpenCL compiler disassembly for the diagnostic
QK256 debug kernel and left the lowered strict-f32 barrier operation sequence
unclassified.

This slice consumes the committed `.asm` artifact and classifies the lowered
operation sequence. It does not change production QK256 dispatch, runtime math,
answer scoring, sampling, model loading, route policy, or any A770 quality,
residency, speed, trusted partial-acceleration, or full-inference claim.

## Receipt

The new receipt is written by:

```text
cargo run --locked -p bitnet-kernels --no-default-features --bin a770-opencl-strict-f32-disassembly-frontier -- --asm ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv_debug.asm --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-strict-f32-disassembly-frontier.json
```

The receipt records:

```text
proof_family = a770_opencl_qk256_strict_f32_disassembly_frontier
proof_stage = diagnostic_strict_f32_barrier_sequence_classified
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
fallback_used = false
cpu_fallback_allowed = false
bitnet_inference = false
qk256_decode = false
production_qk256_policy_change = false
claim_allowed = false
diagnostic_only = true
```

Rows include the assembly path, assembly byte/line count, assembly hash,
compact operation counts, strict-f32 store/load-sequence detection, finite
guard-sequence detection, and compact evidence lines.

## Classification

```text
a770_qk256_strict_f32_disassembly_frontier_missing_context
a770_qk256_strict_f32_disassembly_frontier_barrier_preserving_f32_sequence
a770_qk256_strict_f32_disassembly_frontier_compiler_runtime_reassociation_or_collapse
a770_qk256_strict_f32_disassembly_frontier_unrecognized_lowered_sequence
```

## Live Result

The committed A770-057 assembly classifies as:

```text
classification =
  a770_qk256_strict_f32_disassembly_frontier_barrier_preserving_f32_sequence

asm_path =
  ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv_debug.asm

asm_bytes = 33135
asm_lines = 390
asm_fnv1a64 = a0eaab939147c32a
f32_mul_count = 27
f32_mov_count = 23
direct_div_count = 0
ugm_d32_store_count = 6
ugm_d32_load_count = 6
strict_f32_barrier_store_load_sequence = true
finite_guard_sequence_present = true
```

Interpretation: the lowered assembly does not expose a direct `div`
instruction for the debug expression, but it does preserve a compact f32
store/load sequence around the strict-f32 barrier context and includes the
finite-guard constants in the captured lowering. This is useful disassembly
context only. It does not justify changing production QK256 policy by itself.

The next diagnostic should inspect production-policy impact only after this
strict-f32 disassembly context is reviewed against the focused QK256 replay
frontier.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-kernels/src/bin/a770_opencl_strict_f32_disassembly_frontier.rs
cargo check --locked -p bitnet-kernels --no-default-features --bin a770-opencl-strict-f32-disassembly-frontier
cargo run --locked -p bitnet-kernels --no-default-features --bin a770-opencl-strict-f32-disassembly-frontier -- --asm ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv_debug.asm --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-strict-f32-disassembly-frontier.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-strict-f32-disassembly-frontier.json -Raw | ConvertFrom-Json | Out-Null"
```

The Windows `cargo test --locked -p bitnet-kernels --no-default-features --bin
a770-opencl-strict-f32-disassembly-frontier -- --nocapture` path timed out
before reaching the focused Rust helper tests because the package test profile
entered the C/C++ dev-dependency build path. The same Windows limitation was
already recorded for A770-057. The non-test binary check and live receipt
generation completed successfully.

## Claim Boundary

This report makes no new claim:

- no production QK256 dispatch change;
- no answer scoring or sampling change;
- no CPU/A770 answer parity claim;
- no reference parity claim;
- no strict A770 answer-readiness claim;
- no broad A770 quality claim;
- no selected attention, resident KV, attention score, softmax, value-mix, or
  full-residency claim;
- no speedup or trusted partial-acceleration claim;
- no full BitNet inference claim.
