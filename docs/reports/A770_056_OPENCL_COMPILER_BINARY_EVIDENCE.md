# A770-056 OpenCL Compiler Binary Evidence

## Scope

A770-056 is a diagnostic-only follow-on to A770-055. A770-055 classified the
focused layer-0 `q_proj` QK256 strict-f32 boundary as a host compiler/codegen
collapse while the selected OpenCL device strict-f32 barrier expression still
matched the A770 output bit.

This slice captures selected-device OpenCL program-build and binary metadata
for the same diagnostic QK256 debug kernel. It records the driver-exposed
program binary by size, deterministic hash, and prefix only. It does not
disassemble the binary, change production QK256 dispatch, change runtime math,
or promote any inference, quality, residency, or performance claim.

## Receipt

The new receipt is written by:

```text
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-compiler-evidence -- --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-binary-evidence.json
```

The receipt records:

```text
proof_family = a770_opencl_qk256_compiler_binary_evidence
proof_stage = diagnostic_compiler_binary_captured
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

Rows include selected platform/device identity, driver version, OpenCL build
options/log, binary type, kernel names, program device count, binary sizes,
deterministic binary hashes, binary prefixes, source hash, source byte count,
and strict-f32 source-context flags.

## Classification

```text
a770_qk256_opencl_compiler_binary_evidence_missing_program_binary
a770_qk256_opencl_compiler_binary_evidence_missing_strict_f32_source_context
a770_qk256_opencl_compiler_binary_evidence_program_binary_captured_disassembly_missing
a770_qk256_opencl_compiler_binary_evidence_program_binary_and_disassembly_captured
a770_qk256_opencl_compiler_binary_evidence_disassembly_captured_missing_strict_f32_source_context
```

## Live Result

The selected A770 run produced:

```text
classification =
  a770_qk256_opencl_compiler_binary_evidence_program_binary_captured_disassembly_missing

runtime_device = Intel(R) Arc(TM) A770 Graphics
platform_name = Intel(R) OpenCL Graphics
vendor = Intel(R) Corporation
driver_version = 32.0.101.8801

binary_type = executable
kernel_names = qk256_i2s_i8s_scaled_gemv_debug
program_device_count = 1
binary_sizes = [18824]
binary_fnv1a64 = ["e5a3d44d991982f9"]
binary_prefix_hex =
  ["7f454c460201010001000000000000000100cd00010000000000000000000000"]
source_bytes = 2417
source_fnv1a64 = 1bc2a9662219c6f1
strict_f32_barrier_source_present = true
program_binary_captured = true
disassembly_captured = false
```

Interpretation: the selected Intel OpenCL driver exposes an executable program
binary for the diagnostic QK256 debug kernel, and the source context still
contains the strict-f32 barrier. This receipt does not include vendor
disassembly, so it cannot prove the exact lowered strict-f32 operation sequence
and does not justify a production QK256 policy change by itself.

The next diagnostic remains vendor/offline compiler disassembly evidence for
the strict-f32 barrier before any production QK256 policy change.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-kernels/src/a770_opencl_runtime.rs crates/bitnet-kernels/src/bin/a770_opencl_compiler_evidence.rs
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-compiler-evidence
cargo test --locked -p bitnet-kernels --no-default-features --features opencl --lib compiler_binary_evidence -- --nocapture
cargo test --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-compiler-evidence -- --nocapture
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-compiler-evidence -- --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-binary-evidence.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-binary-evidence.json -Raw | ConvertFrom-Json | Out-Null"
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

The focused `cargo test -p bitnet-kernels` commands and `xtask` campaign
commands passed under WSL. The same Windows test-profile and `xtask` Cargo
build paths were not completed because the package dev-dependency path entered
`sentencepiece-sys` Visual Studio CMake configuration and timed out before
reaching the focused A770 helper tests. The Windows non-test-profile OpenCL
check and live selected-device receipt command completed successfully.

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
