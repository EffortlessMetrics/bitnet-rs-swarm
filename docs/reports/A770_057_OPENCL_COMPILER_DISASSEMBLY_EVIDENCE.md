# A770-057 OpenCL Compiler Disassembly Evidence

## Scope

A770-057 is a diagnostic-only follow-on to A770-056. A770-056 captured the
selected-device Intel Arc A770 OpenCL executable program binary for the
diagnostic QK256 debug kernel and classified the remaining gap as missing
vendor/offline disassembly evidence.

This slice writes the driver-returned program binary to a receipt artifact,
runs Intel `ocloc disasm` against that binary, and records compact kernel
assembly metadata. It does not change production QK256 dispatch, runtime math,
answer scoring, sampling, model loading, or any A770 quality, residency, or
performance claim.

## Receipt

The new receipt is written by:

```text
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence -- --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly-evidence.json --ocloc "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin\ocloc.exe" --device dg2-g10
```

The receipt records:

```text
proof_family = a770_opencl_qk256_compiler_disassembly_evidence
proof_stage = diagnostic_compiler_disassembly_captured
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
options/log, binary type, binary size/hash/prefix, the written binary path,
`ocloc` path/device/command/result, dump directory, kernel assembly path,
kernel assembly size/hash/prefix, source hash, and strict-f32 source-context
flags.

## Classification

```text
a770_qk256_opencl_disassembly_evidence_missing_program_binary
a770_qk256_opencl_disassembly_evidence_missing_strict_f32_source_context
a770_qk256_opencl_disassembly_evidence_ocloc_missing
a770_qk256_opencl_disassembly_evidence_disasm_failed
a770_qk256_opencl_disassembly_evidence_kernel_asm_missing
a770_qk256_opencl_disassembly_evidence_captured
```

## Live Result

The selected A770 run produced:

```text
classification =
  a770_qk256_opencl_disassembly_evidence_captured

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

ocloc_path =
  C:/Program Files (x86)/Intel/oneAPI/ocloc/latest/bin/ocloc.exe
ocloc_device = dg2-g10
ocloc_exit_code = 0
ocloc_stdout = "Warnings: unexpected padding at end of kernel\n"

binary_path =
  ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly/qk256_i2s_i8s_scaled_gemv_debug.bin
kernel_asm_path =
  ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv_debug.asm
kernel_asm_bytes = 33135
kernel_asm_fnv1a64 = a0eaab939147c32a
kernel_asm_trailing_whitespace_trimmed = true
disassembly_captured = true
```

Interpretation: the selected Intel OpenCL driver exposes the executable program
binary, and the local Intel `ocloc` tool can dump compact kernel assembly for
the diagnostic QK256 debug kernel. This receipt does not yet interpret the
lowered strict-f32 barrier operation sequence, does not prove production QK256
policy correctness, and does not justify changing production runtime math.

The next diagnostic is to inspect the lowered strict-f32 barrier operation
sequence in the captured kernel assembly before any production QK256 policy
change.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-kernels/src/a770_opencl_runtime.rs crates/bitnet-kernels/src/bin/a770_opencl_disassembly_evidence.rs
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence -- --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly-evidence.json --ocloc "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin\ocloc.exe" --device dg2-g10
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-compiler-disassembly-evidence.json -Raw | ConvertFrom-Json | Out-Null"
cargo test --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence -- --nocapture
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

The focused `cargo test -p bitnet-kernels --bin
a770-opencl-disassembly-evidence` command passed under WSL. The same Windows
test-profile path was not completed because the package dev-dependency path
entered `sentencepiece-sys` Visual Studio CMake configuration and timed out
before reaching the focused helper tests. The Windows non-test-profile OpenCL
check and live selected-device disassembly receipt command completed
successfully.

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
