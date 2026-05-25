# A770-060 Production Kernel Disassembly Evidence

## Scope

A770-060 is a diagnostic-only follow-on to A770-059. A770-059 classified the
strict-f32 production-policy impact frontier as requiring production-kernel
disassembly or production-kernel replay context before any production QK256
policy change.

This slice captures selected-device Intel Arc A770 OpenCL compiler binary and
`ocloc` disassembly evidence for the production
`qk256_i2s_i8s_scaled_gemv` kernel. It does not change production QK256
dispatch, runtime math, answer scoring, sampling, model loading, route policy,
or any A770 quality, residency, speed, trusted partial-acceleration, or
full-inference claim.

## Receipt

The new receipt is written by:

```text
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence -- --kernel production --artifact-dir ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly-evidence.json --ocloc "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin\ocloc.exe" --device dg2-g10
```

The receipt records:

```text
proof_family = a770_opencl_qk256_production_kernel_disassembly_evidence
proof_stage = diagnostic_production_kernel_disassembly_context_captured
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
kernel_name = qk256_i2s_i8s_scaled_gemv
fallback_used = false
cpu_fallback_allowed = false
bitnet_inference = false
qk256_decode = false
production_qk256_policy_change = false
claim_allowed = false
diagnostic_only = true
```

Rows include selected platform/device identity, driver version, OpenCL build
metadata, production kernel binary size/hash/prefix, production kernel assembly
path/size/hash/prefix, and the exact `ocloc` invocation.

## Classification

```text
a770_qk256_production_kernel_disassembly_evidence_missing_program_binary
a770_qk256_production_kernel_disassembly_evidence_ocloc_missing
a770_qk256_production_kernel_disassembly_evidence_disasm_failed
a770_qk256_production_kernel_disassembly_evidence_kernel_asm_missing
a770_qk256_production_kernel_disassembly_evidence_captured
```

## Live Result

The selected A770 run produced:

```text
classification =
  a770_qk256_production_kernel_disassembly_evidence_captured

runtime_device = Intel(R) Arc(TM) A770 Graphics
platform_name = Intel(R) OpenCL Graphics
vendor = Intel(R) Corporation
driver_version = 32.0.101.8801

binary_type = executable
kernel_names = qk256_i2s_i8s_scaled_gemv
program_device_count = 1
binary_sizes = [11568]
binary_fnv1a64 = ["e43804852ba461f8"]
binary_prefix_hex =
  ["7f454c460201010001000000000000000100cd00010000000000000000000000"]
source_bytes = 1021
source_fnv1a64 = 9362326a33f953ba
program_binary_captured = true

ocloc_path =
  C:/Program Files (x86)/Intel/oneAPI/ocloc/latest/bin/ocloc.exe
ocloc_device = dg2-g10
ocloc_exit_code = 0
ocloc_stdout = "Warnings: unexpected padding at end of kernel\n"

binary_path =
  ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly/qk256_i2s_i8s_scaled_gemv.bin
kernel_asm_path =
  ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv.asm
kernel_asm_bytes = 10782
kernel_asm_fnv1a64 = 830e45d2eab44634
kernel_asm_trailing_whitespace_trimmed = true
disassembly_captured = true
```

Interpretation: the selected A770 OpenCL driver exposes an executable program
binary for the production `qk256_i2s_i8s_scaled_gemv` kernel, and local Intel
`ocloc` successfully dumps the production kernel assembly. This closes the
missing production-kernel disassembly context from A770-059, but it does not by
itself justify a production QK256 policy change.

The next diagnostic should inspect the production-kernel lowered operation
sequence and tie it back to the focused replay context before any runtime
policy change.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-kernels/src/a770_opencl_runtime.rs crates/bitnet-kernels/src/bin/a770_opencl_disassembly_evidence.rs
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence -- --kernel production --artifact-dir ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly-evidence.json --ocloc "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\bin\ocloc.exe" --device dg2-g10
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly-evidence.json -Raw | ConvertFrom-Json | Out-Null"
wsl bash -lc 'cd /mnt/e/Code/Rust/bitnet-rs-swarm && cargo test --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-disassembly-evidence -- --nocapture'
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

The Windows test-profile path timed out in the known C/C++ dev-dependency
build path before reaching the focused helper tests. The same focused binary
tests completed successfully under WSL.

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
