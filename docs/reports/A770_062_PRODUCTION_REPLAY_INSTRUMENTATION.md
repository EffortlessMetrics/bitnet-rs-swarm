# A770-062 Production Replay Instrumentation

## Scope

A770-062 is a diagnostic-only follow-up to A770-061. A770-061 classified the
selected-device production `qk256_i2s_i8s_scaled_gemv` lowered operation
sequence as requiring replay instrumentation before any production QK256 policy
change.

This slice adds a bounded production replay capture for the selected A770. It
runs the unchanged production kernel and a separate replay kernel on the same
fixture, then records adjusted-dot, scale, reciprocal-path, final-value, and
production output-store bits.

It does not change production QK256 dispatch, runtime math, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipt

The new receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-replay-instrumentation.json
```

The receipt records:

```text
proof_family = a770_opencl_qk256_production_replay_instrumentation
proof_stage = diagnostic_production_replay_instrumentation_captured
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
kernel_name = qk256_i2s_i8s_scaled_gemv
replay_kernel_name = qk256_i2s_i8s_scaled_gemv_production_replay
fallback_used = false
cpu_fallback_allowed = false
bitnet_inference = false
qk256_decode = false
production_qk256_policy_change = false
claim_allowed = false
diagnostic_only = true
```

## Classification

```text
a770_qk256_production_replay_instrumentation_output_store_matches_replay
```

The diagnostic fixture captured:

```text
fixture_source = diagnostic_minimal_qk256_fixture
focused_first_mismatch_operands_available = false
focused_case_id = a770_summary_seed770024_keywords_014
rows = 2
cols = 256
row_stride_bytes = 64
sample_limit = 2
kernel_invocations = 2
```

For both sampled rows:

```text
all_output_store_matches_replay_output = true
all_output_store_matches_final_scaled_value = true
```

The receipt also records compact host-side comparison bits. The diagnostic
fixture is intentionally not the focused first-mismatch operand set, so this
receipt does not resolve production QK256 policy for the answer-readiness
divergence by itself.

## Interpretation

The production output-store bits match the new replay kernel's reciprocal-path
final scaled value on the committed diagnostic fixture. That proves the
instrumentation path captures the requested production replay evidence on the
selected A770 device.

It does not prove the focused generated-output mismatch root cause. The next
useful diagnostic is to capture focused production operands for the first
mismatch before any production QK256 policy change.

## Validation

```text
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-replay-instrumentation.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-replay-instrumentation.json -Raw | ConvertFrom-Json | Out-Null"
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo fmt -p bitnet-kernels --check
git diff --check
```

Local note:

```text
cargo test --locked -p bitnet-kernels --lib --no-default-features --features opencl production_replay -- --nocapture
```

timed out in this Windows checkout while building through MSVC/CMake compiler
probe state; stale cargo/cl helper processes were stopped. The bin compile and
selected-device receipt capture completed successfully.

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
