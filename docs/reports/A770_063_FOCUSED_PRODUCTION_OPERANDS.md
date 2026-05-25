# A770-063 Focused Production Operands

## Scope

A770-063 is a diagnostic-only follow-up to A770-062. A770-062 proved that the
selected-device production replay instrumentation can capture adjusted-dot,
scale, reciprocal-path, final-value, and production output-store bits on a
diagnostic QK256 fixture. That fixture was not the focused generated-output
first-mismatch operand set.

This slice points the production replay instrumentation tool at the committed
focused CPU/A770 parity receipt for:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj
```

It does not change production QK256 dispatch, runtime math, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipt

The new receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-production-operands.json
```

The receipt records:

```text
proof_family = a770_opencl_qk256_focused_production_operands
proof_stage = diagnostic_focused_production_operand_context_classified
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
a770_qk256_focused_production_operands_summary_context_only_raw_operands_missing
```

The focused source has compact QK256 replay context:

```text
qkv_projection_dispatch_replay_context_available = true
device_expression_trace_available = true
device_intermediate_trace_available = true
summary_qk256_trace_available = true
cols = 2560
row_stride_bytes = 640
sample_count = 8
sample_limit = 8
```

The focused source does not contain the raw operand bytes required to run the
unchanged production kernel and replay kernel on the exact focused first
mismatch:

```text
focused_first_mismatch_operands_available = false
raw_activation_i8_available = false
raw_packed_qk256_available = false
missing_raw_operand_fields = ["activations_i8", "packed_qk256"]
production_replay_executed = false
kernel_invocations = 0
```

The summary trace still preserves the already-known focused one-bit split:

```text
output_index = 0
activation_sum = 1063
activation_scale_bits = 1155033426
weight_scale_bits = 1067189103
int_dot = -860
adjusted_dot = -1923
focused_device_output_bits = 3215804927
focused_policy_bits = 3215804926
focused_summary_device_vs_policy_bits_match = false
```

## Interpretation

A770-063 narrows the current blocker to missing raw focused operands, not to a
production replay instrumentation failure. The committed focused source is
useful but summary-only: it carries enough context to identify the selected
case, projection, device, scale bits, adjusted-dot, and one-bit device/policy
split, but not enough raw data to feed `qk256_i2s_i8s_scaled_gemv` and
`qk256_i2s_i8s_scaled_gemv_production_replay` with the exact focused model
operands.

The next useful diagnostic is to capture the raw activation row and packed
QK256 bytes for the focused `q_proj` first mismatch before any production
QK256 policy change.

## Validation

```text
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-production-operands.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-production-operands.json -Raw | ConvertFrom-Json | Out-Null"
```

The focused-source mode does not launch a production replay kernel when raw
focused operands are absent. It classifies that absence explicitly instead of
synthesizing operands.

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
