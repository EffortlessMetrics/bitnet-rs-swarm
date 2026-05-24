# A770-052 QK256 Host Replay F32 Codegen Ordering

## Scope

A770-052 is a diagnostic-only continuation of A770-051. A770-051 classified
the selected layer-0 `q_proj` QK256 host div-mul expression-order frontier as:

```text
generated_output_qk256_host_div_mul_expression_order_frontier_host_expression_order_mismatch
```

This slice recomputes compact host f32 operation-order variants from the
recorded selected dot and scale fields, compares them to the existing host replay
trace and selected A770 OpenCL bits, and keeps production QK256 dispatch,
runtime math, answer scoring, sampling, model loading, and promotion policy
unchanged.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_host_replay_f32_codegen_ordering_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
host/device div-mul classifications, recorded dot/scale fields, compact host
codegen bits, explicit f32 operation-order bits, selected device bits, and the
production-impact availability flag. Full output vectors are not forwarded.

## Classifications

```text
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_missing_context
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_host_replay_codegen_order_mismatch
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_explicit_f32_rounding_gap
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_opencl_frontend_codegen_mismatch
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_host_expression_variants_collapsed_to_policy
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_production_policy_change_not_justified
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_clean
```

## Claim Boundary

This is not a runtime fix and not an A770 readiness promotion. It does not prove
CPU/A770 answer parity, reference parity, strict A770 answer readiness, broad
A770 quality, official BitNet QK256 production semantics, activation
quantization residency, selected attention residency, resident KV, full A770
residency, performance speedup, trusted partial acceleration, or full BitNet
inference.

## Live Result

The refreshed focused receipt reports:

```text
classification =
  generated_output_qk256_host_replay_f32_codegen_ordering_frontier_explicit_f32_rounding_gap

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_host_div_mul_expression_order =
    generated_output_qk256_host_div_mul_expression_order_host_expression_order_mismatch
  qk256_host_device_div_mul =
    generated_output_qk256_host_device_div_mul_host_replay_mismatch
  qk256_device_math_mode =
    generated_output_qk256_device_math_mode_default_div_then_mul

selected sample:
  policy_first_value_bits = 3215804926
  device_output_bits = 3215804927

  host_trace_div_then_mul_bits = 3215804926
  host_trace_mul_then_div_bits = 3215804926
  host_trace_weight_over_activation_bits = 3215804926
  host_trace_f64_div_then_mul_cast_bits = 3215804926

  host_codegen_weight_over_activation_bits = 3215804926
  explicit_f32_reciprocal_then_mul_bits = 3215804927

  host_codegen_matches_trace = true
  host_trace_variants_collapse_to_policy = true
  host_codegen_variants_collapse_to_policy = true
  explicit_f32_operation_order_gap = true
  explicit_f32_variant_matches_selected_output = true
  production_kernel_impact_available = true

next_diagnostic =
  pin selected production QK256 expression policy against explicit f32
  operation-order replay before any runtime change
```

Interpretation: the existing host replay/codegen variants are internally
consistent and still collapse to the host policy bit, but an explicit f32
operation-order replay lands on the selected A770 device output bit. The focused
CPU/A770 answer-parity receipt still fails for the selected case, so this does
not promote parity, quality, residency, speed, trusted partial acceleration, or
full inference claims.
