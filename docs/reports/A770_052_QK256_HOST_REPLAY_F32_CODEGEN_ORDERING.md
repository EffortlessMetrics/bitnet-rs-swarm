# A770-052 QK256 Host Replay F32 Codegen Ordering

## Scope

A770-052 is a diagnostic-only continuation of A770-051. A770-051 classified
the selected layer-0 `q_proj` QK256 host div-mul expression-order frontier as:

```text
generated_output_qk256_host_div_mul_expression_order_frontier_host_expression_order_mismatch
```

This slice inspects whether the committed focused CPU/A770 replay evidence
supports a production QK256 policy change from host replay f32 codegen or
operation ordering. It does not change production QK256 dispatch, runtime math,
answer scoring, sampling, model loading, or promotion policy.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_host_replay_f32_codegen_ordering_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
host div-mul expression-order classification, compact host expression-order
bits, selected device expression bits, explicit f32 bit comparison fields,
production-kernel impact availability, and device identity. Full output
vectors are not forwarded.

## Classifications

```text
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_missing_context
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_production_policy_change_not_justified
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_opencl_frontend_codegen_mismatch
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_explicit_f32_rounding_gap
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_host_replay_codegen_order_mismatch
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_host_expression_variants_collapsed_to_policy
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
  generated_output_qk256_host_replay_f32_codegen_ordering_frontier_host_expression_variants_collapsed_to_policy

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
  output_index = 0
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  policy_first_value_bits = 3215804926
  host_div_then_mul_bits = 3215804926
  host_mul_then_div_bits = 3215804926
  host_reciprocal_then_mul_bits = 3215804926
  host_f64_div_then_mul_cast_bits = 3215804926

  device_output_bits = 3215804927
  device_div_then_mul_bits = 3215804927
  device_mul_then_div_bits = 3215804926
  device_behavior = default_div_then_mul

  host_unique_expression_bits_count = 1
  host_variants_collapse_to_policy = true
  any_host_variant_matches_selected_output = false
  host_policy_matches_device_mul_then_div = true
  host_div_then_mul_matches_device_div_then_mul = false
  explicit_f32_rounding_bits_compared = true
  production_kernel_impact_available = true

next_diagnostic =
  capture host compiler codegen or strict f32 barrier evidence before any
  production QK256 policy change
```

Interpretation: the focused host replay f32 expression variants still collapse
to the host policy bit, and the selected A770 device div-then-mul output stays
one bit away. The receipt therefore records that host replay expression
variants alone do not justify a production QK256 policy change. The focused
CPU/A770 answer-parity receipt still fails for the selected case, so this does
not promote parity, quality, residency, speed, trusted partial acceleration, or
full inference claims.
