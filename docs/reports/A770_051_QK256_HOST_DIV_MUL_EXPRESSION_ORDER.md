# A770-051 QK256 Host Div-Mul Expression Order

## Scope

A770-051 is a diagnostic-only continuation of A770-050. A770-050 classified
the selected layer-0 `q_proj` QK256 host/device div-mul frontier as:

```text
generated_output_qk256_host_device_div_mul_frontier_host_replay_mismatch
```

This slice inspects the compact host replay f32 expression-order variants
against the selected A770 OpenCL device div/mul behavior already present in the
focused CPU/A770 replay context. It does not change production QK256 dispatch,
runtime math, answer scoring, sampling, model loading, or promotion policy.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_host_div_mul_expression_order_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
host/device div-mul classification, compact host expression-order bits, selected
device expression bits, production-kernel impact availability, host expression
collapse flags, and device identity. Full output vectors are not forwarded.

## Classifications

```text
generated_output_qk256_host_div_mul_expression_order_frontier_missing_context
generated_output_qk256_host_div_mul_expression_order_frontier_host_expression_order_mismatch
generated_output_qk256_host_div_mul_expression_order_frontier_opencl_build_or_math_mode_mismatch
generated_output_qk256_host_div_mul_expression_order_frontier_volatile_or_reassociation
generated_output_qk256_host_div_mul_expression_order_frontier_host_policy_match
generated_output_qk256_host_div_mul_expression_order_frontier_production_kernel_impact_missing
generated_output_qk256_host_div_mul_expression_order_frontier_unmatched
generated_output_qk256_host_div_mul_expression_order_frontier_clean
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
  generated_output_qk256_host_div_mul_expression_order_frontier_host_expression_order_mismatch

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
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
  production_kernel_impact_available = true

next_diagnostic =
  inspect host replay codegen and f32 operation ordering before any production
  QK256 policy change
```

Interpretation: the compact host f32 expression-order variants all collapse to
the host policy bit. The selected A770 device div-then-mul output lands one bit
away, while the selected device mul-then-div still lands on the host-policy bit.
The focused CPU/A770 answer-parity receipt still fails for the selected case, so
this does not promote parity, quality, residency, speed, trusted partial
acceleration, or full inference claims.
