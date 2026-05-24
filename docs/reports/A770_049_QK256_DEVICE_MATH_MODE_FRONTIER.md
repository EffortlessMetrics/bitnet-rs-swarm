# A770-049 QK256 Device Math-Mode Frontier

## Scope

A770-049 is a diagnostic-only continuation of A770-048. A770-048 classified
the selected layer-0 `q_proj` QK256 OpenCL device-intermediate frontier as:

```text
generated_output_qk256_device_intermediate_frontier_debug_output_matches_readback
```

This slice keeps production QK256 dispatch, answer scoring, sampling, and
promotion claims unchanged. It extends the bounded selected-device debug trace
with compact device-side expression variant bits so the focused receipt can
separate default OpenCL div-then-mul behavior, volatile div-then-mul replay,
reassociated expressions, host-policy match, unmatched device context, and
missing context.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_device_math_mode_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
device-expression and device-intermediate classifications, compact host policy
and A770 readback bits, device output bits, device-side expression variant
bits, selected device identity, and pairwise CPU/A770 side summaries. Full
output vectors are not forwarded.

## Classifications

```text
generated_output_qk256_device_math_mode_frontier_missing_context
generated_output_qk256_device_math_mode_frontier_default_div_then_mul
generated_output_qk256_device_math_mode_frontier_optimized_div_then_mul
generated_output_qk256_device_math_mode_frontier_volatile_div_then_mul
generated_output_qk256_device_math_mode_frontier_mul_then_div
generated_output_qk256_device_math_mode_frontier_reciprocal_then_mul
generated_output_qk256_device_math_mode_frontier_matches_host_policy
generated_output_qk256_device_math_mode_frontier_unmatched
generated_output_qk256_device_math_mode_frontier_clean
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
  generated_output_qk256_device_math_mode_frontier_default_div_then_mul

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_device_expression =
    generated_output_qk256_device_expression_unmatched_device_value
  qk256_device_intermediate =
    generated_output_qk256_device_intermediate_debug_output_matches_readback

selected sample:
  output_index = 0
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  policy_first_value = -1.353820562362671
  policy_first_value_bits = 3215804926
  a770_first_value = -1.3538206815719604
  a770_first_value_bits = 3215804927

  device_output = -1.3538206815719604
  device_output_bits = 3215804927
  device_div_then_mul = -1.3538206815719604
  device_div_then_mul_bits = 3215804927
  device_mul_then_div = -1.353820562362671
  device_mul_then_div_bits = 3215804926
  device_reciprocal_then_mul = -1.3538206815719604
  device_reciprocal_then_mul_bits = 3215804927
  device_volatile_div_then_mul = -1.3538206815719604
  device_volatile_div_then_mul_bits = 3215804927

  matches_device_div_then_mul = true
  matches_device_mul_then_div = false
  matches_device_reciprocal_then_mul = true
  matches_device_volatile_div_then_mul = true
  matches_host_policy = false

next_diagnostic =
  compare host replay f32 div/mul rounding against selected A770 OpenCL device
  div/mul before any runtime policy change
```

Interpretation: the selected A770 readback follows the device-side
div-then-mul result, the reciprocal form also lands on the same device bit, and
the device-side mul-then-div lands on the host policy bit instead. The host
OpenCL-policy replay remains one `f32` bit away. The next diagnostic should
compare host replay f32 div/mul rounding against selected A770 OpenCL device
div/mul before any runtime policy change. The focused CPU/A770 answer-parity
receipt still fails for the selected case, so this does not promote parity,
quality, residency, speed, or full inference claims.
