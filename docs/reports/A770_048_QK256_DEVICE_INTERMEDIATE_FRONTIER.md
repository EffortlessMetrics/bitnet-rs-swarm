# A770-048 QK256 Device Intermediate Frontier

## Scope

A770-048 is a diagnostic-only continuation of A770-047. A770-047 classified
the selected layer-0 `q_proj` QK256 OpenCL device-expression frontier as:

```text
generated_output_qk256_device_expression_frontier_unmatched_device_value
```

This slice keeps production QK256 dispatch, answer scoring, sampling, and
promotion claims unchanged. It adds bounded selected-device debug context for
the sampled QK256 OpenCL output lane so the next receipt can distinguish
device-side integer dot accumulation, activation-sum correction, scale bits,
debug output/readback, and missing-context buckets.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_device_intermediate_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
device-expression classification, compact policy/readback/debug values, sampled
integer dot product, activation sum, adjusted dot, activation and weight scale
bits, and pairwise CPU/A770 side summaries. Full output vectors are not
forwarded.

## Classifications

```text
generated_output_qk256_device_intermediate_frontier_missing_context
generated_output_qk256_device_intermediate_frontier_int_dot_drift
generated_output_qk256_device_intermediate_frontier_activation_or_adjusted_dot_drift
generated_output_qk256_device_intermediate_frontier_scalar_bits_drift
generated_output_qk256_device_intermediate_frontier_debug_output_matches_readback
generated_output_qk256_device_intermediate_frontier_debug_output_matches_policy
generated_output_qk256_device_intermediate_frontier_debug_output_unmatched
generated_output_qk256_device_intermediate_frontier_clean
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
  generated_output_qk256_device_intermediate_frontier_debug_output_matches_readback

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

qk256_device_expression_classification =
  generated_output_qk256_device_expression_unmatched_device_value

selected sample:
  output_index = 0
  sample_count = 8
  sample_limit = 8
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  host_int_dot = -860
  device_int_dot = -860
  host_activation_sum = 1063
  device_activation_sum = 1063
  host_adjusted_dot = -1923
  device_adjusted_dot = -1923
  host_activation_scale_bits = 1155033426
  device_activation_scale_bits = 1155033426
  host_weight_scale_bits = 1067189103
  device_weight_scale_bits = 1067189103
  device_adjusted_f32_bits = 3304087552

  policy_first_value = -1.353820562362671
  policy_first_value_bits = 3215804926
  a770_first_value = -1.3538206815719604
  a770_first_value_bits = 3215804927
  device_output = -1.3538206815719604
  device_output_bits = 3215804927

  debug_output_matches_policy = false
  debug_output_matches_readback = true

next_diagnostic =
  pin the OpenCL device expression or compiler math-mode policy that produces
  the selected one-bit split
```

Interpretation: the selected device-side integer dot, activation sum, adjusted
dot, and scale-bit inputs match the host trace, and the bounded debug kernel's
output exactly matches the A770 readback. The split is now narrower than input
intermediates and should be pinned to the OpenCL device expression or compiler
math-mode policy that produces the one-bit f32 output difference. The focused
CPU/A770 answer-parity receipt still fails for the selected case, so this does
not promote parity, quality, residency, speed, or full inference claims.
