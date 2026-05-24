# A770-047 QK256 Device Expression Frontier

## Scope

A770-047 is a diagnostic-only continuation of A770-046. A770-046 classified
the selected layer-0 `q_proj` QK256 OpenCL output readback frontier as:

```text
generated_output_qk256_output_readback_trace_frontier_device_side_evaluation
```

This slice keeps runtime math, kernels, dispatch policy, answer scoring, and
sampling unchanged. It adds a compact selected-row device-expression trace to
the QKV dispatch replay context and emits a follow-on frontier that compares
the sampled A770 value against bounded host replays of the OpenCL expression.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_device_expression_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
readback-trace classification, compact policy-versus-A770 sampled values,
sampled integer dot product, activation sum, adjusted dot, activation and
weight scale bits, and the bounded expression variants. Full output vectors are
not forwarded.

## Classifications

```text
generated_output_qk256_device_expression_frontier_missing_context
generated_output_qk256_device_expression_frontier_div_then_mul
generated_output_qk256_device_expression_frontier_mul_then_div
generated_output_qk256_device_expression_frontier_reciprocal_then_mul
generated_output_qk256_device_expression_frontier_f64_cast
generated_output_qk256_device_expression_frontier_unmatched_device_value
generated_output_qk256_device_expression_frontier_clean
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
  generated_output_qk256_device_expression_frontier_unmatched_device_value

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

qk256_output_readback_trace_classification =
  generated_output_qk256_output_readback_trace_device_side_evaluation

selected sample:
  output_index = 0
  int_dot = -860
  activation_sum = 1063
  adjusted_dot = -1923
  activation_scale_bits = 1155033426
  weight_scale_bits = 1067189103

policy_first_value = -1.353820562362671
policy_first_value_bits = 3215804926

a770_first_value = -1.3538206815719604
a770_first_value_bits = 3215804927

div_then_mul_bits = 3215804926
mul_then_div_bits = 3215804926
reciprocal_then_mul_bits = 3215804926
f64_div_then_mul_cast_bits = 3215804926

next_diagnostic =
  capture selected QK256 device-side intermediates with a bounded debug kernel
```

Interpretation: the sampled A770 readback does not match the bounded host
expression variants currently replayed in the receipt. The split is narrower
than expression ordering among the recorded host variants; the next diagnostic
should capture device-side intermediates for the selected output lane instead
of changing runtime math from this evidence alone.
