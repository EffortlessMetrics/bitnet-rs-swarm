# A770-046 QK256 Output Readback Trace

## Scope

A770-046 is a diagnostic-only continuation of A770-045. A770-045 classified
the selected layer-0 `q_proj` QK256 OpenCL output casting frontier as:

```text
generated_output_qk256_output_casting_frontier_device_value_summary_drift
```

This slice keeps runtime math, kernels, dispatch policy, answer scoring, and
sampling unchanged. It adds compact first-value samples to the selected QKV
dispatch replay receipt and emits a follow-on frontier that distinguishes
device-side evaluation, output casting/sample serialization, host readback
serialization, clean replay, and missing context.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_output_readback_trace_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
QK256 numeric-policy and output-casting classifications, compact policy-versus
A770 first-value samples, sample count comparison, first sampled mismatch, shape
and value-count comparison, device-to-host byte-count comparison, and SHA/mean
/RMS deltas. Full output vectors are still not forwarded.

## Classifications

```text
generated_output_qk256_output_readback_trace_frontier_missing_context
generated_output_qk256_output_readback_trace_frontier_output_casting
generated_output_qk256_output_readback_trace_frontier_device_side_evaluation
generated_output_qk256_output_readback_trace_frontier_host_readback_serialization
generated_output_qk256_output_readback_trace_frontier_clean
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
  generated_output_qk256_output_readback_trace_frontier_device_side_evaluation

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

qk256_numeric_policy_classification =
  generated_output_qk256_numeric_policy_output_casting_serialization

qk256_output_casting_classification =
  generated_output_qk256_output_casting_device_value_summary_drift

shape_match = true
value_count_match = true
sample_count_match = true
device_to_host_byte_count_match = true
sha256_f32_le_match = false
first_values_match = false
first_mismatch_index = 0
first_mismatch_abs_delta = 1.1920928955078125e-7
mean_abs_delta = 3.257810028689523e-10
rms_abs_delta = 3.9808238039285015e-8

policy_first_values[0] = -1.353820562362671
a770_first_values[0] = -1.3538206815719604

next_diagnostic =
  inspect selected QK256 OpenCL device-side output expression and f32 casting at sampled indices
```

Interpretation: the selected output retains the expected shape, value count,
sample count, and device-to-host byte count. The compact sample itself already
differs at the first sampled value, so the remaining A770-045 output summary
drift is now routed to device-side evaluation or f32 casting at the sampled
indices, not to host readback byte-count or missing trace context.
