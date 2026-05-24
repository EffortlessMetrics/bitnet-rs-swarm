# A770-045 QK256 Output Casting Frontier

## Scope

A770-045 is a diagnostic-only continuation of A770-044. A770-044 classified
the selected layer-0 `q_proj` CPU scalar versus selected-device A770 OpenCL
QK256 replay split as:

```text
generated_output_qk256_numeric_policy_frontier_output_casting_serialization
```

This slice keeps runtime math, kernels, dispatch policy, answer scoring, and
sampling unchanged. It adds a compact receipt frontier for the selected OpenCL
output path so the next boundary can distinguish receipt shape serialization,
device-to-host byte-count/readback shape, and device value-summary drift.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_output_casting_frontier
```

Rows include the selected case, mismatch index, layer/projection, QK256 numeric
policy classification, compact policy-versus-A770 output summaries, shape and
value-count comparisons, expected versus actual device-to-host bytes, and
min/max/mean/RMS deltas. Full output vectors are still not forwarded.

## Classifications

```text
generated_output_qk256_output_casting_frontier_missing_context
generated_output_qk256_output_casting_frontier_receipt_shape_serialization
generated_output_qk256_output_casting_frontier_readback_byte_count
generated_output_qk256_output_casting_frontier_device_value_summary_drift
generated_output_qk256_output_casting_frontier_clean
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
  generated_output_qk256_output_casting_frontier_device_value_summary_drift

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

qk256_numeric_policy_classification =
  generated_output_qk256_numeric_policy_output_casting_serialization

shape_match = true
value_count_match = true
finite_count_match = true
nan_count_match = true
infinite_count_match = true
device_to_host_bytes = 10240
expected_device_to_host_bytes = 10240
device_to_host_byte_count_match = true
sha256_f32_le_match = false
min_abs_delta = 0.0
max_abs_delta = 0.0
mean_abs_delta = 3.257810028689523e-10
rms_abs_delta = 3.9808238039285015e-08

next_diagnostic =
  capture selected QK256 OpenCL output value samples or device readback trace
```

Interpretation: the selected A770 output has the expected shape, finite-count
metadata, and device-to-host byte count for 2,560 `f32` values. The compact
summary still differs from the host OpenCL-policy replay by SHA and tiny
mean/RMS deltas while min/max remain equal. The next useful slice is to capture
selected value samples or a narrow device readback trace so the remaining split
can be pinned to device-side evaluation, output casting, or host readback
serialization without forwarding full vectors.
