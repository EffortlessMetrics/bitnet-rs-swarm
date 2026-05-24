# A770-050 QK256 Host/Device Div-Mul Replay

## Scope

A770-050 is a diagnostic-only continuation of A770-049. A770-049 classified
the selected layer-0 `q_proj` QK256 device math-mode frontier as:

```text
generated_output_qk256_device_math_mode_frontier_default_div_then_mul
```

This slice compares the compact host replay f32 div/mul expression bits against
the selected A770 OpenCL device div/mul bits already present in the committed
focused CPU/A770 replay context. It does not change production QK256 dispatch,
runtime math, answer scoring, sampling, model loading, or promotion policy.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_host_device_div_mul_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
device-expression and device-math-mode classifications, compact host replay
expression bits, selected device expression bits, host/device match flags,
selected device behavior, and device identity. Full output vectors are not
forwarded.

## Classifications

```text
generated_output_qk256_host_device_div_mul_frontier_missing_context
generated_output_qk256_host_device_div_mul_frontier_host_replay_mismatch
generated_output_qk256_host_device_div_mul_frontier_device_default_div_then_mul
generated_output_qk256_host_device_div_mul_frontier_device_optimized_div_then_mul
generated_output_qk256_host_device_div_mul_frontier_volatile_or_reassociation
generated_output_qk256_host_device_div_mul_frontier_host_policy_match
generated_output_qk256_host_device_div_mul_frontier_unmatched
generated_output_qk256_host_device_div_mul_frontier_clean
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
  generated_output_qk256_host_device_div_mul_frontier_host_replay_mismatch

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_device_expression =
    generated_output_qk256_device_expression_unmatched_device_value
  qk256_device_math_mode =
    generated_output_qk256_device_math_mode_default_div_then_mul

selected sample:
  output_index = 0
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  policy_first_value_bits = 3215804926
  host_div_then_mul_bits = 3215804926
  device_div_then_mul_bits = 3215804927
  device_output_bits = 3215804927

  device_behavior = default_div_then_mul
  host_replay_mismatch = true
  host_policy_matches_device_mul_then_div = true
  matches_device_div_then_mul = true
  matches_device_volatile_div_then_mul = true

next_diagnostic =
  inspect host replay f32 div/mul expression ordering against selected A770
  device div/mul before any production policy change
```

Interpretation: the compact host replay div-then-mul expression lands on the
host policy bit, while the selected A770 device div-then-mul and selected device
output land on the A770 readback bit. The device-side mul-then-div still lands
on the host-policy bit. The focused CPU/A770 answer-parity receipt still fails
for the selected case, so this does not promote parity, quality, residency,
speed, trusted partial acceleration, or full inference claims.
