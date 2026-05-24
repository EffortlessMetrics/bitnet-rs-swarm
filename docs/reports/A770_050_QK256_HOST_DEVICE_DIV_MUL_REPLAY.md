# A770-050 QK256 Host/Device Div-Mul Replay

## Scope

A770-050 is a diagnostic-only continuation of A770-049. A770-049 classified
the selected layer-0 `q_proj` QK256 OpenCL device math-mode frontier as:

```text
generated_output_qk256_device_math_mode_frontier_default_div_then_mul
```

This slice keeps production QK256 dispatch, answer scoring, sampling, and
promotion claims unchanged. It adds a compact host replay frontier so the
focused receipt can compare Rust host `f32` div/mul rounding against the
selected A770 OpenCL device div/mul bits before any runtime policy change.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_host_device_div_mul_replay_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
device math-mode classification, compact host replay bits, selected A770
readback bits, device expression bits, selected device identity, and pairwise
CPU/A770 side summaries. Full output vectors are not forwarded.

## Classifications

```text
generated_output_qk256_host_device_div_mul_replay_frontier_missing_context
generated_output_qk256_host_device_div_mul_replay_frontier_host_replay_mismatch
generated_output_qk256_host_device_div_mul_replay_frontier_device_default_div_then_mul
generated_output_qk256_host_device_div_mul_replay_frontier_device_optimized_div_then_mul
generated_output_qk256_host_device_div_mul_replay_frontier_volatile_reassociation
generated_output_qk256_host_device_div_mul_replay_frontier_matches_host_policy
generated_output_qk256_host_device_div_mul_replay_frontier_unmatched
generated_output_qk256_host_device_div_mul_replay_frontier_clean
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
  generated_output_qk256_host_device_div_mul_replay_frontier_host_replay_mismatch

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_device_math_mode =
    generated_output_qk256_device_math_mode_default_div_then_mul

selected sample:
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  policy_first_value_bits = 3215804926
  a770_first_value_bits = 3215804927

  host_div_then_mul = -1.353820562362671
  host_div_then_mul_bits = 3215804926
  host_mul_then_div = -1.353820562362671
  host_mul_then_div_bits = 3215804926
  host_reciprocal_then_mul = -1.353820562362671
  host_reciprocal_then_mul_bits = 3215804926

  device_div_then_mul = -1.3538206815719604
  device_div_then_mul_bits = 3215804927
  device_mul_then_div = -1.353820562362671
  device_mul_then_div_bits = 3215804926
  device_reciprocal_then_mul = -1.3538206815719604
  device_reciprocal_then_mul_bits = 3215804927
  device_volatile_div_then_mul = -1.3538206815719604
  device_volatile_div_then_mul_bits = 3215804927

  host_div_then_mul_matches_readback = false
  matches_device_div_then_mul = true
  matches_device_volatile_div_then_mul = true
  matches_host_policy = false

next_diagnostic =
  capture host compiler and OpenCL build-option div/mul replay for the selected
  QK256 row before any runtime policy change
```

Interpretation: the selected A770 readback still follows the device-side
div-then-mul and volatile div-then-mul bits, while the Rust host replay of the
same compact operands lands on the host policy bit. The selected one-bit split
is therefore not cleared by a local Rust host `f32` div-then-mul replay. The
next diagnostic should pin whether host compiler behavior or OpenCL build
options explain the split before any production QK256 expression-policy change.
The focused CPU/A770 answer-parity receipt still fails for the selected case,
so this does not promote parity, quality, residency, speed, or full inference
claims.
