# A770-044 QK256 Numeric Policy Frontier

## Scope

A770-044 is a diagnostic-only continuation of A770-043. A770-043 showed that
the focused layer-0 `q_proj` replay is stable and that the selected split is
between the CPU scalar QK256 replay and the selected-device A770 OpenCL replay,
using the same materialized projection input and raw QK256 metadata.

This slice adds a host-side replay of the OpenCL kernel expression policy beside
the existing CPU scalar and A770 OpenCL replay outputs. The host OpenCL-policy
replay uses the same prequantized I8_S activation row, packed QK256 bytes,
inline scale, and linear `int` accumulation expression used by the OpenCL kernel
source. It is receipt-only; production runtime math, kernels, dispatch policy,
answer scoring, and sampling are unchanged.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_numeric_policy_frontier
```

The report is compact. It records the focused case, mismatch index, selected
layer/projection, CPU replay output, host OpenCL-policy replay output, A770
OpenCL replay output, and pairwise output fingerprint matches.

## Classifications

```text
generated_output_qk256_numeric_policy_frontier_missing_context
generated_output_qk256_numeric_policy_frontier_raw_input_materialization
generated_output_qk256_numeric_policy_frontier_packed_weight_decode
generated_output_qk256_numeric_policy_frontier_scale_application
generated_output_qk256_numeric_policy_frontier_accumulation_order
generated_output_qk256_numeric_policy_frontier_output_casting_serialization
generated_output_qk256_numeric_policy_frontier_clean
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
  generated_output_qk256_numeric_policy_frontier_output_casting_serialization

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

qkv_projection_dispatch_replay_classification =
  generated_output_qkv_projection_dispatch_replay_cpu_a770_output_drift

left_cpu_opencl_policy_output_match = true
right_cpu_opencl_policy_output_match = true
left_opencl_policy_a770_output_match = false
right_opencl_policy_a770_output_match = false
opencl_policy_output_match_across_receipts = true

next_diagnostic =
  inspect selected QK256 OpenCL output casting and receipt serialization
```

Interpretation: the selected CPU replay output matches the host-side replay of
the OpenCL expression policy across both focused receipts. The A770 device
output still differs from that host policy. At this boundary the live evidence
does not implicate raw input materialization, packed QK256 metadata, inline
scale selection, or the CPU scalar accumulation policy. The next useful slice is
to inspect the selected OpenCL device output path: kernel expression codegen,
device-side float evaluation, readback, and receipt serialization.
