# A770-053 QK256 Strict F32 Barrier Evidence

## Scope

A770-053 is a diagnostic-only continuation of A770-052. A770-052 classified
the selected layer-0 `q_proj` QK256 host replay f32 codegen-ordering frontier
as:

```text
generated_output_qk256_host_replay_f32_codegen_ordering_frontier_host_expression_variants_collapsed_to_policy
```

This slice records compact strict f32 barrier evidence for the selected QK256
expression. It derives whether the selected OpenCL device div-then-mul value
matches the A770 output, whether the host replay value matches that same output,
and whether the host replay variants still collapse to policy. It does not
change production QK256 dispatch, runtime math, answer scoring, sampling, model
loading, or promotion policy.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_strict_f32_barrier_evidence_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
host replay f32 codegen-ordering classification, selected device and host
expression bits, strict-f32 barrier match flags, host compiler/codegen collapse
flags, production-kernel impact availability, and device identity. Full output
vectors are not forwarded.

## Classifications

```text
generated_output_qk256_strict_f32_barrier_evidence_frontier_missing_context
generated_output_qk256_strict_f32_barrier_evidence_frontier_production_policy_change_not_justified
generated_output_qk256_strict_f32_barrier_evidence_frontier_opencl_frontend_codegen_split
generated_output_qk256_strict_f32_barrier_evidence_frontier_host_compiler_codegen_collapse
generated_output_qk256_strict_f32_barrier_evidence_frontier_strict_f32_barrier_matches_selected_device_output
generated_output_qk256_strict_f32_barrier_evidence_frontier_clean
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
  generated_output_qk256_strict_f32_barrier_evidence_frontier_strict_f32_barrier_matches_selected_device_output

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_host_replay_f32_codegen_ordering =
    generated_output_qk256_host_replay_f32_codegen_ordering_host_expression_variants_collapsed_to_policy
  qk256_host_div_mul_expression_order =
    generated_output_qk256_host_div_mul_expression_order_host_expression_order_mismatch
  qk256_host_device_div_mul =
    generated_output_qk256_host_device_div_mul_host_replay_mismatch

selected sample:
  output_index = 0
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  policy_first_value_bits = 3215804926
  host_div_then_mul_bits = 3215804926
  device_output_bits = 3215804927
  device_div_then_mul_bits = 3215804927
  device_mul_then_div_bits = 3215804926
  device_behavior = default_div_then_mul

  strict_f32_barrier_bits_compared = true
  strict_f32_barrier_source = selected_opencl_device_div_then_mul
  device_div_then_mul_matches_selected_output = true
  host_div_then_mul_matches_selected_output = false
  host_compiler_codegen_collapse = true
  production_kernel_impact_available = true
  production_policy_change_justified = false

next_diagnostic =
  capture host compiler strict-f32 barrier codegen evidence before any
  production QK256 policy change
```

Interpretation: the selected OpenCL device div-then-mul value matches the A770
output bit, while the host replay div-then-mul value remains on the host policy
bit. The host replay variants still collapse to policy, so this compact
evidence is not enough to justify a production QK256 policy change. The focused
CPU/A770 answer-parity receipt still fails for the selected case, so this does
not promote parity, quality, residency, speed, trusted partial acceleration, or
full inference claims.
