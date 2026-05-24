# A770-054 Host Compiler Strict F32 Barrier

## Scope

A770-054 is a diagnostic-only follow-on to A770-053. A770-053 showed that the
selected OpenCL device div-then-mul value matches the A770 output bit while the
host replay div-then-mul value remains on the host policy bit.

This slice records compact host compiler strict-f32 barrier context for the same
focused QK256 expression. It does not change production QK256 dispatch, runtime
math, answer scoring, sampling, model loading, route policy, or any promotion
state.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_host_compiler_strict_f32_barrier_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
strict-f32 barrier evidence classification, selected device and host expression
bits, explicit barrier-match flags, host compiler/codegen collapse flags,
production-policy-change availability, and device identity. Full output vectors
are not forwarded.

## Classifications

```text
generated_output_qk256_host_compiler_strict_f32_barrier_frontier_missing_context
generated_output_qk256_host_compiler_strict_f32_barrier_frontier_production_policy_change_not_justified
generated_output_qk256_host_compiler_strict_f32_barrier_frontier_opencl_frontend_codegen_split
generated_output_qk256_host_compiler_strict_f32_barrier_frontier_host_compiler_codegen_collapse
generated_output_qk256_host_compiler_strict_f32_barrier_frontier_explicit_f32_barrier_codegen_match
generated_output_qk256_host_compiler_strict_f32_barrier_frontier_clean
```

## Live Result

The refreshed focused receipt reports:

```text
classification =
  generated_output_qk256_host_compiler_strict_f32_barrier_frontier_host_compiler_codegen_collapse

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_strict_f32_barrier_evidence =
    generated_output_qk256_strict_f32_barrier_evidence_strict_f32_barrier_matches_selected_device_output
  qk256_host_replay_f32_codegen_ordering =
    generated_output_qk256_host_replay_f32_codegen_ordering_host_expression_variants_collapsed_to_policy

selected sample:
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  host_div_then_mul_bits = 3215804926
  device_div_then_mul_bits = 3215804927
  device_mul_then_div_bits = 3215804926
  device_output_bits = 3215804927

  explicit_f32_barrier_codegen_match = true
  device_div_then_mul_matches_selected_output = true
  host_div_then_mul_matches_selected_output = false
  host_compiler_codegen_collapse = true
  production_policy_change_justified = false

next_diagnostic =
  capture compiler-level strict-f32 barrier codegen before any production QK256
  policy change
```

Interpretation: the selected device strict-f32 barrier expression matches the
A770 output bit, but the host compiler/replay side still collapses to the host
policy bit. This is useful diagnostic evidence, not a policy change. The
focused CPU/A770 answer-parity receipt still fails for the selected case, so it
does not prove CPU/A770 answer parity, reference parity, strict A770 answer
readiness, broad A770 quality, residency, speed, trusted partial acceleration,
or full BitNet inference.

## Claim Boundary

This report makes no new claim:

- no production QK256 dispatch change;
- no answer scoring or sampling change;
- no CPU/A770 parity claim;
- no strict A770 answer-readiness claim;
- no broad A770 quality claim;
- no residency, speedup, or trusted partial acceleration claim;
- no full BitNet inference claim.
