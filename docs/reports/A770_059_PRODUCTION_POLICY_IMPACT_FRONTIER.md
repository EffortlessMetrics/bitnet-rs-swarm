# A770-059 Production Policy Impact Frontier

## Scope

A770-059 is a diagnostic-only follow-on to A770-058. A770-058 classified the
selected-device OpenCL disassembly for the diagnostic QK256 debug kernel as a
barrier-preserving strict-f32 sequence.

This slice reviews that committed disassembly against the focused QK256 replay
receipt chain. It decides whether the new disassembly context is enough to
change production QK256 policy or whether the next boundary still needs
production-kernel context.

It does not change production QK256 dispatch, runtime math, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Inputs

```text
A770-053:
  generated_output_qk256_strict_f32_barrier_evidence_frontier

A770-054:
  generated_output_qk256_host_compiler_strict_f32_barrier_frontier

A770-055:
  generated_output_qk256_compiler_strict_f32_codegen_frontier

A770-057:
  a770_opencl_qk256_compiler_disassembly_evidence

A770-058:
  a770_opencl_qk256_strict_f32_disassembly_frontier
```

Committed receipt:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-policy-impact-frontier.json
```

## Classification

```text
a770_qk256_production_policy_impact_frontier_missing_context
a770_qk256_production_policy_impact_frontier_bounded_debug_kernel_context
a770_qk256_production_policy_impact_frontier_requires_production_kernel_disassembly_replay_context
a770_qk256_production_policy_impact_frontier_production_policy_change_not_justified
a770_qk256_production_policy_impact_frontier_clean
```

## Live Result

The committed frontier classifies as:

```text
classification =
  a770_qk256_production_policy_impact_frontier_requires_production_kernel_disassembly_replay_context

focused case:
  case_id = a770_summary_seed770024_keywords_014
  first_mismatch_index = 9
  target_layer_idx = 0
  projection = q_proj

upstream:
  qk256_device_expression =
    generated_output_qk256_device_expression_unmatched_device_value
  qk256_device_math_mode =
    generated_output_qk256_device_math_mode_default_div_then_mul
  qk256_host_device_div_mul =
    generated_output_qk256_host_device_div_mul_host_replay_mismatch
  qk256_host_replay_f32_codegen_ordering =
    generated_output_qk256_host_replay_f32_codegen_ordering_host_expression_variants_collapsed_to_policy
  qk256_strict_f32_barrier_evidence =
    generated_output_qk256_strict_f32_barrier_evidence_strict_f32_barrier_matches_selected_device_output
  qk256_host_compiler_strict_f32_barrier =
    generated_output_qk256_host_compiler_strict_f32_barrier_host_compiler_codegen_collapse
  qk256_compiler_strict_f32_codegen =
    generated_output_qk256_compiler_strict_f32_codegen_host_compiler_elides_explicit_f32_barrier
  strict_f32_disassembly_frontier =
    a770_qk256_strict_f32_disassembly_frontier_barrier_preserving_f32_sequence

debug_kernel_name = qk256_i2s_i8s_scaled_gemv_debug
production_kernel_name = qk256_i2s_i8s_scaled_gemv
debug_kernel_disassembly_available = true
debug_kernel_barrier_sequence_preserved = true
production_kernel_disassembly_available = false
production_kernel_replay_context_available = false
production_policy_change_justified = false
```

Interpretation: the A770-058 disassembly proves that the diagnostic debug
kernel preserves a strict-f32 barrier sequence in the lowered A770 assembly.
That is useful context, but the committed assembly is for
`qk256_i2s_i8s_scaled_gemv_debug`, not the production
`qk256_i2s_i8s_scaled_gemv` kernel. The focused replay chain also still records
`production_policy_change_justified=false`.

Therefore A770-059 does not justify a production QK256 policy change. The next
diagnostic boundary is production-kernel disassembly or production-kernel replay
context, still claim-closed.

## Validation

```text
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-policy-impact-frontier.json -Raw | ConvertFrom-Json | Out-Null"
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Claim Boundary

This report makes no new claim:

- no production QK256 dispatch change;
- no answer scoring or sampling change;
- no CPU/A770 answer parity claim;
- no reference parity claim;
- no strict A770 answer-readiness claim;
- no broad A770 quality claim;
- no selected attention, resident KV, attention score, softmax, value-mix, or
  full-residency claim;
- no speedup or trusted partial-acceleration claim;
- no full BitNet inference claim.
