# A770-055 Compiler Strict F32 Codegen

## Scope

A770-055 is a diagnostic-only follow-on to A770-054. A770-054 classified the
selected layer-0 `q_proj` QK256 row as host compiler/codegen collapse while the
selected OpenCL device strict-f32 barrier expression still matched the A770
output bit.

This slice records compact compiler-level strict-f32 codegen context for that
same focused expression. It does not change production QK256 dispatch, runtime
math, answer scoring, sampling, model loading, route policy, or any promotion
state.

## Receipt

The focused answer-parity receipt now emits:

```text
generated_output_qk256_compiler_strict_f32_codegen_frontier
```

Rows include the selected case, mismatch index, layer/projection, upstream
QK256 source-frontier classifications, selected host/device expression bits,
explicit barrier-match flags, inferred host compiler preserve/elide flags,
OpenCL frontend/device codegen match flags, production-policy availability, and
device identity. Full output vectors are not forwarded.

## Classifications

```text
generated_output_qk256_compiler_strict_f32_codegen_frontier_missing_context
generated_output_qk256_compiler_strict_f32_codegen_frontier_production_policy_change_not_justified
generated_output_qk256_compiler_strict_f32_codegen_frontier_opencl_frontend_device_codegen_split
generated_output_qk256_compiler_strict_f32_codegen_frontier_host_compiler_elides_explicit_f32_barrier
generated_output_qk256_compiler_strict_f32_codegen_frontier_host_compiler_preserves_explicit_f32_barrier
generated_output_qk256_compiler_strict_f32_codegen_frontier_opencl_frontend_device_codegen_matches_selected_output
generated_output_qk256_compiler_strict_f32_codegen_frontier_clean
```

## Live Result

The refreshed focused receipt reports:

```text
classification =
  generated_output_qk256_compiler_strict_f32_codegen_frontier_host_compiler_elides_explicit_f32_barrier

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

upstream:
  qk256_host_compiler_strict_f32_barrier =
    generated_output_qk256_host_compiler_strict_f32_barrier_host_compiler_codegen_collapse
  qk256_strict_f32_barrier_evidence =
    generated_output_qk256_strict_f32_barrier_evidence_strict_f32_barrier_matches_selected_device_output
  qk256_host_replay_f32_codegen_ordering =
    generated_output_qk256_host_replay_f32_codegen_ordering_host_expression_variants_collapsed_to_policy

selected sample:
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  driver_version = 32.0.101.8801

  device_output_bits = 3215804927
  host_div_then_mul_bits = 3215804926
  device_div_then_mul_bits = 3215804927
  device_mul_then_div_bits = 3215804926

  compiler_codegen_bits_compared = true
  explicit_f32_barrier_codegen_match = true
  opencl_frontend_device_codegen_matches_selected_output = true
  host_compiler_elides_explicit_f32_barrier = true
  host_compiler_preserves_explicit_f32_barrier = false
  production_policy_change_justified = false

next_diagnostic =
  capture compiler/disassembly evidence for explicit f32 barrier elision before
  any production QK256 policy change
```

Interpretation: the selected OpenCL device strict-f32 barrier value still
matches the A770 output bit, but the host side remains on the policy bit and is
classified as eliding/collapsing the explicit strict-f32 barrier at this
diagnostic boundary. This is useful attribution evidence, not a runtime policy
change. The focused CPU/A770 answer-parity receipt still fails for the selected
case, so it does not prove CPU/A770 answer parity, reference parity, strict A770
answer readiness, broad A770 quality, residency, speed, trusted partial
acceleration, or full BitNet inference.

## Validation

```text
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli qk256_compiler_strict_f32_codegen -- --nocapture

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- answer-parity --left ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json --right ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json --left-label amd-5700x-cpu-avx2 --right-label intel-a770-opencl --machine amd-5700x-intel-a770 --json-out ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json
```

The answer-parity command is expected to exit nonzero while CPU/A770 parity
remains divergent; it still writes the refreshed diagnostic receipt.

## Claim Boundary

This report makes no new claim:

- no production QK256 dispatch change;
- no answer scoring or sampling change;
- no CPU/A770 parity claim;
- no strict A770 answer-readiness claim;
- no broad A770 quality claim;
- no residency, speedup, or trusted partial acceleration claim;
- no full BitNet inference claim.
