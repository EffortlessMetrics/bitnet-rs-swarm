# A770-065 Focused Host Policy Expression Split

## Scope

A770-065 is a diagnostic-only follow-up to A770-064. A770-064 showed that the
focused raw QK256 row can be fed into the selected-device production replay
kernel and that the production replay reproduces the A770 device output bit.

This slice localizes the remaining one-bit split for:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj
```

It does not change production QK256 dispatch, runtime math, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipt

The focused host-policy expression-split receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-host-policy-expression-split.json
```

It records:

```text
work_item = A770-065
proof_family = a770_opencl_qk256_focused_host_policy_expression_split
proof_stage = diagnostic_focused_host_policy_expression_split_classified
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
fallback_used = false
cpu_fallback_allowed = false
bitnet_inference = false
qk256_decode = false
production_qk256_policy_change = false
claim_allowed = false
diagnostic_only = true
```

## Classification

```text
a770_qk256_focused_host_policy_expression_split_host_summary_policy_replay_one_bit
```

The focused selected-device replay still matches the A770 output bit:

```text
focused_device_output_bits = 3215804927
selected_device_production_output_bits = 3215804927
selected_device_replay_output_bits = 3215804927
selected_device_final_scaled_value_bits = 3215804927
selected_device_replay_matches_device_bits = true
```

The source host-policy expression trace is one bit lower:

```text
focused_host_policy_bits = 3215804926
device_vs_host_policy_bit_delta = 1
selected_device_replay_matches_host_policy_bits = false
source_host_policy_variants_all_match = true
```

The scalar oracle and selected-device replay agree on the integer side of the
focused row:

```text
activation_sum_from_raw = 1063
activation_sum_matches_receipt = true
int_dot = -860
adjusted_dot = -1923
row_stride_matches_expected = true
int_dot_matches_selected_device_replay = true
adjusted_dot_matches_selected_device_replay = true
```

The focused row has no tail or padding ambiguity:

```text
cols = 2560
used_payload_bytes = 640
row_padding_bytes = 0
unused_tail_columns = 0
no_tail_or_padding_for_focused_row = true
```

## Interpretation

A770-065 rules out missing raw operands, QK256 packing/decode, activation-sum
correction, row stride, tail padding, and selected-device production replay as
the cause of this focused one-bit split.

The selected A770 OpenCL production replay follows the reciprocal-path final
scaled value and stores bit `3215804927`. The host summary policy trace records
bit `3215804926` for the same focused row. The next useful work item is a
bounded host summary-policy semantic fix, followed by a selected-device focused
parity replay before any broader production QK256 promotion.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-kernels/src/bin/a770_opencl_production_replay_instrumentation.rs
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/cpu-avx2-vs-a770-summary-logits-raw-operands-parity.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-host-policy-expression-split.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-host-policy-expression-split.json -Raw | ConvertFrom-Json | Out-Null"
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
