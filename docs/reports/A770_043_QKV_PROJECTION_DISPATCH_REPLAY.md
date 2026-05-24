# A770-043 QKV Projection Dispatch Replay

## Scope

A770-043 is a diagnostic-only continuation of A770-042. A770-042 localized the
focused `a770_summary_seed770024_keywords_014` mismatch to the selected layer-0
`q_proj` QKV projection source frontier and reported CPU versus A770 dispatch
path drift with matching projection input and raw QK256 metadata.

This slice adds a focused replay receipt for that selected QKV projection. The
replay uses the same materialized projection input and raw QK256 tensor metadata
when the focused source context is available, then compares a CPU scalar
reference replay against an A770 OpenCL replay without changing the production
dispatch decision.

## Receipt

The focused answer-parity receipt emits:

```text
generated_output_qkv_projection_dispatch_replay_frontier
```

The report is compact. It records the focused case, mismatch index, layer,
projection, input/raw metadata equality, CPU/A770 replay fingerprints, CPU/A770
replay counters, and the selected classification.

## Classifications

```text
generated_output_qkv_projection_dispatch_replay_frontier_missing_context
generated_output_qkv_projection_dispatch_replay_frontier_runtime_replay_mismatch
generated_output_qkv_projection_dispatch_replay_frontier_cpu_a770_output_drift
generated_output_qkv_projection_dispatch_replay_frontier_clean
```

## Claim Boundary

This is not a runtime fix and not an A770 readiness promotion. It does not prove
CPU/A770 answer parity, reference parity, strict A770 answer readiness, broad
A770 quality, official BitNet QK256 production semantics, selected attention
residency, resident KV, full A770 residency, performance speedup, trusted
partial acceleration, or full BitNet inference.

## Live Result

The refreshed focused receipt reports:

```text
classification =
  generated_output_qkv_projection_dispatch_replay_frontier_cpu_a770_output_drift

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

qkv_projection_source_classification =
  generated_output_qkv_projection_source_dispatch_path_drift

qkv_projection_dispatch_replay_context_available = true
left_runtime_output_matches_cpu_replay = true
right_runtime_output_matches_a770_replay = true
cpu_replay_output_match_across_receipts = true
a770_replay_output_match_across_receipts = true

left_cpu_a770_replay_output_match = false
right_cpu_a770_replay_output_match = false
cpu_a770_output_rms_abs_delta = 3.980823781724041e-08

next_diagnostic =
  inspect selected QK256 CPU scalar versus A770 OpenCL GEMV numeric policy
```

Interpretation: the focused replay is available and stable. The CPU receipt's
runtime selected projection output matches the CPU replay, and the A770 receipt's
runtime selected projection output matches the A770 replay. The remaining
selected projection split is therefore localized to CPU scalar replay versus
A770 OpenCL replay numeric output, not to missing context or replay capture
scope.
