# A770-158 Projection Replay Hook Boundary

## Scope

A770-158 uses the committed A770-157 projection-level Q/K/V boundary receipt to
add the smallest bounded projection-level replay hook. It does not capture full
projection operands yet, and it does not run a broad corpus, model download,
hardware matrix, full workspace, Mac lane, Windows lane, production QK256
policy change, answer scoring change, sampling change, or benchmark.

The proof stays narrow:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection_targets = q_proj, k_proj, v_proj
source_packet = A770-157 projection-level Q/K/V boundary receipt
```

## Receipts

The source packet is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-qkv-projection-level-replay-boundary/a770-opencl-qkv-projection-level-replay.json
```

The A770-158 bounded-hook manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-projection-replay-hook-boundary/a770-opencl-qkv-projection-replay-hook-manifest.json
```

The A770-158 receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-projection-replay-hook-boundary/a770-opencl-qkv-projection-replay-hook.json
```

It records:

```text
work_item = A770-158
proof_family = a770_opencl_qk256_projection_level_qkv_replay
proof_stage = diagnostic_projection_level_qkv_replay_boundary
requested_backend = intel-arc-a770
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
fallback_used = false
claim_allowed = false
diagnostic_only = true
projection_replay_kernel_name = qk256_i2s_i8s_scaled_gemv_production_replay
projection_replay_kernel_executed = false
```

## Classification

```text
a770_qk256_projection_level_qkv_replay_blocked_on_projection_operands
```

The manifest targets the layer-0 Q/K/V projection boundary for the same case
and first mismatch:

```text
target_count = 3
projection_replay_target_count = 3
projection_replay_executed_count = 0
projection_replay_blocked_count = 3
projection_replay_hook_available_count = 3
projection_level_full_operands_available_count = 0
projection_replay_fallback_false_count = 3
row_evidence_target_count = 3
clean_row_evidence_count = 3
row_selected_device_match_count = 3
row_fallback_false_count = 3
all_row_evidence_clean = true
all_projection_replay_targets_blocked = true
```

The remaining blocker is deliberately narrow:

```text
projection_level_full_operands_missing
```

The previous `projection_level_replay_hook_missing` blocker is removed by this
work item. The hook is available through the selected-device A770 OpenCL
production replay instrumentation, but it is not executed by this receipt
because the committed A770-157 source packet still contains row-level evidence,
not full projection Q/K/V operands.

Each projection target preserves the row-level A770-157 evidence:

```text
row_evidence.available = true
row_evidence.executed = true
row_evidence.selected_backend = intel-arc-a770-opencl
row_evidence.runtime_api = opencl
row_evidence.fallback_used = false
row_evidence.production_output_matches_selected_device_bits = true
row_evidence.clean_for_projection_boundary = true
projection_replay.projection_level_replay_hook_available = true
projection_replay.projection_level_full_operands_available = false
projection_replay.executed = false
```

## Interpretation

A770-158 lands one half of the A770-158 acceptance criteria: the bounded
projection-level replay hook exists and is ledgered in a receipt. The proof
does not claim projection-level replay success because the full projection Q/K/V
operand capture is still missing. The next runtime step is to capture those
operands for the same one case, one first-mismatch index, and layer-0 Q/K/V
surface, then execute the bounded hook on selected-device Intel Arc A770 OpenCL
with `fallback_used=false`.

## Validation

```text
cargo fmt --package bitnet-kernels
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --projection-source ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-qkv-projection-level-replay-boundary/a770-opencl-qkv-projection-level-replay.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --projection-layer 0 --work-item A770-158 --manifest ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-projection-replay-hook-boundary/a770-opencl-qkv-projection-replay-hook-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-projection-replay-hook-boundary/a770-opencl-qkv-projection-replay-hook.json
```

The receipt is a local bounded-hook classifier output. It does not run a live
projection replay, broad answer corpus, model download, benchmark, or hardware
matrix.

## Claim Boundary

This report makes no promotion claim:

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
