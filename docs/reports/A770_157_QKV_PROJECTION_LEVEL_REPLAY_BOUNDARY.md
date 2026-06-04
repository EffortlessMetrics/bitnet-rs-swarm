# A770-157 Q/K/V Projection-Level Replay Boundary

## Scope

A770-157 starts the projection-level replay frontier from the committed
A770-156 focused Q/K/V packet. It does not run a broad corpus, model download,
hardware matrix, full workspace, Mac lane, Windows lane, production QK256
policy change, answer scoring change, sampling change, or benchmark.

The proof stays narrow:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection_targets = q_proj, k_proj, v_proj
source_packet = A770-156 selected-device focused Q/K/V replay
```

## Receipts

The source packet is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-layer29-v-proj-qk256-replay-target/a770-opencl-qk256-layer29-v-proj-target-replay.json
```

The A770-157 projection-level boundary manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-qkv-projection-level-replay-boundary/a770-opencl-qkv-projection-level-replay-manifest.json
```

The A770-157 receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-qkv-projection-level-replay-boundary/a770-opencl-qkv-projection-level-replay.json
```

It records:

```text
work_item = A770-157
proof_family = a770_opencl_qk256_projection_level_qkv_replay
proof_stage = diagnostic_projection_level_qkv_replay_boundary
requested_backend = intel-arc-a770
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
fallback_used = false
claim_allowed = false
diagnostic_only = true
```

## Classification

```text
a770_qk256_projection_level_qkv_replay_blocked_on_projection_operands_and_hook
```

The manifest targets the layer-0 Q/K/V projection boundary for the same case
and first mismatch:

```text
target_count = 3
projection_replay_target_count = 3
projection_replay_executed_count = 0
projection_replay_blocked_count = 3
row_evidence_target_count = 3
clean_row_evidence_count = 3
row_selected_device_match_count = 3
row_fallback_false_count = 3
all_row_evidence_clean = true
all_projection_replay_targets_blocked = true
```

The blocker is deliberately explicit:

```text
projection_level_full_operands_missing
projection_level_replay_hook_missing
```

Each projection target preserves the row-level A770-156 evidence:

```text
row_evidence.available = true
row_evidence.executed = true
row_evidence.selected_backend = intel-arc-a770-opencl
row_evidence.runtime_api = opencl
row_evidence.fallback_used = false
row_evidence.production_output_matches_selected_device_bits = true
row_evidence.clean_for_projection_boundary = true
```

The projection-level replay itself did not execute because the committed
A770-156 packet only contains single focused output-row operands. It does not
contain full projection output-row operands, and the instrumentation does not
yet expose a bounded projection-level replay hook.

## Interpretation

A770-157 is useful because it turns "move to projection replay" into a concrete
receipt boundary. The row evidence is clean enough to start from, but the next
runtime step is not a performance run and not a production dispatch promotion.
The next runtime step is to capture full projection Q/K/V operands and add the
smallest projection-level replay hook that can run on selected-device Intel Arc
A770 OpenCL with `fallback_used=false`.

## Validation

```text
cargo fmt --package bitnet-kernels
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --projection-source ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-layer29-v-proj-qk256-replay-target/a770-opencl-qk256-layer29-v-proj-target-replay.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --projection-layer 0 --work-item A770-157 --manifest ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-qkv-projection-level-replay-boundary/a770-opencl-qkv-projection-level-replay-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-06-04/a770-qkv-projection-level-replay-boundary/a770-opencl-qkv-projection-level-replay.json
```

The receipt is a local classifier output. It does not run a live hardware
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
