# A770-160 Full Projection Packed Row Capture Source Boundary

## Scope

A770-160 uses the committed A770-159 full projection operand source boundary
receipt to inspect whether the same narrow projection surface now has a full
projection packed QK256 row capture source:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection_targets = q_proj, k_proj, v_proj
source_packet = A770-159 full projection operand source boundary receipt
```

The work does not run a broad corpus, model download, hardware matrix, full
workspace, Mac lane, Windows lane, production QK256 policy change, answer
scoring change, sampling change, or benchmark.

## Receipts

The source packet is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-operands-capture-boundary/a770-opencl-qkv-full-projection-operands-capture.json
```

The A770-160 manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-packed-row-capture-source-boundary/a770-opencl-qkv-full-projection-packed-row-capture-manifest.json
```

The A770-160 receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-packed-row-capture-source-boundary/a770-opencl-qkv-full-projection-packed-row-capture.json
```

It records:

```text
work_item = A770-160
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
a770_qk256_projection_level_qkv_replay_blocked_on_full_projection_packed_row_capture_source
```

The receipt preserves the A770-158 replay hook and the A770-159 operand source
boundary, then makes the remaining source blocker more specific:

```text
target_count = 3
projection_replay_target_count = 3
projection_replay_executed_count = 0
projection_replay_blocked_count = 3
projection_replay_hook_available_count = 3
projection_level_full_operands_available_count = 0
projection_replay_fallback_false_count = 3
projection_operand_capture_source_count = 3
projection_focused_operand_source_count = 3
projection_full_operand_source_count = 0
row_evidence_target_count = 3
clean_row_evidence_count = 3
row_selected_device_match_count = 3
row_fallback_false_count = 3
all_row_evidence_clean = true
all_projection_replay_targets_blocked = true
```

The blockers are:

```text
projection_level_full_operands_missing
projection_level_full_projection_weight_rows_missing
projection_level_full_projection_packed_row_capture_source_missing
```

Each target still has a committed focused source packet, but each packet
contains only single-output-row packed QK256 bytes:

| Projection | Source | Packed bytes present | Required packed bytes |
| --- | --- | ---: | ---: |
| q_proj | `ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json` | 640 | 1,638,400 |
| k_proj | `ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-summary-logits-k-proj-raw-operands.json` | 640 | 409,600 |
| v_proj | `ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-summary-logits-v-proj-raw-operands.json` | 640 | 409,600 |

For all three targets:

```text
projection_operand_capture.current_operand_scope = single_output_row
projection_operand_capture.required_operand_scope = full_projection_output_rows
projection_operand_capture.required_packed_qk256_len_available = false
projection_replay.executed = false
projection_replay.fallback_used = false
projection_replay.projection_level_replay_hook_available = true
projection_replay.projection_level_full_operands_available = false
projection_replay.missing_full_operand_fields = projection_operands.packed_qk256_full_projection_rows
```

## Interpretation

A770-160 does not execute projection-level replay because the committed source
boundary still lacks a full projection packed-row capture source. This is a
source-blocker ledger, not a failed replay. The selected-device route remains
Intel Arc A770 OpenCL with `fallback_used=false`, and no CPU fallback, generic
OpenCL, OpenVINO GPU, Arc 140V, NPU, or CUDA result is substituted.

The next runtime step is to add a capture hook or source packet that records
the full projection packed QK256 rows for this same layer-0 Q/K/V surface, then
run the existing bounded projection replay hook fallback-free.

## Validation

```text
cargo fmt --package bitnet-kernels
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --projection-source ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-operands-capture-boundary/a770-opencl-qkv-full-projection-operands-capture.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --projection-layer 0 --work-item A770-160 --manifest ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-packed-row-capture-source-boundary/a770-opencl-qkv-full-projection-packed-row-capture-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-packed-row-capture-source-boundary/a770-opencl-qkv-full-projection-packed-row-capture.json
git diff --check
parse the A770-160 manifest and receipt with ConvertFrom-Json
assert selected_backend=intel-arc-a770-opencl, runtime_api=opencl, fallback_used=false, projection_replay_kernel_executed=false, and projection_level_full_projection_packed_row_capture_source_missing is present
check touched Markdown reports for balanced code fences
```

The receipt is a local bounded classifier output. It does not run a live
projection replay, broad answer corpus, model download, benchmark, or hardware
matrix.

Local `xtask` campaign validation could not complete in this Windows checkout.
`cargo run --locked -p xtask --no-default-features -- campaign check
intel-a770` timed out after 304 seconds while building `xtask`, and the
leftover Cargo/CMake children were stopped. `campaign generate --check` was not
rerun after the same `xtask` build blocker because this PR does not edit
campaign source files or generated dashboards. Substitute evidence is the
focused `bitnet-kernels` binary check, JSON parsing and assertions for the new
manifest and receipt, Markdown fence checks, and `git diff --check`.

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
