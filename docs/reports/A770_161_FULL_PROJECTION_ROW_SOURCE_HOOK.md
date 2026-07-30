# A770-161 Full Projection Row Source Hook

## Scope

A770-161 closes the handoff gap between the projection source packet and the
selected-device replay hook for the existing narrow target:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection_targets = q_proj, k_proj, v_proj
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
fallback_used = false
```

The replay instrumentation previously classified full operands discovered in
the external projection source file, but then attempted to replay the stale
row manifest. It now retains the validated external full-operand packet and
uses it when the row manifest does not embed the full operands.

## Source boundary

The existing A770-160 source packet remains:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-operands-capture-boundary/a770-opencl-qkv-full-projection-operands-capture.json
```

That packet contains focused single-output-row evidence only. It does not
contain the physical full projection packed QK256 rows required for replay.
The instrumentation therefore continues to preserve these blockers when the
source packet is unchanged:

```text
projection_level_full_operands_missing
projection_level_full_projection_weight_rows_missing
projection_level_full_projection_packed_row_capture_source_missing
```

No packed weights are synthesized from the focused row evidence.

## Proof

The focused regression test constructs an external projection source packet
with a small valid full-row operand set and verifies that capture evidence
retains the parsed operands for the replay handoff. The production source
packet remains diagnostic-only until a real physical full-row capture is
available.

Local hardware verification was run on the physical Intel Arc A770 in this
checkout. The existing focused A770 probes passed:

```text
BITNET_RUN_A770_OPENCL_SMOKE=1 cargo test --locked --offline -p bitnet-device-probe --no-default-features --features opencl --test a770_opencl_smoke a770_selected_opencl_tiny_vector_add_smoke_runs_when_enabled -- --nocapture
  passed = true
  runtime_device = Intel(R) Arc(TM) A770 Graphics
  platform_name = Intel(R) OpenCL Graphics
  kernel_execution = true
  max_abs_error = 0.0
  fallback_used = false

target/debug/a770-opencl-parity.exe --receipt target/a770-local-matmul-i2s-parity.json
  passed = true
  kernel_name = matmul_i2s
  cpu_opencl_parity = true
  max_abs_error = 0.0
  fallback_used = false
```

A single seeded `answer-corpus --case-id
a770_summary_seed770024_keywords_014` run also selected
`intel-a770-opencl`, generated 15 tokens, and recorded 9,030 A770 OpenCL QK256
linear invocations with zero CPU linear fallbacks. Its receipt remains
diagnostic-only and explicitly makes no answer-quality, residency, speed, or
full-inference claim.

The current PR binary was compiled successfully in an isolated Cargo target:

```text
CARGO_TARGET_DIR=E:\Code\Rust\target-a770-161-local-2
cargo check --locked --offline -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
  finished = true

E:\Code\Rust\target-a770-161-local-2\debug\a770-opencl-production-replay-instrumentation.exe
  --projection-source ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-packed-row-capture-source-boundary/a770-opencl-qkv-full-projection-packed-row-capture.json
  --manifest ci/hardware/amd-5700x-intel-a770/2026-06-05/a770-full-projection-packed-row-capture-source-boundary/a770-opencl-qkv-full-projection-packed-row-capture-manifest.json
  --receipt target/a770-local-161-current-runtime-receipt.json
  --case-id a770_summary_seed770024_keywords_014
  --first-mismatch-index 9
  --projection-layer 0
  --work-item A770-161
  classification = a770_qk256_projection_level_qkv_replay_blocked_on_projection_operands
  target_count = 3
  projection_replay_executed_count = 0
  projection_replay_fallback_false_count = 3
  all_row_evidence_clean = true
  blockers = projection_level_full_operands_missing, projection_level_full_projection_weight_rows_missing
```

The current-branch replay receipt therefore proves the bounded blocker path
against the current A770-161 binary; it does not substitute for physical full
projection packed-row capture.

The regenerated manifest preserves the source identity as
`a770_160_full_projection_packed_row_source_boundary` and records the remaining
physical blocker as missing full projection operands/weight rows rather than
mislabeling the packet as an earlier projection boundary.

## Claim boundary

This report does not claim CPU/A770 parity, reference parity, answer readiness,
broad quality, residency, speed, trusted partial acceleration, full BitNet
inference, or a production QK256 policy change.
