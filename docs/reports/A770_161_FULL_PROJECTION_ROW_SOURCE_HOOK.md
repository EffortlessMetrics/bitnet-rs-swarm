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

The current local Windows Cargo build is slow and was not completed within the
five-minute command budget. Formatting and static diff checks remain required;
hosted exact-head checks are the authoritative compile/test proof for this PR.

## Claim boundary

This report does not claim CPU/A770 parity, reference parity, answer readiness,
broad quality, residency, speed, trusted partial acceleration, full BitNet
inference, or a production QK256 policy change.
