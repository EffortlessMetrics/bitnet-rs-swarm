# A770-067 Multi-Case Focused QK256 Replay

## Scope

A770-067 turns the focused A770-064 through A770-066 evidence into a
manifest-bound replay packet. The packet is built from committed raw operand
receipts and names every Q/K/V projection target available at the first
CPU/A770 summary-logits mismatch for:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
```

This is still diagnostic evidence. It does not change production QK256
dispatch, answer scoring, sampling, model loading, route policy, or any A770
quality, residency, speed, trusted partial-acceleration, or full-inference
claim.

## Receipts

The manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay-manifest.json
```

The selected-device A770 OpenCL replay receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay.json
```

It records:

```text
work_item = A770-067
proof_family = a770_opencl_qk256_multi_case_focused_replay
proof_stage = diagnostic_multi_case_focused_qk256_replay_packet
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
a770_qk256_multi_case_focused_replay_partial_manifest_blocked_on_missing_raw_operands
```

The manifest identifies the whole first-mismatch Q/K/V target set for the
committed case:

```text
target_count = 90
dispatch_replay_target_count = 1
runnable_target_count = 1
```

The one runnable target is:

```text
target_id = a770_summary_seed770024_keywords_014:step9:0:q_proj
tensor_name = layers.0.attention.q_proj.weight
qk256_key = layers.0.attention.q_proj.weight.qk256_qs
```

The selected-device production replay executed that target on Intel Arc A770
OpenCL with no fallback and matched the selected-device bits:

```text
executed_target_count = 1
blocked_target_count = 89
failed_target_count = 0
matched_selected_device_bits_count = 1
all_executed_targets_match_selected_device_bits = true
kernel_invocations = 2
host_to_device_bytes = 3200
device_to_host_bytes = 56
```

The blocked targets are ledgered, not ignored:

```text
dispatch_replay_missing = 89
```

## Interpretation

A770-067 improves the lane by making the next replay frontier explicit. The
repo now has a manifest shape that can scale from one focused row to additional
projection targets without conflating missing receipts with correctness.

This is not yet broad multi-case proof. The committed evidence currently spans
one divergent case and one first-mismatch step. Only one target has
object-valued dispatch replay and raw focused operands. The remaining targets
need raw operand capture before they can become selected-device replay evidence.

## Path To Effective Fast Inference

The A770 lane should move toward fast inference in this order:

1. Correctness frontier: capture raw focused operands for more Q/K/V targets,
   then expand from Q/K/V to O projection and MLP QK256 linears. Promotion
   should require selected-device replay matches with `fallback_used=false`.
2. One-step model frontier: after focused projection replay is clean, prove a
   single model step can route the intended QK256 linears through A770 OpenCL
   while CPU reference remains available for comparison.
3. Residency frontier: keep OpenCL program, command queue, QK256 weights,
   activations, and output buffers alive across steps. Track transfer bytes and
   kernel invocations per token.
4. Decode frontier: only after one-step correctness and residency are clean,
   run five-case answer parity and warm decode profiles. These are promotion
   receipts, not default PR checks.
5. Performance frontier: benchmark against the AMD 5700X CPU baseline with the
   same model, tokenizer, prompts, sampling, and fallback policy. A speedup
   claim needs quality passing, selected `intel-arc-a770-opencl`, no fallback,
   and measured low transfer volume.

The near-term engineering work is therefore not broad CI. It is better replay
coverage, resident runtime shape, transfer accounting, and narrow timing
receipts that explain whether the A770 is doing useful work per token.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-kernels/src/bin/a770_opencl_production_replay_instrumentation.rs
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --manifest ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay.json
pwsh -NoProfile -Command "$m = Get-Content 'ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay-manifest.json' -Raw | ConvertFrom-Json; $r = Get-Content 'ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-multi-case-focused-qk256-replay/a770-opencl-qk256-multi-case-focused-replay.json' -Raw | ConvertFrom-Json; if ($m.target_count -ne 90 -or $m.dispatch_replay_target_count -ne 1 -or $r.summary.executed_target_count -ne 1 -or $r.summary.blocked_target_count -ne 89 -or $r.fallback_used -ne $false) { throw 'A770-067 receipt summary mismatch' }"
wsl bash -lc "cd /mnt/e/Code/Rust/bitnet-rs-swarm && ./target/debug/xtask campaign check intel-a770"
wsl bash -lc "cd /mnt/e/Code/Rust/bitnet-rs-swarm && ./target/debug/xtask campaign generate"
wsl bash -lc "cd /mnt/e/Code/Rust/bitnet-rs-swarm && ./target/debug/xtask campaign generate --check"
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
