# A770-068 One More QK256 Replay Target

## Scope

A770-068 advances the A770-067 focused replay packet by converting exactly one
additional `dispatch_replay_missing` Q/K/V target into selected-device Intel Arc
A770 OpenCL production replay.

The proof remains intentionally small:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
new_target = a770_summary_seed770024_keywords_014:step9:0:k_proj
kernel_family = qk256_i2s_i8s_scaled_gemv
```

This work does not change production QK256 dispatch policy, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipts

The new focused source receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-summary-logits-k-proj-raw-operands.json
```

The combined manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-qk256-one-more-target-replay-manifest.json
```

It explicitly points back to the committed A770-067 manifest and selected-device
replay packet, then overlays one supplemental focused source for the new
`k_proj` target.

The selected-device A770 OpenCL replay receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-qk256-one-more-target-replay.json
```

It records:

```text
work_item = A770-068
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

The manifest still identifies the same first-mismatch Q/K/V target set from
A770-067:

```text
target_count = 90
dispatch_replay_target_count = 2
runnable_target_count = 2
```

The runnable targets are:

```text
a770_summary_seed770024_keywords_014:step9:0:q_proj
a770_summary_seed770024_keywords_014:step9:0:k_proj
```

The selected-device production replay executed both targets on Intel Arc A770
OpenCL with no fallback and matched the selected-device bits:

```text
executed_target_count = 2
blocked_target_count = 88
failed_target_count = 0
matched_selected_device_bits_count = 2
all_executed_targets_match_selected_device_bits = true
kernel_invocations = 4
host_to_device_bytes = 6400
device_to_host_bytes = 112
```

The remaining blockers are still ledgered, not ignored:

```text
dispatch_replay_missing = 88
```

## Interpretation

A770-068 proves cumulative focused replay progress beyond A770-067 without
broadening the proof loop. The replay tool now keeps the A770-067 target set and
can overlay raw-operand dispatch replay from supplemental focused sources only
when a source matches an existing manifest target.

This is still not broad multi-case proof. The committed evidence still spans one
divergent case, one first-mismatch step, and two layer-0 Q/K targets. The next
correctness step is to capture and replay the matching `v_proj` target before
expanding to later Q/K/V projections, O projection, or MLP linears.

## Path To Effective Fast Inference

The lane should keep advancing in this order:

1. Finish focused Q/K/V replay for layer 0 under the same selected-device,
   fallback-free receipt rules.
2. Expand replay coverage to additional Q/K/V targets, then O projection and
   MLP QK256 linears.
3. Reconnect clean projection replay to one-step logits before answer scoring or
   sampling changes.
4. Prove a small bounded answer-parity packet only after the focused projection
   path is clean.
5. Build residency and performance evidence after correctness: persistent
   OpenCL program and queue, resident QK256 weights, reusable buffers, transfer
   accounting, kernel timing, and warm decode profiles.

## Validation

```text
rustfmt --edition 2024 crates/bitnet-kernels/src/bin/a770_opencl_production_replay_instrumentation.rs
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
target/release/bitnet.exe answer-corpus --corpus ci/quality/a770-bitnet-answer-readiness-corpus.yaml --model E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/tokenizer.json --device intel-a770-opencl --case-id a770_summary_seed770024_keywords_014 --dump-logit-steps 24 --logits-topk 20 --json-out ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-summary-logits-k-proj-raw-operands.json --per-prompt-timeout-seconds 300
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json --focused-source ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-summary-logits-k-proj-raw-operands.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --work-item A770-068 --manifest ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-qk256-one-more-target-replay-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-qk256-one-more-target-replay.json
pwsh -NoProfile -Command "$m = Get-Content 'ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-qk256-one-more-target-replay-manifest.json' -Raw | ConvertFrom-Json; $r = Get-Content 'ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-qk256-one-more-target-replay.json' -Raw | ConvertFrom-Json; $newTarget = $r.target_results | Where-Object { $_.target_id -eq 'a770_summary_seed770024_keywords_014:step9:0:k_proj' }; if ($m.target_count -ne 90 -or $m.dispatch_replay_target_count -ne 2 -or $m.runnable_target_count -ne 2 -or $r.summary.executed_target_count -ne 2 -or $r.summary.blocked_target_count -ne 88 -or $r.summary.failed_target_count -ne 0 -or $r.summary.matched_selected_device_bits_count -ne 2 -or $r.summary.all_executed_targets_match_selected_device_bits -ne $true -or $r.fallback_used -ne $false -or $r.work_item -ne 'A770-068' -or $newTarget.production_replay.executed -ne $true -or $newTarget.production_replay.fallback_used -ne $false) { throw 'A770-068 receipt summary mismatch' }"
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
