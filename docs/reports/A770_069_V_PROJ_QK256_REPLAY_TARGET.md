# A770-069 V Projection QK256 Replay Target

## Scope

A770-069 advances the focused QK256 replay packet by converting the matching
layer-0 `v_proj` target into selected-device Intel Arc A770 OpenCL production
replay.

The proof stays deliberately small:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
new_target = a770_summary_seed770024_keywords_014:step9:0:v_proj
kernel_family = qk256_i2s_i8s_scaled_gemv
```

This work does not change production QK256 dispatch policy, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipts

The new focused source receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-summary-logits-v-proj-raw-operands.json
```

The combined manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-qk256-v-proj-target-replay-manifest.json
```

It preserves the A770-067 90-target manifest shape, keeps the A770-068 `k_proj`
overlay, and adds one supplemental focused source for the new `v_proj` target.

The selected-device A770 OpenCL replay receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-qk256-v-proj-target-replay.json
```

It records:

```text
work_item = A770-069
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

The manifest still identifies the same first-mismatch Q/K/V target set:

```text
target_count = 90
dispatch_replay_target_count = 3
runnable_target_count = 3
```

The runnable targets are:

```text
a770_summary_seed770024_keywords_014:step9:0:q_proj
a770_summary_seed770024_keywords_014:step9:0:k_proj
a770_summary_seed770024_keywords_014:step9:0:v_proj
```

The selected-device production replay executed all three targets on Intel Arc
A770 OpenCL with no fallback and matched the selected-device bits:

```text
executed_target_count = 3
blocked_target_count = 87
failed_target_count = 0
matched_selected_device_bits_count = 3
all_executed_targets_match_selected_device_bits = true
kernel_invocations = 6
host_to_device_bytes = 9600
device_to_host_bytes = 168
```

The remaining blockers are still ledgered:

```text
dispatch_replay_missing = 87
```

## Interpretation

A770-069 completes the focused layer-0 Q/K/V replay trio for the committed
first-mismatch case without broadening the proof loop. This is useful correctness
evidence, not speed evidence.

The next narrow frontier is to convert one more target from the remaining
`dispatch_replay_missing` set. The most direct continuation is layer-1 Q/K/V or
the next projection boundary selected by the manifest, still one target at a
time until the replay surface is broad enough to justify a projection-level
promotion review.

## Path To Effective Fast Inference

The lane should keep advancing in this order:

1. Burn down focused Q/K/V replay blockers under the same selected-device,
   fallback-free receipt rules.
2. Expand replay coverage to O projection and MLP QK256 linears.
3. Reconnect clean projection replay to one-step logits before answer scoring or
   sampling changes.
4. Prove a small bounded answer-parity packet only after the focused projection
   path is clean.
5. Build residency and performance evidence after correctness: persistent
   OpenCL program and queue, resident QK256 weights, reusable buffers, transfer
   accounting, kernel timing, and warm decode profiles.

## Validation

```text
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
target/release/bitnet.exe answer-corpus --corpus ci/quality/a770-bitnet-answer-readiness-corpus.yaml --model E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/tokenizer.json --device intel-a770-opencl --case-id a770_summary_seed770024_keywords_014 --dump-logit-steps 24 --logits-topk 20 --json-out ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-summary-logits-v-proj-raw-operands.json --per-prompt-timeout-seconds 300
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json --focused-source ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-one-more-qk256-replay-target/a770-opencl-summary-logits-k-proj-raw-operands.json --focused-source ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-summary-logits-v-proj-raw-operands.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --work-item A770-069 --manifest ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-qk256-v-proj-target-replay-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-06-02/a770-v-proj-qk256-replay-target/a770-opencl-qk256-v-proj-target-replay.json
pwsh receipt assertion for target_count=90, dispatch_replay_target_count=3, runnable_target_count=3, executed_target_count=3, blocked_target_count=87, failed_target_count=0, matched_selected_device_bits_count=3, all_executed_targets_match_selected_device_bits=true, fallback_used=false, and v_proj executed fallback-free with selected-device bit match
```

## Claim Boundary

This report makes no new promotion claim:

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
