# A770-114 Layer 15 V Projection QK256 Replay Target

## Scope

A770-114 advances the focused QK256 replay packet by converting one more
remaining target, layer-15 `v_proj`, into selected-device Intel Arc A770 OpenCL
production replay.

The proof stays deliberately small:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
new_target = a770_summary_seed770024_keywords_014:step9:15:v_proj
kernel_family = qk256_i2s_i8s_scaled_gemv
```

This work does not change production QK256 dispatch policy, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipts

The new focused source receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-summary-logits-layer15-v-proj-raw-operands.json
```

The combined manifest is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-qk256-layer15-v-proj-target-replay-manifest.json
```

It preserves the 90-target focused manifest shape, keeps the previous
forty-seven focused replay sources, and adds one supplemental focused source
for the new layer-15 `v_proj` target.

The selected-device A770 OpenCL replay receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-qk256-layer15-v-proj-target-replay.json
```

It records:

```text
work_item = A770-114
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
dispatch_replay_target_count = 48
runnable_target_count = 48
```

The runnable targets are the layer-0 through layer-15 Q/K/V trios.

The selected-device production replay executed all forty-eight runnable targets
on Intel Arc A770 OpenCL with no fallback and matched the selected-device bits:

```text
executed_target_count = 48
blocked_target_count = 42
failed_target_count = 0
matched_selected_device_bits_count = 48
all_executed_targets_match_selected_device_bits = true
kernel_invocations = 96
host_to_device_bytes = 153600
device_to_host_bytes = 2688
```

The new target result records:

```text
classification = a770_qk256_multi_case_focused_replay_matches_selected_device_output
target_id = a770_summary_seed770024_keywords_014:step9:15:v_proj
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
fallback_used = false
production_output_matches_selected_device_bits = true
replay_output_matches_selected_device_bits = true
final_scaled_value_matches_selected_device_bits = true
selected_device_output_bits = 3220222710
```

The remaining blockers are still ledgered:

```text
dispatch_replay_missing = 42
first_blocked_target = a770_summary_seed770024_keywords_014:step9:16:q_proj
first_blocked_reason = dispatch_replay_missing
```

## Interpretation

A770-114 completes the layer-15 focused replay Q/K/V trio by adding `v_proj`
after A770-112 added `q_proj` and A770-113 added `k_proj`. This is useful
correctness evidence, not speed evidence, and not a production QK256 policy
change.

The next narrow frontier is to convert one more target from the remaining
`dispatch_replay_missing` set. The direct continuation is layer-16 `q_proj`
under the same one-case, one-mismatch, selected-device, fallback-free receipt
rules.

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
target/release/bitnet.exe answer-corpus --corpus ci/quality/a770-bitnet-answer-readiness-corpus.yaml --model E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/tokenizer.json --device intel-a770-opencl --case-id a770_summary_seed770024_keywords_014 --dump-logit-steps 24 --logits-topk 20 --json-out ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-summary-logits-layer15-v-proj-raw-operands.json --per-prompt-timeout-seconds 300
target/debug/a770-opencl-production-replay-instrumentation.exe --focused-source ... --focused-source ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-summary-logits-layer15-v-proj-raw-operands.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --work-item A770-114 --manifest ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-qk256-layer15-v-proj-target-replay-manifest.json --receipt ci/hardware/amd-5700x-intel-a770/2026-06-03/a770-layer15-v-proj-qk256-replay-target/a770-opencl-qk256-layer15-v-proj-target-replay.json
pwsh receipt assertion for target_count=90, dispatch_replay_target_count=48, runnable_target_count=48, executed_target_count=48, blocked_target_count=42, failed_target_count=0, matched_selected_device_bits_count=48, all_executed_targets_match_selected_device_bits=true, fallback_used=false, work_item=A770-114, and layer-15 v_proj executed fallback-free with selected-device bit matches
```

The answer-corpus capture was run with:

```text
BITNET_QKV_PROJECTION_DISPATCH_REPLAY=1
BITNET_QKV_PROJECTION_DISPATCH_REPLAY_LAYER=15
BITNET_QKV_PROJECTION_DISPATCH_REPLAY_PROJECTION=v_proj
BITNET_QKV_PROJECTION_DISPATCH_REPLAY_RAW_OPERANDS=1
BITNET_QKV_PROJECTION_DISPATCH_REPLAY_RAW_OPERAND_INPUT_ROW=0
BITNET_QKV_PROJECTION_DISPATCH_REPLAY_RAW_OPERAND_OUTPUT_INDEX=0
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
