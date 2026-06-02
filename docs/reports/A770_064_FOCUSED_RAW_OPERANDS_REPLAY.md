# A770-064 Focused Raw Operands Replay

## Scope

A770-064 is a diagnostic-only follow-up to A770-063. A770-063 proved that the
focused generated-output first-mismatch source carried compact QK256 replay
context, but not the raw activation row or packed QK256 bytes needed to feed
the selected-device production replay kernel.

This slice adds opt-in raw operand capture for the existing QKV projection
dispatch replay path and carries that payload into the focused parity receipt
only when explicitly requested by diagnostic environment variables. It then
runs the unchanged selected-device production replay instrumentation on the
captured focused operand row for:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj
```

It does not change production QK256 dispatch, runtime math, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipts

The new focused source and replay receipts are:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/cpu-avx2-vs-a770-summary-logits-raw-operands-parity.json
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-raw-operands-replay.json
```

The replay receipt records:

```text
proof_family = a770_opencl_qk256_focused_raw_operand_replay
proof_stage = diagnostic_focused_raw_operands_production_replay_classified
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
kernel_name = qk256_i2s_i8s_scaled_gemv
replay_kernel_name = qk256_i2s_i8s_scaled_gemv_production_replay
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
a770_qk256_focused_raw_operands_replay_matches_focused_device_output
```

The focused raw operand context is now present:

```text
focused_first_mismatch_operands_available = true
raw_activation_i8_available = true
raw_packed_qk256_available = true
activation_i8_len = 2560
packed_qk256_len = 640
production_replay_executed = true
kernel_invocations = 2
```

The selected-device production replay reproduces the focused device output
store bit:

```text
focused_device_output_bits = 3215804927
focused_policy_bits = 3215804926
production_output_bits = 3215804927
replay_output_bits = 3215804927
final_scaled_value_bits = 3215804927
production_output_matches_focused_device_bits = true
output_store_matches_replay_output = true
output_store_matches_final_scaled_value = true
```

## Interpretation

A770-064 moves the blocker past missing raw focused operands. The exact focused
raw row can now be fed to the selected-device production replay kernel, and the
production replay matches the A770 device output bit that differs from the
host-side policy summary by one bit.

This is still not a production QK256 policy fix. The next useful diagnostic is
to localize the focused host-policy versus selected-device production replay
expression split before any production QK256 policy change.

## Validation

```text
cargo check --locked -p bitnet-qk256-dispatch --no-default-features --features opencl
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli,opencl
target/release/bitnet.exe answer-corpus --corpus ci/quality/a770-bitnet-answer-readiness-corpus.yaml --model E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --tokenizer E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/tokenizer.json --device intel-a770-opencl --case-id a770_summary_seed770024_keywords_014 --dump-logit-steps 24 --logits-topk 20 --json-out ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json --per-prompt-timeout-seconds 300
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli,opencl -- answer-parity --left ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json --right ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json --left-label amd-5700x-cpu-avx2 --right-label intel-a770-opencl --machine amd-5700x-intel-a770 --json-out ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/cpu-avx2-vs-a770-summary-logits-raw-operands-parity.json
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/cpu-avx2-vs-a770-summary-logits-raw-operands-parity.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-raw-operands-replay.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-focused-raw-operands-replay.json -Raw | ConvertFrom-Json | Out-Null"
```

`answer-parity` still exits non-zero because CPU/A770 answer parity remains
unproven; the command nevertheless writes the focused diagnostic parity receipt
used by the replay classifier.

The release binary was rebuilt for the local A770 receipt. The initial
`cargo build --release` wrapper timed out while `rustc` continued in the
background; the updated `target/release/bitnet.exe` was present before the
`answer-corpus` receipt was captured.

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
