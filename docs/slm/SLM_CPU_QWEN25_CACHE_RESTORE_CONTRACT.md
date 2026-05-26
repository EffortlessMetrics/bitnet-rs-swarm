# Qwen2.5 Q8_0 Cache Restore Contract

Status: `SLM-CPU-140`

This contract defines the exact model-cache and receipt-capture gate required
before the Kaby Lake SLM lane can resume source-order q_proj evidence for the
i5-8250U. It is a restoration and capture contract only. It does not commit a
model binary, promote the source-order Q8_0 matvec candidate, claim timing
improvement, or broaden the runtime beyond the strict CPU proof lane.

## Required Model Cache

The missing exact model is:

```text
model_id = qwen2.5-0.5b-instruct-q8_0
repo = Qwen/Qwen2.5-0.5B-Instruct-GGUF
revision = 9217f5db79a29953eb74d5343926648285ec7e67
file = qwen2.5-0.5b-instruct-q8_0.gguf
sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
runtime_path = target/slm-cpu-140/cache/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf
committed_path = none
```

`models/**` and `target/**` artifacts must not be committed by this lane. The
model may be restored into a local cache only for receipt capture, then verified
by SHA256 before any run uses it.

## Restore And Verify

Use the CLI model cache flow and keep the binary out of git:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  model fetch qwen2.5-0.5b-instruct-q8_0 `
  --cache-dir target\slm-cpu-140\cache `
  --json

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  model verify qwen2.5-0.5b-instruct-q8_0 `
  --cache-dir target\slm-cpu-140\cache `
  --json
```

The verification step must report the SHA256 above. If it does not, stop and
record a blocker instead of capturing receipts.

## Source-Order Receipt Fields

Fresh Qwen3 and Qwen2.5 receipts must include all of these fields before any
source-order q_proj selector use is considered:

```text
model.sha256
tokenizer.source
tokenizer.strict
prompt_ids
generated_ids
decoded_text
selected_backend
selected_kernel
dense_hook_identity
dense_q8_hook_selection.payload_bearing_boundary.source_order_q8_matvec_candidate
dense_q8_hook_selection.payload_bearing_boundary.source_order_selected_path
dense_q8_hook_selection.payload_bearing_boundary.source_order_selected_kernel
dense_q8_hook_selection.payload_bearing_boundary.source_order_input_dim
dense_q8_hook_selection.payload_bearing_boundary.source_order_output_dim
dense_q8_hook_selection.payload_bearing_boundary.source_order_candidate_receipt_identity
dense_q8_hook.source_order_q8_matvec_candidate
dense_q8_hook.source_order_selected_path
dense_q8_hook.source_order_selected_kernel
dense_q8_hook.source_order_candidate_receipt_identity
dense_q8_hook.source_order_candidate_runtime_enabled
q_proj_numeric_evidence
fallback_used
```

The source-order candidate identity is:

```text
candidate_path = source_order_q8_0_qproj_matvec
candidate_kernel = dense-q8-source-order-qproj-matvec
candidate_receipt_identity = layers.0.attention.q_proj.weight:source_order_q8_0_qproj_matvec:runtime_disabled
candidate_runtime_enabled = false
default_runtime = eager_f32_candle
```

## Capture Commands

Capture fresh before/after receipt pairs on current main. The before run must
use the default eager F32 Candle path. The after run may enable only the
existing exact-tensor Q8_0 runtime identity surface for `blk.0.attn_q.weight`;
the source-order candidate itself must remain runtime-disabled unless a later
item explicitly accepts behavior-equivalent receipts.

Qwen3 before:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  --device cpu `
  slm-warm-session `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --corpus ci\quality\slm-warm-session-corpus.yaml `
  --corpus-repeat-runs 2 `
  --max-new-tokens 8 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --threads 4 `
  --strict-loader `
  --strict-tokenizer `
  --prompt-template qwen `
  --require-determinism `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-26\qwen3-slm-cpu-140-before-run.json
```

Qwen3 after:

```powershell
$env:BITNET_DENSE_Q8_RUNTIME_ENABLE = "1"
$env:BITNET_DENSE_Q8_RUNTIME_TENSOR = "blk.0.attn_q.weight"

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  --device cpu `
  slm-warm-session `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --corpus ci\quality\slm-warm-session-corpus.yaml `
  --corpus-repeat-runs 2 `
  --max-new-tokens 8 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --threads 4 `
  --strict-loader `
  --strict-tokenizer `
  --prompt-template qwen `
  --require-determinism `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-26\qwen3-slm-cpu-140-after-run.json

Remove-Item Env:\BITNET_DENSE_Q8_RUNTIME_ENABLE
Remove-Item Env:\BITNET_DENSE_Q8_RUNTIME_TENSOR
```

Qwen2.5 before:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  --device cpu `
  answer-corpus `
  --corpus ci\quality\slm-second-model-sanity-corpus.yaml `
  --model target\slm-cpu-140\cache\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf `
  --model-id qwen2.5-0.5b-instruct-q8_0 `
  --threads 4 `
  --case-id math_2_plus_2_brief `
  --dump-logit-steps 1 `
  --logits-topk 5 `
  --strict-loader `
  --strict-tokenizer `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-26\qwen25-slm-cpu-140-before-run.json
```

Qwen2.5 after:

```powershell
$env:BITNET_DENSE_Q8_RUNTIME_ENABLE = "1"
$env:BITNET_DENSE_Q8_RUNTIME_TENSOR = "blk.0.attn_q.weight"

cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- `
  --device cpu `
  answer-corpus `
  --corpus ci\quality\slm-second-model-sanity-corpus.yaml `
  --model target\slm-cpu-140\cache\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf `
  --model-id qwen2.5-0.5b-instruct-q8_0 `
  --threads 4 `
  --case-id math_2_plus_2_brief `
  --dump-logit-steps 1 `
  --logits-topk 5 `
  --strict-loader `
  --strict-tokenizer `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-26\qwen25-slm-cpu-140-after-run.json

Remove-Item Env:\BITNET_DENSE_Q8_RUNTIME_ENABLE
Remove-Item Env:\BITNET_DENSE_Q8_RUNTIME_TENSOR
```

## Validation Gate

The next evidence slice must fail closed unless all of the following are true:

```text
Qwen3 model SHA = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
Qwen2.5 model SHA = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
tokenizer.source = gguf_metadata
tokenizer.strict = true
selected_backend = cpu-rust
fallback_used = false
prompt IDs match before/after for each model
generated IDs match before/after for each model
decoded text matches before/after for each model
source-order candidate identity fields are present
source-order candidate runtime remains disabled
q_proj numeric evidence is attached or explicitly referenced
```

If any required field is absent, or if generated IDs/text drift, the result is a
blocker artifact. It is not a performance result.

## Claim Boundary

This contract does not claim:

```text
fresh source-order Qwen3/Qwen2.5 receipt pairs
runtime selection for source-order q_proj
allocation reduction
timing improvement
speedup
sustained 8250U throughput
default-runtime promotion
Q4_K_M or Q4_K_S support
server, GPU, NPU, OpenVINO, or UHD 620 execution
Qwen3.5 or hybrid architecture support
BitNet QK256/I2_S behavior
```
