# CUDA-MODEL-017W Qwen3 Warm Session Source Receipt

Status: `warm_session_3_turns` source receipt 2 / 3; aggregate still blocked

## Scope

`CUDA-MODEL-017` requires repeated current-source Qwen3 comparator receipts for
these profiles:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This report records the second current-source `warm_session_3_turns` strict CUDA
source receipt on the Windows 9950X3D + RTX 5070 Ti lane. It advances the warm
session source set only. It does not satisfy the full CUDA-MODEL-017 aggregate
because the `warm_session_3_turns` set is not complete and the
`decode_128_from_warm_context` profile still has no source receipts.

## Source Receipt

Receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-warm-session-3-cuda.json
```

Capture command:

```powershell
rtk powershell -Command '$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02"; $model = "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf"; $base = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15"; .\target\debug\bitnet.exe dense-gguf-qwen-warm-session-strict-cuda --model $model --device nvidia-rtx-5070-ti-cuda --device-index 0 --turns 3 --max-new-tokens 8 --top-k 10 --all-layer-plan "$base\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "$base\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "$base\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "$base\qwen3-0_6b-sampling-policy.json" --one-token-proof "$root\qwen3-0_6b-one-token-cuda.json" --short-decode-proof "$root\qwen3-0_6b-short-decode-8-cuda.json" --json-out "$root\qwen3-0_6b-warm-session-3-cuda.json"'
```

The default logits transfer mode remains the full-logits D2H source mode. The
CUDA device top-k mode remains unclaimed; this receipt records the blocker as
`cpu_sampler_requires_full_logits_until_device_top_k_sampler`.

## Recorded Fields

```text
artifact_kind: dense_gguf_qwen_warm_session_strict_cuda_proof
model: qwen3-0.6b-instruct-q8_0
model_sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
model_coverage_row: dense_qwen3_06b_q8_candidate
model_coverage_tier: accelerator_answer_ready
selected_backend: nvidia-rtx-5070-ti-cuda
selected_route: dense_regular_llm_cuda
fallback_used: false
turns_count: 3
generated_tokens_total: 24
quality_gate: qwen_warm_session_cuda_parity
quality_gate.passed: true
generated_token_ids_match: true
top_k_all_match: false
first_top_k_divergence: turn=1 step=6
speedup_claim: false
claim_boundary.bitnet_packed_i2s_qk256_proof: false
claim_boundary.full_cuda_residency_claimed: false
claim_boundary.server_ready_claimed: false
```

Session lifecycle:

```text
model_loaded_once: true
tokenizer_loaded_once: true
cuda_context_initialized_once: true
weights_uploaded_once: true
per_turn_weight_upload: false
runtime_buffers_reused: true
workspace_reused: true
per_request_model_load: false
```

Logits transfer accounting:

```text
transfer_mode: full_logits_download_cpu_sampler
sampling_location: cpu
device_to_host_bytes_reduced: false
actual_device_to_host_bytes: 14585856
full_logits_download_bytes: 14585856
reduction_blocker: cpu_sampler_requires_full_logits_until_device_top_k_sampler
```

This is strict CUDA route evidence with explicit full-logits D2H sampling. It is
not a logits-transfer-reduction, full-residency, server, or speed claim.

Observed timing envelope:

```text
run-02 total_ms=483428.7471 first_token_ms=1755.5940 prefill_ms=410944.6459 decode_total_ms=56693.5086 kernel_time_ms=463947.3937 kernel_launches=9456 H2D_bytes=639446688 D2H_bytes=14585856
```

The slow CUDA timing is recorded as evidence, not as a speed claim.

## Validation

```powershell
rtk powershell -Command '$bat = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $sep = [char]0x1f; $env:CARGO_ENCODED_RUSTFLAGS = "-L" + $sep + "native=$cuda\lib\x64"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; cmd.exe /c "call `"$bat`" -arch=x64 -host_arch=x64 && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli"'
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-warm-session-3-cuda.json --format json'
rtk powershell -Command '.\target\debug\qwen3_cuda_repeated_comparator_receipt.exe --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-one-token-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-8-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-32-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-32-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-32-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-warm-session-3-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-warm-session-3-cuda.json --receipt-out target\qwen3-017-partial-aggregate-after-017w.json'
rtk cargo test --locked -p bitnet-bench-receipts --no-default-features qwen3
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
rtk git diff --check
```

Result: the CUDA build and `receipts explain` command passed. The partial
aggregate command failed closed as expected because
`decode_128_from_warm_context` source receipts are still missing. The
CUDA-MODEL-017 acceptance still requires one additional `warm_session_3_turns`
source receipt before the warm-session source set is complete. The focused
`bitnet-bench-receipts` Qwen3 tests, focused `bitnet-cli` receipt-explain tests,
and `git diff --check` passed.

Local campaign validation is left to the remote Campaign Tracker gate for this
PR; local `campaign check nvidia-5070ti` has repeatedly timed out in this
checkout on the same campaign lane.

## Remaining Blocker

The aggregate generator remains correctly fail-closed. The current committed
source set is:

```text
one_token: 3 / 3
short_decode_8: 3 / 3
short_decode_32: 3 / 3
warm_session_3_turns: 2 / 3
decode_128_from_warm_context: 0 / 3
```

The next source-capture work should capture the third
`warm_session_3_turns` receipt with the same artifact, route, fallback, and
claim boundaries.

## Claim Boundary

This report proves only that CUDA-MODEL-017 has two current-source Qwen3
`warm_session_3_turns` strict CUDA source receipts.

These remain false:

- `warm_session_3_turns` source set completion
- any `decode_128_from_warm_context` source receipt
- `qwen3_cuda_repeated_comparator` aggregate availability
- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen3 logits transfer reduction
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Any claim that these two warm-session receipts satisfy the remaining profile
  receipts
