# CUDA-MODEL-017Z Qwen3 Decode-128 Source Receipt

Status: `decode_128_from_warm_context` source receipt 2 / 3; aggregate still blocked

## Scope

`CUDA-MODEL-017` requires repeated current-source Qwen3 comparator receipts for
these profiles:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This report records the second current-source `decode_128_from_warm_context`
strict CUDA source receipt on the Windows 9950X3D + RTX 5070 Ti lane. It does
not satisfy the full CUDA-MODEL-017 aggregate because the decode-128 source set
still has only two of the required three source receipts.

## Source Receipt

Receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-decode-128-from-warm-context-cuda.json
```

Capture command:

```powershell
rtk powershell -Command '$ErrorActionPreference = "Stop"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02"; New-Item -ItemType Directory -Force -Path $root | Out-Null; $model = "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf"; $base = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15"; .\target\debug\bitnet.exe dense-gguf-qwen-warm-decode-strict-cuda --model $model --device nvidia-rtx-5070-ti-cuda --device-index 0 --max-new-tokens 128 --top-k 10 --all-layer-plan "$base\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "$base\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "$base\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "$base\qwen3-0_6b-sampling-policy.json" --one-token-proof "$root\qwen3-0_6b-one-token-cuda.json" --short-decode-proof "$root\qwen3-0_6b-short-decode-8-cuda.json" --json-out "$root\qwen3-0_6b-decode-128-from-warm-context-cuda.json"'
```

The default logits transfer mode remains the full-logits D2H source mode. The
CUDA device top-k mode remains unclaimed; this receipt records the blocker as
`cpu_sampler_requires_full_logits_until_device_top_k_sampler`.

## Recorded Fields

```text
artifact_kind: dense_gguf_qwen_warm_decode_strict_cuda_proof
model: qwen3-0.6b-instruct-q8_0
model_sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
model_coverage_row: dense_qwen3_06b_q8_candidate
model_coverage_tier: accelerator_answer_ready
selected_backend: nvidia-rtx-5070-ti-cuda
selected_route: dense_regular_llm_cuda
runtime_api: cuda
fallback_used: false
unsupported_ops: 0
cuda_ops: 50432
profile_id: decode_128_from_warm_context
proof_scope: qwen3_decode_128_from_warm_context
requested_new_tokens: 128
generated_tokens_count: 128
quality_gate: qwen_warm_decode_cuda_parity
quality_gate.passed: true
generated_token_ids_match: true
top_k_all_match: false
first_top_k_divergence_index: 15
top_k_max_abs_error: 0.013857841491699219
top_k_mean_abs_error: 0.0011177495121955869
speedup_claim: false
claim_boundary.bitnet_packed_i2s_qk256_proof: false
claim_boundary.full_cuda_residency_claimed: false
claim_boundary.server_ready_claimed: false
claim_boundary.dense_regular_llm_cuda_claimed: true
```

Warm-context lifecycle:

```text
model_loaded_once: true
cuda_context_initialized_once: true
weights_uploaded_once: true
warm_context_reused: true
decode_started_from_prefilled_context: true
fallback_used: false
```

Logits transfer accounting:

```text
transfer_mode: full_logits_download_cpu_sampler
sampling_location: cpu
device_to_host_bytes_reduced: false
actual_device_to_host_bytes: 77791232
full_logits_download_bytes: 77791232
reduction_blocker: cpu_sampler_requires_full_logits_until_device_top_k_sampler
```

This is strict CUDA route evidence with explicit full-logits D2H sampling. It is
not a logits-transfer-reduction, full-residency, server, or speed claim.

Observed timing envelope:

```text
run-02 total_ms=801628.2489 first_token_ms=1130.4720 decode_total_ms=171788.1877 kernel_time_ms=772215.2241 kernel_launches=50432 H2D_bytes=639446688 D2H_bytes=77791232
```

The slow CUDA timing is recorded as evidence, not as a speed claim.

## Validation

```powershell
rtk powershell -Command '$bat = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; cmd.exe /c "call `"$bat`" -arch=x64 -host_arch=x64 && set CUDA_PATH=$cuda && set PATH=$cuda\bin;%PATH% && set LIB=$cuda\lib\x64;%LIB% && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli"'
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-decode-128-from-warm-context-cuda.json --format json > target\qwen3-decode128-run02-explain.json'
rtk powershell -Command 'cargo run --locked -p bitnet-bench-receipts --no-default-features --bin qwen3_cuda_repeated_comparator_receipt -- --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-one-token-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-8-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-32-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-32-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-32-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-warm-session-3-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-warm-session-3-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-warm-session-3-cuda.json --decode-128-from-warm-context-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-decode-128-from-warm-context-cuda.json --decode-128-from-warm-context-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-decode-128-from-warm-context-cuda.json --receipt-out target\qwen3-repeated-comparator-run02-probe.json'
```

Result: the CUDA build passed after the Visual Studio environment explicitly
included `CUDA\v12.9\lib\x64` in `LIB`, the strict decode-128 command emitted
the run-02 source receipt, and `receipts explain` accepted it. The aggregate
command failed closed as expected because `decode_128_from_warm_context` has
only two of the required three runs.

An initial capture attempt failed closed because the local debug CLI was
CPU-only. The first CUDA rebuild attempt then failed at link time with
`LNK1181: cannot open input file 'cuda.lib'`; extending `LIB` fixed the local
build environment. No source code changes were made to fix either local setup
issue.

## Remaining Blocker

The aggregate generator remains correctly fail-closed. The current committed
source set is:

```text
one_token: 3 / 3
short_decode_8: 3 / 3
short_decode_32: 3 / 3
warm_session_3_turns: 3 / 3
decode_128_from_warm_context: 2 / 3
```

The next source-capture work should capture run-03
`decode_128_from_warm_context` with the same artifact, route, fallback, and
claim boundaries.

## Claim Boundary

This report proves only that CUDA-MODEL-017 has two current-source Qwen3
`decode_128_from_warm_context` strict CUDA source receipts.

These remain false:

- complete `decode_128_from_warm_context` source set
- `qwen3_cuda_repeated_comparator` aggregate availability
- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen3 logits transfer reduction
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Any claim that two decode-128 source receipts satisfy the remaining profile
  receipt
