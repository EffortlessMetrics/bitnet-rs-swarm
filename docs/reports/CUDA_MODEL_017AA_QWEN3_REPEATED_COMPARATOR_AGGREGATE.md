# CUDA-MODEL-017AA Qwen3 Repeated Comparator Aggregate

Status: `decode_128_from_warm_context` source receipt 3 / 3; aggregate generated

## Scope

`CUDA-MODEL-017` requires repeated current-source Qwen3 comparator receipts for
these profiles:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This report records the third current-source `decode_128_from_warm_context`
strict CUDA source receipt on the Windows 9950X3D + RTX 5070 Ti lane and the
first generated `qwen3_cuda_repeated_comparator` aggregate for the 2026-05-23
capture set.

## Source Receipts

New strict CUDA receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-decode-128-from-warm-context-cuda.json
```

Aggregate receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/qwen3-0_6b-repeated-comparator.json
```

The default logits transfer mode remains the full-logits D2H source mode. The
CUDA device top-k mode remains unclaimed; the receipt records the blocker as
`cpu_sampler_requires_full_logits_until_device_top_k_sampler`.

## Run-03 Recorded Fields

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

Observed timing envelope:

```text
run-03 total_ms=751787.0991 first_token_ms=1262.6874 decode_total_ms=254691.6469 kernel_time_ms=720431.7760 kernel_launches=50432 H2D_bytes=639446688 D2H_bytes=77791232
```

## Aggregate Summary

The aggregate validates as `qwen3_cuda_repeated_comparator` and records:

```text
profiles_recorded: 5
total_cpu_runs: 15
total_cuda_runs: 15
min_runs_per_backend: 3
same_artifact_sha: true
same_tokenizer_prompt_policy: true
fallback_free: true
generated_tokens_compared: true
speedup_claim_allowed: false
status: repeated_comparator_only
next_step: Qwen3 exact-profile benchmark qualification review after repeated hardware receipts land
```

Per-profile run counts:

```text
one_token: cpu_runs=3 cuda_runs=3
short_decode_8: cpu_runs=3 cuda_runs=3
short_decode_32: cpu_runs=3 cuda_runs=3
warm_session_3_turns: cpu_runs=3 cuda_runs=3
decode_128_from_warm_context: cpu_runs=3 cuda_runs=3
```

The aggregate is repeated same-artifact CPU/CUDA comparator evidence only. It is
not a speedup, benchmark-qualified, full-residency, server-readiness, broad dense
GGUF, Qwen2.5 inheritance, logits-transfer-reduction, or BitNet packed
I2_S/QK256 proof.

## Validation

```powershell
rtk powershell -Command '$bat = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; cmd.exe /c "call `"$bat`" -arch=x64 -host_arch=x64 && set CUDA_PATH=$cuda && set PATH=$cuda\bin;%PATH% && set LIB=$cuda\lib\x64;%LIB% && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli"'
rtk powershell -Command '$ErrorActionPreference = "Stop"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03"; $model = "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf"; $base = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15"; .\target\debug\bitnet.exe dense-gguf-qwen-warm-decode-strict-cuda --model $model --device nvidia-rtx-5070-ti-cuda --device-index 0 --max-new-tokens 128 --top-k 10 --all-layer-plan "$base\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "$base\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "$base\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "$base\qwen3-0_6b-sampling-policy.json" --one-token-proof "$root\qwen3-0_6b-one-token-cuda.json" --short-decode-proof "$root\qwen3-0_6b-short-decode-8-cuda.json" --json-out "$root\qwen3-0_6b-decode-128-from-warm-context-cuda.json"'
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-decode-128-from-warm-context-cuda.json --format json > target\qwen3-decode128-run03-explain.json'
rtk powershell -Command 'cargo run --locked -p bitnet-bench-receipts --no-default-features --bin qwen3_cuda_repeated_comparator_receipt -- --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-one-token-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-8-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-32-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-32-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-32-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-warm-session-3-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-warm-session-3-cuda.json --warm-session-3-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-warm-session-3-cuda.json --decode-128-from-warm-context-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-decode-128-from-warm-context-cuda.json --decode-128-from-warm-context-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-decode-128-from-warm-context-cuda.json --decode-128-from-warm-context-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-decode-128-from-warm-context-cuda.json --receipt-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\qwen3-0_6b-repeated-comparator.json'
```

Result: the CUDA build passed with `CUDA\v12.9\lib\x64` included in `LIB`, the
strict run-03 decode-128 command emitted the source receipt, `receipts explain`
accepted it, and the aggregate generator validated and wrote the repeated
comparator aggregate.

## Current Source Set

```text
one_token: 3 / 3
short_decode_8: 3 / 3
short_decode_32: 3 / 3
warm_session_3_turns: 3 / 3
decode_128_from_warm_context: 3 / 3
```

## Claim Boundary

This report proves only that CUDA-MODEL-017 has the required repeated
same-artifact Qwen3 CPU/CUDA comparator source receipts and a validated
`qwen3_cuda_repeated_comparator` aggregate.

These remain false:

- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen3 logits transfer reduction
- Qwen3 server readiness
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Runtime math, tokenizer, loader, kernel, or server behavior changes
