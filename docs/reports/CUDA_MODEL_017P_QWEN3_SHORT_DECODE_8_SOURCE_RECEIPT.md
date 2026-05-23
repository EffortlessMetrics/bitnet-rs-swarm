# CUDA-MODEL-017P Qwen3 Short-Decode 8 Source Receipt

Status: `short_decode_8` source receipt 1 / 3; aggregate still blocked

## Scope

`CUDA-MODEL-017` requires at least three source receipts for each Qwen3 repeated
comparator profile:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This report records the first current-source `short_decode_8` strict CUDA source
receipt on the Windows 9950X3D + RTX 5070 Ti lane. It does not satisfy the full
CUDA-MODEL-017 aggregate because the `short_decode_8` set still needs two more
source receipts and the remaining profiles still have no source receipts.

## Source Receipt

Receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-short-decode-8-cuda.json
```

Capture command:

```powershell
rtk powershell -Command '$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01"; New-Item -ItemType Directory -Force -Path $root | Out-Null; $model = "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf"; $base = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15"; .\target\debug\bitnet.exe dense-gguf-qwen-short-decode-strict-cuda --model $model --device nvidia-rtx-5070-ti-cuda --device-index 0 --capture-profile short-decode --max-new-tokens 8 --all-layer-plan "$base\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "$base\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "$base\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "$base\qwen3-0_6b-sampling-policy.json" --one-token-proof "$root\qwen3-0_6b-one-token-cuda.json" --json-out "$root\qwen3-0_6b-short-decode-8-cuda.json"'
```

The default logits transfer mode is now explicitly the full-logits D2H source
mode. The CUDA device top-k mode remains an opt-in CLI value for future
qualification; it is not used for this source receipt.

## Recorded Fields

```text
artifact_kind: dense_gguf_qwen_short_decode_strict_cuda_proof
model: qwen3-0.6b-instruct-q8_0
model_sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
model_coverage_row: dense_qwen3_06b_q8_candidate
model_coverage_tier: accelerator_answer_ready
selected_backend: nvidia-rtx-5070-ti-cuda
selected_route: dense_regular_llm_cuda
fallback_used: false
generated_tokens_count: 8
speedup_claim: false
claim_boundary.bitnet_packed_i2s_qk256_proof: false
claim_boundary.full_cuda_residency_claimed: false
claim_boundary.server_ready_claimed: false
```

Logits transfer accounting:

```text
transfer_mode: full_logits_download_cpu_sampler
sampling_location: cpu
device_to_host_bytes_reduced: false
actual_device_to_host_bytes: 4861952
full_logits_download_bytes: 4861952
reduction_blocker: cpu_sampler_requires_full_logits_until_device_top_k_sampler
device_to_host_ms_source: wall_clock_extract_logits_2d_local
```

This is strict CUDA route evidence with explicit full-logits D2H sampling. It is
not a logits-transfer-reduction, full-residency, or speed claim.

Observed timing envelope:

```text
run-01 total_ms=660486.3470 first_token_ms=1105.6506 prefill_ms=639864.4496 decode_total_ms=9213.2813 kernel_time_ms=648138.9998 kernel_launches=3152 H2D_bytes=639446688 D2H_bytes=4861952
```

The slow CUDA timing is recorded as evidence, not as a speed claim.

## Source Adjustment

The first `short_decode_8` attempt failed closed during the CUDA target run with
`CUDA_ERROR_INVALID_VALUE` from the CUDA device top-k sampler path. The failing
path was the CUDA full-vocab sort used to reduce logits transfer before copying
top-k results back to host.

The source command now defaults governed Qwen short-decode capture to
`full_logits_download_cpu_sampler`. The path still requires CUDA residency before
download for embedding, hidden, last-hidden, and logits tensors, and the receipt
records full D2H bytes plus the reduction blocker. `device-top-k-cuda-sampler`
remains available only as an explicit opt-in mode until it has its own
receipt-backed qualification.

## Validation

```powershell
rtk cargo fmt -p bitnet-cli -- --check
rtk cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli dense_qwen_full_logits_transfer_records_non_reduced_blocker
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
rtk cargo test --locked -p bitnet-bench-receipts --no-default-features qwen3
rtk powershell -Command '$bat = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $sep = [char]0x1f; $env:CARGO_ENCODED_RUSTFLAGS = "-L" + $sep + "native=$cuda\lib\x64"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; cmd.exe /c "call `"$bat`" -arch=x64 -host_arch=x64 && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli"'
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-8-cuda.json --format json'
```

Result: passed. The build produced the existing CUDA deprecation warnings in
`bitnet-kernels`.

`rtk cargo fmt --all -- --check` was attempted first and failed on Windows with
`The filename or extension is too long. (os error 206)`, so the touched package
formatter check was used.

`rtk cargo run --locked -p xtask --no-default-features -- campaign check
nvidia-5070ti` was attempted locally, but the local xtask build failed while
compiling `sentencepiece-sys` before the campaign check could run. Remote
campaign/doctor checks remain required for the PR.

## Remaining Blocker

The aggregate generator remains correctly fail-closed. The current committed
source set is:

```text
one_token: 3 / 3
short_decode_8: 1 / 3
short_decode_32: 0 / 3
warm_session_3_turns: 0 / 3
decode_128_from_warm_context: 0 / 3
```

The next source-capture work should collect two more `short_decode_8` receipts
with the same artifact, route, fallback, and claim boundaries.

## Claim Boundary

This report proves only that the current-source Qwen3 `short_decode_8` strict
CUDA source receipt has one valid run for CUDA-MODEL-017.

These remain false:

- `short_decode_8` source set completion
- `qwen3_cuda_repeated_comparator` aggregate availability
- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen3 logits transfer reduction
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Any claim that this one `short_decode_8` receipt satisfies the remaining runs
