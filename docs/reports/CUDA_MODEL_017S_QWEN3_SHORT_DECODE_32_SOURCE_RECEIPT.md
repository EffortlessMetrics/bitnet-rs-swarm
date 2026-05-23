# CUDA-MODEL-017S Qwen3 Short-Decode 32 Source Receipt

Status: `short_decode_32` source receipt 1 / 3; aggregate still blocked

## Scope

`CUDA-MODEL-017` requires repeated current-source Qwen3 hardware comparator
receipts before the aggregate `qwen3_cuda_repeated_comparator` receipt can be
generated. This report records the first strict CUDA `short_decode_32` source
receipt for the Windows 9950X3D + RTX 5070 Ti lane.

This completes no aggregate profile set. It moves only this counter:

```text
short_decode_32: 1 / 3
```

## Source Receipt

Receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-short-decode-32-cuda.json
```

Capture command:

```powershell
rtk powershell -Command '$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01"; .\target\debug\bitnet.exe dense-gguf-qwen-short-decode-strict-cuda --model "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf" --device nvidia-rtx-5070-ti-cuda --device-index 0 --capture-profile qwen3-short-decode-32 --max-new-tokens 32 --all-layer-plan "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json" --one-token-proof "$root\qwen3-0_6b-one-token-cuda.json" --json-out "$root\qwen3-0_6b-short-decode-32-cuda.json"'
```

## Recorded Fields

```text
artifact_kind: dense_gguf_qwen_short_decode_strict_cuda_proof
model: qwen3-0.6b-instruct-q8_0
model_sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
model_coverage_row: dense_qwen3_06b_q8_candidate
model_coverage_tier: accelerator_answer_ready
selected_backend: nvidia-rtx-5070-ti-cuda
runtime_api: cuda
selected_route: dense_regular_llm_cuda
fallback_used: false
generated_tokens_count: 32
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
actual_device_to_host_bytes: 19447808
full_logits_download_bytes: 19447808
reduction_blocker: cpu_sampler_requires_full_logits_until_device_top_k_sampler
device_to_host_ms_source: wall_clock_extract_logits_2d_local
```

Observed timing envelope:

```text
run-01 total_ms=727586.5555 first_token_ms=1054.2888 prefill_ms=680824.9448 decode_total_ms=33866.3852 kernel_time_ms=711498.3125 kernel_launches=12608 H2D_bytes=639446688 D2H_bytes=19447808
```

The slow timing is recorded as source evidence. It is not a speedup,
benchmark-qualified speed, or full-residency claim.

## Validation

```powershell
rtk powershell -Command '$bat = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $sep = [char]0x1f; $env:CARGO_ENCODED_RUSTFLAGS = "-L" + $sep + "native=$cuda\lib\x64"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; cmd.exe /c "call `"$bat`" -arch=x64 -host_arch=x64 && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli"'
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-32-cuda.json --format json'
rtk powershell -Command '.\target\debug\qwen3_cuda_repeated_comparator_receipt.exe --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-one-token-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-short-decode-8-cuda.json --short-decode-8-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-short-decode-8-cuda.json --short-decode-32-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-short-decode-32-cuda.json --receipt-out target\qwen3-017-partial-aggregate-after-017s.json'
rtk cargo test --locked -p bitnet-bench-receipts --no-default-features qwen3
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain
rtk git diff --check
```

Result: passed where success is expected. The partial aggregate command failed
closed as expected because `warm_session_3_turns` and
`decode_128_from_warm_context` source receipts are still absent. The committed
source-of-truth tracker still requires two additional `short_decode_32` source
receipts before that profile set is complete. The CUDA build produced the
existing CUDA deprecation warnings in `bitnet-kernels`.

`rtk cargo run --locked -p xtask --no-default-features -- campaign generate
--check` and `rtk cargo run --locked -p xtask --no-default-features -- campaign
check nvidia-5070ti` were attempted locally, but each exceeded the local
10-minute validation window. Remote campaign/doctor checks remain required for
the PR.

## Remaining Blocker

The aggregate remains blocked. The current committed source set is:

```text
one_token: 3 / 3
short_decode_8: 3 / 3
short_decode_32: 1 / 3
warm_session_3_turns: 0 / 3
decode_128_from_warm_context: 0 / 3
```

The next source-capture work should collect `short_decode_32` run 02 with the
same artifact, route, fallback, and claim boundaries.

## Claim Boundary

This report proves only that one current-source Qwen3 `short_decode_32` strict
CUDA receipt exists for CUDA-MODEL-017.

These remain false:

- `short_decode_32` source set completion
- `qwen3_cuda_repeated_comparator` aggregate availability
- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen3 logits transfer reduction
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Any claim that this receipt satisfies the remaining profiles
