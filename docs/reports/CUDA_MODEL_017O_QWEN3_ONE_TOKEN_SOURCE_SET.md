# CUDA-MODEL-017O Qwen3 One-Token Source Set

Status: `one_token` source receipt set complete; aggregate still blocked

## Scope

`CUDA-MODEL-017` requires at least three source receipts for each Qwen3 repeated
comparator profile:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This report records the completed current-source `one_token` strict CUDA source
set on the Windows 9950X3D + RTX 5070 Ti lane. It does not satisfy the full
CUDA-MODEL-017 aggregate because the remaining profiles still have no source
receipts.

## Source Receipts

Receipts:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-one-token-cuda.json
```

Capture command pattern:

```powershell
rtk powershell -Command '$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-0N"; New-Item -ItemType Directory -Force -Path $root | Out-Null; $model = "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf"; $base = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15"; .\target\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda --model $model --device nvidia-rtx-5070-ti-cuda --device-index 0 --all-layer-plan "$base\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "$base\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "$base\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "$base\qwen3-0_6b-sampling-policy.json" --json-out "$root\qwen3-0_6b-one-token-cuda.json"'
```

All three receipts recorded:

```text
artifact_kind: dense_gguf_qwen_one_token_strict_cuda_proof
model: qwen3-0.6b-instruct-q8_0
model_sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
selected_backend: nvidia-rtx-5070-ti-cuda
selected_route: dense_regular_llm_cuda
fallback_used: false
quality_gate.passed: true
parity.passed: true
generated_tokens_count: 1
cuda_selected_token_id: 3555
decoded_token_text: " What"
speedup_claim: false
claim_boundary.bitnet_packed_i2s_qk256_proof: false
claim_boundary.full_cuda_residency_claimed: false
claim_boundary.server_ready_claimed: false
```

Observed timing envelope:

```text
run-01 total_ms=952519.4266 kernel_time_ms=936719.1936 kernel_launches=394 H2D_bytes=639446688 D2H_bytes=607744
run-02 total_ms=672506.4677 kernel_time_ms=657209.0666 kernel_launches=394 H2D_bytes=639446688 D2H_bytes=607744
run-03 total_ms=773968.3111 kernel_time_ms=758824.5092 kernel_launches=394 H2D_bytes=639446688 D2H_bytes=607744
```

The slow CUDA timing is recorded as evidence, not as a speed claim.

## Receipt Explain Smoke

```powershell
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-one-token-cuda.json --format json'
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-one-token-cuda.json --format json'
```

Result: passed. The normalized explanations reported the exact Qwen3 coverage
row, `dense_regular_llm_cuda` route, selected RTX 5070 Ti CUDA backend,
`fallback_used=false`, `speedup_claim=false`,
`full_residency_claim=false`, BitNet QK256 proof false, and dense CUDA proof
true.

## Aggregate Fail-Closed Probe

```powershell
rtk powershell -Command '.\target\debug\qwen3_cuda_repeated_comparator_receipt.exe --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-02\qwen3-0_6b-one-token-cuda.json --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-03\qwen3-0_6b-one-token-cuda.json --receipt-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-0_6b-repeated-comparator.json'
```

Result: failed closed as expected. The generator accepted the completed
`one_token` source set and refused to emit an aggregate while the remaining
profile receipts were absent.

## Remaining Blocker

The aggregate generator remains correctly fail-closed. The current committed
source set is:

```text
one_token: 3 / 3
short_decode_8: 0 / 3
short_decode_32: 0 / 3
warm_session_3_turns: 0 / 3
decode_128_from_warm_context: 0 / 3
```

The next source-capture work should collect the three `short_decode_8` receipts
or first reduce the strict CUDA Qwen3 runtime bottleneck without weakening
fallback, quality, or proof-family boundaries.

## Claim Boundary

This report proves only that the current-source Qwen3 `one_token` strict CUDA
source receipt set is complete for CUDA-MODEL-017.

These remain false:

- `qwen3_cuda_repeated_comparator` aggregate availability
- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Any claim that `one_token` receipts satisfy the remaining profiles
