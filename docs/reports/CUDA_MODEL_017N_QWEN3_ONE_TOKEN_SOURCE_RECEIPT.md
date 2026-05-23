# CUDA-MODEL-017N Qwen3 One-Token Source Receipt

Status: first current-source CUDA-MODEL-017 source receipt captured; aggregate
still blocked

## Scope

`CUDA-MODEL-017` requires at least three source receipts for each Qwen3 repeated
comparator profile:

- `one_token`
- `short_decode_8`
- `short_decode_32`
- `warm_session_3_turns`
- `decode_128_from_warm_context`

This report records the first current-source `one_token` strict CUDA source
receipt on the Windows 9950X3D + RTX 5070 Ti lane. It does not satisfy the full
CUDA-MODEL-017 aggregate.

## Build Precondition

The current-source debug CLI was rebuilt with CUDA after adding the CUDA 12.9
library path through `CARGO_ENCODED_RUSTFLAGS` so the MSVC linker could resolve
`cuda.lib`, `nvrtc.lib`, `curand.lib`, `cublas.lib`, `cublasLt.lib`, and
`cudart.lib`.

```powershell
rtk powershell -Command '$bat = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"; $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $sep = [char]0x1f; $env:CARGO_ENCODED_RUSTFLAGS = "-L" + $sep + "native=$cuda\lib\x64"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; cmd.exe /c "call `"$bat`" -arch=x64 -host_arch=x64 && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli"'
```

Result: passed.

## Source Receipt

Receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-one-token-cuda.json
```

Capture command:

```powershell
rtk powershell -Command '$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"; $env:CUDA_PATH = $cuda; $env:PATH = "$cuda\bin;" + $env:PATH; $root = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01"; New-Item -ItemType Directory -Force -Path $root | Out-Null; $model = "C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf"; $base = "ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15"; .\target\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda --model $model --device nvidia-rtx-5070-ti-cuda --device-index 0 --all-layer-plan "$base\qwen3-0_6b-cuda-all-layer-plan.json" --model-boundary-fixtures "$base\qwen3-0_6b-model-boundary-fixtures.json" --kv-cache-policy "$base\qwen3-0_6b-kv-cache-policy.json" --sampling-policy "$base\qwen3-0_6b-sampling-policy.json" --json-out "$root\qwen3-0_6b-one-token-cuda.json"'
```

Result: passed and emitted the receipt.

Key receipt fields:

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
speedup_claim: false
claim_boundary.bitnet_packed_i2s_qk256_proof: false
claim_boundary.server_ready_claimed: false
```

Observed timing envelope:

```text
cpu_reference_total_ms: 13552.0028
total_ms / first_token_ms: 952519.4266
kernel_time_ms: 936719.1936
host_to_device_bytes: 639446688
host_to_device_ms: 15692.9332
device_to_host_bytes: 607744
device_to_host_ms: 1.3855
```

The slow CUDA timing is recorded as evidence, not as a speed claim.

## Receipt Explain Smoke

```powershell
rtk powershell -Command '.\target\debug\bitnet.exe receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --format json'
```

Result: passed. The normalized explanation reported:

```text
model_coverage_row: dense_qwen3_06b_q8_candidate
current_tier: product_cli_ready
selected_backend: nvidia-rtx-5070-ti-cuda
selected_route: dense_regular_llm_cuda
fallback_used: false
speedup_claim: false
full_residency_claim: false
bitnet_packed_i2s_qk256_proof: false
dense_regular_llm_cuda_proof: true
```

## Aggregate Fail-Closed Probe

```powershell
rtk powershell -Command '.\target\debug\qwen3_cuda_repeated_comparator_receipt.exe --one-token-run ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\run-01\qwen3-0_6b-one-token-cuda.json --receipt-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-0_6b-repeated-comparator.json'
```

Result: failed closed as expected. The generator accepted the supplied
`one_token` source path but refused to emit an aggregate while the remaining
profile receipts were absent.

## Remaining Blocker

The aggregate generator remains correctly fail-closed. It requires at least
three source receipts per profile, so this run leaves the required source set at:

```text
one_token: 1 / 3
short_decode_8: 0 / 3
short_decode_32: 0 / 3
warm_session_3_turns: 0 / 3
decode_128_from_warm_context: 0 / 3
```

At the observed one-token runtime, collecting the full 15-receipt set from the
debug CLI is expected to be a multi-hour hardware campaign. The next work should
either run the remaining captures in a deliberate long hardware window or first
reduce the Qwen3 strict CUDA runtime bottleneck without weakening fallback,
quality, or proof-family boundaries.

## Claim Boundary

This report proves only that one current-source Qwen3 one-token strict CUDA
source receipt was emitted for CUDA-MODEL-017.

These remain false:

- `qwen3_cuda_repeated_comparator` aggregate availability
- Qwen3 speedup
- Qwen3 benchmark-qualified speed
- Qwen3 full CUDA residency
- Qwen3 broad dense GGUF readiness
- Qwen2.5 proof inheritance
- BitNet packed I2_S/QK256 proof
- Any claim that a one-token receipt satisfies the remaining profiles
