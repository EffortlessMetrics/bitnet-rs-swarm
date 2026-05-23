# CUDA-MODEL-018 Qwen3 Exact-Profile Benchmark Review

Date: 2026-05-23
Campaign item: CUDA-MODEL-018
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0
Coverage row: `dense_qwen3_06b_q8_candidate`
Linked plan: `plans/native-rust-inference/dense-qwen3.md`
Linked spec: `docs/specs/BITNET-SPEC-0014-runtime-performance-contract.md`

## Summary

This review consumes the Qwen3 repeated comparator aggregate from
CUDA-MODEL-017. It does not run new inference, benchmark, server, tokenizer,
loader, transformer, or kernel code.

Decision:

- accepted speed profiles: none;
- `benchmark_qualified=false`;
- `speedup_claim=false`;
- `full_residency_claim=false`;
- Qwen3 exact-profile server readiness remains separate and unchanged;
- `dense_regular_llm_cuda_proof=true` remains Qwen3-specific;
- `bitnet_packed_i2s_qk256_proof=false`;
- `qwen25_proof_inherited=false`;
- broad dense GGUF readiness remains false.

The aggregate is useful repeated fallback-free comparator evidence, but it does
not qualify speed or residency. Every reviewed CUDA profile is slower than the
same-artifact CPU AVX-512 comparator by total wall-clock time, the current H2D
values are model-load and upload envelopes rather than pure transfer timing,
and the decode profiles still use full-logits D2H sampling by the CPU sampler.

## Evidence

Primary aggregate:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/qwen3-0_6b-repeated-comparator.json
```

Source receipt set:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-one-token-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-short-decode-8-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-short-decode-8-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-short-decode-8-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-short-decode-32-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-short-decode-32-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-short-decode-32-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-warm-session-3-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-warm-session-3-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-warm-session-3-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-01/qwen3-0_6b-decode-128-from-warm-context-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-02/qwen3-0_6b-decode-128-from-warm-context-cuda.json
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-23/qwen3-perf-017/run-03/qwen3-0_6b-decode-128-from-warm-context-cuda.json
```

Aggregate facts:

```text
artifact_kind = qwen3_cuda_repeated_comparator
selected_backend = nvidia-rtx-5070-ti-cuda
selected_route = dense_regular_llm_cuda
runtime_api = cuda
fallback_used = false
profiles_recorded = 5
total_cpu_runs = 15
total_cuda_runs = 15
min_runs_per_backend = 3
same_artifact_sha = true
same_tokenizer_prompt_policy = true
generated_tokens_compared = true
speedup_claim_allowed = false
benchmark_qualified_speedup = false
full_cuda_residency_claimed = false
bitnet_packed_i2s_qk256_proof = false
```

## Exact-Profile Decisions

| Profile | CPU runs | CUDA runs | CPU mean total ms | CUDA mean total ms | CUDA/CPU total ratio | TTFT mean ms | Decode mean ms | CUDA tok/s mean | H2D MB | D2H MB | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `one_token` | 3 | 3 | 13,359.3404 | 799,664.7351 | 59.8581 | 799,664.7351 | 864.9529 | 1.1585 | 609.82 | 0.58 | `not_accepted` |
| `short_decode_8` | 3 | 3 | 14,065.9288 | 521,462.6648 | 37.0728 | 1,066.8107 | 8,598.5311 | 0.9343 | 609.82 | 4.64 | `not_accepted` |
| `short_decode_32` | 3 | 3 | 19,989.0101 | 600,859.9468 | 30.0595 | 1,241.2419 | 46,149.8972 | 0.7723 | 609.82 | 18.55 | `not_accepted` |
| `warm_session_3_turns` | 3 | 3 | 23,455.6711 | 505,861.7307 | 21.5667 | 1,269.6433 | 40,165.1138 | 0.6726 | 609.82 | 13.91 | `not_accepted` |
| `decode_128_from_warm_context` | 3 | 3 | 55,892.6993 | 776,494.5614 | 13.8926 | 1,213.5498 | 203,811.2175 | 0.6466 | 609.82 | 74.19 | `not_accepted` |

Every profile has repeated CPU and CUDA evidence for the exact Qwen3 artifact,
the selected RTX 5070 Ti CUDA backend, `dense_regular_llm_cuda`, and
`fallback_used=false`. Every profile remains rejected for speed and
benchmark-qualified speed because CUDA total time is slower than CPU total time
for the exact profile.

## TTFT And Throughput

TTFT and throughput values are reported as receipt fields, not accepted as
performance claims. The review rejects TTFT and throughput qualification for
all profiles because the current profile totals do not beat the CPU reference
and because the timing decomposition is not yet strong enough to isolate pure
H2D transfer from the model-load and upload envelope.

The H2D value is still an envelope:

```text
host_to_device_ms_source = wall_clock_model_load_with_cuda_weight_upload
host_to_device_ms_scope = model_load_wall_clock_envelope
pure_host_to_device_ms_source = not measured by the Qwen3 runtime receipts
```

The D2H values are measured wall-clock logits extraction, but the decode
profiles still copy full logits for CPU-side sampling. That makes the D2H
numbers useful bottleneck evidence, not a logits-transfer-reduction proof.

## Residency Review

Full residency is rejected.

The source receipts record useful CUDA execution and launch evidence, but full
residency requires phase-level proof for the request lifecycle: model handles,
CUDA context, uploaded weights, workspace reuse, KV cache, norm, RoPE,
attention, MLP, LM head, logits or device-side selection, and session reuse.
The aggregate keeps `full_cuda_residency_claimed=false`, and this review keeps
`full_residency_claim=false` in the model coverage row.

## Server Boundary

Qwen3 server readiness remains exact-profile only and separate from this review.
The existing server-ready claim is scoped to the current-source non-streaming
shared-engine `/v1/chat/completions` receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json
```

CUDA-MODEL-018 does not widen server readiness, streaming readiness,
concurrency readiness, deployment readiness, or broad dense GGUF serving.

## Claim Boundary

May claim:

- Qwen3 repeated same-artifact CPU/CUDA comparator evidence exists for five
  exact profiles on Windows 9950X3D + RTX 5070 Ti.
- The reviewed source receipts are fallback-free and route
  `dense_regular_llm_cuda`.
- Generated token IDs match between CPU comparator and CUDA source receipts.
- CUDA-MODEL-018 reviewed Qwen3 speed, benchmark-qualified speed, TTFT,
  throughput, and residency by exact profile.
- Every reviewed speed and residency promotion was rejected.

Must not claim:

- accepted Qwen3 CUDA speedup;
- `benchmark_qualified=true`;
- benchmark-qualified Qwen3 speed;
- full CUDA residency;
- pure H2D event transfer timing;
- logits-transfer reduction;
- broad dense GGUF readiness;
- Qwen2.5 proof inheritance;
- BitNet packed I2_S/QK256 proof from Qwen3 dense CUDA evidence.

## Next Proof

The next proof should be a Qwen3 optimization and requalification lane, not
another promotion review over the same evidence. It needs at least one real
runtime improvement or sharper measurement boundary before speed or residency
can be reconsidered:

- reduce CUDA total time for one or more exact profiles;
- separate pure H2D transfer timing from model-load and weight-upload envelope
  timing;
- reduce D2H bytes with a device top-k or greedy sampler, or explicitly justify
  full logits;
- record a phase residency map for KV, attention, MLP, LM head, logits, and
  request/session lifecycle;
- rerun repeated same-artifact CPU and RTX 5070 Ti CUDA comparators.

Until that lands, the model coverage matrix should keep
`benchmark_qualified=false`, `speedup_claim=false`, and
`full_residency_claim=false` for Qwen3.

## Validation

This report is derived from committed receipts and reports only. It did not run
new inference, benchmark, server, CUDA, tokenizer, loader, transformer, or
kernel code.

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-23\qwen3-perf-017\qwen3-0_6b-repeated-comparator.json
rtk cargo test --locked -p bitnet-bench-receipts --no-default-features qwen3
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk git diff --check
```
