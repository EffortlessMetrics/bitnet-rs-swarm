# CUDA-BITNET-PERF-005 Profile Manifest

## Summary

`CUDA-BITNET-PERF-005` is the next official BitNet performance collection
step after the earlier repeated strict-ask and warm-session benchmark review.
The first slice added a governed source-capture manifest for the current-source
profile set. The follow-up slice enables aggregate receipt generation once the
profile source receipts exist. It still does not add hardware receipts and does
not promote speed, benchmark, residency, or server claims.

The manifest is emitted by:

```bash
cargo run --locked -p bitnet-bench-receipts --no-default-features --bin strict_bitnet_cuda_repeated_profiles_receipt -- --print-manifest
```

## Profiles

The manifest names the required official BitNet I2_S/QK256 profiles:

```text
one_token
short_decode_8
short_decode_32
prefill_128_decode_16
prefill_512_decode_32
warm_session_3_turns
warm_session_10_turns
decode_128_from_warm_context
```

Each profile requires at least three current-source receipts before an aggregate
receipt can be generated.

The aggregate builder writes:

```text
artifact_kind: strict_bitnet_cuda_repeated_profiles
claim: strict_bitnet_cuda_repeated_profiles_baseline
selected_backend: nvidia-rtx-5070-ti-cuda
selected_route: bitnet_qk256_cuda
kernel_id: qk256_gemv_cuda
fallback_used: false
speedup_claim: false
benchmark_qualified_speedup: false
full_cuda_residency_claimed: false
server_ready_claimed: false
bitnet_packed_i2s_qk256_proof: true
dense_regular_llm_cuda_proof: false
```

## Required Boundaries

The manifest fixes the proof family and strict route:

```text
model: microsoft/bitnet-b1.58-2B-4T-gguf / ggml-model-i2_s.gguf
route: bitnet_qk256_cuda
backend: nvidia-rtx-5070-ti-cuda
runtime_api: cuda
kernel: qk256_gemv_cuda
fallback_used: false
kernel fallback invocations: 0
```

Dense regular-LLM CUDA receipts are rejected for this proof family. Generic
`cuda` backend labels are not strict RTX 5070 Ti proof.

## Claim Boundary

This PR may claim:

- the CUDA-BITNET-PERF-005 capture contract is executable as a manifest;
- the official BitNet profile set and required receipt fields are explicit;
- missing source receipts fail closed with a per-profile report;
- a repeated-profile aggregate can be generated and schema-validated after all
  required source receipts are present.

This PR must not claim:

- current-source hardware receipts were collected;
- speedup is accepted;
- benchmark-qualified speed is accepted;
- full CUDA residency is proven;
- broad server readiness is proven;
- dense SLM CUDA proof satisfies BitNet packed I2_S/QK256 proof.

## Next Work

The next PR should collect or commit the current-source receipts named by the
manifest, then run the aggregate builder against those receipts. The aggregate
becomes input to `CUDA-BITNET-PERF-006`; it does not itself accept speedup or
full residency.
