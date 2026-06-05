# BitNet-rs Support Matrix

This page is the user-facing entry point for v0.3 usable-preview support
posture. It summarizes current model and device rows without changing any
claim. The machine-readable authority remains
`ci/model-artifacts/model-coverage-matrix.toml`; receipts and hardware matrices
prove what actually ran.

## Source Of Truth

| Surface | Owns |
| --- | --- |
| `ci/model-artifacts/model-coverage-matrix.toml` | Model tier, claim booleans, proof-family booleans, next proof, and forbidden claims. |
| [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md) | Human-readable model coverage summary. |
| [BitNet Capability Matrix](BITNET_CAPABILITY_MATRIX.md) | Official BitNet 2B I2_S/QK256 posture and route boundaries. |
| [CUDA Capability Matrix](CUDA_CAPABILITY_MATRIX.md) | RTX 5070 Ti CUDA model, route, server, and speed boundaries. |
| [Apple Capability Matrix](APPLE_CAPABILITY_MATRIX.md) | Apple M4 CPU/NEON, Metal phase, MPSGraph, and Neural Engine boundaries. |
| [OpenVINO Capability Matrix](OPENVINO_CAPABILITY_MATRIX.md) | Lunar Lake OpenVINO route-promotion and diagnostic boundaries. |
| [Hardware Matrix](../hardware/HARDWARE_MATRIX.md) | Device labels and hardware proof identity. |
| [Receipt Explain Schema](../specs/BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md) | Support summary fields for raw receipts. |
| [Support Bundle Spec](../specs/BITNET-SPEC-SUPPORT-BUNDLE.md) | Issue-ready support bundle fields. |

If this page disagrees with a model coverage row, receipt, route matrix, or
hardware matrix, keep the narrower claim and repair the mismatch before
promoting user-facing wording.

## Model Support Summary

| Model or family | Current posture | Supported devices or routes | Proof pointer | Claim boundary |
| --- | --- | --- | --- | --- |
| Official Microsoft BitNet 2B I2_S/QK256 | Supported preview | CPU answer lane and strict `nvidia-rtx-5070-ti-cuda` route where receipts match `bitnet_qk256_cuda`; Apple CPU/NEON only where Apple receipts say so. | `bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s`; `bitnet receipts explain --latest`; [BitNet matrix](BITNET_CAPABILITY_MATRIX.md). | No TL1/TL2, BF16/GPU-int2, dense SLM, speedup, full residency, Metal, A770, or broad server inheritance. |
| Qwen2.5 0.5B Q8_0 dense SLM | Supported preview | `dense_regular_llm_cuda` on RTX 5070 Ti; Apple M4 CPU/NEON dense SLM default through Apple docs. | `bitnet model verify qwen2.5-0.5b-instruct-q8_0`; [CUDA matrix](CUDA_CAPABILITY_MATRIX.md); [Apple matrix](APPLE_CAPABILITY_MATRIX.md). | Dense SLM proof is not BitNet QK256 proof. Speedup and full residency remain false unless exact benchmark/residency receipts promote them. |
| Qwen3 0.6B Q8_0 dense SLM | Supported preview for bounded CUDA CLI paths | `dense_regular_llm_cuda` on RTX 5070 Ti. | `bitnet model verify qwen3-0.6b-instruct-q8_0`; [CUDA matrix](CUDA_CAPABILITY_MATRIX.md). | Does not inherit Qwen2.5 or BitNet proof; speedup, benchmark-qualified speed, full residency, and broad dense support remain false. |
| Qwen2.5 0.5B Q4_K_M dense SLM | Supported Apple preview, non-default | `apple-m4-cpu-neon` only. | [Apple dense SLM matrix](../slm/apple-m4-dense-slm-model-support-matrix.md); [Apple matrix](APPLE_CAPABILITY_MATRIX.md). | Storage-conscious Apple CPU/NEON answer path only; no CUDA, Metal, MPSGraph, Neural Engine, BitNet, or speed claim. |
| Qwen2.5 1.5B Q4_K_M dense SLM | Supported Apple preview, non-default | `apple-m4-cpu-neon` only. | [Apple dense SLM matrix](../slm/apple-m4-dense-slm-model-support-matrix.md); model coverage row `small_llm_qwen25_15b_q4km_candidate`. | Apple CPU/NEON evidence only; CUDA currently fails closed for this artifact; no broad Apple Silicon or speed claim. |
| SmolLM2 360M Q8_0 | Candidate / structurally valid | No supported answer route. | Model coverage row `dense_smollm2_360m_candidate`; next proof is same-prompt reference/top-k or checkpoint comparator. | No CPU answer, CUDA answer, product CLI, speed, server, full residency, or BitNet proof. |
| Llama 3.2, Gemma, Phi, later Qwen, larger SmolLM2 | Registered or candidate | No supported answer route unless a later exact row is promoted. | [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md). | Needs exact artifact, tokenizer, prompt, CPU sanity, backend route, quality, and receipt ladder before support. |
| TL1, TL2, BF16/GPU-int2 BitNet routes | Registered candidates | No supported answer route. | [BitNet matrix](BITNET_CAPABILITY_MATRIX.md). | These routes do not inherit I2_S/QK256 proof. |
| A770, ROCm, WGPU/Vulkan, OpenVINO, Metal full inference, MPSGraph, Neural Engine | Diagnostic, candidate, or planned unless an exact matrix row says otherwise. | No broad supported route. | [Hardware Matrix](../hardware/HARDWARE_MATRIX.md), [OpenVINO matrix](OPENVINO_CAPABILITY_MATRIX.md), [Apple matrix](APPLE_CAPABILITY_MATRIX.md). | Detection, smoke, subgraph, or static-shape proof is not full local-answer support. |

## Device Support Summary

| Device label | Current posture | Proof command or receipt family | Must not claim |
| --- | --- | --- | --- |
| `cpu` and exact CPU labels | Supported preview only for rows with accepted CPU answer receipts. | `bitnet model status --device cpu --format json`; model-specific receipt explanation. | Accelerator execution, speedup, server readiness, or another CPU profile. |
| `nvidia-rtx-5070-ti-cuda` | Supported preview for exact CUDA rows in [CUDA Capability Matrix](CUDA_CAPABILITY_MATRIX.md). | `bitnet model status --device nvidia-rtx-5070-ti-cuda --format json`; `bitnet receipts explain --latest`. | Generic CUDA/GPU support, WGPU/Vulkan, speedup, full residency, or broad server readiness. |
| `apple-m4-cpu-neon` | Supported preview for exact Apple M4 CPU/NEON rows in [Apple Capability Matrix](APPLE_CAPABILITY_MATRIX.md). | `bitnet mac ask`, `bitnet mac validate`, and `bitnet mac receipts-check` receipt families. | Metal, MPSGraph, Neural Engine, MacBook, broad Apple Silicon, or speed claims. |
| `apple-m4-metal` | Phase/subgraph diagnostic unless a future full-model receipt promotes it. | Apple Metal probe, smoke, parity, prefill, and projection-residual receipts. | Full model inference, QK256 acceleration, Neural Engine, MPSGraph, or performance. |
| `apple-m4-mpsgraph` | Graph/reference diagnostic. | MPSGraph smoke receipts and resolved-target fields. | Native Metal or Neural Engine proof without separate receipt evidence. |
| OpenVINO CPU/GPU/NPU labels | Candidate or diagnostic. | `bitnet validate open-vino-lunar-lake --receipt <receipt.json>` and route-promotion ledgers. | Route promotion, speed, low-power, server, native OpenCL, or BitNet QK256 proof. |
| A770, ROCm, WGPU, Vulkan, generic `gpu` | Diagnostic or registered unless exact rows are promoted. | Hardware-specific receipts and status docs. | Answer readiness, speedup, full residency, server readiness, or generic GPU support. |

## User Commands

Start with status before running model or hardware work:

```bash
bitnet model status --format json
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
bitnet model status --device apple-m4-cpu-neon --format json
```

Then use the model-specific fetch, verify, answer, and receipt commands from
the linked device matrix. A supported preview answer path must emit or point to
a receipt that `bitnet receipts explain` can summarize:

```bash
bitnet model fetch <supported-model>
bitnet model verify <supported-model>
bitnet ask --model <supported-model> --device <supported-device> "What is 2+2?"
bitnet receipts explain --latest --format json
bitnet support bundle --latest --device <supported-device> --format json
```

If a command is hardware-specific, the linked matrix must say whether it can run
without that hardware. Status commands may summarize support without executing
inference; receipts prove what actually happened.

## Claim Boundary Checklist

Do not promote support unless the exact row has proof for all relevant fields:

```text
model coverage row
artifact identity and checksum
tokenizer and prompt authority
requested backend
selected backend
selected route
fallback_used
quality gate result
speedup_claim
server_ready and server_ready_scope
full_residency_claim
proof family booleans
receipt path
next proof or blocker
```

## Validation

Run after editing this page or linked status pages:

```powershell
cargo run --locked -p xtask --no-default-features -- check-model-coverage
npx --yes markdownlint-cli2@0.18.1 --config .markdownlint.jsonc docs/status/SUPPORT_MATRIX.md docs/status/README.md
git diff --check
```
