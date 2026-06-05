# Apple Capability Matrix

This page is the user-facing status map for Apple Silicon support. It
summarizes the current Apple M4 Mac mini rows without changing any model,
backend, speed, server, or release claim.

The operational source of truth remains the Apple Silicon source map, the Apple
M4 runbook, the dense SLM support matrix, model coverage, campaign trackers,
and committed receipts.

## Source Of Truth

| Surface | Source |
| --- | --- |
| Apple route and proof-family map | [Apple Silicon source-of-truth map](../apple-silicon/README.md) |
| Apple M4 operator path | [Apple M4 Mac mini operator runbook](../hardware/apple-m4-mac-mini-operator-runbook.md) |
| Apple M4 hardware profile | [Apple M4 Mac mini validation profile](../hardware/apple-m4-mac-mini-validation.md) |
| Dense SLM model support | [Apple M4 dense SLM model support matrix](../slm/apple-m4-dense-slm-model-support-matrix.md) |
| Release go/no-go rules | [Apple M4 release go/no-go matrix](../slm/apple-m4-release-go-no-go.md) |
| Model coverage authority | [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md) and `ci/model-artifacts/model-coverage-matrix.toml` |
| Hardware labels | [Hardware Matrix](../hardware/HARDWARE_MATRIX.md) |
| Apple campaign state | `docs/tracking/campaigns/apple-m4*/` |
| Receipts | `ci/hardware/apple-m4-mac-mini/**` |

## Current Apple Rows

| Model or route | Backend label | Status | Proof command or receipt family | Boundary |
| --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Q8_0 dense SLM | `apple-m4-cpu-neon` | Supported preview, default dense SLM Mac path. | `bitnet model fetch qwen2.5-0.5b-instruct-q8_0`; `bitnet model verify qwen2.5-0.5b-instruct-q8_0`; `bitnet mac ask`; `bitnet mac validate`. | Dense Apple CPU/NEON only; not BitNet, CUDA, Metal, MPSGraph, Neural Engine, MacBook, broad Apple Silicon, or speed proof. |
| Qwen2.5 0.5B Q4_K_M dense SLM | `apple-m4-cpu-neon` | Supported preview, non-default. | `bitnet model verify qwen2.5-0.5b-instruct-q4_k_m`; Apple dense SLM quality and determinism receipts. | Explicit model ID only; no CUDA, Metal, BitNet, speed, server, or broad Apple claim. |
| Qwen2.5 1.5B Q4_K_M dense SLM | `apple-m4-cpu-neon` | Supported preview, non-default larger dense SLM. | `bitnet model verify qwen2.5-1.5b-instruct-q4_k_m`; Apple M4 model-breadth quality receipts; model coverage row `small_llm_qwen25_15b_q4km_candidate`. | Apple CPU/NEON answer model only; the 5070 Ti strict CUDA all-layer plan fails closed, so no CUDA claim. |
| Official Microsoft BitNet 2B I2_S | `apple-m4-cpu-neon` | Supported preview only for exact receipt-backed one-shot, warm, eval, or benchmark paths. | `bitnet mac bitnet-proof`; `bitnet mac ask --model-id microsoft-bitnet-b1.58-2B-4T-i2s`; Apple BitNet eval and benchmark receipts. | BitNet CPU/NEON only; no chat, serve, full Metal inference, QK256 acceleration, Neural Engine, MPSGraph, MacBook, broad Apple Silicon, or speedup claim. |
| Apple Metal phase/subgraph work | `apple-m4-metal` | Diagnostic phase proof. | Metal probe, tiny smoke, I2_S parity, prefill contribution, and projection-residual receipts. | Metal visibility or phase parity is not full model inference, QK256 on Metal, speedup, or server readiness. |
| MPSGraph reference work | `apple-m4-mpsgraph` | Diagnostic graph/reference proof. | MPSGraph smoke receipts with resolved-target fields. | Not native Metal proof and not Neural Engine proof unless a separate receipt proves the resolved target. |
| Neural Engine | future explicit route | Unsupported. | No current accepted proof command. | Do not infer Neural Engine execution from Apple Silicon, MPSGraph, Metal, or unified memory. |
| MacBook auxiliary lanes | MacBook-specific labels | Separate auxiliary evidence only. | Apple MacBook lane specs and receipts when present. | MacBook evidence does not prove M4 Mac mini behavior; M4 Mac mini evidence does not prove MacBook behavior. |

## Route Status

| Route family | Current state | Next proof |
| --- | --- | --- |
| `apple_m4_cpu_neon_dense_slm` | Supported for exact Qwen2.5 rows listed above. | Keep model support matrix, cache verification, quality receipts, and deterministic receipts current for any new model ID. |
| `apple_m4_cpu_neon_bitnet` | Supported preview for exact accepted BitNet artifact receipts. | Keep one-shot, warm, eval, benchmark, and release go/no-go evidence aligned before widening public wording. |
| `apple_m4_metal_phase` | Diagnostic phase/subgraph proof. | Full-model Metal inference requires a separate receipt family with no CPU fallback and matching generated-token behavior. |
| `apple_m4_mpsgraph_reference` | Diagnostic graph/reference proof. | Any Neural Engine or native Metal wording requires separate resolved-target proof. |
| Apple service/chat for BitNet | Disabled or no-go unless route-state rows and receipts promote it. | BitNet chat and serve need their own ready gates; ask, warm, eval, or benchmark proof is not enough. |

## User Commands

Status and artifact checks:

```bash
bitnet model status --device apple-m4-cpu-neon --format json
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
bitnet model verify qwen2.5-0.5b-instruct-q8_0
bitnet mac check
```

Dense SLM answer and validation:

```bash
bitnet mac ask \
  --question "What is 2+2? Answer briefly." \
  --json-out target/apple-m4-productization/mac-ask.json

bitnet mac validate \
  --json-out target/apple-m4-productization/mac-validate.json
```

BitNet CPU/NEON proof and receipt checks:

```bash
bitnet mac bitnet-proof \
  --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --proof-receipt ci/hardware/apple-m4-mac-mini/YYYY-MM-DD/bitnet-local-answer/bitnet-answer-corpus-full-release.json \
  --strict \
  --json-out target/apple-m4-bitnet-proof-receipt-check.json

bitnet mac receipts-check <receipt-or-bundle> --json
```

These commands are examples of the accepted surfaces. Use current receipt paths
from the Apple runbook or campaign when citing proof.

## Claim Boundaries

- Apple CPU/NEON proof is not Metal proof.
- Dense Qwen proof is not BitNet proof.
- BitNet CPU/NEON proof is not dense SLM proof.
- Metal visibility, smoke, or subgraph parity is not full Metal inference.
- MPSGraph smoke is not native Metal or Neural Engine proof.
- CPU fallback cannot count as Metal execution.
- MacBook evidence is not M4 Mac mini proof.
- M4 Mac mini evidence is not broad Apple Silicon proof.
- QK256 x86/CUDA evidence is not QK256-on-Metal evidence.
- No Apple row currently claims speedup, full residency, production serving,
  broad server readiness, or broad Apple Silicon performance.

## Validation

Run after editing this page or the underlying Apple status docs:

```powershell
npx --yes markdownlint-cli2@0.18.1 --config .markdownlint.jsonc docs/status/APPLE_CAPABILITY_MATRIX.md docs/status/SUPPORT_MATRIX.md docs/status/README.md
git diff --check
```

If this page changes model coverage or campaign status, also run the exact
model coverage and campaign checks named by those changed authorities.
