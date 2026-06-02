# OpenVINO Capability Matrix

This page is the user-facing status map for the Lunar Lake OpenVINO lane. It
summarizes the current Qwen2.5 dense SLM OpenVINO CPU/GPU/NPU evidence and the
separate BitNet OpenVINO subgraph research boundary without changing route
promotion.

The operational source of truth remains the Intel 258V campaign tracker, the
OpenVINO Lunar Lake specs, the route-promotion ledger, and the committed
receipts. Start here before running:

```powershell
bitnet lunar-lake validate --strict
bitnet lunar-lake regress --strict
bitnet lunar-lake compare --strict
bitnet validate open-vino-lunar-lake --receipt <receipt.json>
bitnet receipts explain <receipt.json>
```

## Source Of Truth

| Surface | Source |
| --- | --- |
| OpenVINO route identity and fallback rules | `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md` |
| Dense SLM artifact/export contract | `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md` |
| NPU cold/cache/warm evidence contract | `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md` |
| Quality corpus and token-ID visibility contract | `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md` |
| Phase timing contract | `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md` |
| Route promotion gates | `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md` |
| BitNet proof boundary | `docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md` |
| Lunar Lake route ID/proof-family map | `docs/reviews/lunar-lake-route-id-proof-family-map.md` |
| Rust bridge boundary | `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md` |
| Server boundary | `docs/specs/BITNET-SPEC-OPENVINO-SERVER.md` |
| Implementation plan | `plans/openvino-lunar-lake/implementation-plan.md` |
| Route promotion ledger | `ci/hardware/intel-258v/2026-05-08/lunar-lake-route-promotion.json` |
| Route profile comparison | `ci/hardware/intel-258v/2026-05-08/lunar-lake-route-profile-comparison.json` |
| OpenVINO corpus-v2 receipt | `ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json` |
| GPU corpus-v2 diagnosis status refresh | `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-gpu-corpus-v2-diagnosis-status-refresh.json` |
| OpenVINO phase runner | `ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-phase-runner.json` |
| OpenVINO generated-token visibility boundary | `docs/reviews/lunar-lake-openvino-token-visibility.md`, [#1244](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244) |
| CPU slow-path and OpenVINO CPU decision boundary | `docs/research/lunar-lake-cpu-slow-path.md`, `docs/reviews/lunar-lake-cpu-route-decision.md`, [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232) |
| GPU profile promotion review boundary | `docs/reviews/lunar-lake-openvino-gpu-promotion-review.md`, `docs/reviews/lunar-lake-openvino-gpu-phase-profile-review.md`, [#1241](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1241) |
| NPU cold/cache and warm-resident boundary | `docs/research/lunar-lake-npu-cold-start.md`, `docs/reviews/lunar-lake-npu-cold-cache-evidence.md`, `docs/reviews/lunar-lake-npu-warm-resident-acceptance.md`, [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119) |
| NPU runtime AUTO selected-device boundary | `docs/reviews/lunar-lake-openvino-auto-selected-device.md`, [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149), [#1242](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1242) |
| Low-power evidence boundary | [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064) |

## Current OpenVINO Rows

| Row | Model / proof family | Backend | Runtime device | Status | Evidence | Boundary |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 OpenVINO CPU | Dense SLM OpenVINO GenAI | `openvino-cpu` | `CPU` | candidate | IR manifest, LLMPipeline smoke, corpus-v2 receipt, phase runner, Rust-vs-OpenVINO CPU comparison | Not the promoted default; dense GGUF CPU remains default until a route-promotion review changes it. |
| Qwen2.5 OpenVINO GPU | Dense SLM OpenVINO GenAI | `openvino-gpu` | `GPU.0` / Arc 140V | profile-promoted | Arc 140V OpenVINO ask receipts, corpus-v2 receipt, phase runner, route-promotion ledger, route-profile comparison, validator gate | Promoted only for `ask_short`, `ask_normal`, `prefill_heavy`, and `decode_heavy`; no native OpenCL, BitNet, low-power, speedup, power-advantage, or broad quality claim. |
| Qwen2.5 OpenVINO NPU | Dense SLM OpenVINO GenAI | `openvino-npu` | `NPU` | profile-promoted | NPU ask receipt, corpus-v2 receipt, phase runner, cold-start diagnosis, resident evidence, route-promotion ledger, route-profile comparison, validator gate | Promoted only for `warm_resident`; no cold one-off, dynamic decode, beam/parallel sampling, low-power, native NPU, or BitNet claim. |
| Qwen2.5 OpenVINO server | Dense SLM exact-profile server | OpenVINO route under server | exact profile only | planned | Server spec only | No server readiness until ask/chat readiness, underlying route linkage, exposure fields, cold/warm timing, streaming/concurrency boundaries, and exact-profile receipts exist. |
| BitNet OpenVINO NPU subgraphs | BitNet-shaped static subgraphs | `openvino-npu` | `NPU` | diagnostic | RMSNorm, linear projection, and FFN/static subgraph parity receipts in the 258V lane | No full BitNet inference, QK256 decode, dynamic decode, or acceleration claim. |
| BitNet OpenVINO GPU/subgraphs | BitNet-shaped OpenVINO research | `openvino-gpu` | `GPU.0` | planned/diagnostic | Current Arc GPU proof is OpenVINO smoke plus separate native OpenCL parity outside this row | No full BitNet GPU inference, no native OpenCL proof, and no QK256 accelerator decode claim. |

## Route Status

| Route | Promotion status | Promoted profiles | Current blockers |
| --- | --- | --- | --- |
| `dense_slm_default_cpu` | promoted | `regression_tiny`, `structured` | CPU remains the default route ID and regression baseline; OpenVINO routes supersede it only for their promoted profiles. |
| `dense_slm_openvino_gpu_candidate` | profile-promoted | `ask_normal`, `ask_short`, `decode_heavy`, `prefill_heavy` | Not promoted for `regression_tiny`, `structured`, `warm_resident`, `low_power`, or BitNet reference; no native OpenCL, power, or broad acceleration claim. |
| `dense_slm_openvino_npu_candidate` | profile-promoted | `warm_resident` | Cold one-off use, `low_power`, dynamic decode, beam/parallel sampling, and BitNet/QK256 execution remain blocked or unproven. |
| `bitnet_reference_cpu` | promoted for BitNet reference only | `bitnet_strict_reference` | Not a dense SLM route; dense SLM success never counts as BitNet proof. |

## Claim Boundaries

- OpenVINO dense SLM receipts are not BitNet packed I2_S/QK256 proof.
- OpenVINO GPU receipts are not native OpenCL execution proof.
- OpenVINO NPU receipts are not native NPU kernel proof.
- OpenVINO GPU/NPU candidate evidence is not route promotion; the
  route-promotion ledger and route-profile comparison are the route-status
  authority.
- `fallback_used=false` or an explicit strict no-fallback policy is mandatory
  for OpenVINO status rows.
- Retokenized generated token IDs must be labeled as retokenized and must not
  be described as direct OpenVINO GenAI pipeline internals.
- NPU hot-path timing does not prove cold one-off usability.
- NPU route promotion requires cache identity plus warm or resident evidence.
- Speedup, power advantage, low-power promotion, and server readiness require
  separate exact-profile receipts.

## Validator

The receipt-boundary validator rejects hidden fallback, backend/device drift,
retokenized-token ambiguity, dense-SLM-to-BitNet claim leakage,
OpenVINO-GPU-to-native-OpenCL claim leakage, and premature NPU promotion
without cache plus warm/resident evidence:

```powershell
bitnet validate open-vino-lunar-lake `
  --receipt ci\hardware\intel-258v\2026-05-08\slm-openvino-cpu-gpu-npu-corpus-v2.json
```

## Next Proofs

| Row | Next proof |
| --- | --- |
| Qwen2.5 OpenVINO CPU | Use #1232 resident Rust GGUF phase evidence to isolate per-prompt cost before any matched Rust GGUF versus OpenVINO CPU comparison. No OpenVINO CPU promotion or speedup claim is valid while model-format, timing-scope, prompt-render, tokenization, and runtime-scope mismatches remain unresolved or unaccepted. |
| Qwen2.5 OpenVINO GPU | Keep current profile-scoped promotion. #1241 owns any future phase-split receipts or status fields that refine prefill/decode claim boundaries, but no broader GPU promotion or power claim follows without a fresh route review. `low_power` remains blocked by #1064. |
| Qwen2.5 OpenVINO NPU | Keep `warm_resident` scoped promotion only. Cold/cache work stays under #1119 until direct cache truth, an accepted stricter proxy policy, or profile-matched phase evidence exists for the exact mode. #1242 is only AUTO debug-log parser/source planning; no cold one-off, `ask_short`, `ask_normal`, or `low_power` expansion follows from cache timing or AUTO debug-log evidence alone. |
| OpenVINO generated-token visibility | Use #1244 for future schema or checker work that distinguishes direct pipeline IDs from retokenized or text-only evidence. Token visibility alone does not promote routes, prove matched CPU/OpenVINO format parity, or satisfy BitNet QK256/I2_S evidence. |
| OpenVINO server | Add exact-profile server receipts only after the underlying ask/chat route is promoted or explicitly candidate-scoped. |
| BitNet OpenVINO subgraphs | Continue static-shape CPU-reference parity for selected subgraphs; keep QK256/dynamic decode out of scope until separately proven. |

## Validation

Run these checks after editing this page or the underlying status surfaces:

```powershell
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
npx --yes markdownlint-cli2@0.18.1 --config .markdownlint.jsonc docs/status/OPENVINO_CAPABILITY_MATRIX.md docs/status/README.md plans/openvino-lunar-lake/implementation-plan.md
git diff --check
```
