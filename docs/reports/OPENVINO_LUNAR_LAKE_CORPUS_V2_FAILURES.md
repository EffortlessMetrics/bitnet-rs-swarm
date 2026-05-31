# OpenVINO Lunar Lake Corpus V2 Failure Report

Status: diagnostic report
Created: 2026-05-18
Machine: intel-258v

## Scope

This report summarizes Lunar Lake OpenVINO dense SLM corpus-v2 candidate-route
failures after rerunning the corpus under the accepted one-token
`yes_no_clear_sky` fixture policy and the cross-runtime exact-text
`stop_token_one_word_done` fixture, plus the cross-runtime prefill-heavy and
decode-heavy fixture updates. It does not promote any route, claim speedup or
power advantage, claim native Arc/NPU acceleration, or change BitNet QK256/I2_S
behavior.

## Source Evidence

```text
ci/hardware/intel-258v/2026-05-08/slm-openvino-cpu-gpu-npu-corpus-v2.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-cpu-corpus-v2-diagnosis.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-gpu-corpus-v2-diagnosis.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-corpus-v2-diagnosis.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-generation-budget-sensitivity.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-gpu-corpus-v2-diagnosis-status-refresh.json
ci/hardware/intel-258v/2026-05-08/lunar-lake-route-profile-comparison.json
ci/quality/lunar-lake-answer-corpus-v2.yaml
```

The CPU/GPU/NPU corpus-v2 quality diagnoses now record 14/14 passing cases,
`fallback_used=false`, and direct OpenVINO GenAI generated-token IDs. The older
quality-diagnosis `promotion_status=candidate_only_not_promoted` field is not
the route-status authority. Current route status comes from
`lunar-lake-route-promotion.json` and
`lunar-lake-route-profile-comparison.json`.

## Route Summary

| Route | Corpus v2 result | Failed profiles | Promotion result |
| --- | ---: | --- | --- |
| OpenVINO CPU | 14/14 pass, 0 fail | none | Candidate; dense GGUF CPU remains default for promoted CPU profiles |
| OpenVINO GPU.0 / Arc 140V | 14/14 pass, 0 fail | none | Profile-promoted for `ask_short`, `ask_normal`, `prefill_heavy`, and `decode_heavy` |
| OpenVINO NPU | 14/14 pass, 0 fail | none | Profile-promoted for `warm_resident` only |

Unpromoted profiles still remain blocked by their own route-profile evidence:
`low_power` lacks battery-mode power evidence, `structured` and
`regression_tiny` remain CPU-promoted, and BitNet reference behavior is not a
dense SLM OpenVINO claim.

## Failure Classification

The current OpenVINO CPU/GPU/NPU corpus-v2 rerun has no failed cases. The prior
prefill-heavy and decode-heavy answer-content blockers were fixture/prompt
stability issues: the canonical corpus now asks the prefill-heavy summary to
include the exact words `Lunar`, `CPU`, and `route`, and asks the decode-heavy
case to use a stable route-check phrase bank containing `fallback` and `model`.

## Budget Sensitivity

The generation-budget sensitivity receipt isolates the normalized-match cases:

| Case | CPU | GPU.0 | NPU | Interpretation |
| --- | --- | --- | --- | --- |
| yes_no_clear_sky | passes at max_new_tokens=1 only | passes at max_new_tokens=1 only | passes at max_new_tokens=1 only | Accepted one-token fixture policy; rerun now passes |
| stop_token_one_word_done | passes at max_new_tokens=1/2/4 | passes at max_new_tokens=1/2/4 | passes at max_new_tokens=1/2/4 | Cross-runtime exact-text fixture now passes |

This means the yes/no failure was a stop/max-token fixture-budget issue rather
than a standing route-quality blocker after rerun. The one-word `done` case is
no longer an OpenVINO candidate blocker under the cross-runtime fixture wording.

## Profile Blockers

OpenVINO GPU.0 remains unpromoted for:

- `regression_tiny` and `structured`, where CPU remains the promoted route.
- `low_power`, where battery-mode telemetry, a valid energy proxy, and
  benchmark-qualified power advantage are still missing.
- `warm_resident`, where NPU is the promoted route.

OpenVINO NPU remains unpromoted for:

- Cold one-off `ask_short`, `ask_normal`, `prefill_heavy`, and `decode_heavy`,
  where cold-load and cache behavior remain bounded by separate NPU evidence.
- `low_power`, where battery-mode telemetry, a valid energy proxy, and
  benchmark-qualified power advantage are still missing.
- Dynamic decode, beam search, parallel sampling, native NPU, and BitNet/QK256
  execution claims.

OpenVINO CPU remains unpromoted for:

- OpenVINO CPU-specific promotion. Dense GGUF CPU remains the promoted CPU
  default for `regression_tiny` and `structured` until OpenVINO CPU proves a
  profile-specific advantage.

## Next Actions

1. Use the route-promotion ledger and route-profile comparison as the
   route-status authority.
2. Preserve direct versus retokenized generated-token visibility in every
   OpenVINO candidate receipt.
3. Keep `low_power` blocked until POWER-006 battery-mode telemetry and
   energy-proxy evidence prove a benchmark-qualified power advantage.

## Claim Boundary

This report supports only the following claim:

```text
Existing Lunar Lake OpenVINO CPU/GPU/NPU corpus-v2 routes pass the bounded
quality fixture. Route promotion remains profile-scoped and is governed by the
route-promotion ledger and route-profile comparison.
```

It does not prove unscoped OpenVINO GPU/NPU route promotion, speedup, power
advantage, native OpenCL execution, native NPU execution, full BitNet
accelerator inference, packed QK256 accelerator decode, or BitNet QK256/I2_S
behavior.
