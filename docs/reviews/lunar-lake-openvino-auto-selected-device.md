# Lunar Lake OpenVINO AUTO Selected-Device Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149), [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064), [#1123](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123), [#1135](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1135)
Linked PRs: n/a
Support-tier impact: no promotion; review-only AUTO selected-device evidence contract
Policy impact: no policy exception

## Recommendation

Keep OpenVINO `AUTO` diagnostic until a receipt proves the device OpenVINO
actually executed for every route-relevant phase.

This review records the #1149 evidence boundary only. It does not run
inference, add an `AUTO` runtime command, refresh receipts, change route policy,
promote NPU or GPU, claim speedup, claim power advantage, or add BitNet
QK256/I2_S behavior evidence.

The key distinction is:

```text
CLI --device auto route selection != OpenVINO runtime AUTO selected-device proof
```

Existing Lunar Lake `--device auto` ask receipts show the campaign route
selector choosing a promoted route. They do not by themselves prove what the
OpenVINO runtime would select if `AUTO` were requested at the OpenVINO layer.

## Current Evidence Snapshot

| Evidence | Review consequence |
| --- | --- |
| `BITNET-SPEC-OPENVINO-ROUTE-CONTRACT` says `--device openvino-auto` is diagnostic unless execution devices are recorded | This review follows the existing spec; it does not create a new promotion rule |
| `BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE` says AUTO receipts without actual execution devices are diagnostic only | NPU cold/cache or resident evidence cannot inherit AUTO proof |
| `lunar-lake-route-id-proof-family-map.md` maps `openvino-auto` to no proof family until selected execution devices are recorded | Route identity stays unresolved without runtime-selected-device fields |
| Current ask receipts with `requested_device=auto` are CLI route-selector receipts | They are useful route-policy artifacts, not runtime `AUTO` selected-device evidence |
| No committed receipt currently records OpenVINO `EXECUTION_DEVICES` or an equivalent selected-device property for runtime `AUTO` | AUTO remains diagnostic even when answer gates pass |

## Required Evidence Package

Before any `AUTO` receipt is cited as GPU, NPU, `low_power`, speed, or route
promotion evidence, one package must compare:

- explicit `openvino-gpu`;
- explicit `openvino-npu`;
- runtime-layer OpenVINO `AUTO`;
- the same model/export/tokenizer/template;
- the same prompt, generation config, answer gate, and cache settings;
- the same artifact root and claim boundary.

The receipt or review package must record:

| Field | Required value or handling |
| --- | --- |
| `auto_scope` | `cli_route_selector` or `openvino_runtime_auto`; only the latter can satisfy #1149 |
| CLI request | Requested CLI device, requested route, selected route, and profile |
| OpenVINO request | Requested OpenVINO device string, including `AUTO` when tested |
| Runtime identity | Runtime API, selected backend, runtime device, resolved device name, OpenVINO version, and relevant device properties |
| Execution devices | `EXECUTION_DEVICES`, equivalent selected-device property, or explicit `not_exposed` |
| Phase scope | Selected-device visibility for pipeline construction, compile/load/cache, first generate, and warm generate when exposed |
| Fallback | `fallback_used=false` and no hidden CPU/GPU/NPU drift |
| Quality | Answer gate result and generated-token visibility source |
| Timing | Pipeline construction, OpenVINO load/cache timing or explicit unavailable state, first token, decode/generation, and total response |
| Cache | Cache directory, cache settings, cache classification source, and whether classification is runtime truth or diagnostic |
| Claim boundary | Exact statement of what the receipt does not prove |

If OpenVINO does not expose execution devices, record `not_exposed`. That still
creates useful diagnostic evidence, but it does not prove selected-device
identity and cannot support a route promotion.

## Fail-Closed Conditions

| Condition | Required decision |
| --- | --- |
| CLI `--device auto` route selection is cited as OpenVINO runtime `AUTO` proof | Reject selected-device claim |
| Runtime `AUTO` receipt lacks `EXECUTION_DEVICES`, equivalent evidence, or explicit `not_exposed` | Mark incomplete; no selected-device proof |
| Runtime `AUTO` records `not_exposed` for selected-device visibility | Diagnostic only |
| Runtime `AUTO` selects CPU for a GPU or NPU claim | No accelerator proof |
| Runtime `AUTO` changes devices by phase and phase-specific evidence is missing | Diagnostic only for route policy |
| Explicit GPU/NPU and `AUTO` runs use different model, prompt, generation config, cache, tokenizer, or artifact root | Comparison invalid |
| Answer gate passes but selected-device visibility is absent | Quality diagnostic only; no GPU/NPU route proof |
| Fallback appears or selected backend is inconsistent | Block route evidence |
| Direct generated-token IDs are unavailable | No direct pipeline-internal token claim |
| Battery telemetry or energy proxy is missing | No `low_power` or power-advantage claim |
| Dense SLM `AUTO` evidence is cited as BitNet proof | Reject claim boundary |

## Route Consequences

### CLI `--device auto`

Keep the existing campaign route selector behavior separate from OpenVINO
runtime `AUTO`. The selector may choose a profile-promoted route from the route
ledger, but that receipt should not be reused as proof of OpenVINO automatic
device selection.

### OpenVINO Runtime `AUTO`

Keep runtime `AUTO` diagnostic until selected-device visibility is present and
unambiguous. If visibility is unavailable, preserve the `not_exposed` gap and
use explicit `openvino-gpu` or explicit `openvino-npu` receipts for promotion
review.

### GPU And NPU Promotion

Do not broaden GPU or NPU route promotion from `AUTO` evidence unless the
receipt proves the selected execution device for the exact profile and all
other route-promotion gates still pass.

### `low_power`

Keep `low_power` blocked by #1064. `AUTO` selected-device proof would still be
only one identity gate; it would not provide battery-mode route samples, energy
proxy evidence, thermal context, or a power-advantage claim.

### BitNet

Dense SLM OpenVINO `AUTO` evidence remains in the dense SLM proof family. It
does not prove native OpenCL, native NPU kernels, full BitNet inference, packed
QK256 decode, or BitNet QK256/I2_S parity.

## Next Smallest PR

No route-policy PR is required from this review alone.

The next implementation PR, if needed, should be one narrow receipt/schema or
measurement step scoped by #1149. It should add only enough structure to record:

- `auto_scope`;
- requested CLI route/device versus requested OpenVINO runtime device;
- selected backend/runtime device/resolved device;
- execution-device property value or `not_exposed`;
- phase applicability for selected-device visibility;
- fallback, answer-gate, token-visibility, timing, cache, and claim-boundary
  fields.

Do not combine that implementation with route promotion, new inference surface
work, low-power promotion, benchmark matrices, generated-dashboard churn, or
unrelated hardware lanes.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- new generated dashboards;
- route-policy mutation;
- route promotion or revocation;
- OpenVINO runtime `AUTO` selected-device proof;
- GPU, NPU, or `low_power` promotion;
- speedup or power-advantage evidence;
- battery-mode evidence;
- measured-temperature evidence;
- native OpenCL proof;
- native NPU kernel proof;
- BitNet QK256/I2_S behavior proof.

It only defines the selected-device evidence contract required before runtime
`AUTO` can move from diagnostic context into route-policy evidence.
