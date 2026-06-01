# Lunar Lake OpenVINO AUTO Selected-Device Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149), [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064), [#1123](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123), [#1135](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1135)
Linked PRs: [#1158](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1158), [#1159](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1159)
Support-tier impact: no promotion; review-only AUTO selected-device evidence contract
Policy impact: no policy exception

## Recommendation

Keep OpenVINO `AUTO` diagnostic until a receipt proves the device OpenVINO
actually executed for every route-relevant phase.

This review records the #1149 evidence boundary only. It does not run
inference, add an `AUTO` runtime command, refresh receipts, change route policy,
promote NPU or GPU, claim speedup, claim power advantage, or add BitNet
QK256/I2_S behavior evidence.

The review contract is now landed by #1158, and #1159 added the fail-closed
receipt validator guard for runtime `AUTO` selected-device proof. Issue #1149
remains open for actual runtime/physical `AUTO` measurement evidence.

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
| #1159 receipt validation requires runtime `AUTO` receipts to identify `auto_scope=openvino_runtime_auto` and execution-device visibility or explicit `not_exposed` | The guard preserves the boundary, but does not collect selected-device measurement evidence |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-auto-phase-gpu-npu-auto-20260601.json` | Runtime-layer `AUTO` was requested alongside explicit `GPU.0` and explicit `NPU`; bounded phase prompts passed with `fallback_used=false`, but `AUTO` selected-device visibility is `not_exposed` |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-auto-corpus-v2-gpu-npu-auto-20260601.json` | Runtime-layer `AUTO`, explicit `GPU.0`, and explicit `NPU` ran all corpus-v2 cases with no failed answer gates and direct generated-token IDs; `AUTO` selected-device proof remains false |
| `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-auto-phase-validation-20260601.json` and `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-auto-corpus-v2-validation-20260601.json` | Both diagnostic packages validate under `lunar_lake_openvino_route_boundary` |
| OpenVINO `AUTO` rejects the safe `EXECUTION_DEVICES` property probe in the 2026-06-01 diagnostic package | AUTO remains diagnostic even when answer gates pass |

## 2026-06-01 Runtime AUTO Diagnostic Package

The current 258V diagnostic package deliberately requests OpenVINO runtime
`AUTO`, not CLI route-selector `auto`, and compares it with explicit `GPU.0`
and explicit `NPU` on the same exported Qwen2.5 INT4_SYM model.

The phase receipt records:

- `artifact_kind=intel_258v_dense_slm_openvino_phase_runner`;
- OpenVINO `2026.2.0-21903-52ddc073857-releases/2026/2`;
- OpenVINO GenAI `2026.2.0.0-3121-adf73e80e66`;
- available devices `CPU`, `GPU`, and `NPU`;
- explicit GPU, explicit NPU, and runtime `AUTO` bounded prompt execution;
- passing answer gates, `fallback_used=false`, and direct generated-token IDs;
- `auto_scope=openvino_runtime_auto` for the `AUTO` device row;
- `selected_backend=openvino-auto`;
- `selected_device_visibility_status=not_exposed`;
- `execution_devices_status=not_exposed`;
- `selected_device_proof=false`;
- `openvino_runtime_auto_selected_device_proof=false`.

The corpus-v2 receipt records the same runtime `AUTO` boundary across all 14
bounded corpus-v2 cases. The explicit GPU, explicit NPU, and runtime `AUTO`
rows all pass the bounded answer gates with `fallback_used=false` and direct
generated-token IDs available from `openvino_genai_encoded_results_tokens`.

The safe selected-device property probe against `AUTO` reports that the
OpenVINO AUTO plugin does not support `EXECUTION_DEVICES`. That is a useful
negative finding: the current receipt source can prove that runtime `AUTO` was
requested and that answer gates passed, but it still cannot prove which device
OpenVINO selected internally.

Treat this package as a diagnostic closeout for the currently exposed
runtime-layer `AUTO` surface, not as selected-device proof. It does not support
GPU promotion, NPU promotion, `low_power`, speedup, power advantage, native
accelerator execution, or BitNet QK256/I2_S claims.

## Official API Boundary

OpenVINO documents selected-device visibility for `AUTO` through the compiled
model property `EXECUTION_DEVICES`, queried from a compiled model after
`core.compile_model(..., "AUTO")`. The current Lunar Lake GenAI receipts do
not own that object directly. They construct `openvino_genai.LLMPipeline`, and
the current Python `LLMPipeline` surface does not expose a compiled model or a
generic `get_property` accessor.

That means the 2026-06-01 `AUTO` diagnostics should be read carefully:

- the failed `EXECUTION_DEVICES` probe is a plugin/core-level diagnostic from
  the receipt source, not proof that a GenAI-internal compiled model was queried
  through the official compiled-model property path;
- the official OpenVINO property path remains a promising research direction,
  but it needs a receipt source that can access the GenAI pipeline's compiled
  model, an equivalent lower-level OpenVINO model run for the same tuple, a
  supported plugin log, or a future GenAI API;
- until one of those paths records actual execution devices for the governed
  Qwen export/profile/cache tuple, runtime `AUTO` remains diagnostic only.

## Current Command Surface

The current repo can validate and commit runtime `AUTO` diagnostic receipts,
but the current OpenVINO GenAI receipt source still does not expose actual
selected-device identity for the governed Qwen export/profile/cache tuple.

- `crates/bitnet-receipts-core/src/lib.rs` accepts
  `auto_scope=openvino_runtime_auto` only when the receipt records
  `execution_devices`, an equivalent selected-device evidence field, or
  `selected_device_visibility_status=not_exposed`.
- The same validator keeps `not_exposed` diagnostic: a receipt that cannot see
  execution devices must not claim selected-device proof, promotion,
  acceleration, power, or `low_power` evidence.
- `scripts/openvino_genai_phase_receipt.py` and
  `scripts/openvino_genai_corpus_v2.py` treat `--devices AUTO` as runtime-layer
  OpenVINO `AUTO` diagnostics by recording `auto_scope=openvino_runtime_auto`,
  requested runtime `AUTO`, explicit `not_exposed` selected-device visibility,
  false selected-device proof booleans, and the `EXECUTION_DEVICES` property
  probe result when available. This makes future `AUTO` script receipts
  validator-compatible, but still does not collect physical selected-device
  evidence or close #1149 by itself.
- `crates/bitnet-cli/src/commands/lunar_lake.rs` currently indexes CLI
  `--device auto` operator-ask receipts and explicit OpenVINO GPU/NPU route
  evidence. That is route-selector evidence, not OpenVINO runtime-layer `AUTO`
  selected-device proof.
- Current operator ask routing maps explicit OpenVINO routes to `GPU.0` or
  `NPU`; it does not deliberately request OpenVINO GenAI `AUTO` for the Qwen
  export/profile/cache tuple and record which device OpenVINO executed.
- Generic device-smoke and Intel NPU probe surfaces remain visibility checks.
  They are not runtime `AUTO` selected-device evidence for the governed Qwen
  export, prompt, generation config, answer gate, and cache settings.

After the 2026-06-01 diagnostic package, another #1149 PR should only proceed
if it identifies a different OpenVINO API, a way to query the GenAI
pipeline's compiled model, a lower-level equivalent run that preserves the same
tuple, a runtime property, a supported plugin log, or another receipt source
that can expose actual selected-device identity for the same tuple. A generic
schema, validator, route-policy, benchmark, or repeat diagnostic PR is not the
next useful step while selected-device visibility remains `not_exposed`.

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

The review and validator guard are already landed by #1158/#1159. The receipt
sources now have a diagnostic `--devices AUTO` shape that records
`not_exposed` instead of silently implying selected-device proof. The
2026-06-01 diagnostic package exercises that shape on the 258V host and
preserves the fail-closed result.

The next useful PR is not another generic runtime `AUTO` rerun. It should be
opened only if a concrete OpenVINO selected-device source is found. The most
promising source to investigate is the official compiled-model
`EXECUTION_DEVICES` property path, but the PR must also show how that path maps
back to the GenAI `LLMPipeline` tuple or explicitly document that it cannot.
That PR should keep the same Qwen export/profile/cache tuple and replace the
current `not_exposed` visibility with actual selected-device evidence, or prove
that a newly inspected source is also unavailable.

Do not open another generic schema or validator PR unless the measurement run
exposes a missing or ambiguous field that #1159 cannot represent. Do not
combine the measurement with route promotion, new inference surface work,
low-power promotion, benchmark matrices, generated-dashboard churn, or
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
