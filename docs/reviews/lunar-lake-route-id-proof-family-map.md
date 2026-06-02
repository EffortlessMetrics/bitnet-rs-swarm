# Lunar Lake Route ID Proof Family Map

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md), [BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](../specs/BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1135](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1135), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1108](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1108), [#1178](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1178), [#1263](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1263)
Linked PRs: [#1137](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1137)
Support-tier impact: no promotion; review-only route identity map
Policy impact: no policy exception

## Question

How should Lunar Lake map current campaign-local route IDs to canonical route
IDs and proof families before future validators, receipt explainers, or route
promotion reviews rely on route identity?

This review answered #1135 and landed in #1137. It does not rename committed
receipts, run inference, change route policy, promote or revoke a profile,
refresh benchmarks, mutate generated dashboards, claim a speedup, claim a power
advantage, prove native accelerator kernels, or prove BitNet QK256/I2_S
behavior.

## Recommendation

Keep the current campaign route IDs in existing ledgers and receipts. When a
new receipt, validator, or explanation surface needs promotion-grade route
identity, add canonical fields alongside the existing route ID:

```json
{
  "route_id": "dense_slm_openvino_gpu_candidate",
  "canonical_route_id": "openvino_dense_slm_gpu_arc140v",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "route_id_source": "intel-258v campaign ledger",
  "proof_family_source": "BITNET-SPEC-OPENVINO-ROUTE-CONTRACT"
}
```

For non-OpenVINO rows, `canonical_route_id` should remain `null` unless a
narrower non-OpenVINO route contract defines a canonical route ID. Do not map
Rust GGUF CPU evidence to `openvino_dense_slm_cpu`, and do not map dense SLM
evidence to `bitnet_cpu_reference`.

## Current Evidence Snapshot

The current committed route-promotion ledger records these profile routes:

| Profile | Campaign route ID | Current status |
| --- | --- | --- |
| `regression_tiny` | `dense_slm_default_cpu` | promoted CPU regression baseline |
| `structured` | `dense_slm_default_cpu` | promoted CPU structured-output baseline |
| `ask_short` | `dense_slm_openvino_gpu_candidate` | profile-promoted OpenVINO GPU route |
| `ask_normal` | `dense_slm_openvino_gpu_candidate` | profile-promoted OpenVINO GPU route |
| `prefill_heavy` | `dense_slm_openvino_gpu_candidate` | profile-promoted, phase split still review-watch |
| `decode_heavy` | `dense_slm_openvino_gpu_candidate` | profile-promoted, phase split still review-watch |
| `warm_resident` | `dense_slm_openvino_npu_candidate` | profile-promoted OpenVINO NPU resident route |
| `bitnet_strict_reference` | `bitnet_reference_cpu` | promoted BitNet CPU reference route; semantic freshness owned by #1178 |
| `low_power` | none | blocked by POWER-006 battery-mode evidence |

The current CPU runtime comparison also records
`dense_slm_openvino_cpu_candidate` as an OpenVINO CPU candidate/control path.
It is useful route context, but it is not the default CPU route and not a
matched-format speedup proof against Rust GGUF CPU.

## Mapping Table

| Campaign route or lane | Canonical route ID | Proof family | Required identity checks | Claim boundary |
| --- | --- | --- | --- | --- |
| `dense_slm_openvino_gpu_candidate` | `openvino_dense_slm_gpu_arc140v` | `openvino_dense_slm_gpu_arc140v` | `selected_backend=openvino-gpu`, `runtime_api=openvino_genai`, Arc 140V / `GPU.0`, fallback false, dense SLM model/export identity | OpenVINO GenAI dense SLM only; no native OpenCL, NPU, BitNet, low-power, or broad speedup claim |
| `dense_slm_openvino_npu_candidate` | `openvino_dense_slm_npu` | `openvino_dense_slm_npu` | `selected_backend=openvino-npu`, `runtime_api=openvino_genai`, Intel AI Boost / `NPU`, fallback false, cold/cache/warm/resident mode explicit | Warm/resident dense SLM only unless later cold or low-power evidence lands; no native NPU kernel or BitNet claim |
| `dense_slm_openvino_cpu_candidate` | `openvino_dense_slm_cpu` | `openvino_dense_slm_cpu` | `selected_backend=openvino-cpu`, `runtime_api=openvino_genai`, 258V CPU resolved device, fallback false, OpenVINO IR model/export identity | OpenVINO CPU dense SLM candidate/control only; not Rust GGUF CPU and not a matched-format speedup claim |
| `dense_slm_default_cpu` | `null` until a non-OpenVINO route contract defines one | `dense_slm_gguf_cpu_reference` as the current receipt provenance/evidence family | `selected_backend=cpu-rust`, `runtime_api=cpu`, GGUF Q8_0 model identity, fallback false, CPU answer/phase evidence | Rust GGUF dense SLM baseline only; not OpenVINO CPU, not accelerator proof, and not BitNet proof |
| `bitnet_reference_cpu` | `null` until a BitNet route contract defines one | `bitnet_cpu_reference` | `selected_backend=intel-258v-cpu-avx2` or accepted CPU reference backend, `runtime_api=cpu`, corrected CPU reference bundle, fallback false | BitNet strict-reference only; not dense SLM usability, OpenVINO, GPU, NPU, speedup, or low-power proof |
| `openvino-auto` or campaign `auto` selection | `null` until selected execution devices are recorded | diagnostic only | Actual execution device for every required phase, fallback false, no unresolved AUTO/HETERO ambiguity | Diagnostic only without selected-device proof |
| blocked `low_power` profile | none | none | Battery-mode route samples, telemetry, energy proxy, fallback false, and accepted thermal context if exposed | No low-power promotion or power-advantage claim until POWER-006 evidence passes |
| future server exact profile | `openvino_server_exact_profile` | `openvino_model_server` | Server endpoint/profile identity plus underlying route identity, fallback false, exposure, timing, and concurrency/streaming boundaries as applicable | Exact endpoint/profile only; no broad server readiness or BitNet claim by inheritance |
| future OpenVINO BitNet subgraph | `openvino_bitnet_subgraph_reference` | `openvino_bitnet_subgraph_reference` | Static subgraph definition, OpenVINO runtime device, CPU reference receipt, parity tolerance, fallback false | Static subgraph parity only; no full BitNet inference, QK256 decode, or speedup claim |

`dense_slm_gguf_cpu_reference` is not a new OpenVINO proof family. It names the
existing Rust GGUF CPU receipt provenance so future schemas do not incorrectly
collapse the CPU baseline into `openvino_dense_slm_cpu`.

## Receipt Guidance

Future receipts that need route identity hardening should preserve the existing
campaign-local route ID and add canonical fields:

```json
{
  "route_id": "dense_slm_openvino_npu_candidate",
  "canonical_route_id": "openvino_dense_slm_npu",
  "proof_family": "openvino_dense_slm_npu",
  "model_family": "qwen",
  "requested_backend": "openvino-npu",
  "selected_backend": "openvino-npu",
  "runtime_api": "openvino_genai",
  "runtime_device": "NPU",
  "resolved_device": "Intel(R) AI Boost",
  "fallback_used": false,
  "profile_id": "warm_resident",
  "residency_scope": "same_process_resident",
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false,
  "native_npu_kernel_proof": false
}
```

For the Rust GGUF CPU baseline:

```json
{
  "route_id": "dense_slm_default_cpu",
  "canonical_route_id": null,
  "proof_family": "dense_slm_gguf_cpu_reference",
  "model_family": "qwen",
  "requested_backend": "cpu-rust",
  "selected_backend": "cpu-rust",
  "runtime_api": "cpu",
  "fallback_used": false,
  "bitnet_qk256_proof": false
}
```

For the BitNet CPU reference:

```json
{
  "route_id": "bitnet_reference_cpu",
  "canonical_route_id": null,
  "proof_family": "bitnet_cpu_reference",
  "model_family": "bitnet",
  "runtime_api": "cpu",
  "fallback_used": false,
  "dense_slm_proof": false,
  "openvino_proof": false
}
```

Old receipts do not need a rewrite solely to add these fields. New validators
may infer the map from this review only for read-only explanation, but any new
claim-bearing receipt should record the canonical fields directly.

## Fail-Closed Rules

| Conflict | Required decision |
| --- | --- |
| Campaign route ID maps to one proof family but `selected_backend` names another backend | Fail route identity validation |
| OpenVINO GPU route resolves to CPU, NPU, AUTO without selected-device proof, or a non-Arc device | No `openvino_dense_slm_gpu_arc140v` proof |
| OpenVINO NPU route resolves to CPU, GPU, AUTO without selected-device proof, or an unknown device | No `openvino_dense_slm_npu` proof |
| OpenVINO CPU route is used to summarize Rust GGUF CPU evidence | Fail mapping or keep the evidence diagnostic |
| Rust GGUF CPU route is summarized as OpenVINO CPU proof | Fail mapping |
| Dense SLM proof family is used as BitNet QK256/I2_S proof | Reject the claim boundary |
| BitNet CPU reference evidence is used as dense SLM route quality | Reject the claim boundary |
| Shared BitNet semantic fix lands after current CPU reference evidence | Rerun affected BitNet CPU reference evidence through #1178 before changing BitNet route claims |
| Diagnostic-only shared BitNet-adjacent instrumentation touches route/proof surfaces | Record the reviewed non-trigger through #1263 before treating it as non-stale; ambiguous scope remains blocked |
| `openvino-auto` lacks per-phase execution-device evidence | Diagnostic only |
| `low_power` has no battery-mode route samples or energy proxy | Keep profile blocked before promotion |
| `fallback_used=true` in any selected route evidence | Block selected-route promotion or proof-family claim |
| `canonical_route_id` and `proof_family` disagree with the route contract | Fail validation until the receipt is corrected |

Unknown facts must stay unknown or `null`; missing route identity must not be
treated as proof of strict execution.

## Route Consequences

### GPU Profiles

`dense_slm_openvino_gpu_candidate` maps to
`openvino_dense_slm_gpu_arc140v` only for OpenVINO GenAI Arc 140V dense SLM
evidence with fallback false. This mapping does not strengthen the current
profile promotions and does not prove native OpenCL.

### NPU Resident Profile

`dense_slm_openvino_npu_candidate` maps to `openvino_dense_slm_npu` only under
the NPU dense SLM route identity. Current policy remains resident-scoped for
`warm_resident`; this mapping does not promote cold one-off, `ask_short`,
`ask_normal`, `low_power`, native NPU kernels, or BitNet QK256/I2_S behavior.

### OpenVINO CPU Candidate

`dense_slm_openvino_cpu_candidate` maps to `openvino_dense_slm_cpu` when the
receipt is OpenVINO GenAI CPU evidence. It remains a separate candidate/control
path and does not replace `dense_slm_default_cpu`.

### Rust GGUF CPU Baseline

`dense_slm_default_cpu` stays in the Rust GGUF dense SLM lane. It must not be
mapped to `openvino_dense_slm_cpu` just because both execute on the CPU.

### BitNet CPU Reference

`bitnet_reference_cpu` maps to the `bitnet_cpu_reference` proof family. Dense
SLM CPU/GPU/NPU evidence must never close BitNet strict-reference gates.
Issue #1178 owns semantic-intake freshness if future shared BitNet semantic
changes or receipt/validator gaps require a targeted CPU reference rerun. The
diagnostic-only non-trigger classification in #1263 covers merged
BitNet-adjacent instrumentation touches that are reviewed as not changing Lunar
Lake BitNet CPU reference semantics.

### Low Power

`low_power` still has no promoted route. Candidate route mappings can describe
what a future receipt would mean, but they cannot supply the missing battery
telemetry, route samples, or energy proxy.

## Next Smallest PR

No route-policy PR is required from this review alone.

If a later validator or receipt explainer needs this map in code, the next
small implementation PR should add a route-ID helper or schema check that:

- records `canonical_route_id` and `proof_family` alongside campaign route IDs;
- keeps old receipts readable without rewriting them;
- rejects backend/device/runtime/proof-family conflicts;
- preserves `null` for non-OpenVINO canonical route IDs until a narrower route
  contract defines them;
- preserves the dense SLM, OpenVINO, BitNet, native OpenCL, native NPU, server,
  and low-power claim boundaries.

Do not combine that helper with route-policy mutation, new inference, benchmark
matrices, POWER-006 battery evidence, CPU optimization, or generated-dashboard
churn.

If the question is stale BitNet CPU reference evidence rather than route-ID
mapping, route it to #1178. The next change should be a targeted freshness
rerun, guard, or receipt-note update only after that issue names the evidence
gap.

## Acceptance For #1135

Issue #1135 closed when this review landed because it:

- maps current Lunar Lake campaign-local route IDs to canonical OpenVINO route
  IDs and proof families where applicable;
- explicitly handles CPU dense SLM, OpenVINO GPU Arc 140V, OpenVINO NPU
  resident, OpenVINO CPU candidate, BitNet CPU reference, blocked `low_power`,
  server exact-profile, and BitNet subgraph cases;
- gives receipt guidance for `canonical_route_id` and `proof_family`;
- defines fail-closed rules for route ID, backend, device, runtime API, and
  proof-family conflicts;
- keeps all promotion, speed, power, native accelerator, and BitNet claims
  unchanged.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- route-policy mutation;
- route promotion or revocation;
- benchmark refresh;
- generated dashboard churn;
- speedup or acceleration claims;
- power-advantage evidence;
- battery-mode evidence;
- measured-temperature evidence;
- native OpenCL proof;
- native NPU kernel proof;
- full BitNet accelerator inference;
- BitNet QK256/I2_S behavior proof.

It only defines how future receipts, validators, and reviews should relate
campaign-local route IDs to canonical route and proof-family fields.
