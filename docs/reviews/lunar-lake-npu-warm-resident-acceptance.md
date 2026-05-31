# Lunar Lake NPU Warm Resident Acceptance Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1120](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1120), [#1139](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1139), [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119), [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064), [#1123](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124)
Linked PRs: n/a
Support-tier impact: no promotion; review-only warm resident acceptance rule
Policy impact: no policy exception

## Recommendation

A same-process OpenVINO NPU resident profile may be accepted independently from
cold one-off NPU promotion, but only as `warm_resident` evidence.

The acceptance boundary is narrow: the route may rely on one already
constructed NPU `LLMPipeline` that stays resident across a bounded warm loop.
Pipeline construction, the first ask after construction, cache context, memory
growth, answer drift, generated-token drift, route drift, device drift, and
fallback status must remain visible. The same evidence must not promote NPU for
`ask_short`, `ask_normal`, `low_power`, cold one-off asks, native NPU kernels,
or BitNet QK256/I2_S behavior.

This review does not change route policy. It defines when existing or future
resident evidence is sufficient for a route-facing `warm_resident` acceptance
review and when that evidence must fail closed.

Issue #1120 is closed as answered by this acceptance rule. Issue #1139 closed
the phase-timing schema work used when future receipts need tighter host setup,
tokenizer/template, pipeline, compile/load/cache, first-ask, warm-ask, or
receipt-overhead ownership.

## Current Evidence Snapshot

`lunar-lake-openvino-npu-resident-session.json` records:

| Field | Current value | Acceptance relevance |
| --- | --- | --- |
| Requested backend | `openvino-npu` | Explicit NPU route, not AUTO |
| Selected backend | `openvino-npu` | No CPU/GPU selected-device drift |
| Runtime device | `NPU` | Matches NPU proof family |
| Resolved device | `Intel(R) AI Boost` | Lunar Lake NPU identity present |
| Fallback used | `false` | Required for any route evidence |
| Pipeline construct | 1470.051 ms | Must be recorded outside warm timing |
| Cold first ask | 1 passed, 0 failed | Separated from warm loop |
| Cold first ask generation wall | 1004.710 ms | Not a cold one-off route claim |
| Cold first ask OpenVINO TTFT | 294.261 ms | First ask is not merged into warm p95 |
| Warm repeats | 30 requested, 30 passed, 0 failed | Meets promotion-grade warm loop count |
| Warm generation wall | min 571.886 ms, mean 764.148 ms, p95 1171.061 ms, max 1244.647 ms | Warm timing distribution present |
| Warm OpenVINO TTFT | min 200.292 ms, mean 220.584 ms, p95 257.297 ms, max 309.611 ms | Warm first-token distribution present |
| Warm throughput | min 14.573 t/s, mean 18.511 t/s, p95 19.844 t/s, max 19.927 t/s | Warm throughput distribution present |
| Answer drift | `false` | Stability gate satisfied |
| Generated-token drift | `false` | Token stability gate satisfied |
| Fallback drift | `false` | Strict route gate satisfied |
| Route drift | `false` | Resident session stayed on the intended route |
| Direct generated-token IDs | `true` | Promotion-grade token visibility for this route/profile |
| Resident memory growth | 692248576 bytes | Memory growth is recorded, not hidden |

This evidence is enough to support a warm-resident route review. It is not
enough to support cold one-off NPU routing because the route requires a
constructed resident pipeline before the warm loop begins.

## Acceptance Rule

NPU `warm_resident` acceptance requires one review or receipt package with all
of these fields:

| Gate | Required evidence |
| --- | --- |
| Profile scope | `profile=warm_resident`; no alias or inherited `ask_short` / `ask_normal` profile |
| Route identity | Requested and selected backend are `openvino-npu`, runtime device is `NPU`, resolved device names Intel AI Boost or the accepted exact NPU identity |
| Residency | `resident_session_ready=true` and `same_process_pipeline_reused=true` |
| Pipeline separation | Pipeline construction, cache context, cold first ask, and warm repeats are separate fields |
| Warm sample count | At least 30 same-process warm asks for promotion-grade warm resident evidence |
| Quality | Every warm ask passes its answer gate, or failures are classified and the route remains candidate-only |
| Drift checks | Answer, generated-token, fallback, route, and device drift are all false |
| Token visibility | Direct pipeline-generated token IDs are available, or the review explicitly lowers the claim and blocks token-drift proof |
| Timing | Warm generation wall time, OpenVINO TTFT, and throughput include min, mean, p95, and max |
| Memory | Samples before construct, after construct, after first ask, after warm loop, plus resident growth bytes |
| Telemetry | Power and thermal context are recorded or explicitly marked unavailable; unavailable telemetry blocks `low_power` only |
| Claim boundary | Receipt says resident evidence does not remove cold-start, low-power, native NPU, or BitNet blockers |

The package may keep `warm_resident` promoted or promotion-eligible only under
that exact profile. Missing required fields force the NPU route to
candidate-only for `warm_resident` until the evidence is refreshed.

## Fail-Closed Conditions

| Condition | Required decision |
| --- | --- |
| Pipeline construction is absent or merged into warm timing | Block warm-resident acceptance |
| Cold first ask is not separated from warm repeats | Block cold-start caveat and keep candidate-only |
| Fewer than 30 warm asks in a promotion-grade run | Candidate-only unless a policy explicitly accepts the smaller sample |
| `fallback_used=true` or fallback drift appears | Block NPU route evidence |
| Requested NPU resolves to CPU, GPU, AUTO without execution-device proof, or an unknown device | Block selected-device proof |
| Any answer-gate failure is unclassified | Candidate or blocked until diagnosis explains it |
| Direct generated-token IDs are missing | No token-drift claim; promotion-grade route review must block unless the claim is explicitly lowered |
| Generated-token drift appears | Block warm-resident acceptance until rerun or diagnosed |
| Route or device drift appears | Block resident-session stability claim |
| Memory growth is unrecorded | Candidate-only because residency stability is not bounded |
| Memory growth contradicts the intended long-lived workflow | Reopen review before route selection relies on residency |
| Power source is AC or battery telemetry is missing | No `low_power` promotion or power-advantage claim |
| Dense SLM NPU evidence is cited as BitNet proof | Reject the claim boundary |

## Route Consequences

### `warm_resident`

Keep NPU eligible only when the user workflow is explicitly resident and the
receipt proves the same process keeps one NPU pipeline alive. The route reason
must include the cold-start caveat and must not describe this as cold default
NPU readiness.

### `ask_short` And `ask_normal`

Keep NPU blocked for one-off `ask_short` and `ask_normal`. Warm loop timing can
explain why a resident route is attractive, but it cannot ignore pipeline
construction for ordinary one-off asks.

### Cold One-Off And Cache

Keep cold one-off NPU blocked until #1119 defines accepted cache and phase
evidence. Timing-derived cache classification is diagnostic unless a later
review accepts it as promotion-grade.

### `low_power`

Keep `low_power` blocked by #1064. Resident latency and stability evidence are
not battery-mode energy evidence.

### BitNet

Dense Qwen OpenVINO NPU evidence stays in the
`openvino_dense_slm_npu` proof family. It does not prove native NPU kernels,
full BitNet inference, packed QK256 decode, or BitNet QK256/I2_S parity.

## Next Smallest PR

No route-policy PR is required from this review alone.

The next small implementation PR, if needed, should be a validation guard that
blocks a `warm_resident` NPU route review unless the package includes:

- pipeline construction and cold first ask separated from warm repeats;
- at least 30 warm repeats for promotion-grade acceptance;
- false fallback, answer, generated-token, route, and device drift;
- direct token visibility or an explicit lower claim boundary;
- memory samples and resident growth bytes;
- a cold-start caveat and no low-power or BitNet claim leakage.

Use the #1139/#1141 phase-timing contract if the guard needs phase-timer
ownership beyond the resident-session fields above. Do not combine that guard
with NPU route-policy mutation, new inference surfaces, benchmark matrices,
low-power promotion, or
generated-dashboard churn.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- route-policy mutation;
- NPU cold one-off promotion;
- default NPU routing;
- `ask_short` or `ask_normal` NPU promotion;
- `low_power` promotion;
- speedup or power-advantage claims;
- cache-hit runtime truth claims;
- native NPU kernel proof;
- BitNet QK256/I2_S behavior proof.

It only defines the route-facing acceptance rule for same-process OpenVINO NPU
warm-resident evidence.
