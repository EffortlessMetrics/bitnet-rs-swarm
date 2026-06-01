# Lunar Lake NPU Cold And Cache Evidence Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119), [#1139](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1139), [#1143](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1143), [#1120](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1120), [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064), [#1123](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149), [#1160](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1160), [#1162](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1162)
Linked PRs: [#1163](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1163)
Support-tier impact: no promotion; review-only cold/cache evidence contract
Policy impact: no policy exception

## Recommendation

Keep OpenVINO NPU cold and cache evidence diagnostic until a later route review
accepts direct cache telemetry or a stricter proxy policy.

The current cache receipts are useful: they separate first-process cache-miss
construction from second-process cache reuse, keep explicit NPU routing, pass
answer gates, record direct generated-token IDs, and mark cache-hit runtime
metrics as unavailable. They are not enough to promote NPU for cold one-off
asks, `ask_short`, `ask_normal`, `low_power`, native NPU kernels, or BitNet
QK256/I2_S behavior.

This review supports #1119 by naming the evidence gate that must exist before
any NPU cold/cache route-policy change is considered. It does not run
inference, refresh receipts, or change route policy.

Issue #1139 closed the narrow phase-timing schema gap. Issue #1143 is also
closed after #1145 aligned the committed NPU diagnosis and route-promotion
receipts for direct validation. #1154 is closed because existing validator
coverage already rejects treating timing-derived cache diagnostics as direct
runtime cache-hit truth without direct runtime evidence fields. #1160 is the
current cache-rerun evidence issue for collecting a fresh cache snapshot package
under those boundaries. #1162 is closed by #1163 after the warm-resident
validator started enforcing the resident-session acceptance boundary. Keep any
future follow-up limited to the specific cache, phase, resident, or validation
gap being measured instead of widening this review into route policy.

## Current Evidence Snapshot

| Receipt | Finding | Review consequence |
| --- | --- | --- |
| `lunar-lake-openvino-npu-cache-experiment.json` | Two separate OpenVINO GenAI NPU processes use one cache directory; first construct is 28103.867 ms, second construct is 872.662 ms, ratio is 0.031, improvement is 27231.205 ms | Strong timing-derived cache effect, not direct runtime cache-hit truth |
| `lunar-lake-openvino-npu-cache-experiment.json` | Cache starts empty, then one 154693720-byte blob exists after both process runs | File evidence supports cache reuse diagnosis |
| `lunar-lake-openvino-npu-cache-experiment.json` | `cache_hit_runtime_metric_available=false` | Promotion-grade cache-hit claims remain blocked |
| `lunar-lake-openvino-npu-cache-experiment.json` | First and second process answer gates pass, fallback is false, direct generated-token IDs are available | Route identity and quality are suitable for diagnostic comparison |
| `lunar-lake-openvino-npu-cold-start-diagnosis.json` | `cold_load_decomposition` records first-process cache miss, second-process cache reuse, timing-derived cache classification, and missing direct cache metrics | Decomposition is review-ready, but still diagnostic |
| `lunar-lake-npu-cold-start.md` | Cold startup remains dominated by pipeline construction, compile/load, transfer, cache, tokenizer/setup, and first-ask costs | Do not promote cold one-off NPU from cache evidence alone |

The useful current claim is:

```text
OpenVINO NPU cache reuse materially reduces second-process pipeline construction
for this exact model/export/device/cache directory, with fallback false and
passing answer gates.
```

The unsafe claim is:

```text
OpenVINO exposed a promotion-grade cache hit, so NPU is ready for cold/default
or low_power routing.
```

## Required Cold/Cache Evidence Gate

Before any NPU cold/cache route-policy PR, one package must record:

| Gate | Requirement |
| --- | --- |
| Route identity | Requested backend, selected backend, runtime API, runtime device, resolved device, route ID, proof family, and fallback status |
| Model/export identity | Source model, OpenVINO IR XML/BIN hashes, tokenizer hashes, OpenVINO version, driver/device context, and cache key basis |
| Cache setup | Cache directory, enabled/writable status, permissions, pre-run snapshot, after-first-process snapshot, after-second-process snapshot |
| Cache-hit evidence | Runtime metric, runtime log, stable file reuse, or explicit `not_exposed`; timing-derived classification must say so |
| Phase split | Pipeline construction, OpenVINO load time, cache lookup or `not_exposed`, tokenizer/setup or `not_exposed`, first ask, TTFT, decode/generation, and total response |
| Process split | First process cache-miss and second process cache-reuse runs recorded separately |
| Quality | Answer gate and generated-token visibility for both processes |
| Drift and fallback | Fallback false, selected device stable, generated-token source explicit, no hidden CPU/GPU fallback |
| Promotion posture | Whether evidence is diagnostic, candidate, conditional, blocked, or promotion-grade |
| Claim boundary | No cold one-off, low-power, speedup, native NPU, or BitNet claim unless separately proven |

Missing direct cache-hit metrics do not make the receipt useless. They make the
cache classification timing-derived and prevent it from becoming runtime truth
or promotion-grade cache evidence by itself.

## Route Decisions

### Cold One-Off NPU

Keep blocked. A second-process cache improvement does not prove the first
fresh interactive process is acceptable, and current evidence still shows a
large first-process construct/load cost.

### Cached Cold Process

Keep candidate/diagnostic. The evidence may support a future cached-process
route only if the product contract defines that the cache is prewarmed,
available, writable, and bound to the exact model/export/runtime/device tuple.

### Warm Resident

Use #1120 and `lunar-lake-npu-warm-resident-acceptance.md` for warm-resident
acceptance. Cache evidence can explain why resident setup is attractive, but it
does not replace the same-process resident loop acceptance gates.

### `low_power`

Keep blocked by #1064. Cache reuse and warm timing are not battery-mode energy
evidence and do not prove a power advantage.

### BitNet

Keep blocked. Dense Qwen OpenVINO NPU cache behavior is not native NPU kernel
proof, full BitNet inference, packed QK256 decode, or BitNet QK256/I2_S parity.

## Fail-Closed Conditions

| Condition | Required decision |
| --- | --- |
| Cache directory is missing, unwritable, or not bound to the model/export/device tuple | No cached-process claim |
| First and second process use different model, prompt, generation config, device, OpenVINO version, or cache directory | Comparison invalid |
| Cache-hit metric is unavailable and timing/file evidence is absent | Diagnostic cache claim blocked |
| Cache-hit metric is unavailable but timing/file evidence exists | Timing-derived diagnosis only |
| Fallback appears in either process | Block NPU route evidence |
| Selected device is missing or not NPU | Block selected-device proof |
| Answer gate fails without accepted diagnosis | Candidate or blocked until rerun/diagnosis |
| Direct generated-token IDs are missing | No token-drift or promotion-grade token claim |
| Cache timing improves but first-process cold path remains large | No cold one-off promotion |
| Battery telemetry or energy proxy is missing | No `low_power` promotion |
| Dense SLM NPU evidence is cited as BitNet proof | Reject claim boundary |

## Next Smallest PR

No route-policy PR is required from this review alone.

The cache-truth guard and direct-validation alignment have already landed: #1145
aligned current receipts for direct validation, and existing validator coverage
closed #1154 by enforcing the timing-derived-versus-runtime-cache-truth
boundary. The next small implementation PR, if needed, should be scoped through
#1160 or the #1119 parent issue. It should preserve these constraints when new
cache evidence is added or when a newly exposed field creates a more specific
gap:

- requires cache classification source to be `runtime_metric`, `runtime_log`,
  `file_reuse`, `timing_derived`, or `not_exposed`;
- fails closed when a receipt treats timing-derived cache classification as a
  direct runtime cache hit;
- requires first-process and second-process phase fields to remain separate;
- preserves host setup, tokenizer/template, `LLMPipeline`, compile/load/cache,
  first-ask, warm-ask, and receipt-overhead ownership when those timers are
  present;
- preserves fallback, answer-gate, generated-token, route identity, and claim
  boundary fields.

`AUTO` selected-device work is intentionally separate: #1149 owns receipts or
validators that prove what `AUTO` actually executed, with the current contract
recorded in
[lunar-lake-openvino-auto-selected-device.md](lunar-lake-openvino-auto-selected-device.md).
Do not use cache evidence or `AUTO` selection as route-policy evidence while
selected-device proof is missing or ambiguous.

Do not combine that guard with NPU route-policy mutation, new inference
surfaces, low-power promotion, benchmark matrices, or generated-dashboard churn.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- route-policy mutation;
- NPU cold one-off promotion;
- default NPU routing;
- `ask_short` or `ask_normal` NPU promotion;
- `low_power` promotion;
- speedup or power-advantage claims;
- direct cache-hit runtime truth;
- native NPU kernel proof;
- BitNet QK256/I2_S behavior proof.

It only defines the cold/cache evidence gate required before NPU route-policy
work can safely use cache evidence.
