# Lunar Lake NPU Cold And Cache Evidence Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1119](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1119), [#1371](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1371), [#1139](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1139), [#1143](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1143), [#1120](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1120), [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064), [#1123](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123), [#1244](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149), [#1216](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1216), [#1160](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1160), [#1162](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1162), [#1189](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1189)
Linked PRs: [#1163](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1163), [#1174](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1174), [#1191](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1191), [#1217](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1217), [#1282](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1282), [#1286](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1286)
Support-tier impact: no promotion; review-only cold/cache evidence contract
Policy impact: no policy exception
Cache-truth child refresh: 2026-06-03

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

Issue #1371 is the current child issue for the direct cache-hit truth boundary.
It is the right owner for future work that distinguishes `runtime_metric`,
`runtime_log`, `file_reuse`, `timing_derived`, and `not_exposed` evidence
sources, ties cache status to the exact model/export/device/cache tuple, and
fails closed when timing/file-derived cache evidence is labeled as direct
runtime cache-hit truth. It is not a route-policy, benchmark, low-power, or
promotion queue.

Issue #1139 closed the narrow phase-timing schema gap. Issue #1143 is also
closed after #1145 aligned the committed NPU diagnosis and route-promotion
receipts for direct validation. #1154 is closed because existing validator
coverage already rejects treating timing-derived cache diagnostics as direct
runtime cache-hit truth without direct runtime evidence fields. #1160 is closed
by #1174 after collecting the current cache-rerun evidence package under those
boundaries. The 2026-06-01 rerun provides a current diagnostic cache snapshot
package, but it still uses timing-derived cache classification and does not
expose direct runtime cache-hit truth. #1162 is closed by #1163 after the
warm-resident validator started enforcing the
resident-session acceptance boundary. #1189 is closed by #1191 after the
existing OpenVINO GenAI phase receipt path gained a `host_phase_timing` block
and validator guard. That closes the immediate host phase schema/guard gap, but
does not add physical timing evidence, direct runtime cache-hit truth, or route
promotion evidence. Keep any future follow-up limited to the specific cache,
phase, resident, or validation gap being measured instead of widening this
review into route policy.

PR #1282 added the current OpenVINO cache source-boundary audit to the NPU
cold-start research note, and #1286 aligned this review with that boundary. The
public source boundary supports cache configuration and provenance fields such
as `CACHE_DIR`, `NPUW_CACHE_DIR`, `EXPORT_BLOB` / `BLOB_PATH`,
supported-property enumeration, and release-note cache/import compatibility
context. It still does not expose a direct GenAI receipt field that proves a
specific `LLMPipeline` run hit cache, so the committed cache receipts remain
timing/file-derived diagnostics.

Generated-token visibility uses the same successor split: #1123 is the closed
historical review issue, and #1244 is the live watch issue for future direct,
retokenized, or text-only token-visibility schema/checker gaps. Direct NPU
generated-token IDs keep cache and resident diagnostics honest, but they do not
turn timing-derived cache evidence into direct cache-hit truth or route
promotion evidence.

## Current Evidence Snapshot

| Receipt | Finding | Review consequence |
| --- | --- | --- |
| `lunar-lake-openvino-npu-cache-experiment.json` | Two separate OpenVINO GenAI NPU processes use one cache directory; first construct is 28103.867 ms, second construct is 872.662 ms, ratio is 0.031, improvement is 27231.205 ms | Strong timing-derived cache effect, not direct runtime cache-hit truth |
| `lunar-lake-openvino-npu-cache-experiment.json` | Cache starts empty, then one 154693720-byte blob exists after both process runs | File evidence supports cache reuse diagnosis |
| `lunar-lake-openvino-npu-cache-experiment.json` | `cache_hit_runtime_metric_available=false` | Promotion-grade cache-hit claims remain blocked |
| `lunar-lake-openvino-npu-cache-experiment.json` | First and second process answer gates pass, fallback is false, direct generated-token IDs are available | Route identity and quality are suitable for diagnostic comparison |
| `lunar-lake-openvino-npu-cache-rerun-20260601.json` | Closed #1160 / #1174 rerun uses explicit NPU on OpenVINO 2026.2.0 / GenAI 2026.2.0.0; first construct is 11163.864 ms, second construct is 944.647 ms, ratio is 0.085, improvement is 10219.218 ms | Committed receipt confirms the cache effect remains material under the new receipt contract |
| `lunar-lake-openvino-npu-cache-rerun-20260601.json` | Cache starts empty, then one stable 158052779-byte blob remains after both process runs | File evidence supports cache reuse diagnosis for the current model/export/runtime/device tuple |
| `lunar-lake-openvino-npu-cache-rerun-20260601.json` | `cache_evidence_source=timing_derived`; `direct_runtime_cache_hit_status.available=false`; command provenance uses repo-relative replay commands | Cache-hit truth remains diagnostic, while the committed receipt avoids stale local checkout paths |
| `lunar-lake-openvino-npu-cache-rerun-20260601.json` | First and second process answer gates pass, fallback is false, direct generated-token IDs are available, and profile applicability is marked non-promotion smoke evidence | Suitable closed #1160 diagnostic package; still not route-promotion evidence |
| `lunar-lake-openvino-npu-cache-probe-20260601T1323Z.json` | Current-main explicit-NPU cache probe uses one cache directory; first construct is 10838.986 ms, second construct is 892.883 ms, ratio is 0.082, improvement is 9946.102 ms | Confirms the material cache effect is still visible on the current script/runtime path, but remains timing/file-derived diagnosis |
| `lunar-lake-openvino-npu-cache-probe-20260601T1323Z.json` | Answer gates pass in both processes, `fallback_used=false`, generated-token IDs come from `openvino_genai_encoded_results_tokens`, and `direct_runtime_cache_hit_status.available=false` | Valid diagnostic evidence under `lunar_lake_openvino_route_boundary`; still not direct runtime cache-hit truth or route-promotion evidence |
| `lunar-lake-openvino-npu-resident-session-20260601T1325Z.json` | Paired same-cache resident diagnostic constructs the pipeline in 864.753 ms and completes 10/10 warm asks with fallback false, no answer/token/fallback/route drift, 314.739 ms mean generation wall, and 161.381 ms mean OpenVINO TTFT | Useful current-main resident context, but not a replacement for the existing 30/30 warm-resident acceptance receipt and not a new promotion claim |
| `lunar-lake-openvino-npu-cold-start-diagnosis.json` | `cold_load_decomposition` records first-process cache miss, second-process cache reuse, timing-derived cache classification, and missing direct cache metrics | Decomposition is review-ready, but still diagnostic |
| `lunar-lake-npu-cold-start.md` | Cold startup remains dominated by pipeline construction, compile/load, transfer, cache, tokenizer/setup, and first-ask costs | Do not promote cold one-off NPU from cache evidence alone |
| `lunar-lake-npu-cold-start.md` and this review | #1282/#1286 record the OpenVINO cache source boundary: current docs justify cache configuration/provenance receipt fields, but do not expose direct GenAI runtime cache-hit truth for these receipts | Future cache work needs a documented runtime cache-hit property, parseable runtime log, or precise cache/blob/compiler/compatibility provenance gap |

The useful current claim is:

```text
OpenVINO NPU cache reuse materially reduces second-process pipeline construction
for this exact model/export/device/cache directory, with fallback false and
passing answer gates.
```

The paired 2026-06-01 resident diagnostic adds only this bounded context:

```text
With the same cache directory already present, a current-main explicit-NPU
resident session can keep one pipeline alive for 10 warm asks with fallback
false, stable answers/tokens/routes, and direct generated-token IDs.
```

It does not replace the existing 30/30 warm-resident acceptance receipt and does
not expand route policy.

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
| Phase split | Pipeline construction, OpenVINO load time, cache lookup or `not_exposed`, tokenizer/setup, prompt render/tokenization, first ask, TTFT, decode/generation, receipt overhead, telemetry timing, and total response |
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

The receipt-only #1160 PR shape has landed in #1174: the dated cache rerun
artifact is committed and the evidence remains diagnostic. No follow-up PR is
needed unless #1371 identifies a newly exposed direct cache metric, parseable
runtime log, explicit unavailable truth-source handling, cache provenance gap,
or stricter missing field that creates a specific cache-truth problem.

The cache-truth guard, direct-validation alignment, and host phase timing guard
have already landed: #1145 aligned current receipts for direct validation,
existing validator coverage closed #1154 by enforcing the
timing-derived-versus-runtime-cache-truth boundary, and #1191 closed #1189 by
adding `host_phase_timing` value/status/source entries plus fail-closed
validation for sentinels, coarse-versus-narrow phase ownership, and
timing-derived cache status. The next small implementation PR, if needed,
should be scoped through the #1119 parent issue or a new precise cache-evidence
issue, not another broad rerun of #1160, repeat of the #1189 schema guard, or
repeat of the #1282/#1286 source-boundary audit. It should preserve these
constraints when new cache evidence is added or when a newly exposed field
creates a more specific gap:

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

Do not treat the #1191 receipt surface as a physical evidence run. Future
route-policy work still needs fresh receipts with measured or explicitly
unavailable phase fields for the target model/export/runtime/device/profile.

`AUTO` selected-device work is intentionally separate: #1149 owns receipts or
validators that prove what `AUTO` actually executed, with the current contract
recorded in
[lunar-lake-openvino-auto-selected-device.md](lunar-lake-openvino-auto-selected-device.md).
PR #1217 validator-admits the current OpenVINO GenAI debug-log evidence
artifact for one stateful LLM model block, but it remains selected-device
review evidence only. Do not use cache evidence or `AUTO` selection as
route-policy evidence while selected-device, profile, fallback, timing, and
power gates are missing or ambiguous.

Do not combine that guard with NPU route-policy mutation, new inference
surfaces, low-power promotion, benchmark matrices, or generated-dashboard churn.

## Claim Boundary

This review does not add:

- route-policy inference beyond the cited #1160 diagnostic cache rerun receipt;
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
