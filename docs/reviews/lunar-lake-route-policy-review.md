# Lunar Lake Route Policy Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149)
Linked PRs: [#1137](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1137), [#1138](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1138), [#1141](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1141), [#1156](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1156), [#1158](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1158), [#1159](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1159), [#1163](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1163), [#1165](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1165)
Support-tier impact: no promotion; review-only route policy guard
Policy impact: no policy exception

## Recommendation

Keep the current Lunar Lake route policy in force, but treat every promoted
route as profile-scoped and receipt-invalidatable.

Do not open a route-policy mutation PR from this review alone. The current
smallest useful follow-up is to keep the profile reviews and evidence contracts
current:

- #1064 for battery-mode `low_power` evidence;
- #1119 for NPU cold/cache decomposition;
- #1120 for NPU warm-resident acceptance, now closed as defined in
  [lunar-lake-npu-warm-resident-acceptance.md](lunar-lake-npu-warm-resident-acceptance.md),
  with the #1162 follow-up guard landed by #1163;
- #1139 for the NPU phase-timing schema contract, now closed by #1141, before
  host setup, tokenizer/template, pipeline, compile/load/cache, first-ask,
  warm-ask, or receipt-overhead timers are used for route decisions;
- #1121 for OpenVINO GPU `ask_short` / `ask_normal` review, now closed with
  the keep decision recorded in
  [lunar-lake-openvino-gpu-promotion-review.md](lunar-lake-openvino-gpu-promotion-review.md);
- #1122 and #1132 for the CPU route posture decision, with #1069 and #1071
  as live measurement follow-ups;
- #1123 for generated-token visibility rules, now closed by #1138 and defined
  in [lunar-lake-openvino-token-visibility.md](lunar-lake-openvino-token-visibility.md);
- #1135 for route ID and canonical proof-family mapping, now closed by #1137
  in [lunar-lake-route-id-proof-family-map.md](lunar-lake-route-id-proof-family-map.md);
- #1149 for `AUTO` selected-device evidence before any `AUTO` receipt is used
  as route-policy evidence, with the current evidence contract recorded by
  #1158 in
  [lunar-lake-openvino-auto-selected-device.md](lunar-lake-openvino-auto-selected-device.md)
  and the runtime `AUTO` proof-boundary guard landed by #1159;
- #1154 / existing validator coverage for the NPU cache-classification guard:
  timing-derived cache evidence stays diagnostic and cannot become direct
  runtime cache-hit truth without direct runtime evidence fields.
- #1156 for the CPU comparison qualification guard: model-format or timing-scope
  mismatch keeps `benchmark_qualified=false` while preserving diagnostic CPU
  candidate context.
- #1160 for the current OpenVINO NPU cache-rerun evidence package;
- #1165 for operator receipt coverage closeout, which leaves the remaining
  operator-appliance gaps in #1064, #1069/#1071, #1149, and #1160 rather than a
  new broad route-policy or receipt cleanup.

This review adds a decision table and shared fail-closed rules only. It does
not run inference, refresh receipts, promote a route, revoke a route, claim a
speedup, claim a power advantage, or alter BitNet QK256/I2_S behavior.

## Current Ledger Snapshot

The committed `lunar-lake-route-promotion.json` currently records:

| Profile | Current promoted route | Review posture |
| --- | --- | --- |
| `regression_tiny` | `dense_slm_default_cpu` | keep CPU as cheap strict regression route |
| `ask_short` | `dense_slm_openvino_gpu_candidate` | keep after #1121 confirmed corpus, timing, fallback, and token visibility remain valid |
| `ask_normal` | `dense_slm_openvino_gpu_candidate` | keep after #1121 confirmed corpus, timing, fallback, and token visibility remain valid |
| `prefill_heavy` | `dense_slm_openvino_gpu_candidate` | review-watch because prefill/decode split evidence is weaker than total-response evidence |
| `decode_heavy` | `dense_slm_openvino_gpu_candidate` | review-watch because prefill/decode split evidence is weaker than total-response evidence |
| `structured` | `dense_slm_default_cpu` | keep CPU until structured OpenVINO evidence has its own promotion package |
| `low_power` | none | blocked by #1064 until real battery-mode route samples and energy proxy exist |
| `warm_resident` | `dense_slm_openvino_npu_candidate` | keep as resident-only after #1163 guarded the acceptance boundary; does not imply cold one-off or low-power promotion |
| `bitnet_strict_reference` | `bitnet_reference_cpu` | keep separate from dense SLM OpenVINO evidence |

NPU phase-timing schema work is defined by #1139/#1141 and must not become a
route-policy shortcut. New timer fields can support review only after their
scope, source, unavailable handling, and claim boundary are explicit.

Recent guard closeouts make the current route posture easier to preserve but do
not change it. #1159 blocks ambiguous runtime `AUTO` selected-device proof,
while #1163 blocks incomplete `warm_resident` NPU evidence, and #1165 confirms
that operator receipt follow-ups are now physical evidence or deferred
measurement items rather than another route-policy cleanup.

The ledger's route IDs remain campaign-local names. The route-ID proof-family
map records how future OpenVINO receipts and validators should relate those
campaign route IDs to canonical proof families before any future promotion
review depends on route identity:

```text
docs/reviews/lunar-lake-route-id-proof-family-map.md
```

## Decision Table

| Decision | Use when | Required route effect | Next smallest PR |
| --- | --- | --- | --- |
| Keep | Exact-profile quality, route identity, fallback=false, timing applicability, benchmark-qualified advantage, and negative claim fields still match current receipts | Leave the existing profile promotion in place | Docs/review closeout only |
| Keep with review-watch | Evidence still supports auto-route behavior, but a known gap limits claim strength, such as missing prefill/decode split or unavailable OpenVINO perf metric | Leave route policy unchanged, but require the gap to stay visible in receipts and reviews | Narrow review note or receipt-schema blocker field |
| Conditional | Evidence supports diagnosis or bounded comparison but depends on proxy timing, timing-derived cache classification, retokenized token IDs, or model-format mismatch | Keep route selectable only under the exact condition; do not broaden the claim | Schema hardening or issue update naming the condition |
| Narrow | A promoted route still passes for some profiles but lacks promotion evidence for others | Remove only the unsupported profiles from auto-route eligibility | Route-policy PR scoped to the named profiles |
| Revoke | Fallback appears, answer gates fail, route/device identity drifts, direct evidence is contradicted, or invalidation requires rerun before trust | Move the route/profile to `candidate` or `blocked` until evidence is rerun | Route-policy PR plus regression receipt update |
| Candidate-only | Visibility, smoke, bounded answer, or hot-path timing exists without a full promotion package | Keep explicit-request only; auto-route must not select it | Research or receipt contract PR, not promotion |
| Blocked | A hard gate is absent, such as battery-mode power evidence for `low_power` or selected-device proof for AUTO | Fail closed before model load or auto selection | Guard or preflight PR |

## Shared Fail-Closed Rules

These rules apply before profile-specific promotion language:

| Evidence condition | Required policy consequence |
| --- | --- |
| `fallback_used=true` anywhere in required route evidence | Block promotion for that route/profile |
| Requested accelerator resolves to CPU or a different accelerator | Block selected-device proof |
| AUTO/HETERO lacks actual execution-device evidence | Diagnostic only; no selected-device proof |
| Corpus-v2 profile has an unclassified failure | Candidate or blocked until rerun or accepted diagnostic-only policy |
| Direct generated-token IDs are required but unavailable | Block token-parity or promotion-grade token-visibility claims |
| Generated IDs are retokenized from text | Allow output accounting only; no direct pipeline-internal token claim |
| Timing uses another profile, proxy prompt, or missing token bounds | Block promotion; may remain diagnostic |
| OpenVINO metrics are unavailable or sentinel values | Keep the gap explicit; do not coerce into numeric summaries |
| Model format differs, such as GGUF Q8_0 versus OpenVINO INT4_SYM | Allow route/profile comparison only; no engine parity or matched-format speed claim |
| Power source is AC or battery telemetry is missing | No `low_power` promotion or power-advantage claim |
| Thermal readings are unavailable | Keep thermal as an explicit gap; do not invent measured temperature evidence |
| NPU evidence excludes pipeline construction | May support `warm_resident`; must not support cold one-off promotion |
| Cache hit is timing-derived rather than runtime-reported | Use for diagnosis only unless a later accepted policy allows it |
| Shared BitNet semantic fix lands after current CPU reference evidence | Rerun affected BitNet CPU reference evidence before changing BitNet route policy |
| Dense SLM OpenVINO evidence passes | Do not infer BitNet QK256/I2_S behavior, native OpenCL, native NPU kernels, or full BitNet accelerator inference |
| Old-repo Lunar Lake text or stale generated dashboard disagrees with swarm receipts | Treat swarm receipts and swarm issues as current; do not update route policy from old-repo wording |

## Profile-Specific Boundaries

### CPU Dense SLM

Dense Qwen CPU remains the correctness and fallback plate for profiles where no
accelerator has a stronger exact-profile package. CPU route evidence is not an
acceleration claim. #1122, closed by #1132, keeps Rust GGUF CPU as the dense
SLM correctness and fallback baseline while #1069 and #1071 remain the live
measurement follow-ups.

### OpenVINO GPU

OpenVINO GPU may stay promoted only for profiles whose evidence still has:

- exact OpenVINO IR model/export identity;
- Arc 140V / `GPU.0` selected-device identity;
- corpus-v2 pass for the same profile;
- `fallback_used=false`;
- direct generated-token visibility or an accepted lower claim boundary;
- timing that fits the profile bounds;
- benchmark-qualified advantage against the relevant CPU baseline;
- no native OpenCL, BitNet, power, or broad acceleration claim.

The `ask_short` and `ask_normal` profile decision lives in #1121. The current
ledger also promotes `prefill_heavy` and `decode_heavy`; keep those on
review-watch until prefill/decode split evidence is refreshed or a later review
decides whether to keep, narrow, or mark them conditional.

### OpenVINO NPU

OpenVINO NPU remains a resident-profile route, not a cold one-off route. Warm
resident evidence must keep these boundaries visible:

- one resident pipeline is reused;
- warm asks are separate from pipeline construction and first cold ask;
- answer, generated-token, fallback, route, and device drift are absent;
- memory growth and power/thermal context are recorded or blocked explicitly;
- cold-start caveats remain in route reasons.

Phase-timing work must keep host setup, tokenizer/template, pipeline
construction, compile/load/cache behavior, first ask, warm asks, and receipt
overhead in separate fields or explicit unavailable states before those timers
can affect route policy. Timing-derived cache classification remains
diagnostic unless a later accepted policy names it as sufficient for a narrower
claim.

Hot-path latency alone does not promote NPU for `ask_short`, `ask_normal`, or
`low_power`. Battery or energy-proxy evidence for `low_power` remains blocked
by #1064.

### BitNet CPU Reference

`bitnet_reference_cpu` is a separate specialist route. Dense Qwen CPU, GPU, or
NPU success must not be treated as BitNet QK256/I2_S proof. Shared BitNet
semantic-intake changes must rerun the affected CPU reference receipts before
BitNet route policy changes.

## Route Mutation Checklist

Any future route-policy PR must state:

1. exact profile or profiles changed;
2. current ledger state and proposed ledger state;
3. evidence receipt paths for quality, route identity, timing, telemetry, and
   regression;
4. whether generated-token IDs are direct, retokenized, or unavailable;
5. model/export/tokenizer/template identity and any format mismatch;
6. fallback status for every required receipt;
7. battery, power, and thermal status if `low_power` is affected;
8. cold/cache/warm/resident mode if NPU is affected;
9. BitNet claim-boundary fields and semantic-intake status;
10. rollback or fail-closed behavior if a later receipt contradicts the change.

Do not combine route-policy mutation with new inference surface work, broad
benchmark matrices, generated dashboard churn, or unrelated hardware lanes.

## Next Work

- Future token-visibility schema or validator work should use the #1123/#1138
  strategy before token-ID gaps become one-off wording in each receipt.
- #1121 has kept GPU `ask_short` / `ask_normal` promotion with a current
  evidence map; future GPU mutation needs a new concrete regression or review
  finding.
- If `prefill_heavy` or `decode_heavy` become active route-review targets,
  open a focused profile-phase issue instead of bundling them into #1121.
- #1120 has defined the NPU `warm_resident` route acceptance rule, and
  #1162/#1163 landed the current guard. Future resident-session policy changes
  should cite that review directly only if new evidence exposes a new gap.
- Use the #1139/#1141 NPU phase-timing schema before host setup,
  tokenizer/template, pipeline, compile/load/cache, first-ask, warm-ask, or
  receipt-overhead timings become route-policy evidence.
- #1119 should keep NPU cold/cache evidence diagnostic until cache, phase, and
  cold-start gates are accepted. The current cache-classification guard is
  already covered by existing validator behavior and #1154 is closed; #1160
  owns the current cache-rerun evidence package.
- Future receipt or validator work should use the #1135/#1137 route-ID
  proof-family map before depending on route identity.
- #1149 should own any runtime `AUTO` selected-device measurement follow-up;
  [lunar-lake-openvino-auto-selected-device.md](lunar-lake-openvino-auto-selected-device.md)
  records the current fail-closed contract from #1158, and #1159 has landed the
  validator guard. Route policy must keep `AUTO` diagnostic while
  selected-device proof is missing or ambiguous.
- #1156 has landed the current CPU comparison qualification guard; CPU follow-up
  work should focus on #1069/#1071 measurement or a newly exposed evidence gap,
  not another generic non-equivalence guard.
- #1165 closed the current operator receipt follow-up review. Future operator
  work should cite one of #1064, #1069/#1071, #1149, or #1160 rather than
  opening another broad operator coverage cleanup.
- #1064 remains the only current path to `low_power` promotion evidence.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- new generated dashboards;
- route-policy mutation;
- route promotion;
- route revocation;
- speedup or acceleration claims;
- power-advantage evidence;
- battery-mode evidence;
- measured-temperature evidence;
- native OpenCL proof;
- native NPU kernel proof;
- BitNet QK256/I2_S behavior proof.

It only defines when future Lunar Lake route-policy work must keep, narrow,
condition, revoke, or block a profile based on evidence.
