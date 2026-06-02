# Lunar Lake Route Policy Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1069](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1069), [#1071](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1071), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149), [#1160](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1160), [#1178](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1178), [#1186](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1186), [#1195](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1195), [#1209](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1209), [#1232](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1232), [#1241](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1241), [#1242](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1242), [#1244](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244), [#1245](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1245), [#1251](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1251), [#1263](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1263)
Linked PRs: [#1137](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1137), [#1138](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1138), [#1141](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1141), [#1156](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1156), [#1158](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1158), [#1159](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1159), [#1163](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1163), [#1165](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1165), [#1174](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1174), [#1182](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1182), [#1194](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1194), [#1208](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1208), [#1233](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1233), [#1248](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1248), [#1252](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1252), [#1254](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1254), [#1267](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1267), [#1294](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1294), [#1298](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1298)
Support-tier impact: no promotion; review-only route policy guard
Policy impact: no policy exception

## Recommendation

Keep the current Lunar Lake route policy in force, but treat every promoted
route as profile-scoped and receipt-invalidatable.

Do not open a route-policy mutation PR from this review alone. Issue #1245 is
the live watch issue for future keep, narrow, conditional, revoke, or blocked
decisions after #1124 closed. It requires a concrete linked evidence finding
before any route-policy PR.

The current smallest useful follow-up is to keep the profile reviews and
evidence contracts current:

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
  #1241 is closed by #1268 for the current GPU `prefill_heavy` /
  `decode_heavy` phase-claim-boundary hardening; future true phase-split or
  route-policy decisions should use a new narrow evidence issue or #1245;
- #1122 and #1132 for the CPU route posture decision; #1069 is now closed by
  #1182 as a resident-session command-surface review, #1186 is closed by #1194
  as the thread/core matrix builder contract, #1071 is closed by #1208 as the
  physical thread/core matrix evidence package, #1232 is the parent resident
  Rust GGUF phase evidence successor, #1280 is closed by #1334 as the physical
  resident package, and #1291 is closed by #1292 as the receipt-write /
  telemetry scope contract;
- #1123 for generated-token visibility rules, now closed by #1138 and defined
  in [lunar-lake-openvino-token-visibility.md](lunar-lake-openvino-token-visibility.md);
  #1244 owns future token-visibility schema or checker work if a later receipt
  exposes ambiguity;
- #1135 for route ID and canonical proof-family mapping, now closed by #1137
  in [lunar-lake-route-id-proof-family-map.md](lunar-lake-route-id-proof-family-map.md);
- #1178 for BitNet semantic-intake freshness; current intake remains ready, but
  future shared BitNet semantic changes must rerun affected CPU reference
  evidence before BitNet route-policy changes. #1263 is closed by #1267 as the
  diagnostic-only shared-surface classification path; future ambiguous
  BitNet-adjacent instrumentation touches should open a new narrow child under
  #1178 rather than weaken true semantic rerun behavior;
- #1149 for `AUTO` selected-device evidence before any `AUTO` receipt is used
  as route-policy evidence, with the current evidence contract recorded by
  #1158 in
  [lunar-lake-openvino-auto-selected-device.md](lunar-lake-openvino-auto-selected-device.md)
  and the runtime `AUTO` proof-boundary guard landed by #1159; #1242 is closed
  by #1248 and #1251 is closed by #1252, which landed the narrower GenAI
  debug-log parser helper and repeatable capture wrapper without changing
  route policy or generated phase-receipt selected-device status; #1254 then
  preserved SDPA warning and AUTO fallback-disabled line refs without changing
  the paired phase receipt's application `fallback_used=false` decision;
- #1154 / existing validator coverage for the NPU cache-classification guard:
  timing-derived cache evidence stays diagnostic and cannot become direct
  runtime cache-hit truth without direct runtime evidence fields.
- #1156 for the CPU comparison qualification guard: model-format or timing-scope
  mismatch keeps `benchmark_qualified=false` while preserving diagnostic CPU
  candidate context.
- #1160 for the current OpenVINO NPU cache-rerun evidence package, now closed
  by #1174 with a dated diagnostic receipt at
  `ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-npu-cache-rerun-20260601.json`;
- #1165 for operator receipt coverage closeout, which leaves the remaining
  operator-appliance gaps in #1064, #1149, #1178, and #1232 rather than a new
  broad route-policy or receipt cleanup.

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
| `prefill_heavy` | `dense_slm_openvino_gpu_candidate` | keep with review-watch after #1268: total-response evidence remains route evidence, isolated prefill/decode split claims remain blocked |
| `decode_heavy` | `dense_slm_openvino_gpu_candidate` | keep with review-watch after #1268: total-response evidence remains route evidence, isolated prefill/decode split claims remain blocked |
| `structured` | `dense_slm_default_cpu` | keep CPU until structured OpenVINO evidence has its own promotion package |
| `low_power` | none | blocked by #1064 until real battery-mode route samples and energy proxy exist |
| `warm_resident` | `dense_slm_openvino_npu_candidate` | keep as resident-only after #1163 guarded the acceptance boundary; does not imply cold one-off or low-power promotion |
| `bitnet_strict_reference` | `bitnet_reference_cpu` | keep separate from dense SLM OpenVINO evidence; #1178 owns semantic-intake freshness |

NPU phase-timing schema work is defined by #1139/#1141 and must not become a
route-policy shortcut. New timer fields can support review only after their
scope, source, unavailable handling, and claim boundary are explicit.

Recent guard closeouts make the current route posture easier to preserve but do
not change it. #1159 blocks ambiguous runtime `AUTO` selected-device proof,
while #1163 blocks incomplete `warm_resident` NPU evidence, #1165 confirms
that operator receipt follow-ups are now physical evidence or deferred
measurement items rather than another route-policy cleanup, #1182/#1194
separate CPU command-surface and receipt-builder closeouts from physical CPU
measurement evidence, #1268 keeps GPU heavy-profile phase-split claims
machine-readable and false when splits are absent, and #1292 keeps CPU resident
receipt-write / telemetry scope from being backfilled into per-profile timing.

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
| Shared BitNet semantic fix lands after current CPU reference evidence | Rerun affected BitNet CPU reference evidence through #1178 before changing BitNet route policy |
| Merged BitNet-adjacent diagnostic-only instrumentation touches shared surfaces | Classify through the #1263/#1267 reviewed non-trigger pattern before treating it as safe; ambiguous classification remains blocked and needs a new narrow #1178 child |
| Dense SLM OpenVINO evidence passes | Do not infer BitNet QK256/I2_S behavior, native OpenCL, native NPU kernels, or full BitNet accelerator inference |
| Old-repo Lunar Lake text or stale generated dashboard disagrees with swarm receipts | Treat swarm receipts and swarm issues as current; do not update route policy from old-repo wording |

## Profile-Specific Boundaries

### CPU Dense SLM

Dense Qwen CPU remains the correctness and fallback plate for profiles where no
accelerator has a stronger exact-profile package. CPU route evidence is not an
acceleration claim. #1122, closed by #1132, keeps Rust GGUF CPU as the dense
SLM correctness and fallback baseline. #1069 is closed by #1182 as historical
resident-session command-surface work, #1186 is closed by #1194 as matrix
builder work, #1071 is closed by #1208 as physical thread/core evidence, and
issue #1232 owns any further resident Rust GGUF phase/no-reload evidence before
CPU optimization, matched OpenVINO CPU comparison, or route-policy work. The
physical resident CPU package from #1334 closes #1280 and remains diagnostic,
not route-policy evidence. PR #1292 closed the #1291 scope decision by keeping
profile `receipt_write_ms` and `telemetry_ms` not backfilled unless a later
contract defines their source, scope, and summarizer rule. That contract is not
route-policy evidence and does not make Rust GGUF CPU resident timing
benchmark-qualified.

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
review-watch with the #1268 machine-readable phase boundary unless a later
review links true isolated prefill/decode phase evidence or a concrete receipt
contradiction and decides whether to keep, narrow, or mark them conditional.

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
BitNet route policy changes. #1178 owns that freshness contract. #1263 is
closed by #1267 as the current diagnostic-only shared-surface non-trigger
classification path, which records reviewed merged instrumentation without
turning it into a stale BitNet CPU reference trigger. The current #1178 audit
through swarm `main` `d6197833b299ea3b5d547f32f72cc8eef2ed88bc` keeps
`rerun_required=false`, `intake_ready=true`, and
`dense_slm_as_bitnet_proof=false`; the post-`c8076ea` A770-071 replay and
closeout commits (#1305/#1306) are docs/receipts/tracking-only non-triggers.
This review does not require a rerun while the current semantic intake remains
ready.

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

- Future token-visibility schema or validator work should use #1244 and the
  #1123/#1138 strategy before token-ID gaps become one-off wording in each
  receipt.
- #1121 has kept GPU `ask_short` / `ask_normal` promotion with a current
  evidence map; future GPU mutation needs a new concrete regression or review
  finding.
- #1241 is closed by #1268 for the current `prefill_heavy` and `decode_heavy`
  phase-claim-boundary hardening. Future GPU heavy-profile mutation needs a new
  concrete evidence issue or a #1245 keep, conditional, narrow, revoke, or
  blocked finding; do not bundle that work into #1121.
- #1120 has defined the NPU `warm_resident` route acceptance rule, and
  #1162/#1163 landed the current guard. Future resident-session policy changes
  should cite that review directly only if new evidence exposes a new gap.
- Use the #1139/#1141 NPU phase-timing schema before host setup,
  tokenizer/template, pipeline, compile/load/cache, first-ask, warm-ask, or
  receipt-overhead timings become route-policy evidence.
- #1119 should keep NPU cold/cache evidence diagnostic until cache, phase, and
  cold-start gates are accepted. The current cache-classification guard is
  already covered by existing validator behavior and #1154 is closed. #1160 is
  now closed by #1174 with current timing-derived diagnostic cache evidence,
  not direct runtime cache-hit truth or route-policy evidence.
- Future receipt or validator work should use the #1135/#1137 route-ID
  proof-family map before depending on route identity.
- #1178 owns BitNet semantic-intake freshness. Current intake stays ready
  unless a shared BitNet semantic change, validator gap, or receipt gap makes a
  targeted CPU reference rerun necessary. #1263 is closed by #1267 as the
  current diagnostic-only classification path for reviewed merged
  shared-surface touches such as #1257/#1264. Current post-`3080b3cca` movement
  through #1338 is A770 diagnostic replay/tracker work plus Lunar Lake CPU
  resident status/docs/evidence work. The only matched code surface in that
  range is #1319's `crates/bitnet-cli/src/commands/lunar_lake.rs` resident
  status update, which preserves BitNet claim boundaries and does not change
  the BitNet semantic intake state. Future ambiguous diagnostic replay indexing
  should be another narrow diagnostic-only refresh under #1178, not a BitNet
  reference rerun.
- #1149 should own any runtime `AUTO` selected-device measurement follow-up;
  [lunar-lake-openvino-auto-selected-device.md](lunar-lake-openvino-auto-selected-device.md)
  records the current fail-closed contract from #1158, and #1159 has landed the
  validator guard. Route policy must keep `AUTO` diagnostic while
  selected-device proof is missing or ambiguous. #1242/#1248 and #1251/#1252
  cover the accepted `genai_debug_log` parser shape and capture wrapper, while
  #1254 preserves warning/internal-policy line context without changing
  fallback status. Future #1149 work should only persist materially useful
  wrapper-generated evidence, test a concrete selected-device API/bridge, or
  perform route review after the selected-device, quality, timing, fallback,
  profile, and power gates exist.
- #1156 has landed the current CPU comparison qualification guard; CPU follow-up
  work should use #1232 for further resident Rust GGUF phase/no-reload evidence
  after the closed #1280/#1334 physical package, with #1292 as the current
  receipt-write / telemetry scope boundary, or a separate narrow issue for
  matched OpenVINO CPU comparison or later topology evidence once the target is
  concrete. Do not repeat the generic non-equivalence guard or backfill
  unavailable profile fields from aggregate observations.
- #1165 closed the current operator receipt follow-up review. Future operator
  work should cite one of #1064, #1149, #1178, #1232, or a new narrow physical
  CPU measurement/comparison issue rather than opening another broad operator
  coverage cleanup. Use #1119 for any broader NPU cold/cache research follow-up
  now that #1160 has closed.
- #1064 remains the only current path to `low_power` promotion evidence.
- #1245 owns future route-policy watch updates. Use it only after a linked
  evidence issue names a concrete keep, conditional, narrow, revoke, or blocked
  decision; it is not a standing route-policy mutation queue.

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
