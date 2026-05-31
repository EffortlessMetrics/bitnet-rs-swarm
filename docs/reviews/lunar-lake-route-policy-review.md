# Lunar Lake Route Policy Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124)
Linked PRs: n/a
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
- #1120 for NPU warm-resident acceptance;
- #1121 for OpenVINO GPU `ask_short` / `ask_normal` review;
- #1122 and #1132 for the CPU route posture decision, with #1069 and #1071
  as live measurement follow-ups;
- #1123 for generated-token visibility rules;
- #1135 for route ID and canonical proof-family mapping.

This review adds a decision table and shared fail-closed rules only. It does
not run inference, refresh receipts, promote a route, revoke a route, claim a
speedup, claim a power advantage, or alter BitNet QK256/I2_S behavior.

## Current Ledger Snapshot

The committed `lunar-lake-route-promotion.json` currently records:

| Profile | Current promoted route | Review posture |
| --- | --- | --- |
| `regression_tiny` | `dense_slm_default_cpu` | keep CPU as cheap strict regression route |
| `ask_short` | `dense_slm_openvino_gpu_candidate` | keep while #1121 confirms corpus, timing, fallback, and token visibility remain valid |
| `ask_normal` | `dense_slm_openvino_gpu_candidate` | keep while #1121 confirms corpus, timing, fallback, and token visibility remain valid |
| `prefill_heavy` | `dense_slm_openvino_gpu_candidate` | review-watch because prefill/decode split evidence is weaker than total-response evidence |
| `decode_heavy` | `dense_slm_openvino_gpu_candidate` | review-watch because prefill/decode split evidence is weaker than total-response evidence |
| `structured` | `dense_slm_default_cpu` | keep CPU until structured OpenVINO evidence has its own promotion package |
| `low_power` | none | blocked by #1064 until real battery-mode route samples and energy proxy exist |
| `warm_resident` | `dense_slm_openvino_npu_candidate` | keep as resident-only; does not imply cold one-off or low-power promotion |
| `bitnet_strict_reference` | `bitnet_reference_cpu` | keep separate from dense SLM OpenVINO evidence |

The ledger's route IDs remain campaign-local names. New OpenVINO receipts and
validators still need to map them to the canonical proof families defined by
`BITNET-SPEC-OPENVINO-ROUTE-CONTRACT` before any future promotion review.

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

- #1123 should define a central generated-token visibility strategy before
  token-ID gaps become one-off wording in each receipt.
- #1121 should either keep, narrow, or mark conditional the GPU
  `ask_short` / `ask_normal` promotion with a current evidence map.
- If `prefill_heavy` or `decode_heavy` become active route-review targets,
  open a focused profile-phase issue instead of bundling them into #1121.
- #1120 should define the NPU `warm_resident` route acceptance rule before any
  resident-session policy change.
- #1119 should keep NPU cold/cache evidence diagnostic until cache, phase, and
  cold-start gates are accepted.
- #1135 should map campaign route IDs to canonical proof families before new
  receipts or validators depend on route identity.
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
