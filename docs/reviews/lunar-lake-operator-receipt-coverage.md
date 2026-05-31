# Lunar Lake Operator Receipt Coverage Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1108](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1108), [#1110](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1110)
Linked PRs: n/a
Support-tier impact: no promotion; review-only operator receipt coverage audit
Policy impact: no policy exception

## Question

Do the current Lunar Lake operator commands record enough receipt fields to make
profile-aware routing explainable from repo-native evidence?

This review covers the operator surfaces that should make boring decisions:

- `lunar-lake ask`;
- `lunar-lake validate`;
- `lunar-lake regress`;
- `lunar-lake profile-compare`;
- `lunar-lake compare`;
- adjacent route, telemetry, and power-profile receipts consumed by those
  commands.

It does not run inference, regenerate artifacts, promote or revoke routes, or
change route policy.

## Evidence Inspected

Current committed receipt root:

```text
ci/hardware/intel-258v/2026-05-08
```

Representative artifacts:

```text
lunar-lake-operator-ask-auto-gpu-ask-short-math-brief.json
lunar-lake-operator-ask-auto-gpu-ask-normal-math-brief.json
lunar-lake-operator-ask-auto-npu-warm-resident-math-brief.json
lunar-lake-operator-ask-auto-low-power-blocked.json
lunar-lake-operator-readiness.json
lunar-lake-regression-bundle-v2.json
lunar-lake-route-profile-comparison.json
lunar-lake-operator-comparison.json
lunar-lake-power-profile-evidence.json
lunar-lake-power-thermal-context.json
```

The artifacts were inspected as committed JSON. No hardware run, model load,
benchmark refresh, or generated dashboard edit was performed for this review.

## Coverage Legend

| Status | Meaning |
| --- | --- |
| Present | The operator-facing receipt has the field directly. |
| Linked | The aggregate receipt links a source receipt that has the field. |
| Blocked | The field is intentionally absent until a named physical evidence gate passes. |
| Gap | A small schema or guard PR should make the field harder to misuse. |

## Surface Map

| Surface | Current artifact | Coverage | Review decision |
| --- | --- | --- | --- |
| Successful `ask` | auto GPU `ask_short` / `ask_normal`, auto NPU `warm_resident` ask receipts | Present for selected route, profile, backend, runtime, model/export, tokenizer/template, prompt IDs, generated IDs, answer gate, fallback, route reason, claim boundary, and timing | Keep; add telemetry and timing-status hardening before using asks as complete operator context |
| Blocked `ask` | `lunar-lake-operator-ask-auto-low-power-blocked.json` | Present for fail-closed route selection, `fallback_used=false`, no model load, no new inference, runbook, and next evidence | Keep; this is correct blocker evidence, not low-power proof |
| `validate --strict` | `lunar-lake-operator-readiness.json` | Linked aggregate for operator readiness, route policy, route model identity coverage, power-profile blockers, thermal availability, and blocked ask evidence | Keep as readiness aggregate; do not treat it as a live answer receipt |
| `profile-compare --strict` | `lunar-lake-route-profile-comparison.json` | Present for per-profile route evidence, model identity, quality, timing, telemetry, blockers, and promotion eligibility | Keep as route-profile authority; continue mapping campaign route IDs to canonical proof families in future schemas |
| `regress --strict` | `lunar-lake-regression-bundle-v2.json` | Linked aggregate for corpus-v2, route profile, durability, BitNet semantic intake, power-profile evidence, ask receipts, blocked low-power, and claim boundary | Keep as regression bundle; use it to detect drift, not to replace source receipt fields |
| `compare --strict` | `lunar-lake-operator-comparison.json` | Linked aggregate for readiness and regression surfaces, no hidden fallback, route scope, and low-power blocker state | Keep as comparison summary; do not use it for new route claims |
| Power and telemetry | `lunar-lake-power-profile-evidence.json`, `lunar-lake-power-thermal-context.json` | Present for AC-only context, battery blocker, thermal availability, energy-proxy status, and no power claim | Keep; POWER-006 still needs physical battery-mode evidence |

## Field Checklist

| Field family | Current coverage | Notes |
| --- | --- | --- |
| Model/export identity | Present or linked | OpenVINO ask receipts include model file identities and export metadata; profile comparison carries route model identity. |
| Tokenizer/template identity | Present or linked | Ask receipts include `tokenizer_source`, `prompt_template`, rendered prompt, and prompt token IDs. |
| Backend/runtime/device identity | Present | Ask receipts expose requested and selected backend, runtime API, runtime device, and selected runtime. |
| Route reason | Present | Ask receipts carry `route_reason` and route-selection why-not fields. |
| Fallback status | Present | Operator ask and aggregate receipts preserve `fallback_used=false`; blocked asks also preserve no fallback. |
| Answer gate | Present or linked | Successful asks record bounded answer gates; regression and comparison index ask summaries. |
| Generated-token visibility | Present | Successful OpenVINO asks carry direct generated IDs from the source receipt path and token counts. |
| Timing | Present with caveat | Ask receipts include OpenVINO wall and perf metric timing; tokenization and detokenization metrics can appear as `-1.0` sentinels. |
| Power and thermal context | Linked, not direct on ask | Aggregate route/profile/regression receipts index telemetry, but successful ask receipts do not carry a power or thermal context summary. |
| Known blockers and next evidence | Present for blocked low-power, linked in aggregates | Low-power blocker fields are clear and should remain the active POWER-006 evidence contract. |
| BitNet proof boundary | Present | Claim-boundary fields keep dense SLM evidence separate from BitNet QK256/I2_S proof. |

## Findings

### 1. Successful asks need a telemetry summary

Successful `lunar-lake ask` receipts already prove route identity, fallback
status, answer gate, token visibility, and timing for the selected profile.
They do not directly carry power scheme, AC/battery state, thermal availability,
or a telemetry receipt pointer.

That is acceptable for the current non-low-power profile decisions because the
route-profile, regression, comparison, and power-profile receipts link the
telemetry context. It is still a coverage gap for the operator end state where
each answer path should explain power and thermal context where available.

Next smallest PR:

```text
LNL258V-OP-TELEMETRY-001:
  add a non-promotional telemetry_context summary to successful operator ask
  receipts, or explicitly mark it not_sampled/not_exposed.
```

The exact field contract for that PR is now defined in
[lunar-lake-operator-ask-telemetry-context.md](lunar-lake-operator-ask-telemetry-context.md).

Required boundaries for that PR:

- no battery-mode evidence claim;
- no `low_power` promotion;
- no benchmark refresh;
- no inference rerun requirement unless the field cannot be derived honestly;
- no speedup, power-advantage, accelerator, or BitNet claim.

### 2. OpenVINO timing sentinels need status fields

OpenVINO ask timing includes useful wall timing and OpenVINO perf metrics, but
some OpenVINO metrics use `-1.0` for unavailable tokenization or detokenization
timing. The token visibility review already says sentinel values must not be
coerced into numeric summaries.

Next smallest PR:

```text
LNL258V-OP-TIMING-STATUS-001:
  preserve raw OpenVINO perf metrics, but add per-metric status fields such as
  measured, not_exposed, not_reported_by_openvino, or not_applicable.
```

Required boundaries for that PR:

- no timing improvement claim;
- no profile promotion;
- no benchmark-qualified advantage claim;
- no generated-token or answer-quality claim expansion.

### 3. Campaign route IDs now have a canonical mapping note

The route policy review already records that current ledger route IDs are
campaign-local names. The operator receipts are usable because route reasons and
claim-boundary fields are explicit. The route-ID map now records how campaign
IDs such as `dense_slm_openvino_gpu_candidate` should relate to canonical
route IDs and proof families:

```text
docs/reviews/lunar-lake-route-id-proof-family-map.md
```

Next smallest implementation PR only if this becomes a validator or
support-tier blocker:

```text
LNL258V-ROUTE-ID-MAP-001:
  add canonical_route_id and proof_family fields alongside existing campaign
  route IDs.
```

That implementation should use the mapping note, reject backend/device/runtime
conflicts, and keep existing receipts readable without rewriting them.

## Non-Findings

This review does not find a reason to mutate route policy.

- OpenVINO GPU `ask_short` and `ask_normal` asks still have selected route,
  model/export identity, direct generated IDs, answer gates, fallback false, and
  timing evidence.
- OpenVINO NPU `warm_resident` remains resident-scoped and does not erase the
  cold-start or low-power blockers.
- `low_power` fails closed before model load and remains blocked on POWER-006.
- Dense SLM receipts remain separate from BitNet QK256/I2_S proof.

## Acceptance For #1108

Issue #1108 can close when this review lands because it answers the coverage
question and names the next smallest PRs. It must not be used to close
POWER-006, #1069, #1071, or any physical measurement issue.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- new benchmark or route-profile execution;
- generated dashboard churn;
- route-policy mutation;
- route promotion or revocation;
- `low_power` battery evidence;
- power-advantage evidence;
- measured-temperature evidence;
- OpenVINO speedup or acceleration claims;
- native OpenCL or native NPU proof;
- BitNet QK256/I2_S behavior proof.

It only audits current operator receipt coverage and defines narrow schema
hardening candidates.
