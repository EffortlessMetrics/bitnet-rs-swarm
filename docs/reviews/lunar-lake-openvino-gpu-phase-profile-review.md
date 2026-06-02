# Lunar Lake OpenVINO GPU Phase Profile Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1113](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1113), [#1241](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1241)
Linked PRs: [#1114](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1114)
Support-tier impact: no promotion; review-only claim boundary for existing GPU phase profiles
Policy impact: no policy exception

## Recommendation

Keep OpenVINO GPU promoted for `prefill_heavy` and `decode_heavy`, but keep
both profiles on review-watch for phase attribution.

The current receipts support exact-profile auto routing for these two dense
Qwen OpenVINO GPU profiles. They do not support a narrower claim that the
profile win is proven by isolated prefill-phase speed or isolated decode-phase
speed. The route evidence is profile-level total-response evidence with
profile-matching prompt/output token counts, fallback-free GPU identity,
passing answer gates, direct generated-token visibility, and a
benchmark-qualified lower total response than the CPU profile baseline.

No route-policy PR follows from this review. The next smallest useful
implementation, if any, is receipt/status hardening that distinguishes:

- `profile_timing_available`;
- `profile_total_response_benchmark_qualified`;
- `prefill_split_available`;
- `decode_split_available`.

Do not use this review to claim native OpenCL execution, broad acceleration,
power advantage, BitNet QK256/I2_S behavior, or detailed prefill/decode phase
speedup.

The review landed in #1114 and closed #1113. It remains a profile-evidence
review-watch boundary, not an active route-policy or runtime implementation
queue.

Issue #1241 is the live evidence-contract follow-up for this boundary. It owns
the question of whether future receipts or status surfaces should make the
profile-level total-response evidence versus isolated prefill/decode phase
claim boundary machine-readable.

The route-profile receipt now serializes that boundary per route as
`phase_claim_boundary`. For the OpenVINO GPU `prefill_heavy` and
`decode_heavy` rows, the field records profile timing and benchmark-qualified
total-response evidence while keeping `prefill_split_available=false`,
`decode_split_available=false`, and `phase_split_claim_allowed=false`.

## Current Ledger State

`ci/hardware/intel-258v/2026-05-08/lunar-lake-route-promotion.json` records
`dense_slm_openvino_gpu_candidate` as promoted for:

- `ask_short`;
- `ask_normal`;
- `prefill_heavy`;
- `decode_heavy`.

This review covers only `prefill_heavy` and `decode_heavy`. The `ask_short` and
`ask_normal` decision remains in
[lunar-lake-openvino-gpu-promotion-review.md](lunar-lake-openvino-gpu-promotion-review.md).

## Evidence Map

| Profile | Required shape | Measured OpenVINO GPU shape | Corpus-v2 | Fallback | Direct token IDs | Total response | CPU baseline | Route posture |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `prefill_heavy` | prompt `>=2048`, output `<=64` | prompt `2731`, output `64` | 1/1 passed | false | yes | 6161.384 ms | 1373681.117 ms | keep with review-watch |
| `decode_heavy` | prompt `<=256`, output `>=512` | prompt `66`, output `512` | 1/1 passed | false | yes | 13423.639 ms | 123115.592 ms | keep with review-watch |

The GPU route evidence is profile-specific:

- selected backend: `openvino-gpu`;
- runtime API: `openvino_genai`;
- runtime device: `GPU.0`;
- resolved device: Intel Arc 140V iGPU;
- `fallback_used=false`;
- `answer_gate_passed=true`;
- `phase_timing_present=true`;
- `benchmark_qualified_advantage=true`;
- `promotion_eligible_for_profile=true`.

The profile run records explicit same-machine OpenVINO timing cases for these
two profiles so route-profile comparison no longer has to borrow tiny
corpus-v2 cases for timing applicability.

## What This Evidence Proves

The current receipts prove enough to keep the existing route-policy state for
these exact profiles:

- The GPU profile timing cases satisfy the token bounds for `prefill_heavy` and
  `decode_heavy`.
- The same profile rows pass bounded answer gates and corpus-v2 profile gates.
- The selected runtime is OpenVINO GenAI on `GPU.0`, with no CPU fallback.
- Direct generated-token IDs are available from
  `openvino_genai_encoded_results_tokens`.
- Profile-level total response is lower than the CPU profile baseline in the
  route comparison, and the receipt marks the comparison benchmark-qualified.

This supports exact-profile route selection. It is not a broad speedup claim.

## What Remains Weaker

The phase attribution remains weaker than the profile-level route evidence:

- OpenVINO timing has `tokenize_ms=-1.0` for these rows, meaning the metric was
  not reported by OpenVINO.
- Route-profile timing records `prefill_ms=null` for OpenVINO GPU.
- The current OpenVINO receipts repeatedly state that they do not expose
  `prefill_512` or `decode_128` splits for every profile.
- `prefill_heavy` total response captures a long-prompt workload, but it does
  not isolate prefill cost as a standalone measured phase.
- `decode_heavy` total response captures a long-output workload, but it does
  not isolate steady decode throughput as a standalone promotion claim.

Those gaps do not contradict the existing profile promotion. They limit the
claim boundary and justify keeping both profiles on review-watch until a later
phase receipt can express prefill and decode splits directly.

## Decision

Keep with review-watch.

| Decision candidate | Applies? | Reason |
| --- | --- | --- |
| Keep unqualified | no | The phase split gap is real and should remain visible. |
| Keep with review-watch | yes | Exact-profile total-response evidence supports route selection, while phase attribution remains incomplete. |
| Conditional | not needed yet | The current auto route can stay profile-scoped without adding a new runtime condition. |
| Narrow | no | No current receipt contradiction, fallback, answer failure, or device drift was found for these profiles. |
| Revoke | no | Required profile-level route evidence is present and benchmark-qualified. |

If a later receipt shows fallback, answer-gate failure, route/device drift,
missing token-bound applicability, or invalidated model/tokenizer identity,
then route policy should narrow or revoke only the affected profile.

## Next Smallest PR

No immediate route-policy PR is recommended.

Issue #1113 is closed by #1114. Issue #1241 owns the machine-readable
profile-level versus phase-split distinction, and the current receipt hardening
adds or derives fields such as:

- `profile_timing_available=true`;
- `profile_total_response_benchmark_qualified=true`;
- `prefill_split_available=false`;
- `decode_split_available=false`;
- `phase_split_claim_allowed=false`.

These fields make future reviews and operator summaries avoid treating
profile-level total response as isolated prefill/decode phase evidence. Future
work under #1241 should only continue if a concrete evidence regression appears
or a later receipt can expose true isolated prefill/decode splits.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- generated dashboard refreshes;
- route-policy mutation;
- route promotion;
- route revocation;
- speedup or acceleration claims;
- power-advantage evidence;
- battery-mode evidence;
- native OpenCL proof;
- NPU evidence;
- BitNet QK256/I2_S behavior proof.

It only reviews the current OpenVINO GPU `prefill_heavy` and `decode_heavy`
profile evidence and recommends keeping the existing promotions with explicit
phase-attribution review-watch.
