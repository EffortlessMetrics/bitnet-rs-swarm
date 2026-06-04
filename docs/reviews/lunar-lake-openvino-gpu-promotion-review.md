# Lunar Lake OpenVINO GPU Promotion Review

Review issues:

- Original review: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1034
- Current refresh: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1121
- Closed duplicate refresh: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1225
- Closed phase-boundary follow-up: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1241
- Current token-visibility watch: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244
- Current route-policy watch: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1245
- Current ask-profile guard: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1373

Review date: 2026-05-30
Refresh date: 2026-05-31
Post-phase-boundary refresh: 2026-06-02
Ask-guard refresh: 2026-06-03
Post-#1373 question matrix refresh: 2026-06-04

Repository: `EffortlessMetrics/bitnet-rs-swarm`

## Recommendation

Keep OpenVINO GPU promotion for `ask_short` and `ask_normal`.

Do not open a route-policy PR for those profiles. Current swarm artifacts show
fallback-free OpenVINO GenAI execution on Arc 140V GPU, passing corpus-v2
profile evidence, direct generated-token visibility, profile-matched timing,
and benchmark-qualified lower total response than the CPU baseline.

The 2026-05-31 refresh keeps that conclusion. The route-policy review now
records that `ask_short` and `ask_normal` should remain profile-scoped and
receipt-invalidatable, and the token-visibility review records that current
OpenVINO GPU corpus-v2 evidence has direct generated-token IDs from
`openvino_genai_encoded_results_tokens`. Those reviews remove route-policy and
token-visibility drift as reasons to narrow the two GPU ask profiles.

The 2026-06-02 phase-boundary refresh also keeps the conclusion unchanged. Issue
`#1241` is now closed by #1268, which added machine-readable
`phase_claim_boundary` fields to route-profile rows. For OpenVINO GPU
`prefill_heavy` and `decode_heavy`, those fields keep profile-level
total-response route evidence separate from isolated prefill/decode phase
claims with `prefill_split_available=false`,
`decode_split_available=false`, and `phase_split_claim_allowed=false`. That
hardening protects the broader GPU ledger without changing the `ask_short` /
`ask_normal` promotion decision.

The 2026-06-03 ask-guard refresh adds #1373 as the active receipt-invalidation
contract for `ask_short` and `ask_normal`. That issue does not request new
inference or a route-policy mutation. It says the two ask profiles may stay
promoted only while route identity, ledger/profile state, `fallback_used=false`,
answer gates, corpus-v2 quality, direct generated-token visibility,
profile-matched timing, benchmark qualification, selected-device evidence, and
negative claim-boundary fields remain present and uncontradicted. Any future
contradiction should first produce a narrow guard, status, or review update; a
route-policy mutation still goes through #1245.

This is a narrow dense SLM OpenVINO recommendation. It is not:

- native OpenCL proof;
- BitNet QK256/I2_S behavior proof;
- a broad acceleration claim;
- a broad chat-quality claim;
- a `low_power` promotion;
- a reason to promote GPU for `structured`, `regression_tiny`, or
  `warm_resident`.

## Post-#1373 GPU Promotion Question Matrix

This matrix routes the remaining GPU promotion and guard questions to current
evidence, missing receipt shape, owner issues, forbidden claims, and next
smallest PRs. It is a review aid only; it does not refresh hardware receipts,
run inference, change route policy, promote new profiles, claim speedup, claim
power advantage, prove native OpenCL, or prove BitNet QK256/I2_S behavior.

| Question | Current evidence | Missing receipt shape | Owner issue | Forbidden claims | Next smallest PR |
| --- | --- | --- | --- | --- | --- |
| Can `ask_short` and `ask_normal` stay GPU-promoted? | #1121 kept both profiles promoted, and #1373 records the current receipt-invalidation guard: route identity, ledger/profile state, fallback false, answer gates, corpus-v2 quality, direct token IDs, profile timing, benchmark qualification, selected device, and negative claim boundaries must remain present and uncontradicted. | Future receipts must retain those fields for each ask profile, including selected backend/device, `fallback_used=false`, direct token source, profile-matched timing, and benchmark-qualified comparison. | [#1373](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1373), [#1245](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1245) | No new route-policy mutation, broader GPU promotion, speedup claim, native OpenCL claim, or BitNet claim from the guard alone. | None while the guard fields stay present; open a narrow guard/status/review PR only if a future receipt drops or contradicts one required field. |
| Does corpus-v2 quality support the promoted ask profiles? | The GPU corpus-v2 package records 14/14 GPU cases passed, with `ask_short` 2/2 and `ask_normal` 3/3, fallback false, and no current answer-gate failures. | A rerun only becomes useful if corpus, model/export, tokenizer/template, answer gates, or profile mapping changes and the receipt preserves per-profile pass/fail and fallback fields. | [#1373](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1373), [#1244](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244) | No broad chat-quality or all-profile quality claim from bounded corpus-v2 pass counts. | No PR unless a future corpus or fixture change needs a profile-quality status refresh or validator anchor. |
| Are generated token IDs promotion-grade? | Current GPU corpus-v2 evidence records direct generated-token IDs from `openvino_genai_encoded_results_tokens` and marks retokenized generated IDs false. | Future receipts must either use the shared direct-token helper or explicitly record direct, proxy, text-retokenized, or unavailable token-source status. | [#1244](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244), [#1373](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1373) | No promotion-grade token-visibility claim from text-only or retokenized IDs; no route-policy change from token visibility alone. | A checker/schema PR only if a future GPU receipt bypasses the helper or makes token-source status ambiguous. |
| Is route/device identity clear enough for ask-profile evidence? | The cited receipts name requested and selected backend `openvino-gpu`, runtime API `openvino_genai`, runtime device `GPU.0`, resolved Arc 140V GPU identity, and `fallback_used=false`. | Future ask receipts must keep requested backend, selected backend, runtime API, runtime device, resolved device, fallback status, and selected runtime in one profile-scoped package. | [#1373](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1373), [#1149](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1149) | No `AUTO` selected-device proof, NPU proof, native OpenCL proof, or CPU-fallback proof from explicit GPU receipts. | No PR unless explicit GPU receipts lose selected-device clarity, or an `AUTO` path later needs #1149 evidence before route-policy use. |
| Do timing and benchmark qualification justify ask-profile routing? | The route-profile comparison marks `ask_short` and `ask_normal` as benchmark-qualified, profile quality passed, fallback false, and lower total response than the CPU baseline for those exact profiles. | Any refresh must preserve profile bounds, CPU baseline identity, total-response timing scope, benchmark qualification, answer gate, fallback status, and model/export identity. | [#1373](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1373), [#1245](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1245) | No broad speedup or acceleration claim; no benchmark claim for profiles outside the measured and promoted ask profiles. | No PR unless route-profile comparison loses benchmark qualification fields or a concrete evidence issue asks #1245 for keep, conditional, narrow, revoke, or blocked review. |
| Do `prefill_heavy` and `decode_heavy` need route-policy action? | #1268 made the phase boundary machine-readable: GPU heavy profiles have profile-level total-response evidence, but isolated prefill/decode splits remain unavailable and `phase_split_claim_allowed=false`. | True isolated prefill/decode phase receipts, or a concrete contradiction in the current `phase_claim_boundary` fields. | [#1241](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1241), [#1245](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1245) | No isolated prefill/decode speedup claim and no bundled ask-profile route mutation. | No PR from current evidence; open a new narrow evidence issue or use #1245 only if true phase splits or contradictions appear. |
| Can GPU satisfy `low_power` today? | GPU `low_power` remains candidate-only and blocked by missing battery-mode and power-advantage evidence; current GPU latency receipts are not battery or energy evidence. | Battery-mode route samples with strict power state, energy proxy, thermal availability or explicit unavailability, answer gates, fallback false, and benchmark-qualified power advantage. | [#1064](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1064) | No low-power route, battery, power-advantage, thermal, or energy claim from AC-only or latency-only GPU evidence. | Wait for the physical POWER-006 battery runbook sequence; do not open another GPU promotion PR for low_power without that evidence. |
| Does OpenVINO GPU evidence prove native OpenCL or BitNet accelerator behavior? | Current GPU evidence is dense Qwen OpenVINO GenAI on Arc 140V `GPU.0`; it is separate from native OpenCL kernels and separate from packed BitNet QK256/I2_S behavior. | Separate native OpenCL or BitNet accelerator proof with model/kernel identity, CPU reference, fallback false, answer gates, and exact route/profile timing. | Future Arc/OpenCL or BitNet accelerator issue only | No native OpenCL, full BitNet accelerator inference, packed QK256/I2_S, or dense-SLM-as-BitNet claim. | No PR in this dense SLM OpenVINO GPU promotion review; keep native/BitNet accelerator proof in separate lanes. |

## Current Route State

`lunar-lake-route-promotion.json` promotes:

- `dense_slm_default_cpu` for `regression_tiny` and `structured`;
- `dense_slm_openvino_gpu_candidate` for `ask_short`, `ask_normal`,
  `prefill_heavy`, and `decode_heavy`;
- `dense_slm_openvino_npu_candidate` for `warm_resident`;
- no route for `low_power`.

This review is about whether `ask_short` and `ask_normal` should remain on the
OpenVINO GPU route. The answer is yes.

This review does not re-open `prefill_heavy` or `decode_heavy`. The route-policy
review keeps those profiles on review-watch until a focused profile-phase review
decides whether their prefill/decode split evidence is sufficient.

## Evidence Summary

### Corpus-v2 Quality

`slm-openvino-cpu-gpu-npu-corpus-v2.json` records the GPU route as:

- requested backend: `openvino-gpu`;
- selected backend: `openvino-gpu`;
- runtime API: `openvino_genai`;
- runtime device: `GPU.0`;
- resolved device: `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)`;
- selected runtime: `openvino-genai-llmpipeline-gpu0`;
- fallback used: `false`;
- pipeline construction: 3644.763 ms;
- 14/14 GPU corpus-v2 cases passed;
- `ask_short`: 2/2 passed;
- `ask_normal`: 3/3 passed;
- direct generated token IDs were captured from OpenVINO GenAI output;
- generated token IDs were not retokenized from generated text.

The same receipt records 42/42 total OpenVINO CPU/GPU/NPU corpus-v2 cases
passed, with no fallback. That is evidence for bounded route regression, not a
broad quality claim.

### Route Profile Comparison

`lunar-lake-route-profile-comparison.json` records these GPU profile rows:

| Profile | Route status | Corpus profile pass | Total response | CPU baseline | Ratio | Blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `ask_short` | promoted | 2/2 | 3720.154 ms | 27986.539 ms | 0.133 | none |
| `ask_normal` | promoted | 3/3 | 4066.433 ms | 27986.539 ms | 0.145 | none |
| `regression_tiny` | candidate | 4/4 | 4203.053 ms | 27986.539 ms | 0.150 | route not promoted for profile |
| `low_power` | candidate | 1/1 | 3776.763 ms | 27986.539 ms | 0.135 | power advantage missing |
| `warm_resident` | candidate | 1/1 | 3803.649 ms | 27986.539 ms | 0.136 | route not promoted for profile |

Both `ask_short` and `ask_normal` have:

- `fallback_used=false`;
- `answer_gate_passed=true`;
- profile quality status `passed`;
- benchmark-qualified latency advantage;
- timing matched to the target profile;
- no route blockers.

### Operator Ask Evidence

The profile-specific operator receipts support the same recommendation:

| Receipt | Profile | Construct | Generation | Load time | TTFT | Generated tokens | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lunar-lake-operator-ask-auto-gpu-ask-short-math-brief.json` | `ask_short` | 1839.544 ms | 328.120 ms | 1819 ms | 160.202 ms | 9 | promoted, fallback false, answer gate passed |
| `lunar-lake-operator-ask-auto-gpu-ask-normal-math-brief.json` | `ask_normal` | 2224.170 ms | 285.518 ms | 2202 ms | 160.490 ms | 8 | promoted, fallback false, answer gate passed |
| `lunar-lake-openvino-operator-ask-gpu-math-brief.json` | bounded GPU ask | 2160.026 ms | 301.604 ms | 2133 ms | 161.787 ms | 9 | fallback false, direct token IDs available |

The profile-specific operator receipts also set `speedup_claim=false`, which is
correct. The route comparison can justify route selection without turning it
into a broad speedup or acceleration claim.

## Issue Scope Findings

### Route-Profile Comparison

The route-profile comparison is sufficient to keep `ask_short` and `ask_normal`
promoted. It names the CPU baseline, records the OpenVINO GPU total response,
marks the comparison benchmark-qualified, and keeps profile scope explicit.

### GPU Corpus-v2 Diagnosis

`lunar-lake-openvino-gpu-corpus-v2-diagnosis.json` contains stale generic text:

- `promotion_status` still says `candidate_only_not_promoted`;
- `recommended_next_actions` still says to keep OpenVINO GPU/NPU routes
  unpromoted until failed corpus-v2 cases are rerun.

The data in the same diagnosis is clean:

- 14 total GPU cases;
- 14 passed;
- 0 failed;
- no timeout or not-run cases;
- no failed profiles or categories;
- direct generated token IDs available;
- no retokenized generated IDs.

Treat the stale wording as diagnostic text drift, not as evidence that
`ask_short` or `ask_normal` must be narrowed. Route policy does not need to
change for this review. The current status-refresh receipt is
`ci/hardware/intel-258v/2026-05-08/lunar-lake-openvino-gpu-corpus-v2-diagnosis-status-refresh.json`.

That status-refresh receipt preserves the useful quality facts from the
diagnosis: 14/14 cases passed, fallback is false, direct generated-token IDs
are available, retokenized generated IDs were not used, and the active fixture
is aligned. It also marks the stale diagnosis fields explicitly:

- `promotion_status=candidate_only_not_promoted` is quality-diagnosis text
  only, not route-status authority;
- `recommended_next_actions[0]` is superseded because GPU corpus-v2 failures
  are cleared;
- `profile_diagnoses[*].route_profile_status=candidate_blocked_by_quality` is
  superseded for route status by profile-scoped route comparison evidence.

Use `lunar-lake-route-promotion.json` and
`lunar-lake-route-profile-comparison.json` as the current GPU route-status
authority. Keep the original diagnosis receipt as a quality-diagnosis artifact,
not a promotion ledger. If a future receipt refresh removes the status-refresh
boundary or changes the preserved quality facts, reopen route-policy review
before narrowing or broadening GPU promotion.

### Answer-Gate Failures

No current GPU corpus-v2 answer-gate failures were found. The issue's
`corpus-v2 cleanup` concern appears resolved by the current committed
receipts.

### Token Visibility Gap

No token visibility blocker was found for GPU corpus-v2. The OpenVINO corpus
receipt records:

- `generated_token_ids_available_from_pipeline=true`;
- `generated_token_ids_retokenized_from_text=false`;
- source `openvino_genai_encoded_results_tokens`.

Preserve this requirement for every future OpenVINO GPU receipt.

The dedicated token-visibility review confirms the same boundary: direct
pipeline IDs can support promotion-grade token visibility inside this exact
dense SLM route/profile package, while retokenized or text-only evidence would
remain diagnostic. The current GPU corpus-v2 diagnosis is in the direct-token
bucket, so token visibility does not require narrowing `ask_short` or
`ask_normal`.

### Prefill/Decode Split Gap

The gap is real but not a blocker for keeping `ask_short` and `ask_normal`
promoted.

Current OpenVINO receipts do not expose `prefill_512` or `decode_128` splits
for every profile. That matters most for future claims about `prefill_heavy`,
`decode_heavy`, and detailed phase attribution. It should remain a follow-up
measurement requirement, not a reason to narrow `ask_short` or `ask_normal`.

### Route Scope Language

The current route language is appropriately scoped:

- GPU is promoted only for named dense Qwen OpenVINO profiles;
- hidden fallback is disallowed;
- CPU remains the default route ID and regression baseline;
- `low_power` remains unpromoted until power evidence exists;
- BitNet remains a CPU reference route until accelerator BitNet parity and
  timing evidence exists.

Keep that language. Do not reword it into an acceleration, native OpenCL, or
BitNet claim.

The route-policy review turns this into a reusable decision rule. Future GPU
route-policy work should keep, narrow, condition, or revoke a profile based on
the exact profile evidence, not on global accelerator status or old generated
dashboard wording.

## Promotion Boundaries

Keep GPU promoted for:

- `ask_short`;
- `ask_normal`.

Also keep the existing ledger state for `prefill_heavy` and `decode_heavy` for
now. #1268 makes their split boundary explicit in the route-profile receipt:
profile-level total-response evidence is benchmark-qualified route evidence,
while isolated prefill/decode split evidence remains unavailable for the
OpenVINO GPU rows. If a later route-policy review targets those two profiles,
require either true isolated prefill/decode phase receipts or a concrete
receipt contradiction before narrowing or broadening the ledger.

Do not promote GPU for:

- `regression_tiny`, where CPU remains the cheap strict smoke route;
- `structured`, which remains CPU-promoted;
- `low_power`, which lacks battery-mode and power-advantage evidence;
- `warm_resident`, where NPU is the promoted route;
- `bitnet_strict_reference`, which is not a dense SLM OpenVINO profile.

## Follow-Up Work

No route-policy PR is recommended from this review.

Small follow-ups that still fit the current research-first operating mode:

1. Use #1373 as the active guard for `ask_short` / `ask_normal` receipt
   invalidation. It is the right place for a future checker, schema, status, or
   review update if those profiles lose route identity, fallback-free evidence,
   corpus quality, direct-token visibility, timing applicability, benchmark
   qualification, selected-device clarity, or claim-boundary fields.
2. Keep the GPU corpus-v2 status refresh tied to the route-promotion ledger and
   route-profile comparison when those artifacts change.
3. Treat #1241 as satisfied by #1268 for the current phase-boundary hardening.
   Open a new narrow issue only if later receipts can expose true isolated
   prefill/decode splits or if a future artifact regresses the
   `phase_claim_boundary` fields.
4. Keep `low_power` blocked until `LNL258V-POWER-006` produces real
   battery-mode samples and energy-proxy evidence.
5. Use #1244 only for future direct-vs-retokenized token-visibility schema or
   checker gaps.
6. Use #1245 only after a linked evidence issue names a concrete keep,
   conditional, narrow, revoke, or blocked decision.

No immediate implementation PR follows from #1121. If `prefill_heavy` or
`decode_heavy` needs more review, open a separate focused profile-phase issue
or route the decision through #1245 rather than expanding this `ask_short` /
`ask_normal` review.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- new generated dashboards;
- new route policy;
- new route promotion;
- speedup or acceleration claims;
- power-advantage evidence;
- native OpenCL proof;
- NPU evidence;
- `low_power` evidence;
- BitNet QK256/I2_S behavior proof.

It only reviews the current OpenVINO GPU promotion evidence and recommends
keeping `ask_short` and `ask_normal` promotion in place.

## References

- OpenVINO documentation: GPU device,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html
- OpenVINO documentation: query device properties,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html
- OpenVINO documentation: optimizing latency,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency.html
- Lunar Lake route policy review:
  [lunar-lake-route-policy-review.md](lunar-lake-route-policy-review.md)
- Lunar Lake OpenVINO token visibility review:
  [lunar-lake-openvino-token-visibility.md](lunar-lake-openvino-token-visibility.md)
