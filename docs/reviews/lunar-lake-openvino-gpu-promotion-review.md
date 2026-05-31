# Lunar Lake OpenVINO GPU Promotion Review

Review issues:

- Original review: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1034
- Current refresh: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1121

Review date: 2026-05-30
Refresh date: 2026-05-31

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

This is a narrow dense SLM OpenVINO recommendation. It is not:

- native OpenCL proof;
- BitNet QK256/I2_S behavior proof;
- a broad acceleration claim;
- a broad chat-quality claim;
- a `low_power` promotion;
- a reason to promote GPU for `structured`, `regression_tiny`, or
  `warm_resident`.

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
now, but treat their prefill/decode phase split as weaker evidence than their
profile-level total response and corpus gates. If a later route-policy review
targets those two profiles, require explicit prefill/decode phase receipts.

Do not promote GPU for:

- `regression_tiny`, where CPU remains the cheap strict smoke route;
- `structured`, which remains CPU-promoted;
- `low_power`, which lacks battery-mode and power-advantage evidence;
- `warm_resident`, where NPU is the promoted route;
- `bitnet_strict_reference`, which is not a dense SLM OpenVINO profile.

## Follow-Up Work

No route-policy PR is recommended from this review.

Small follow-ups that still fit the current research-first operating mode:

1. Keep the GPU corpus-v2 status refresh tied to the route-promotion ledger and
   route-profile comparison when those artifacts change.
2. Add profile-phase receipts that split prefill and decode for
   `prefill_heavy` and `decode_heavy`.
3. Keep `low_power` blocked until `LNL258V-POWER-006` produces real
   battery-mode samples and energy-proxy evidence.

No immediate implementation PR follows from #1121. If `prefill_heavy` or
`decode_heavy` needs review, open a separate focused profile-phase issue rather
than expanding this `ask_short` / `ask_normal` review.

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
