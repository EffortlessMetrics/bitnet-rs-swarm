# Lunar Lake OpenVINO Token Visibility Review

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1123](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1123), [#1121](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1121), [#1124](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1124), [#1160](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1160), [#1244](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1244)
Linked PRs: [#1101](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1101), [#1138](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1138)
Support-tier impact: no promotion; review-only token evidence policy
Policy impact: no policy exception

## Recommendation

Use three evidence levels for generated-token visibility in Lunar Lake
OpenVINO receipts:

| Evidence level | Receipt signal | May support | Must not support |
| --- | --- | --- | --- |
| Direct pipeline IDs | `generated_token_ids_available_from_pipeline=true`, `direct_generated_token_ids_available=true`, or source `openvino_genai_encoded_results_tokens` | corpus-v2 diagnosis, route regression, profile promotion review, first-token comparison within the same model/export scope | BitNet QK256 parity, native OpenCL/NPU proof, cross-format engine parity by itself |
| Retokenized output IDs | source contains `retokenized`, or direct flag is false while decoded text is tokenized again | output accounting, approximate generated-token count, diagnosis of text output | direct pipeline-internal token visibility, token parity, promotion-grade token evidence |
| Text only or unavailable | no token IDs, source `unavailable`, missing direct/proxy status, or sentinel metrics only | answer-gate text checks and blocked diagnostics | promotion-grade token visibility, token parity, first-token evidence, matched runtime comparison |

Direct generated-token IDs are promotion-grade token-visibility evidence only
inside the exact route/model/profile package that produced them. They do not
turn dense Qwen OpenVINO success into BitNet QK256/I2_S proof.

This review does not require a route-policy change because current committed
OpenVINO CPU, GPU, NPU corpus-v2, NPU cache, NPU cache-rerun, NPU resident, and
CPU comparison receipts already expose direct OpenVINO GenAI token IDs where the
current route reviews need them. It does define the fail-closed behavior for
future receipts.

The review note was added by #1101 and the #1123 closeout landed in #1138. It
remains a future fail-closed strategy, not an active implementation queue or a
reason to open another inference PR.

Issue #1244 is the live review-watch issue for future generated-token
visibility schema or checker work. It owns the question of whether a later
receipt needs a central `visibility_level`, a fail-closed promotion-review
guard, or a stale-receipt cleanup without changing route policy.

## Current Evidence Map

| Receipt | Token visibility finding | Policy consequence |
| --- | --- | --- |
| `slm-openvino-cpu-gpu-npu-corpus-v2.json` | scoring policy records `openvino_genai_encoded_results_tokens` captured from `EncodedResults.tokens` after generation from OpenVINO `TokenizedInputs` | usable for OpenVINO CPU/GPU/NPU corpus-v2 profile diagnosis |
| `lunar-lake-openvino-cpu-corpus-v2-diagnosis.json` | `direct_generated_token_ids_available=true`, `retokenized_generated_ids_used=false`, source `openvino_genai_encoded_results_tokens` | OpenVINO CPU diagnosis has direct token visibility |
| `lunar-lake-openvino-gpu-corpus-v2-diagnosis.json` | `direct_generated_token_ids_available=true`, `retokenized_generated_ids_used=false`, source `openvino_genai_encoded_results_tokens` | GPU `ask_short` / `ask_normal` review does not have a token-visibility blocker |
| `lunar-lake-openvino-npu-corpus-v2-diagnosis.json` | `direct_generated_token_ids_available=true`, `retokenized_generated_ids_used=false`, source `openvino_genai_encoded_results_tokens` | NPU corpus diagnosis has direct token visibility, but cold/resident/power blockers still apply |
| `lunar-lake-openvino-npu-cache-experiment.json` | `direct_generated_token_ids_available=true`, source `openvino_genai_encoded_results_tokens` | cache evidence can keep answer/token visibility honest, but cache-hit classification remains separate |
| `lunar-lake-openvino-npu-cache-rerun-20260601.json` | `direct_generated_token_ids_available=true`, source `openvino_genai_encoded_results_tokens`, profile applicability marked non-promotion smoke evidence | #1160 cache-rerun evidence keeps answer/token visibility honest, but timing-derived cache classification and profile scope remain separate |
| `lunar-lake-openvino-npu-resident-session.json` | `direct_generated_token_ids_available=true`, source `openvino_genai_encoded_results_tokens` | resident-session token drift checks can use direct IDs within that resident session |
| `lunar-lake-cpu-slm-runtime-comparison.json` | Rust GGUF CPU direct token evidence and OpenVINO CPU direct token evidence are both present; OpenVINO CPU did not use retokenized IDs | token visibility is not the CPU comparison blocker; model format and timing scope remain blockers |

The current blocker is not absence of direct OpenVINO IDs in the main Lunar
Lake OpenVINO receipts. The blocker is inconsistent claim scope: direct IDs can
support route/profile evidence, while retokenized or text-only evidence must
stay diagnostic.

## Required Receipt Shape

Every Lunar Lake OpenVINO route, diagnosis, comparison, and promotion-review
receipt should expose a token-visibility block or equivalent fields:

```json
{
  "generated_token_visibility": {
    "direct_generated_token_ids_available": true,
    "retokenized_generated_ids_used": false,
    "generated_token_ids_source": "openvino_genai_encoded_results_tokens",
    "visibility_level": "direct_pipeline_ids",
    "promotion_grade": true,
    "claim_boundary": [
      "direct token visibility is scoped to this model/export/profile",
      "does not prove BitNet QK256/I2_S parity"
    ]
  }
}
```

If the receipt uses case-level fields instead of a summary block, each case must
still make the same distinction:

- direct runtime IDs;
- retokenized output IDs;
- text-only or unavailable output.

Missing token-visibility fields must be treated as unavailable for promotion
review, not inferred from answer text.

## Promotion And Regression Use

| Surface | Minimum accepted token visibility |
| --- | --- |
| Corpus-v2 diagnosis | Direct, retokenized, or unavailable may be recorded, but the evidence level must be explicit |
| Regression-v2 route drift | Direct IDs preferred; retokenized or unavailable IDs must not be used as token-drift proof |
| Route promotion review | Direct IDs required unless an accepted review explicitly lowers the claim boundary |
| CPU/OpenVINO benchmark comparison | Direct IDs required for token-level comparison; text answer gates alone can only support route-level comparison |
| NPU resident-session drift | Direct IDs required for generated-token drift claims; text-only evidence can pass answer gates but not token drift |
| BitNet semantic intake | Dense SLM OpenVINO IDs do not satisfy BitNet QK256/I2_S direct-token evidence |

Answer gates and generated-token visibility answer different questions. A text
answer can be correct while direct generated-token visibility is missing.
Likewise, direct token IDs can be available while model-format mismatch still
blocks a fair runtime comparison.

## Fail-Closed Rules

Apply these rules before route promotion or regression acceptance:

| Condition | Required handling |
| --- | --- |
| Direct IDs required but missing | block promotion or token-drift claim |
| Retokenized IDs present and direct IDs absent | allow output accounting only; mark promotion token visibility not satisfied |
| Token source missing | treat as unavailable |
| Sentinel OpenVINO tokenization or detokenization timing values appear | do not coerce to numeric timing summaries; keep token timing as unavailable |
| Direct IDs present for OpenVINO dense SLM | keep scope to dense SLM route/profile; no BitNet QK256/I2_S claim |
| Direct IDs present but model formats differ | allow route/profile comparison; no matched-format engine parity claim |
| Direct IDs present but fallback occurs | block accelerator route evidence despite token visibility |
| Text answer gate passes but token IDs are unavailable | quality can pass; token parity and promotion-grade visibility remain blocked |

## Current Route Consequences

### GPU

The current GPU corpus-v2 and GPU promotion review do not have a token
visibility blocker. GPU `ask_short` and `ask_normal` can continue to be reviewed
against quality, fallback, timing, route identity, and benchmark-qualified
advantage without adding a token-visibility caveat.

Issue #1121 is now closed by the GPU promotion review. Token visibility remains
non-blocking for `ask_short` and `ask_normal` while the current direct-ID
evidence stays valid. Future GPU route mutation still needs a concrete
evidence regression or review finding.

### NPU

NPU corpus, cache, current cache-rerun, and resident receipts expose direct
token IDs, so the NPU blockers remain cold-start decomposition, cache truth,
resident acceptance, and `low_power` battery/power evidence. Direct NPU token
visibility does not promote NPU for cold one-off asks or `low_power`.

### CPU

Issue #1086 showed that OpenVINO CPU direct token visibility is available, but CPU
route optimization remains blocked by model-format mismatch, timing-scope
mismatch, prompt-render/tokenization gaps, and non-equivalent runtime scopes.
Do not use direct token visibility to claim Rust GGUF CPU and OpenVINO CPU are
matched-format benchmark peers.

### BitNet

Dense SLM OpenVINO token IDs are not BitNet QK256/I2_S evidence. BitNet CPU
reference and semantic-intake work still needs its own direct-token and
first-token evidence.

## Next Smallest PR

No immediate implementation PR is required from this review. Issue #1244 owns
future schema or checker work if a concrete ambiguity appears. The next small PR
should be one of:

- add a central schema helper for `visibility_level` if future receipts keep
  duplicating one-off token wording;
- make a promotion-review checker fail closed when direct IDs are required but
  only retokenized or text-only evidence exists;
- update a specific stale receipt only if it lacks explicit direct/proxy/
  unavailable token status.

Do not add a new inference path, route promotion, benchmark matrix, or broad
artifact refresh for token visibility alone.

## Acceptance For #1123

Issue #1123 is closed by #1138 because this review:

- defines the three accepted token-visibility levels: direct pipeline IDs,
  retokenized output IDs, and text-only or unavailable;
- maps current OpenVINO CPU/GPU/NPU receipts to those levels;
- gives a receipt shape for future `generated_token_visibility` fields;
- states when answer gates are enough and when token visibility blocks
  promotion-grade token evidence;
- defines fail-closed behavior when direct IDs are required but only
  retokenized or text-only evidence exists;
- keeps route promotion, benchmark equivalence, speedup, native accelerator,
  and BitNet QK256/I2_S claims unchanged.

## Claim Boundary

This review does not add:

- new Lunar Lake inference;
- route-policy mutation;
- route promotion;
- route revocation;
- speedup or acceleration claims;
- power-advantage evidence;
- OpenVINO correctness beyond the cited receipt fields;
- model equivalence between GGUF and OpenVINO IR;
- native OpenCL proof;
- native NPU kernel proof;
- BitNet QK256/I2_S behavior proof.

It only centralizes how Lunar Lake distinguishes direct, retokenized, and
unavailable generated-token evidence for future route decisions.
