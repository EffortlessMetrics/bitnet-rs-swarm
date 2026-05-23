# A770-034 Model-Forward Source Frontier

Date: 2026-05-23

## Scope

A770-034 adds compact model-forward source context to the focused
A770-030/A770-031/A770-032/A770-033 summary divergence receipts:

```text
a770_summary_seed770024_keywords_014
```

This is diagnostic evidence only. It does not change runtime math, tokenizer
authority, model artifacts, QK256 dispatch, OpenCL kernels, backend routing, or
the seeded answer-readiness corpus.

## Inputs

The refreshed focused receipts are:

- CPU AVX2:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- Intel A770 OpenCL:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Focused parity:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

The answer-corpus runs used:

```text
BITNET_LOGIT_SOURCE_CONTEXT_STEPS=9
```

so only the known first generated-token mismatch carries the internal source
context.

## Live Result

The focused parity receipt still fails for the selected case:

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 1 |
| `summary.first_divergence.kind` | `generated_token_ids` |
| `generated_output_hidden_state_source_frontier.classification` | `generated_output_hidden_state_source_frontier_forward_output_drift` |
| `generated_output_model_forward_source_frontier.classification` | `generated_output_model_forward_source_frontier_prior_layer_output_drift` |

The new frontier records:

| Field | Value |
| --- | --- |
| `first_mismatch_index` | 9 |
| `left_chosen_id` | 40599 |
| `right_chosen_id` | 27252 |
| `prior_layer_output_sha256_match` | `false` |
| `final_norm_output_sha256_match` | `false` |
| `forward_output_sha256_match` | `false` |
| `left_prior_layer_output_sha256_f32_le` | `338c8b6d1164bc13b83c0afd76cc85365fa2e412f8e6c01b7956372919d6c286` |
| `right_prior_layer_output_sha256_f32_le` | `34239dfc1f8d9be61d8d0e43bceb2c17e080379eef013b9a710a590d71d9c233` |
| `left_prior_layer_output_shape` | `[1, 1, 2560]` |
| `right_prior_layer_output_shape` | `[1, 1, 2560]` |
| `left_prior_layer_output_rms` | 5616.558052599897 |
| `right_prior_layer_output_rms` | 5442.889793282263 |
| `prior_layer_output_rms_abs_delta` | 173.66825931763378 |
| `left_final_norm_matches_forward_output` | `true` |
| `right_final_norm_matches_forward_output` | `true` |
| `final_norm_output_rms_abs_delta` | 0.0016226334791995672 |

That moves the selected first-mismatch source upstream of final norm output: the
prior layer output already differs between CPU AVX2 and Intel A770 OpenCL at the
selected step. Final norm preserves the drift into `model.forward` output, so the
next useful boundary is inside the final transformer block.

## Next Frontier

The useful next diagnostic is:

```text
capture final transformer block residual, attention output, and FFN output fingerprints
```

That next slice should stay diagnostic-only until it separates:

- final transformer block residual drift;
- final block attention output drift;
- final block FFN output drift;
- residual-add numeric policy;
- or missing internal trace context.

## Claim Boundary

A770-034 may claim:

- The focused summary first mismatch now carries compact final norm and prior
  layer output fingerprints inside `model.forward`.
- The committed focused receipt classifies the selected first mismatch as
  `generated_output_model_forward_source_frontier_prior_layer_output_drift`.
- The first mismatch is already present in the prior layer output before final
  norm at the committed diagnostic boundary.

A770-034 must not claim:

- CPU/A770 answer parity is proven.
- Reference parity is proven.
- Strict A770 answer readiness is proven.
- Broad A770 answer quality is proven.
- BitNet inference is fully proven on A770.
- Official BitNet QK256 production semantics are fully proven.
- Activation quantization is GPU-resident.
- Selected attention is resident.
- Resident KV is proven.
- Full A770 residency is proven.
- Performance speedup is proven.
- A770 trusted partial acceleration is claim-grade.
