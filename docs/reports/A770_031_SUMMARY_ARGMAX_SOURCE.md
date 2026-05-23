# A770-031 Summary Argmax Source Frontier

Date: 2026-05-23

## Scope

A770-031 adds a compact first-mismatch argmax-source frontier to the focused
A770-030 CPU AVX2 versus Intel A770 OpenCL summary parity receipt:

```text
a770_summary_seed770024_keywords_014
```

This is diagnostic evidence only. It does not change runtime math, tokenizer
authority, model artifacts, QK256 dispatch, OpenCL kernels, backend routing, or
the seeded answer-readiness corpus.

## Inputs

The refreshed parity receipt reuses the committed A770-030 focused receipts:

- CPU AVX2:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- Intel A770 OpenCL:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Focused parity:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

## Live Result

The focused parity receipt still fails for the selected case:

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 1 |
| `summary.first_divergence.kind` | `generated_token_ids` |
| `generated_output_frontier.classification` | `generated_output_frontier_first_mismatch_has_logit_context` |
| `generated_output_logit_margin_frontier.classification` | `generated_output_logit_margin_frontier_opposite_argmax` |
| `generated_output_argmax_source_frontier.classification` | `generated_output_argmax_source_frontier_internal_logit_source_missing_context` |

The new frontier keeps the row compact and preserves the first-mismatch
classification:

| Field | Value |
| --- | --- |
| `first_mismatch_index` | 9 |
| `left_chosen_id` | 40599 |
| `right_chosen_id` | 27252 |
| `left_generated_matches_chosen` | `true` |
| `right_generated_matches_chosen` | `true` |
| `left_chosen_is_top1` | `true` |
| `right_chosen_is_top1` | `true` |
| `prompt_token_ids_match` | `true` |
| `has_cross_chosen_logits` | `true` |
| `opposite_argmax` | `true` |
| `common_top_token_count` | 17 |
| `max_common_token_abs_delta` | 0.6159191131591797 |
| `qk256_operand_context_available` | `false` |
| `output_head_logit_accumulation_context_available` | `false` |

That exonerates prompt-history serialization, sampler/logit extraction policy,
and trace/capture loss at the current receipt boundary. It does not identify
whether the remaining opposite-argmax source is QK256 operand drift or
output-head/logit accumulation drift because the focused receipts do not carry
that internal context.

## Next Frontier

The useful next diagnostic is:

```text
capture first-mismatch QK256 operand and output-head logit accumulation context
```

That next slice should stay diagnostic-only until it separates:

- QK256 operand drift;
- output-head/logit accumulation drift;
- or missing internal trace context.

## Claim Boundary

A770-031 may claim:

- The focused A770-030 summary parity receipt now reports a compact
  first-mismatch argmax-source frontier.
- The committed focused receipt exonerates prompt-history serialization,
  sampler/logit extraction, and trace/capture context loss for the first
  mismatch with available top-k context, while routing the remaining source to
  missing internal QK256/output-head logit accumulation context.

A770-031 must not claim:

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
