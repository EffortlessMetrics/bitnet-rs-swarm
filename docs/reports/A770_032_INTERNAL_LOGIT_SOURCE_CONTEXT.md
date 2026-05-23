# A770-032 Internal Logit Source Context

Date: 2026-05-23

## Scope

A770-032 adds compact first-mismatch internal logit source context to the
focused A770-030/A770-031 summary divergence receipts:

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
| `generated_output_argmax_source_frontier.classification` | `generated_output_argmax_source_frontier_internal_logit_source_missing_context` |
| `generated_output_internal_logit_source_frontier.classification` | `generated_output_internal_logit_source_frontier_hidden_operand_drift` |

The new frontier records:

| Field | Value |
| --- | --- |
| `first_mismatch_index` | 9 |
| `left_chosen_id` | 40599 |
| `right_chosen_id` | 27252 |
| `hidden_operand_sha256_match` | `false` |
| `left_hidden_operand_sha256_f32_le` | `224d8c1fb08d05d0a3ce05e8e8491cbcc46da8a52f25461c3204a3c37993f650` |
| `right_hidden_operand_sha256_f32_le` | `701a390a1879edd1b1df381a90098fec35e76288a38cd40a89fc187a02065114` |
| `left_hidden_operand_rms` | 0.07708468808572078 |
| `right_hidden_operand_rms` | 0.07870732156492034 |
| `hidden_operand_rms_abs_delta` | 0.0016226334791995672 |
| `output_head_logit_accumulation_context_available` | `false` |
| `left_output_head_qk256_dispatch_delta.execution_claim` | `no_qk256_dispatch_observed` |
| `right_output_head_qk256_dispatch_delta.execution_claim` | `no_qk256_dispatch_observed` |

That moves the first-mismatch source upstream of output-head accumulation: the
hidden operand entering logits already differs between CPU AVX2 and Intel A770
OpenCL at the selected step.

## Next Frontier

The useful next diagnostic is:

```text
localize hidden-state operand drift before output-head QK256
```

That next slice should stay diagnostic-only until it separates:

- final norm / last hidden-state drift;
- prior layer output drift;
- QK256 residual contribution drift;
- or missing internal trace context.

## Claim Boundary

A770-032 may claim:

- The focused summary first mismatch now carries compact hidden-operand
  fingerprints at the logits boundary.
- The committed focused receipt classifies the selected first mismatch as
  `generated_output_internal_logit_source_frontier_hidden_operand_drift`.
- The first mismatch is already present before output-head accumulation at the
  committed diagnostic boundary.

A770-032 must not claim:

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
