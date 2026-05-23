# A770-033 Hidden-State Source Frontier

Date: 2026-05-23

## Scope

A770-033 adds compact hidden-state source context to the focused
A770-030/A770-031/A770-032 summary divergence receipts:

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
| `generated_output_internal_logit_source_frontier.classification` | `generated_output_internal_logit_source_frontier_hidden_operand_drift` |
| `generated_output_hidden_state_source_frontier.classification` | `generated_output_hidden_state_source_frontier_forward_output_drift` |

The new frontier records:

| Field | Value |
| --- | --- |
| `first_mismatch_index` | 9 |
| `left_chosen_id` | 40599 |
| `right_chosen_id` | 27252 |
| `forward_output_sha256_match` | `false` |
| `last_hidden_sha256_match` | `false` |
| `left_forward_output_sha256_f32_le` | `224d8c1fb08d05d0a3ce05e8e8491cbcc46da8a52f25461c3204a3c37993f650` |
| `right_forward_output_sha256_f32_le` | `701a390a1879edd1b1df381a90098fec35e76288a38cd40a89fc187a02065114` |
| `left_forward_output_shape` | `[1, 1, 2560]` |
| `right_forward_output_shape` | `[1, 1, 2560]` |
| `left_forward_output_rms` | 0.07708468808572078 |
| `right_forward_output_rms` | 0.07870732156492034 |
| `forward_output_rms_abs_delta` | 0.0016226334791995672 |
| `left_last_hidden_shape` | `[1, 2560]` |
| `right_last_hidden_shape` | `[1, 2560]` |
| `last_hidden_rms_abs_delta` | 0.0016226334791995672 |

That moves the first-mismatch source upstream of last-hidden extraction: the
`model.forward` output already differs between CPU AVX2 and Intel A770 OpenCL
at the selected step. Last-hidden extraction preserves the same fingerprint
delta, so the next useful boundary is inside the model forward path.

## Next Frontier

The useful next diagnostic is:

```text
capture final norm and prior layer output fingerprints before model.forward output
```

That next slice should stay diagnostic-only until it separates:

- final norm output drift;
- prior layer output drift;
- residual contribution drift;
- or missing internal trace context.

## Claim Boundary

A770-033 may claim:

- The focused summary first mismatch now carries compact model.forward output
  and last-hidden fingerprints at the logits boundary.
- The committed focused receipt classifies the selected first mismatch as
  `generated_output_hidden_state_source_frontier_forward_output_drift`.
- The first mismatch is already present in `model.forward` output before
  last-hidden extraction at the committed diagnostic boundary.

A770-033 must not claim:

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
