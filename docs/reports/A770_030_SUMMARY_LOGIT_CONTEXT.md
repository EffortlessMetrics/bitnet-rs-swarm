# A770-030 Summary Logit Context

Date: 2026-05-22

## Scope

A770-030 records focused multi-step CPU AVX2 and Intel A770 OpenCL logits for
the one remaining generated-output divergent A770-029 readiness case:

```text
a770_summary_seed770024_keywords_014
```

This is diagnostic evidence only. It does not change runtime math, tokenizer
authority, model artifacts, QK256 dispatch, OpenCL kernels, backend routing, or
the seeded answer-readiness corpus.

## Inputs

The focused run uses the A770 seeded answer-readiness corpus v1.0.2 and the
official Microsoft BitNet 2B I2_S artifact:

- Corpus:
  `ci/quality/a770-bitnet-answer-readiness-corpus.yaml`
- Model:
  `E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`
- Tokenizer:
  `E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/tokenizer.json`
- Focused case:
  `a770_summary_seed770024_keywords_014`
- Logit dump:
  `--dump-logit-steps 24 --logits-topk 20`

## Receipts

- CPU AVX2:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- Intel A770 OpenCL:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Focused parity:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

## Live Result

Both lanes pass the focused answer-readiness quality rule, but the generated
answer remains divergent:

| Lane | Selected backend | Runtime API | Selected kernel | Fallback | Quality | Generated tokens |
| --- | --- | --- | --- | --- | --- | ---: |
| CPU AVX2 | `cpu-rust` | `cpu` | `i2_s-avx2-reference` | `false` | passed | 18 |
| Intel A770 OpenCL | `intel-a770-opencl` | `opencl` | `a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate` | `false` | passed | 15 |

CPU answer:

```text
Rust enables fast and safe software development by eliminating null pointer exceptions and memory leaks.
```

A770 answer:

```text
Rust enables fast and safe software development by preventing common programming errors.
```

## Parity Frontier

The focused parity receipt fails for the selected case:

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 1 |
| `summary.first_divergence.kind` | `generated_token_ids` |
| `generated_output_frontier.classification` | `generated_output_frontier_first_mismatch_has_logit_context` |
| `generated_output_logit_margin_frontier.classification` | `generated_output_logit_margin_frontier_opposite_argmax` |
| `logits_topk_frontier.classification` | `logits_topk_frontier_missing_context` |
| `generated_output_frontier.rows[0].first_mismatch_index` | 9 |
| `generated_output_frontier.rows[0].left_chosen_id_at_first_mismatch` | 40599 |
| `generated_output_frontier.rows[0].right_chosen_id_at_first_mismatch` | 27252 |
| `logits_topk_frontier.compared_step_count` | 15 |
| `logits_topk_frontier.logits_topk_mismatch_count` | 15 |
| `logits_topk_frontier.different_chosen_token_count` | 6 |
| `logits_topk_frontier.same_chosen_token_count` | 9 |
| `logits_topk_frontier.max_common_token_abs_delta` | 16.969619750976562 |

At the first generated-output mismatch, both lanes have cross-chosen logits:

| Field | Value |
| --- | --- |
| `left_chosen_id` | 40599 |
| `right_chosen_id` | 27252 |
| `left_margin_over_right_chosen_on_left` | 0.009021759033199572 |
| `right_margin_over_left_chosen_on_right` | 0.16872406005859375 |
| `left_chosen_delta_across_lanes` | -0.06170082092284801 |
| `right_chosen_delta_across_lanes` | 0.11604499816894531 |
| `right_margin_near_tie` | false |

The first mismatch is therefore no longer missing logit context. It is an
opposite-argmax generated-output mismatch with a near-tie-sized CPU-side
chosen-token margin and a larger A770-side chosen-token margin.

## Claim Boundary

A770-030 may claim:

- The remaining A770-029 generated-output divergent summary case has focused
  CPU AVX2 and Intel A770 OpenCL receipts with multi-step logit context under
  corpus v1.0.2.
- The focused parity receipt classifies the first generated-output mismatch as
  having logit context and opposite-argmax cross-chosen logits.

A770-030 must not claim:

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

## Next Frontier

The useful next diagnostic is to localize the summary-case first-mismatch
opposite-argmax source. Keep it diagnostic-only until a receipt pins whether
the mismatch is caused by:

- QK256 operand drift,
- output-head/logit accumulation drift,
- sampler/logit extraction policy,
- prompt-history serialization,
- or trace/capture context loss.
