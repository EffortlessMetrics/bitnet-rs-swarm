# A770-029 Readiness Prompt Contract

Date: 2026-05-22

## Scope

A770-029 repairs the seeded A770 BitNet answer-readiness corpus prompt
contract for the five shared CPU/A770 content failures left by A770-028.

This is a corpus prompt/scoring contract update. It does not change runtime
math, tokenizer authority, model artifacts, QK256 dispatch, OpenCL kernels, or
backend routing.

## Corpus Changes

The corpus moves from version `1.0.1` to `1.0.2` and changes only the cases
that remained shared CPU/A770 content failures after A770-028:

| Case | Change |
| --- | --- |
| `a770_extract_seed770024_code_009` | Replace the code-extraction prompt that triggered fenced `R` package text with a `Package ID: R-42` prompt. |
| `a770_sort_seed770024_words_012` | Replace the free-form sorting prompt with an explicit two-list alphabetical choice. |
| `a770_yes_no_seed770024_true_015` | Replace the ambiguous `water_wet` yes/no case with an unambiguous true comparison and normalized yes/no scoring. |
| `a770_yes_no_seed770024_false_016` | Replace the sky-color yes/no case with an unambiguous false comparison and normalized yes/no scoring. |
| `a770_unknown_seed770024_unknown_020` | Replace the unknown-handling prompt with an explicit `Color: not stated` one-word prompt and normalized scoring. |

The two yes/no case IDs changed to reflect the new deterministic fixture
meaning. The other three cases keep their existing IDs and seed indices.

## Live Receipt Summary

Both receipts used the official Microsoft BitNet 2B I2_S artifact and external
tokenizer:

- Model:
  `E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`
- Tokenizer:
  `E:/Code/Rust/BitNet-rs/models/BitNet-b1.58-2B-4T/tokenizer.json`

| Lane | Selected backend | Runtime API | Selected kernel/runtime | Fallback | Passed | Failed |
| --- | --- | --- | --- | --- | ---: | ---: |
| CPU AVX2 | `cpu-rust` | `cpu` | `i2_s-avx2-reference` | `false` | 20 | 0 |
| Intel A770 OpenCL | `intel-a770-opencl` | `opencl` | `a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate` | `false` | 20 | 0 |

Receipts:

- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract/cpu-avx2-answer-readiness-prompt-contract.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract/a770-opencl-answer-readiness-prompt-contract.json`

## Parity Receipt Summary

The repaired prompt contract clears the seeded quality gate, but it does not
clear CPU/A770 parity.

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 20 |
| `logits_topk_frontier.classification` | `logits_topk_frontier_generated_output_divergence` |
| `logits_topk_mismatch_count` | 20 / 20 |
| `same_chosen_token_count` | 20 / 20 |
| `same_generated_output_count` | 19 / 20 |
| `generated_output_mismatch_count` | 1 / 20 |
| `generated_output_frontier.classification` | `generated_output_frontier_first_mismatch_missing_logit_context` |
| `generated_output_logit_margin_frontier.classification` | `generated_output_logit_margin_frontier_missing_context` |

Parity receipt:

- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract/cpu-avx2-vs-a770-answer-readiness-prompt-contract-parity.json`

The remaining generated-output divergent case is:

- `a770_summary_seed770024_keywords_014`

The full readiness receipts dump one logit step per case, so this receipt does
not contain first-mismatch logit context for the summary divergence.

## Claim Boundary

A770-029 may claim:

- The seeded A770 answer-readiness corpus v1.0.2 repaired the five shared
  prompt/scoring content failures left by A770-028.
- Fresh CPU AVX2 and Intel A770 OpenCL receipts both pass the repaired 20-case
  seeded corpus with `fallback_used=false`.

A770-029 must not claim:

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

Keep the tracks split:

1. CPU/A770 parity: record focused multi-step logits for
   `a770_summary_seed770024_keywords_014` under corpus v1.0.2, then localize the
   first-mismatch source before any runtime math change.
2. Answer usability: expand beyond this seeded 20-case corpus only after parity
   work has a stable baseline; do not use this corpus pass as a broad quality
   claim.
