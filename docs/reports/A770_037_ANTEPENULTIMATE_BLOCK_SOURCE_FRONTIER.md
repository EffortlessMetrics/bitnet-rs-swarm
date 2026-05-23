# A770-037 Antepenultimate Block Source Frontier

## Scope

A770-037 adds a diagnostic-only antepenultimate transformer block source
frontier to the focused AMD 5700X CPU AVX2 versus Intel Arc A770 OpenCL summary
parity receipt. It does not change runtime math, dispatch, kernels, scoring, or
sampling.

## Inputs

- CPU receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- A770 receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Parity receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

## Live Result

```text
generated_output_antepenultimate_block_source_frontier.classification =
  generated_output_antepenultimate_block_source_frontier_block_input_drift

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
left_token_id = 40599
right_token_id = 27252

antepenultimate_block_source_context_available = true
block_input_sha256_match = false
attention_output_sha256_match = false
post_attention_residual_sha256_match = false
feed_forward_output_sha256_match = false
block_output_sha256_match = false

block_input_rms_abs_delta = 60.3220771044771
attention_output_rms_abs_delta = 24.25363257348323
post_attention_residual_rms_abs_delta = 97.99798241809276
feed_forward_output_rms_abs_delta = 5.431938802454624
block_output_rms_abs_delta = 109.00388247664432

next_diagnostic =
  capture pre-antepenultimate transformer block source frontier
```

## Interpretation

The focused generated-output mismatch is already present at the antepenultimate
transformer block input. A770-037 therefore keeps routing upstream instead of
promoting any attention, FFN, residual-add, QK256, OpenCL, or runtime math fix
from this evidence.

## Claim Boundary

This report is diagnostic-only.

It does not prove:

- CPU/A770 answer parity
- reference parity
- strict A770 answer readiness
- broad A770 answer quality
- full BitNet inference on A770
- official BitNet QK256 production semantics
- GPU-resident activation quantization
- selected attention residency
- resident KV
- full A770 residency
- performance speedup
- claim-grade trusted partial acceleration
