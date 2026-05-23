# A770-040 Transformer Block Source Stack Frontier

## Scope

A770-040 adds a diagnostic-only transformer block source stack frontier to the
focused AMD 5700X CPU AVX2 versus Intel Arc A770 OpenCL summary parity receipt.
It records compact per-block fingerprints for block input, attention output,
post-attention residual, FFN output, and block output so the receipt can classify
the earliest divergent transformer block and field in one pass.

This does not change runtime math, dispatch, kernels, scoring, or sampling.

## Inputs

- CPU receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- A770 receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Parity receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

## Live Result

```text
generated_output_transformer_block_source_stack_frontier.classification =
  generated_output_transformer_block_source_stack_frontier_attention_output_drift

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
left_token_id = 40599
right_token_id = 27252

transformer_block_source_stack_context_available = true
compared_block_count = 30
earliest_divergent_block_index = 0
earliest_divergent_layer_idx = 0

block_input_sha256_match = true
attention_output_sha256_match = false
post_attention_residual_sha256_match = false
feed_forward_output_sha256_match = false
block_output_sha256_match = false

block_input_rms_abs_delta = 0
attention_output_rms_abs_delta = 3.434592599216302E-08
post_attention_residual_rms_abs_delta = 6.954555153981801E-09
feed_forward_output_rms_abs_delta = 1.0651849233767052E-06
block_output_rms_abs_delta = 1.0528472298432234E-06

next_diagnostic =
  replay earliest divergent transformer block attention output source
```

## Interpretation

The focused generated-output mismatch is not introduced by the layer-0 block
input. The earliest divergent field in the compact transformer block source
stack is layer-0 attention output.

A770-040 therefore routes the next diagnostic to replay the earliest divergent
transformer block attention output source. It does not justify any runtime math,
OpenCL dispatch, QK256, score-input, attention, FFN, residual-add, or sampling
change by itself.

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
