# A770-041 Layer 0 Attention Output Source Frontier

## Scope

A770-041 adds a diagnostic-only attention output source frontier to the focused
AMD 5700X CPU AVX2 versus Intel Arc A770 OpenCL summary parity receipt.

The receipt now records compact layer-local fingerprints for the attention input,
Q/K/V projections, reshaped heads, Q/K norm outputs, RoPE outputs, KV context,
expanded KV, raw scores, probabilities, value mix, output projection input,
optional sub-layernorm output, and final attention output. The comparison only
uses these fingerprints to route the next diagnostic boundary.

This does not change runtime math, dispatch, kernels, scoring, sampling, or
backend selection.

## Inputs

- CPU receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- A770 receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Parity receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

## Live Result

```text
generated_output_attention_output_source_frontier.classification =
  generated_output_attention_output_source_frontier_q_projection_drift

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
left_token_id = 40599
right_token_id = 27252

target_layer_idx = 0
attention_output_source_context_available = true
left_attention_output_source_count = 30
right_attention_output_source_count = 30

attention_input.sha256_match = true
attention_input.left_rms = 0.016132854576369458
attention_input.right_rms = 0.016132854576369458
attention_input.rms_abs_delta = 0

q_projection.sha256_match = false
q_projection.left_rms = 1.1568787420709823
q_projection.right_rms = 1.1568787818792203
q_projection.rms_abs_delta = 3.9808238039285015E-08

next_diagnostic =
  replay earliest divergent block QKV projection source
```

## Interpretation

The focused generated-output mismatch remains rooted inside layer 0 attention
output, but the layer 0 attention input itself matches between CPU AVX2 and A770
OpenCL receipts. The first divergent compact source field is the layer 0
`q_projection` fingerprint.

A770-041 therefore routes the next diagnostic to replay the earliest divergent
block QKV projection source. It does not justify a runtime math, OpenCL dispatch,
QK256, attention score, softmax, value-mix, output projection, residual, FFN,
or sampling change by itself.

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
