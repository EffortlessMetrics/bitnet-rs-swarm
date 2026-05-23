# A770-035 Final Block Source Frontier

Status: diagnostic only

## Scope

A770-035 extends the focused A770 summary first-mismatch receipt after
A770-034 routed the `model.forward` source drift to prior-layer output before
final norm. This slice adds compact final transformer block source fingerprints
for the same focused case and keeps the claim boundary closed.

It does not change runtime math, QK256 dispatch policy, tokenizer behavior,
sampler behavior, prompt scoring, OpenCL routing, residency, or performance
claims.

## Live Receipt

Receipt:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json
```

Focused case:

```text
a770_summary_seed770024_keywords_014
```

Current frontier:

| Field | Value |
| --- | --- |
| `generated_output_model_forward_source_frontier.classification` | `generated_output_model_forward_source_frontier_prior_layer_output_drift` |
| `generated_output_final_block_source_frontier.classification` | `generated_output_final_block_source_frontier_block_input_drift` |
| `generated_output_final_block_source_frontier.next_diagnostic` | `capture penultimate transformer block source frontier` |
| `block_input_sha256_match` | `false` |
| `attention_output_sha256_match` | `false` |
| `post_attention_residual_sha256_match` | `false` |
| `feed_forward_output_sha256_match` | `false` |
| `block_output_sha256_match` | `false` |
| `block_input_rms_abs_delta` | `156.20258235589608` |
| `attention_output_rms_abs_delta` | `3.549200527364519` |
| `feed_forward_output_rms_abs_delta` | `17.84122973484648` |
| `block_output_rms_abs_delta` | `173.66825931763378` |

## Interpretation

The A770-034 prior-layer drift is already present at the final transformer
block input for the focused summary first mismatch. The downstream final-block
attention output, residual, FFN output, and block output fingerprints also
differ, but this receipt routes the first observed final-block boundary to the
block input, not to a final-block attention or FFN runtime policy.

The next diagnostic should move one block earlier and capture a compact
penultimate transformer block source frontier. This result does not justify a
runtime fix by itself.

## Claim Boundary

This report does not prove:

- CPU/A770 answer parity
- reference parity
- strict A770 answer readiness
- broad A770 answer quality
- BitNet inference fully works on A770
- official BitNet QK256 production semantics are fully proven
- activation quantization is GPU-resident
- selected attention is resident
- resident KV is proven
- full A770 residency is proven
- performance speedup is proven
- A770 trusted partial acceleration is claim-grade
