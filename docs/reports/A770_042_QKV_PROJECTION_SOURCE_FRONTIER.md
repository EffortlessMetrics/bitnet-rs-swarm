# A770-042 QKV Projection Source Frontier

## Scope

A770-042 adds a diagnostic-only QKV projection source frontier to the focused
AMD 5700X CPU AVX2 versus Intel Arc A770 OpenCL summary parity receipt.

The receipt now records compact per-projection fingerprints for the selected
layer Q/K/V projection input and output, plus QK256 dispatch deltas, CPU
hot-path deltas, and A770 OpenCL runtime deltas. The comparison uses those
fields only to route the next diagnostic boundary.

This does not change runtime math, dispatch policy, kernels, scoring, sampling,
or backend selection.

## Inputs

- CPU receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json`
- A770 receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/a770-opencl-summary-logits.json`
- Parity receipt:
  `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-vs-a770-summary-logits-parity.json`

## Live Result

```text
generated_output_qkv_projection_source_frontier.classification =
  generated_output_qkv_projection_source_frontier_dispatch_path_drift

case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

attention_output_source_classification =
  generated_output_attention_output_source_q_projection_drift

input.sha256_match = true
input.left_rms = 0.016132854576369458
input.right_rms = 0.016132854576369458
input.rms_abs_delta = 0

metadata_match = true
left_metadata.qk256_key = layers.0.attention.q_proj.weight.qk256_qs
right_metadata.qk256_key = layers.0.attention.q_proj.weight.qk256_qs
left_metadata.qk256_raw_tensor_present = true
right_metadata.qk256_raw_tensor_present = true

dispatch_match = false

left_dispatch.dispatch_delta.execution_claim = cpu_qk256_reference
left_dispatch.dispatch_delta.bitnet_linear_layers_total = 1
left_dispatch.dispatch_delta.bitnet_linear_layers_on_a770_opencl = 0
left_dispatch.cpu_hot_path_delta.qk256_i8s_scaled_scalar_invocations = 1
left_dispatch.cpu_hot_path_delta.selected_kernel =
  qk256-i2s-i8s-scaled-scalar-gemv

right_dispatch.dispatch_delta.execution_claim = a770_opencl_qk256_contribution
right_dispatch.dispatch_delta.bitnet_linear_layers_total = 1
right_dispatch.dispatch_delta.bitnet_linear_layers_on_a770_opencl = 1
right_dispatch.a770_opencl_runtime_delta.kernel_invocations = 1
right_dispatch.a770_opencl_runtime_delta.host_to_device_bytes = 1640960
right_dispatch.a770_opencl_runtime_delta.device_to_host_bytes = 10240

output.sha256_match = false
output.left_rms = 1.1568787420709823
output.right_rms = 1.1568787818792203
output.rms_abs_delta = 3.9808238039285015E-08

next_diagnostic =
  replay selected QKV projection CPU versus A770 dispatch policy
```

## Interpretation

The focused generated-output mismatch is still rooted at layer 0
`q_projection`, but the selected projection input matches exactly and both
lanes use the same raw QK256 tensor key. The first classified source split is
the projection dispatch path:

- CPU AVX2 records one scaled I2_S x I8_S scalar QK256 reference call.
- Intel Arc A770 OpenCL records one selected-device OpenCL QK256 contribution.

This routes the next diagnostic to replay the selected QKV projection CPU versus
A770 dispatch policy and numeric output. A770-042 does not by itself prove which
side is correct, and it does not justify changing runtime math without a
before/after receipt.

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
