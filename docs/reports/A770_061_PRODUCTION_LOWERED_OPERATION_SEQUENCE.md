# A770-061 Production Lowered Operation Sequence

## Scope

A770-061 is a diagnostic-only follow-on to A770-060. A770-060 captured the
selected-device production `qk256_i2s_i8s_scaled_gemv` binary and `ocloc`
assembly. This slice inspects that committed assembly against the focused
QK256 replay context and classifies whether the production-lowered sequence is
enough to change production policy.

It does not change production QK256 dispatch, runtime math, answer scoring,
sampling, model loading, route policy, or any A770 quality, residency, speed,
trusted partial-acceleration, or full-inference claim.

## Receipt

The new receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-lowered-operation-sequence.json
```

The receipt records:

```text
proof_family = a770_opencl_qk256_production_lowered_operation_sequence
proof_stage = diagnostic_production_lowered_operation_sequence_classified
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
kernel_name = qk256_i2s_i8s_scaled_gemv
fallback_used = false
cpu_fallback_allowed = false
bitnet_inference = false
qk256_decode = false
production_qk256_policy_change = false
claim_allowed = false
diagnostic_only = true
```

## Classification

```text
a770_qk256_production_lowered_operation_sequence_requires_replay_instrumentation
```

The production assembly is present and reviewable:

```text
asm_path =
  ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-kernel-disassembly/ocloc-dump/.text.qk256_i2s_i8s_scaled_gemv.asm

asm_lines = 142
asm_fnv1a64 = 830e45d2eab44634
program_binary_fnv1a64 = e43804852ba461f8
direct_div_count = 0
math_inv_count = 1
f32_mul_count = 7
int_mul_count = 7
ugm_d8_load_count = 3
ugm_d32_store_count = 2
finite_guard_constant_count = 3
```

## Interpretation

The lowered sequence exposes the expected production regions:

```text
QK256 byte loads
shift/mask unpack
integer dot accumulation
adjusted-dot subtraction
final float scaling
output store
```

The final scaling region is not a source-shaped direct divide. It lowers to a
guarded reciprocal/multiply sequence with `math.inv`, no direct `div`
instruction, and no production strict-f32 barrier source context.

That is useful production-kernel context, but it does not prove that the
production sequence preserves the expected QK256 scaling/math policy at the bit
level. The focused replay chain still records:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj

policy_first_value_bits = 3215804926
host_div_then_mul_bits = 3215804926
device_output_bits = 3215804927
device_div_then_mul_bits = 3215804927
device_mul_then_div_bits = 3215804926
```

So A770-061 leaves production policy unchanged. The next useful diagnostic is
production replay instrumentation for:

```text
adjusted_dot
activation_scale
weight_scale
reciprocal-path intermediate bits
final scaled value bits
output store bits
```

## Validation

```text
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-production-lowered-operation-sequence.json -Raw | ConvertFrom-Json | Out-Null"
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Claim Boundary

This report makes no new claim:

- no production QK256 dispatch change;
- no answer scoring or sampling change;
- no CPU/A770 answer parity claim;
- no reference parity claim;
- no strict A770 answer-readiness claim;
- no broad A770 quality claim;
- no selected attention, resident KV, attention score, softmax, value-mix, or
  full-residency claim;
- no speedup or trusted partial-acceleration claim;
- no full BitNet inference claim.
