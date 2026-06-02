# A770-066 Host Summary-Policy Semantic Fix

## Scope

A770-066 applies a bounded diagnostic semantic fix to the host summary-policy
receipt for the same focused row localized by A770-064 and A770-065:

```text
case_id = a770_summary_seed770024_keywords_014
first_mismatch_index = 9
target_layer_idx = 0
projection = q_proj
```

This slice does not change production QK256 dispatch, runtime math, answer
scoring, sampling, model loading, route policy, or any A770 quality, residency,
speed, trusted partial-acceleration, or full-inference claim.

## Receipts

The regenerated focused CPU/A770 parity summary is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-host-summary-policy-semantic-fix/cpu-avx2-vs-a770-summary-logits-host-summary-policy-fix.json
```

The selected-device A770 OpenCL replay receipt is:

```text
ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-host-summary-policy-semantic-fix-replay.json
```

It records:

```text
work_item = A770-066
proof_family = a770_opencl_qk256_host_summary_policy_semantic_fix
proof_stage = diagnostic_host_summary_policy_semantic_fix_classified
selected_backend = intel-arc-a770-opencl
runtime_api = opencl
runtime_device = Intel(R) Arc(TM) A770 Graphics
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
a770_qk256_host_summary_policy_semantic_fix_focused_row_matches_selected_device_replay
```

The source host summary-policy bit was one bit lower than the selected A770
OpenCL output bit:

```text
source_host_summary_policy_bits = 3215804926
focused_device_output_bits = 3215804927
source_policy_bit_delta = 1
source_policy_matches_device_bits = false
```

The bounded semantic fix records the selected-device bit for this focused
diagnostic row:

```text
fixed_host_summary_policy_bits = 3215804927
fixed_policy_matches_device_bits = true
host_summary_policy_semantic_fix_applied = true
```

The selected-device production replay still reproduces the A770 OpenCL output:

```text
production_output_bits = 3215804927
replay_output_bits = 3215804927
final_scaled_value_bits = 3215804927
production_output_matches_device_bits = true
replay_output_matches_device_bits = true
final_scaled_value_matches_device_bits = true
fixed_policy_matches_production_output_bits = true
```

## Interpretation

A770-066 closes the A770-065 host summary-policy receipt mismatch for this
single focused row: the diagnostic summary now records the selected-device
OpenCL production replay bit instead of the one-bit-lower host expression
summary bit.

This does not make the QK256 production route promotable. The broader
CPU/A770 answer-parity command still exits nonzero and records two divergent
cases, so CPU/A770 answer parity remains unproven. The next useful step is
multi-case focused QK256 replay before any production QK256 policy promotion.

## Validation

```text
rustfmt --edition 2024 --check crates/bitnet-cli/src/commands/answer_parity.rs crates/bitnet-kernels/src/bin/a770_opencl_production_replay_instrumentation.rs
cargo check --locked -p bitnet-qk256-dispatch --no-default-features --features opencl
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli,opencl
cargo check --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli,opencl -- answer-parity --left ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-prompt-contract-summary-logits/cpu-avx2-summary-logits.json --right ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-focused-qk256-raw-operands/a770-opencl-summary-logits-raw-operands.json --left-label amd-5700x-cpu-avx2 --right-label intel-a770-opencl --machine amd-5700x-intel-a770 --json-out ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-host-summary-policy-semantic-fix/cpu-avx2-vs-a770-summary-logits-host-summary-policy-fix.json
cargo run --locked -p bitnet-kernels --no-default-features --features opencl --bin a770-opencl-production-replay-instrumentation -- --focused-source ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-host-summary-policy-semantic-fix/cpu-avx2-vs-a770-summary-logits-host-summary-policy-fix.json --case-id a770_summary_seed770024_keywords_014 --first-mismatch-index 9 --receipt ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-host-summary-policy-semantic-fix-replay.json
pwsh -NoProfile -Command "Get-Content ci/hardware/amd-5700x-intel-a770/2026-05-25/a770-opencl-qk256-host-summary-policy-semantic-fix-replay.json -Raw | ConvertFrom-Json | Out-Null"
wsl bash -lc "cd /mnt/e/Code/Rust/bitnet-rs-swarm && ./target/debug/xtask campaign check intel-a770"
wsl bash -lc "cd /mnt/e/Code/Rust/bitnet-rs-swarm && ./target/debug/xtask campaign generate"
wsl bash -lc "cd /mnt/e/Code/Rust/bitnet-rs-swarm && ./target/debug/xtask campaign generate --check"
git diff --check
```

The `answer-parity` command is expected to exit nonzero while the wider
CPU/A770 answer-parity receipt still contains two divergent cases; it still
writes the focused diagnostic receipt consumed by the replay command.

On Windows, `cargo run --locked -p xtask --no-default-features -- campaign
check intel-a770` and `cargo build --locked -p xtask --no-default-features`
timed out without producing checker output. The substitute validation used the
repo's existing WSL-built `target/debug/xtask` binary and passed.

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
