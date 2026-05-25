<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-061 | #714 | `codex/intel-a770/A770-061-production-lowered-operation-sequence` | Use the committed A770-060 production-kernel disassembly and focused replay context to inspect the lowered qk256_i2s_i8s_scaled_gemv operation sequence, classifying whether the production sequence preserves the expected QK256 scaling/math policy, requires replay instrumentation, remains missing context, or is clean, without changing production QK256 dispatch, answer scoring, sampling, CPU/A770 parity, strict answer readiness, broad A770 quality, residency, speed, trusted-partial acceleration, or full BitNet inference claims. |
