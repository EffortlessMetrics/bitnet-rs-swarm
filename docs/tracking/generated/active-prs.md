<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-067 | #1276 | `codex/intel-a770/A770-067-multi-case-focused-qk256-replay` | Use the committed A770-066 host summary-policy semantic-fix receipt and existing CPU/A770 answer-readiness parity receipts to build a manifest-bound multi-case focused QK256 replay packet. The packet must identify every focused replay target, capture or consume raw focused operands for at least one additional replay target beyond the A770-064 single q_proj row when available, run selected-device Intel Arc A770 OpenCL production replay for each manifest target that has operands, and explicitly ledger any missing operand/source receipt as a blocker. The work must not change production QK256 dispatch policy, answer scoring, sampling, CPU/A770 parity, strict answer readiness, broad A770 quality, residency, speed, trusted-partial acceleration, or full BitNet inference claims. |
