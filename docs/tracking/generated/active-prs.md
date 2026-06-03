<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-120 | #1451 | `codex/intel-a770/A770-120-layer17-v-proj-replay-target` | Use the committed A770-119 focused replay packet to convert layer-17 v_proj into selected-device Intel Arc A770 OpenCL replay. The implementation must keep the proof to one case, one first-mismatch index, one kernel family, and one newly runnable manifest target beyond A770-119; it must require fallback_used=false for any selected-device replay; and it must ledger the remaining missing operands as blockers. The work must not run broad answer corpora, model downloads, hardware matrices, full-workspace CI, Mac/Windows lanes, production QK256 dispatch policy changes, answer scoring or sampling changes, CPU/A770 answer parity promotion, strict answer readiness, broad quality, residency, speed, trusted partial acceleration, or full BitNet inference claims. |
