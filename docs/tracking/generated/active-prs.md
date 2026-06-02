<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-068 | #1293 | `codex/intel-a770/A770-068-one-more-qk256-replay-target` | Use the committed A770-067 manifest-bound focused QK256 replay packet to move exactly one additional dispatch_replay_missing Q/K/V target into selected-device Intel Arc A770 OpenCL replay when raw operands can be captured or located. The implementation must keep the proof to one case, one first-mismatch index, one kernel family, and one newly runnable manifest target; it must require fallback_used=false for any selected-device replay; and it must ledger the remaining missing operands as blockers. The work must not run broad answer corpora, model downloads, hardware matrices, full-workspace CI, Mac/Windows lanes, production QK256 dispatch policy changes, answer scoring or sampling changes, CPU/A770 answer parity promotion, strict answer readiness, broad quality, residency, speed, trusted-partial acceleration, or full BitNet inference claims. |
