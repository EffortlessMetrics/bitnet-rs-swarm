<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-222 | #985 | `codex/slm-cpu-222-no-bias-strict-receipt-artifact-pair` | Consume the SLM-CPU-221 receipt-gated candidate execution boundary and either capture a real same-callsite candidate-off/candidate-on strict receipt artifact pair for bounded no-bias apply-linear evidence, or record the exact remaining blocker. The artifact pair must bind explicit gate identity, descriptor identity, FeedForward::apply_linear owner/callsite identity, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, and default eager_f32_candle preservation for Qwen3 and Qwen2.5 Q8_0. Candidate execution must remain disabled unless all identity evidence is present, and this item must not claim timing, allocation, speedup, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 changes. |
