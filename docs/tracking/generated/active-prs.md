<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-221 | #983 | `codex/slm-cpu-221-no-bias-receipt-gated-candidate-execution-blocker` | Consume the SLM-CPU-220 candidate-off/on strict receipt boundary and either record the exact remaining blocker for a receipt-gated no-bias candidate execution attempt, or prove that the explicit gate identity, descriptor identity, FeedForward::apply_linear owner/callsite identity, candidate-off/on strict artifact pair, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, and default eager_f32_candle preservation are all present before keeping candidate execution runtime-disabled for a later enablement PR. No default runtime change, candidate execution by default, timing claim, allocation claim, speedup claim, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 change is allowed. |
