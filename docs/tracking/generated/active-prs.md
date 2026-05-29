<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-223 | #989 | `codex/slm-cpu-223-no-bias-strict-artifact-capture` | Consume the SLM-CPU-222 strict receipt artifact-pair boundary and either produce the missing candidate-off/candidate-on same-callsite strict artifact pair for Qwen3 and Qwen2.5 Q8_0 feed_forward.down_proj, or record the exact capture blocker. A valid artifact pair must bind explicit gate identity, descriptor identity, FeedForward::apply_linear owner/callsite identity, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, and default eager_f32_candle preservation. Candidate execution must remain disabled unless the artifact pair is complete and separately gated, and this item must not claim timing, allocation, speedup, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 changes. |
