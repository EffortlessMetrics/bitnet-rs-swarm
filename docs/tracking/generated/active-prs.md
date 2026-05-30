<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-224 | #992 | `codex/slm-cpu-224-no-bias-strict-capture-artifact-pair` | Consume the SLM-CPU-223 strict artifact-capture blocker and either produce the validated candidate-off/candidate-on same-callsite strict capture artifact pair for Qwen3 and Qwen2.5 Q8_0 feed_forward.down_proj, or record the exact remaining capture prerequisite. A valid artifact pair must bind explicit gate identity, descriptor identity, FeedForward::apply_linear owner/callsite identity, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, candidate-off and candidate-on capture commands, model/backend identity, and default eager_f32_candle preservation. Candidate execution and normal runtime selection must remain disabled unless the strict capture pair is complete and separately gated, and this item must not claim timing, allocation, speedup, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 changes. |
