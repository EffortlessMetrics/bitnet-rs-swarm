<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-227 | #1005 | `codex/slm-cpu-227-no-bias-strict-capture-commands` | Consume the SLM-CPU-226 strict capture pair blocker and define the concrete candidate-off and candidate-on strict capture commands plus artifact schema for Qwen3 and Qwen2.5 Q8_0 feed_forward.down_proj. A valid slice must make the capture commands receipt-visible, bind explicit gate identity, descriptor identity, FeedForward::apply_linear owner/callsite identity, prompt/generated/text digests, model/backend identity, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, and default eager_f32_candle preservation, while keeping candidate execution disabled unless a later separately gated runtime PR enables it. This item must not claim generated-ID preservation for a candidate-on runtime experiment unless validated artifacts are present, and must not claim timing, allocation, speedup, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 changes. |
