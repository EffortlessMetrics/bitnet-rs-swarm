<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-235 | #1025 | `codex/slm-cpu-235-no-bias-execution-receipts` | Consume the SLM-CPU-234 Qwen2.5 artifact prerequisite and capture fresh strict Qwen3 and Qwen2.5 Q8_0 candidate-off/candidate-on no-bias execution receipts for feed_forward.down_proj through the SLM-CPU-232 command contract. Valid receipts must preserve model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, selected path/kernel identity, candidate path/kernel identity, FeedForward::apply_linear owner/callsite identity, bias_present=false, and fallback=false. Candidate execution must remain disabled by default and the normal eager_f32_candle path must remain unchanged when the explicit gate is absent. No timing improvement, allocation reduction, speedup, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 change is allowed. |
