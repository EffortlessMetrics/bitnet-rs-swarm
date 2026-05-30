<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-233 | #1021 | `codex/slm-cpu-233-no-bias-execution-receipts` | Consume the SLM-CPU-232 candidate-off/candidate-on execution capture command contract and either capture or precisely block fresh strict Qwen3 and Qwen2.5 Q8_0 no-bias candidate-off/candidate-on execution receipts for feed_forward.down_proj. A valid capture must preserve model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, selected path/kernel identity, candidate path/kernel identity, FeedForward::apply_linear owner/callsite identity, bias_present=false, and fallback=false. Candidate execution must remain disabled by default and the normal eager_f32_candle path must remain unchanged when the explicit gate is absent. |
