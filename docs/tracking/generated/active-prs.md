<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-231 | #1017 | `codex/slm-cpu-231-no-bias-candidate-execution-receipts` | Consume the SLM-CPU-230 runtime hook attachment boundary and capture or precisely block a fresh explicit-gate no-bias candidate-off/candidate-on execution receipt pair for Qwen3 and Qwen2.5 Q8_0 feed_forward.down_proj. A valid slice must preserve prompt IDs, generated IDs, decoded text, model SHA, tokenizer authority, runtime_api=cpu, selected_backend=cpu-rust, selected kernel/path identity, descriptor identity, DenseLinearRuntimeHookRegistry attachment identity, FeedForward::apply_linear owner/callsite identity, bias_present=false, prompt/generated/text digests, and fallback=false. Candidate execution must remain disabled by default and must not affect the normal eager_f32_candle path when the explicit gate is absent. |
