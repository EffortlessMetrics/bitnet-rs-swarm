<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-232 | #1019 | `codex/slm-cpu-232-no-bias-execution-capture-commands` | Consume the SLM-CPU-231 candidate execution receipt gate and define the concrete explicit-gate candidate-off and candidate-on execution capture commands or record the exact remaining blocker. A valid slice must make the capture commands receipt-visible and require the SLM-CPU-230 registry attachment identity, model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, selected path/kernel identity, candidate path/kernel identity, FeedForward::apply_linear owner/callsite identity, bias_present=false, and fallback=false. Candidate execution must remain disabled by default and the normal eager_f32_candle path must remain unchanged when the explicit gate is absent. |
