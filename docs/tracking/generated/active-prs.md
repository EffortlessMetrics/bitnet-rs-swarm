<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-230 | #1013 | `codex/slm-cpu-230-no-bias-runtime-hook-attachment` | Consume the SLM-CPU-229 runtime-attempt blocker and add or precisely block the receipt-bound no-bias selector runtime-hook registry attachment for Qwen3 and Qwen2.5 Q8_0 feed_forward.down_proj. A valid slice must attach a descriptor identity to DenseLinearRuntimeHookRegistry only when the explicit gate, SLM-CPU-228 strict capture artifact pair identity, prompt/generated/text digests, model SHA, tokenizer authority, runtime_api=cpu, selected_backend=cpu-rust, selected kernel identity, FeedForward::apply_linear owner/callsite identity, bias_present=false, and fallback=false are present. Candidate execution and default runtime selection must remain disabled unless a later separately gated PR proves fresh candidate-off/candidate-on generated-ID and decoded-text preservation receipts. |
