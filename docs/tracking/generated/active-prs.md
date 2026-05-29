<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-216 | #965 | `codex/slm-cpu-216-no-bias-candidate-on-evidence` | Capture the first explicit-gated no-bias candidate-on behavior evidence, or record the exact remaining runtime attachment blocker, for Qwen3 Q8_0 and Qwen2.5 Q8_0 feed_forward.down_proj. Candidate-on evidence must preserve identical model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, selected path/kernel, candidate path/kernel, and per-callsite descriptor identity versus candidate-off evidence. The default runtime must remain eager_f32_candle when the gate is absent. If candidate-on execution cannot be safely attempted, the blocker must name the missing CLI/env gate, apply_linear attachment point, receipt field, or runtime ownership boundary. No timing, allocation, speedup, sustained throughput, Q4/Q5, server/accelerator, Qwen3.5, or BitNet QK256 claim is allowed. |
