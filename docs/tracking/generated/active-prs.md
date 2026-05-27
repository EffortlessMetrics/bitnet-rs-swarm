<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-156 | #835 | `codex/slm-cpu-156-source-order-qproj-selector-gate` | Use the SLM-CPU-155 mapped-candidate evidence to add or precisely block the next default-disabled Qwen3 source-order q_proj selector/runtime gate. A valid slice must preserve `eager_f32_candle` as the default runtime unless fresh Qwen3 and Qwen2.5 strict CPU receipts prove unchanged model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_proj numeric evidence, and `fallback_used=false`. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion without receipts, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
