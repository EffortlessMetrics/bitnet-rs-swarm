<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-158 | #839 | `codex/slm-cpu-158-source-order-qproj-runtime-binding-gate` | Consume the SLM-CPU-157 paired receipt evidence and either add or precisely block the next default-disabled generation-time runtime binding for the exact Qwen3 source-order q_proj candidate. A valid slice must keep `eager_f32_candle` as the default runtime, require an explicit opt-in gate for any source-order q_proj runtime path, and preserve Qwen3 and Qwen2.5 strict CPU before/after receipt evidence for model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_proj numeric evidence, and `fallback_used=false`. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
