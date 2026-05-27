<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-157 | #837 | `codex/slm-cpu-157-source-order-qproj-receipt-pair` | Capture or precisely block the paired Qwen3 and Qwen2.5 strict CPU before/after receipt evidence required by the SLM-CPU-156 source-order q_proj selector gate. A valid slice must keep `eager_f32_candle` as the default runtime, keep the source-order q_proj candidate default-disabled, and prove or explicitly block unchanged model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_proj numeric evidence, and `fallback_used=false` for the selector path. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
