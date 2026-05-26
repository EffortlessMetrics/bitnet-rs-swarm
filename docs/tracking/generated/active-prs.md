<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-131 | #784 | `codex/slm-cpu-131-qwen3-qproj-payload-order-proof` | Classify or implement the Qwen3 exact layer-0 attention.q_proj packed-Q8 payload-order proof needed after SLM-CPU-130: Qwen3 must remain on eager_f32_candle unless a machine-checkable payload reorder/runtime-shape proof shows sidecar_payload_order_matches_runtime_shape=true for the same model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, fallback_used=false, allocation-audit counters, and accepted q_proj numeric gate. The slice may add proof artifacts or code for payload-order verification, but must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
