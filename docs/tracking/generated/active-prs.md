<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-138 | #798 | `codex/slm-cpu-138-source-order-qproj-receipt-pairs` | Capture or precisely block Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict CPU receipt pairs that include the SLM-CPU-137 source-order Q8_0 q_proj candidate identity surface. A valid slice must preserve identical model SHA, GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, source-order q_proj candidate path/kernel/receipt identity, q_proj numeric evidence, and fallback_used=false before any source-order selector use. If capture is unsafe or unavailable, emit a machine-checkable blocker naming the exact missing model cache, CLI flag, receipt field, trace identity field, q_proj numeric evidence, or runtime input binding. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
