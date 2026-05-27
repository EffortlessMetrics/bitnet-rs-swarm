<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-154 | #831 | `codex/slm-cpu-154-source-order-row-mapping-proof` | Use the SLM-CPU-153 Candle row-slice comparison to prove or precisely block the source-order payload-to-runtime row mapping for the exact Qwen3-0.6B Q8_0 `blk.0.attn_q.weight` / `layers.0.attention.q_proj.weight` path. A valid slice must either define a machine-checkable mapping from source-order GGUF Q8 block/offset indices to Candle runtime row-major q_proj indices, with selected-row evidence that reconciles source-order payload terms to the Candle-materialized q_proj row slice, or emit a blocker naming the missing GGUF tensor-layout proof, Q8 block regrouping rule, row/column transpose contract, or comparison hook. It must preserve `eager_f32_candle` as the default runtime and must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
