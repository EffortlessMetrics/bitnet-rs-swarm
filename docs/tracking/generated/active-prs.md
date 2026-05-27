<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-155 | #833 | `codex/slm-cpu-155-source-order-mapped-qproj-candidate` | Use the SLM-CPU-154 runtime row-mapping proof to update or precisely block the default-disabled source-order Qwen3 q_proj candidate accumulator for the exact Qwen3-0.6B Q8_0 `blk.0.attn_q.weight` / `layers.0.attention.q_proj.weight` path. A valid slice must preserve `eager_f32_candle` as the default runtime, keep the source-order candidate behind explicit trace/runtime-disabled gates, and capture before/after Qwen3 and Qwen2.5 receipts or a machine-checkable blocker proving why the mapped candidate cannot yet preserve generated IDs. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
