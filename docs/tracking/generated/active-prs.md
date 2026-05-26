<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-137 | #796 | `codex/slm-cpu-137-source-order-qproj-selector-hook` | Add or precisely block a default-disabled exact-tensor selector hook and receipt identity surface for the source-order Q8_0 q_proj matvec candidate. A valid slice may introduce an opt-in selector path for only layers.0.attention.q_proj.weight that preserves eager_f32_candle as default, records selected dense hook/kernel identity and q_proj numeric evidence in receipts, and remains unusable for behavior or performance claims until Qwen3/Qwen2.5 before/after strict CPU receipt pairs pass. If implementation is not safe, emit a machine-checkable blocker naming the exact missing Tensor API, selector integration point, trace identity field, receipt field, or hidden-state input binding. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
