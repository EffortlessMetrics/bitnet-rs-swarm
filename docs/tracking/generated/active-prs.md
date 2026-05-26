<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-135 | #792 | `codex/slm-cpu-135-source-order-qproj-behavior-gate` | Define the exact-model behavior gate required before the SLM-CPU-134 source-order Q8_0 q_proj matvec prototype can be considered for selector use. A valid slice must either capture strict before/after Qwen3 Q8_0 and Qwen2.5 Q8_0 CPU receipts proving unchanged model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, fallback_used=false, and accepted q_proj numeric evidence, or keep runtime selection disabled with the exact missing receipt or trace blocker. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
