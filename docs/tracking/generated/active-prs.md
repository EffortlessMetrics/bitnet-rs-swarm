<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-140 | #802 | `codex/slm-cpu-140-qwen25-cache-restore-contract` | Define the exact Qwen2.5 Q8_0 cache restoration contract needed before source-order q_proj receipt capture can resume. The slice must name the required model path, SHA256, non-committed cache handling, receipt capture commands, source-order q_proj identity fields, and validation gates for Qwen3 Q8_0 plus Qwen2.5 Q8_0 fresh before/after strict CPU receipts. It must not commit models/** or target/** artifacts, claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
