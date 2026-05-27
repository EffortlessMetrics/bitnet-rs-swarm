<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-160 | #843 | `codex/slm-cpu-160-allocation-buffer-reuse-boundary` | Consume the SLM-CPU-159 Kaby Lake performance dashboard and implement or precisely block the first safe allocation/buffer-reuse optimization boundary for Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU runs. A valid implementation must preserve model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and `fallback_used=false` in before/after receipts. A valid blocker must identify the exact missing receipt, trace, or shape contract needed before changing runtime allocation behavior. The slice must not claim speedup, sustained throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
