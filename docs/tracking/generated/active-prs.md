<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-163 | #849 | `codex/slm-cpu-163-residual-output-storage-runtime-slice` | Consume the SLM-CPU-162 residual_block_output_storage_boundary shape/ownership contract and either implement the first behavior-preserving runtime allocation slice at that boundary, or record the exact remaining Candle/tensor API blocker preventing implementation. Any runtime behavior change must include paired Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU before/after receipts preserving model SHA, tokenizer authority, prompt/generated IDs, decoded text, selected CPU backend/kernel, dense hook identity where applicable, and `fallback_used=false`. It must not claim allocation reduction, speedup, sustained throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
