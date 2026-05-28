<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-162 | #847 | `codex/slm-cpu-162-residual-output-storage-contract` | Consume the SLM-CPU-161 allocation-audit-enabled Qwen3 Q8_0 and Qwen2.5 Q8_0 baselines and define or implement the next safe residual_block_output_storage_boundary contract before runtime allocation behavior changes. A valid slice must either produce a behavior-preserving caller-output-storage shape/ownership contract for the Candle tensor add/output boundary, or record the exact remaining blocker preventing that contract. It must preserve strict CPU receipt identity for model SHA, tokenizer authority, prompt/generated IDs, decoded text, selected CPU backend/kernel, dense hook identity where applicable, and `fallback_used=false`. It must not claim allocation reduction, speedup, sustained throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
