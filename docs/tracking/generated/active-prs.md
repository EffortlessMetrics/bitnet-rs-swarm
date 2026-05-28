<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-167 | #858 | `codex/slm-cpu-167-presizing-receipt-comparison` | Consume the SLM-CPU-166 prompt/session buffer pre-sizing implementation and the committed SLM-CPU-161 allocation-audit baselines, then capture or precisely block a committed Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU receipt comparison that proves unchanged model SHA, tokenizer authority, prompt/generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and `fallback_used=false` across the pre-sizing boundary. The slice must classify whether prompt/session buffer pre-sizing changed allocation-audit counters or only made capacity reuse receipt-visible, and it must name the next safe allocation hotspot before any runtime optimization claim. It must not claim allocation reduction, speedup, sustained throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
