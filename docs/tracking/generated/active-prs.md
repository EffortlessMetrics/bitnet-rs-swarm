<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-166 | #856 | `codex/slm-cpu-166-session-buffer-presizing` | Implement the SLM-CPU-165 session-level prompt/buffer pre-sizing boundary by using already-rendered/tokenized warm-session prompt metadata to reserve resident prompt/session buffers before the first prompt loop reset, emit aggregate and per-prompt receipt evidence for the pre-sizing source and capacity sufficiency, and preserve paired Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU receipt identity for model SHA, tokenizer authority, prompt/generated IDs, decoded text, selected CPU backend/kernel, dense hook identity where applicable, and `fallback_used=false`. The slice must not claim allocation reduction, speedup, sustained throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
