<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-136 | #794 | `codex/slm-cpu-136-source-order-qproj-receipt-pairs` | Capture or precisely block exact-model strict CPU before/after receipt pairs for considering the SLM-CPU-134 source-order Q8_0 q_proj matvec prototype as an opt-in selector candidate. A valid slice must cover both Qwen3 Q8_0 and Qwen2.5 Q8_0, preserving model SHA, GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, q_proj numeric evidence, and fallback_used=false. If receipt capture cannot be done safely, emit a machine-checkable blocker naming the missing selector hook, model cache, trace identity, CLI flag, or receipt field. Runtime selection must remain disabled by default and no allocation, timing, speedup, sustained-throughput, Q4/Q5, server/accelerator, Qwen3.5, or BitNet QK256 claim is allowed. |
