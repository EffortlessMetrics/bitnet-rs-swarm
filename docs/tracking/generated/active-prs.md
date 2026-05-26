<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-130 | #782 | `codex/slm-cpu-130-qproj-selector-convergence-gate` | Resolve or explicitly classify the cross-model exact-tensor packed-Q8 selector mismatch after SLM-CPU-129: Qwen3 records allocation counters but stays on eager_f32_candle for the selected path, while Qwen2.5 records opt-in packed_q8_sidecar counter selection for the exact layer-0 attention.q_proj tensor. A valid slice must preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, fallback_used=false, allocation-audit counters, and the accepted q_proj numeric gate. It may narrow the selector or keep the sidecar diagnostic-only, but must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
