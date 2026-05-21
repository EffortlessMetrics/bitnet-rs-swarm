<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-071 | #161 | `codex/slm-cpu-071-post070-timing-gate` | Regenerate or ingest the real i5-8250U Qwen3-0.6B Q8_0 before/after warm-session artifact pack after the SLM-CPU-070 block-local packed Q8_0 matvec prototype, then classify whether the opt-in exact-tensor path preserves behavior and improves, regresses, or remains inconclusive on bounded timing. The artifact must compare eager_f32_candle against the opt-in packed_q8_sidecar path for the exact `layers.0.attention.q_proj.weight` tensor and prove identical model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false before any timing interpretation. The slice must keep eager F32 Candle as the default runtime and must not claim speedup unless the bounded artifact supports it. |
