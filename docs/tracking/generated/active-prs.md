<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-115 | #740 | `codex/slm-cpu-115-qnorm-proof-next-boundary` | Resolve the SLM-CPU-114 q_norm-input proof blocker into the next safe evidence boundary. A valid slice must either add before/after receipt-pair f32-le q_norm_input fingerprint capture for the exact Qwen3 Q8_0 `layers.0.attention.q_proj.weight` boundary, or choose and document a comparable Qwen-family boundary that exists on both Qwen3 Q8_0 and Qwen2.5 Q8_0. It must preserve identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and fallback_used=false before any allocation or timing claim. It must keep eager_f32_candle as the default runtime and must not claim sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
