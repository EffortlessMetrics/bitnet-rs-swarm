<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-107 | #692 | `codex/slm-cpu-107-qnorm-runtime-hook-impl` | Burn down the next q_norm_input_candle_tensor_boundary blocker after SLM-CPU-106 by adding a runtime-disabled hook and receipt tensor-identity surface for the exact Qwen3 Q8_0 layers.0.attention.q_proj.weight packed-Q8 sidecar path, or precisely blocking the missing API surface that prevents that hook. A valid slice must preserve eager_f32_candle as the default runtime, keep packed_q8_sidecar execution disabled unless Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict CPU receipts prove identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_norm_input tensor identity, and fallback_used=false, and must not claim allocation reduction, timing improvement, sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
