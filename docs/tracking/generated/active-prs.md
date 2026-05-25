<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-105 | #675 | `codex/slm-cpu-105-qnorm-input-proof-gate` | Prove or precisely block the q_norm_input_candle_tensor_boundary selected by SLM-CPU-104 for the exact Qwen3 Q8_0 `layers.0.attention.q_proj.weight` packed-Q8 sidecar path. A valid slice must either add before/after strict CPU receipt evidence for Qwen3 Q8_0 and Qwen2.5 Q8_0 showing identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, and fallback_used=false across the selected boundary, or emit a machine-checkable blocker naming the missing runtime hook, receipt field, comparator, tensor identity, accumulator-order, or artifact gap. The slice must keep the default runtime `eager_f32_candle`, must not promote packed_q8_sidecar, and must not claim allocation reduction, timing improvement, sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
