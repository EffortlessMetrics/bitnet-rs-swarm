<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-104 | #671 | `codex/slm-cpu-104-qnorm-materialization-boundary` | Resolve the post-SLM-CPU-103 typed q_norm/RoPE blocker into one precise next boundary for the exact Qwen3 Q8_0 `layers.0.attention.q_proj.weight` packed-Q8 sidecar path: either define a behavior-preserving typed q_norm kernel contract, define a behavior-preserving typed RoPE kernel contract, or select exactly one Candle materialization boundary (`q_norm_input`, `after_q_norm_before_rope`, or `after_q_rope_before_attention_scores`) with the required Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after receipt gates. The slice must name the accepted boundary or remaining blocker, preserve eager F32 default runtime, and make no allocation, timing, sustained-throughput, Q4/Q5, server/GPU/NPU/OpenVINO/UHD 620, Qwen3.5, or BitNet QK256 claim. |
