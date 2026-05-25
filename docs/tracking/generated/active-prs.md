<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-116 | #742 | `codex/slm-cpu-116-qproj-output-pre-qnorm-hook` | Add or precisely block a runtime-disabled diagnostic hook and fail-closed receipt/comparator surface for the shared Qwen-family `attention.q_proj_output_pre_optional_qnorm` boundary selected by SLM-CPU-115. A valid slice must cover the exact Qwen3 Q8_0 and Qwen2.5 Q8_0 artifacts, preserve identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, boundary label, source tensor identity, f32-le tensor fingerprint availability or exact blocker, and fallback_used=false before any allocation or timing claim. It must keep eager_f32_candle as the default runtime and must not claim sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
