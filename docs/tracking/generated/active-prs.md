<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-117 | #749 | `codex/slm-cpu-117-qproj-output-pre-qnorm-hook` | Implement or precisely block the runtime-disabled `attention.q_proj_output_pre_optional_qnorm` diagnostic hook selected by SLM-CPU-115 and gated by SLM-CPU-116. A valid implementation must keep eager_f32_candle as the default runtime, keep packed_q8_sidecar unpromoted, expose only opt-in trace/receipt identity for the exact Qwen3 Q8_0 and Qwen2.5 Q8_0 artifacts, and preserve identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, boundary label, source tensor identity, f32-le tensor fingerprint availability or exact blocker, and fallback_used=false before any allocation or timing claim. The slice must include targeted tests or artifacts for the hook/comparator surface and must not claim sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
