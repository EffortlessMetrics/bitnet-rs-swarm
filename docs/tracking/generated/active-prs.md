<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-118 | #753 | `codex/slm-cpu-118-qproj-output-receipt-pair` | Capture or precisely block the real Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU before/after receipt pairs for the shared `attention.q_proj_output_pre_optional_qnorm` boundary added by SLM-CPU-117. A valid slice must use the exact accepted model artifacts, strict GGUF tokenizer authority, CPU backend, fallback_used=false, identical prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, boundary label, source tensor identity, shape/dtype, and f32-le tensor fingerprint availability or exact blocker. It must keep eager_f32_candle as the default runtime, keep packed_q8_sidecar unpromoted, and must not claim allocation reduction, timing improvement, sustained 8250U throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
