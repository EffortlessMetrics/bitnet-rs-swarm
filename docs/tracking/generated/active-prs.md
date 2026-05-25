<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-114 | #730 | `codex/slm-cpu-114-qnorm-fingerprint-receipt-pair` | Compare the SLM-CPU-113 Qwen3 Q8_0 q_norm_input fingerprint against a before/after strict CPU receipt pair and add the corresponding Qwen2.5 Q8_0 fingerprint artifact or record the exact blocker. A valid slice must preserve identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_norm_input tensor identity/fingerprint, and fallback_used=false across before/after receipts before any packed-Q8 sidecar runtime promotion. It must keep eager_f32_candle as the default runtime and must not claim allocation reduction, timing improvement, sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
