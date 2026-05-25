<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-110 | #710 | `codex/slm-cpu-110-qnorm-receipt-pair-after-identity` | Collect the Qwen3 Q8_0 before/after strict CPU receipt pair now that warm-session receipts emit dense_q8_hook.q_norm_input_tensor_identity, or record the next narrower machine-checkable blocker. A valid proof must preserve identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_norm_input tensor identity, and fallback_used=false across before/after receipts before claiming behavior preservation. It must keep eager_f32_candle as the default runtime, keep packed_q8_sidecar disabled by default unless all proof gates pass, and must not claim allocation reduction, timing improvement, sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes. |
