<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-121 | #764 | `codex/slm-cpu-121-qwen3-preboundary-allocation` | Resolve or precisely isolate the remaining Qwen3 Q8_0 167772160-byte allocation failure that occurs after prompt IDs are emitted but before a post-guard receipt or layer-0 attention.q_proj_output_pre_optional_qnorm fingerprint is written. A valid slice must preserve the SLM-CPU-120 fail-closed packed-Q8 sidecar payload-order guard, keep eager_f32_candle as the default runtime, preserve strict GGUF tokenizer authority, CPU backend, fallback_used=false where receipts exist, and must not claim q_proj sidecar behavior proof, allocation reduction, timing improvement, default-runtime promotion, sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes until accepted Qwen3 and Qwen2.5 before/after proof gates pass. |
