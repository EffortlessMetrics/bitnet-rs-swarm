<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-122 | #766 | `codex/slm-cpu-122-qproj-output-proof-refresh` | Resume the shared attention.q_proj_output_pre_optional_qnorm behavior proof after SLM-CPU-121 makes the Qwen3 Q8_0 post-guard receipt and layer-0 fingerprint reachable. A valid slice must compare accepted Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict CPU receipts at the shared q_proj-output boundary, preserve strict GGUF tokenizer authority, CPU backend, fallback_used=false, model SHA, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, boundary label, source tensor identity, shape/dtype, and must fail closed on missing or mismatched f32-le fingerprints. It must not claim packed-Q8 sidecar behavior proof, allocation reduction, timing improvement, default-runtime promotion, sustained throughput, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes unless the accepted cross-model proof gates pass. |
