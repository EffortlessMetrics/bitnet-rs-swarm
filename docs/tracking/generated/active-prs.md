<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-124 | #770 | `codex/slm-cpu-124-qwen25-qproj-tensor-dump` | Capture a bounded Qwen2.5 Q8_0 attention.q_proj_output_pre_optional_qnorm tensor sample or full 896-f32 diagnostic dump for the strict CPU before/after pair, then classify the numeric mismatch behind the SLM-CPU-123 fingerprint delta before any packed-Q8 sidecar behavior proof, allocation reduction, timing improvement, or default-runtime promotion claim. A valid slice must preserve strict GGUF tokenizer authority, CPU backend, fallback_used=false, model SHA, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, boundary label, source tensor identity, shape/dtype, f32-le fingerprint, and must emit max_abs_diff plus first_differing_index or fail closed with the exact missing dump blocker. |
