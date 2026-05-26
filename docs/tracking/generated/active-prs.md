<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-123 | #768 | `codex/slm-cpu-123-qwen25-qproj-fingerprint-root-cause` | Explain or eliminate the Qwen2.5 Q8_0 attention.q_proj_output_pre_optional_qnorm before/after f32-le fingerprint mismatch recorded by SLM-CPU-122 before any packed-Q8 sidecar behavior proof, allocation reduction, timing improvement, or default-runtime promotion claim. A valid slice must preserve strict GGUF tokenizer authority, CPU backend, fallback_used=false, model SHA, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, boundary label, source tensor identity, shape/dtype, and must fail closed if the fingerprint mismatch remains unexplained. |
