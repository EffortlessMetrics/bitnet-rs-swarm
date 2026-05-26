<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-125 | #772 | `codex/slm-cpu-125-qproj-numeric-tolerance-gate` | Decide the next shared attention.q_proj_output_pre_optional_qnorm proof gate after SLM-CPU-124 classified the Qwen2.5 before/after mismatch as small f32 numeric drift. A valid slice must either define and validate a conservative numeric tolerance policy using accepted Qwen3 and Qwen2.5 strict CPU before/after evidence, or keep the boundary fail-closed with a machine-checkable blocker naming the next implementation fix. It must preserve strict GGUF tokenizer authority, CPU backend, fallback_used=false, model SHA, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, boundary label, source tensor identity, shape/dtype, f32-le fingerprint, max_abs_diff, and first_differing_index, and must not claim packed-Q8 sidecar behavior proof, allocation reduction, timing improvement, or default-runtime promotion unless the accepted numeric proof gate passes. |
