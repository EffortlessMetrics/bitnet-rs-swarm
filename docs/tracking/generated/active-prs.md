<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-119 | #757 | `codex/slm-cpu-119-qproj-output-receipt-pair-capture` | Capture the exact Qwen3 Q8_0 and Qwen2.5 Q8_0 before/after strict CPU receipt pairs with the SLM-CPU-117 `attention.q_proj_output_pre_optional_qnorm` hook active, or provide the exact missing GGUF/cache path, trace activation, receipt field, tensor fingerprint, or command transcript blocker. A valid capture must preserve identical model SHA, strict GGUF tokenizer authority, CPU backend, fallback_used=false, prompt IDs, generated IDs, decoded text, selected backend/kernel identity, dense hook identity, boundary label, source tensor identity, shape/dtype, and f32-le tensor fingerprint equality before any allocation or timing claim. |
