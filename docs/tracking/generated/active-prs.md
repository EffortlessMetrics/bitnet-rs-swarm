<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-128 | #778 | `codex/slm-cpu-128-qproj-allocation-timing-classification` | Use the SLM-CPU-125 exact-boundary numeric gate and the SLM-CPU-127 fresh Qwen3/Qwen2.5 before/after receipt pack to run or classify the next bounded exact-tensor packed-Q8 sidecar allocation or timing experiment. A valid slice must preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, fallback_used=false, and the accepted layer-0 attention.q_proj_output_pre_optional_qnorm numeric gate. It may record a bounded allocation/timing result only if matching receipts prove behavior equivalence; otherwise it must fail closed with the exact blocker. |
