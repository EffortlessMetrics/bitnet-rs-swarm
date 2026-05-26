<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-132 | #786 | `codex/slm-cpu-132-qwen3-qproj-payload-reorder-contract` | Define or precisely block the Qwen3 exact layer-0 attention.q_proj packed-Q8 payload reorder/runtime-shape contract after SLM-CPU-131: the slice may specify a machine-checkable transform from GGUF source-order Q8_0 payload shape [1024, 2048] into runtime matrix order [2048, 1024], or it may keep Qwen3 fail-closed with the exact missing proof. It must preserve Qwen3 and Qwen2.5 behavior oracles with model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, fallback_used=false, allocation-audit counters, and accepted q_proj numeric gate before any runtime selection claim. It must not claim allocation reduction, timing improvement, speedup, sustained throughput, default-runtime promotion, Q4/Q5 runtime support, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
