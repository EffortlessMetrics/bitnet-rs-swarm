<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-129 | #780 | `codex/slm-cpu-129-repeated-allocation-audit-receipts` | Collect or explicitly block repeated warm-session before/after receipts with allocation-audit counters for the exact-tensor packed-Q8 sidecar boundary after SLM-CPU-128 classified the single-sample timing evidence as mixed. A valid slice must preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, fallback_used=false, and the accepted layer-0 attention.q_proj_output_pre_optional_qnorm numeric gate. It must require at least repeated behavior-equivalent receipts before claiming allocation reduction or timing improvement, and must fail closed if allocation-audit counters are unavailable or behavior drifts. |
