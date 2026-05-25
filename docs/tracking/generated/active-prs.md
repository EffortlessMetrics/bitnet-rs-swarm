<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-109 | #703 | `codex/slm-cpu-109-warm-session-qnorm-identity` | Burn down the SLM-CPU-108 blocker by adding the missing warm-session receipt field for the exact Qwen3 Q8_0 layers.0.attention.q_proj.weight packed-Q8 sidecar q_norm_input tensor identity, or record a narrower machine-checkable blocker if the current CLI/model/transformer API cannot safely expose it. A valid slice must preserve eager_f32_candle as the default runtime, keep packed_q8_sidecar disabled by default, and require before/after strict CPU receipts with identical model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, q_norm_input tensor identity, and fallback_used=false before claiming behavior preservation. |
