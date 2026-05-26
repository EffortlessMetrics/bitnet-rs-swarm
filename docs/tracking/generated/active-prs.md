<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-127 | #776 | `codex/slm-cpu-127-qwen25-fresh-receipt-prereq` | Restore or fetch the exact Qwen2.5-0.5B-Instruct Q8_0 GGUF artifact required by SLM-CPU-126, verify its pinned SHA256, and collect or explicitly block the fresh Qwen3/Qwen2.5 before/after strict CPU receipt pack needed before any allocation or timing experiment can claim improvement. A valid slice must not commit model binaries, must preserve strict GGUF tokenizer authority, selected CPU backend/kernel identity, prompt IDs, generated IDs, decoded text, dense hook identity, fallback_used=false, and the SLM-CPU-125 q_proj-output numeric gate, and must fail closed if the artifact or receipt pack cannot be produced. |
