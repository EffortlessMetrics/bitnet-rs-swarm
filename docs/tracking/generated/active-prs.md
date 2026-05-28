<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-164 | #852 | `codex/slm-cpu-164-buffer-capacity-receipts` | Consume the SLM-CPU-163 Candle residual-output storage blocker and queue the next safe Kaby Lake SLM allocation slice at the warm-session prompt/session buffer capacity receipt boundary. A valid implementation must either make prompt/session buffer pre-sizing and reuse more receipt-visible, or record the exact missing before/after allocation evidence needed before changing runtime buffer behavior. Any runtime behavior change must preserve paired Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU receipt identity for model SHA, tokenizer authority, prompt/generated IDs, decoded text, selected CPU backend/kernel, dense hook identity where applicable, and `fallback_used=false`. It must not claim allocation reduction, speedup, sustained throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
