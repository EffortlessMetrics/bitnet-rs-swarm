<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-159 | #841 | `codex/slm-cpu-159-kaby-performance-dashboard` | Formalize the i5-8250U Kaby Lake SLM performance dashboard from existing strict receipts. The slice must summarize the Qwen3-0.6B Q8_0 appliance profile, Qwen2.5-0.5B Q8_0 second-model sanity status, SmolLM2 fail-closed status, 1/2/4/8-thread envelope, selected operator thread count, cold load, warm-session load-once, prefill, first-token, steady decode, per-prompt timing, allocation-buffer reuse fields where available, resident memory, storage context, thermal/power availability, and the next safe optimization targets. It may make only bounded evidence statements backed by committed receipts. It must not claim sustained throughput, broad chat quality, Q4/Q5 support, default-runtime promotion, source-order/packed-Q8 speedup, server/accelerator execution, Qwen3.5, or BitNet QK256 behavior. |
