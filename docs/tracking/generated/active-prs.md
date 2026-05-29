<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-219 | #974 | `codex/slm-cpu-219-no-bias-same-callsite-receipt-emitter` | Consume the SLM-CPU-218 runtime owner boundary and either wire the same-callsite candidate-off/candidate-on receipt emitter needed for bounded no-bias apply-linear evidence, or record the exact receipt-emission blocker. Candidate execution must remain disabled unless explicit gate identity, descriptor identity, same-callsite candidate-off/on strict receipts, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, and default eager_f32_candle preservation are all present. No default runtime change, timing claim, allocation claim, speedup claim, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 change is allowed. |
