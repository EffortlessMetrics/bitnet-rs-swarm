<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-218 | #971 | `codex/slm-cpu-218-no-bias-runtime-owner-boundary` | Consume the SLM-CPU-217 candidate runtime attachment boundary and either add the candidate runtime owner / candidate-on receipt emitter needed for same-callsite no-bias apply-linear evidence, or record the exact remaining ownership blocker. Candidate execution must remain disabled unless explicit gate identity, descriptor identity, strict candidate-off/on receipts, prompt/generated/text digests, runtime_api=cpu, selected_backend=cpu-rust, fallback=false, and default eager_f32_candle preservation are all present. No default runtime change, timing claim, allocation claim, speedup claim, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 change is allowed. |
