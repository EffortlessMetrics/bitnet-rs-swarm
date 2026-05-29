<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-217 | #969 | `codex/slm-cpu-217-no-bias-candidate-runtime-attachment` | Consume the SLM-CPU-216 candidate-on behavior evidence gate and either wire a fail-closed explicit no-bias apply-linear candidate runtime attachment for Qwen3 Q8_0 and Qwen2.5 Q8_0 feed_forward.down_proj, or record the exact missing runtime ownership blocker. Candidate execution must remain disabled unless an explicit CLI/env gate, descriptor identity, strict receipt fields, candidate-off/on evidence, and default eager_f32_candle preservation are all present. No default runtime change, timing claim, allocation claim, speedup claim, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 change is allowed. |
