<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-234 | #1023 | `codex/slm-cpu-234-qwen25-artifact-prereq` | Consume the SLM-CPU-233 execution receipt blocker and either restore/verify the exact pinned Qwen2.5 Q8_0 local artifact prerequisite without committing a model binary, or record the exact storage/acquisition blocker that prevents fresh Qwen3/Qwen2.5 no-bias candidate-off/candidate-on execution receipts. A valid verification must prove the Qwen2.5 artifact SHA ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e and keep the SLM-CPU-232 capture command contract unchanged. No candidate execution, default runtime change, generated-ID preservation claim, timing improvement, allocation reduction, speedup, Q4/Q5 support, server/accelerator execution, Qwen3.5 support, or BitNet QK256 change is allowed. |
