<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-026 | #336 | `codex/intel-a770/A770-026-answer-readiness-failure-frontier` | Classify the committed A770-025 seeded answer-readiness failures and CPU/A770 parity frontier so shared prompt/model/scoring quality failures stay separate from OpenCL logits/output drift, without changing runtime math or promoting readiness, parity, residency, speed, trusted-partial acceleration, or full BitNet inference. |
| slm-cpu | SLM-CPU-080 | #332 | `codex/slm-cpu-080-kaby-performance-dashboard-refresh` | Refresh the Kaby Lake Qwen3 Q8_0 CPU performance dashboard after the merged prompt-token cache, KV-cache reuse, prefill-attribution, and post-aligned packed-matvec evidence. The dashboard must preserve the Qwen3 Q8_0 behavior oracle, record that the current operator default is evidence-scoped to 4 threads, identify the current allocation and packed-matvec next targets, and keep speedup, sustained-throughput, Q4/Q5 runtime, server, accelerator, Qwen3.5, and BitNet QK256 claims false. |
