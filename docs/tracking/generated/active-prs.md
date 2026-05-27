<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-161 | #845 | `codex/slm-cpu-161-allocation-audit-baseline` | Consume the SLM-CPU-160 allocation/buffer-reuse blocker and capture or define the allocation-audit-enabled baseline needed before any runtime allocation behavior changes. A valid slice must produce Qwen3 Q8_0 and Qwen2.5 Q8_0 strict CPU baseline evidence, or record the exact local artifact/cache blocker preventing that capture. The evidence must preserve model SHA, tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and `fallback_used=false`. It must keep allocation-reduction, speedup, sustained-throughput, Q4/Q5 runtime support, default-runtime promotion, server/accelerator execution, Qwen3.5, and BitNet QK256 claims false. |
