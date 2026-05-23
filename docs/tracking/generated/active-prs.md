<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-085 | #411 | `codex/slm-cpu-085-final-norm-output-storage-boundary` | Burn down or further narrow the post-SLM-CPU-084 final-norm/layer-output caller-output-storage boundary for the Qwen3 Q8_0 Kaby performance lane. A valid slice may add a behavior-preserving output-storage API surface or a machine-checkable blocker proving the remaining Candle Tensor ownership/API gap, but any runtime change must preserve the established Qwen3 Q8_0 appliance behavior oracle: model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and fallback_used=false. The slice must not promote packed_q8_sidecar, claim speedup without before/after receipts, claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
