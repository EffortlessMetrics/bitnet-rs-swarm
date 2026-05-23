<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-090 | #474 | `codex/slm-cpu-090-residual-add-output-storage-slice` | Implement or conclusively block the narrow residual-add output-storage slice queued by SLM-CPU-089. A valid slice must either add a behavior-preserving transformer.block.output residual-add helper that can write into caller-provided reusable storage, or produce a machine-checkable blocker naming the exact Candle Tensor API limitation that still prevents reusable storage. Any runtime change must preserve the Qwen3 Q8_0 appliance oracle before claiming even bounded allocation improvement: model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and fallback_used=false. The slice must not promote packed_q8_sidecar, claim speedup without before/after receipts, claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
