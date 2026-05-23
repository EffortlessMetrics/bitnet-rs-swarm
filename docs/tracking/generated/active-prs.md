<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-089 | #471 | `codex/slm-cpu-089-residual-add-storage-gate` | Define the next behavior-preserving residual-add output-storage gate after SLM-CPU-088. The slice must either add or precisely specify a narrow transformer.block.output residual-add storage API that can reuse caller-owned storage, or emit a machine-checkable blocker proving why Candle still forces an owned Tensor output. Any implementation must preserve the established Qwen3 Q8_0 appliance oracle: model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and fallback_used=false before claiming even a bounded allocation improvement. The slice must not promote packed_q8_sidecar, claim speedup without before/after receipts, claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
