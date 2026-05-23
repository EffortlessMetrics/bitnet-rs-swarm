<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-088 | #459 | `codex/slm-cpu-088-residual-block-output-boundary` | Implement or record the residual-add / transformer.block.output allocation-layout boundary queued by SLM-CPU-087. A valid slice may add a behavior-preserving block-output or residual-add workspace/storage API surface, or produce a machine-checkable blocker that identifies the exact remaining Candle Tensor ownership/API gap. Any runtime change must preserve the established Qwen3 Q8_0 appliance oracle: model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and fallback_used=false. The slice must not promote packed_q8_sidecar, claim speedup without before/after receipts, claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
