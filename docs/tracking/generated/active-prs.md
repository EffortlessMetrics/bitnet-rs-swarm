<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-086 | #414 | `codex/slm-cpu-086-final-norm-output-storage-gate` | Burn down or further narrow the SLM-CPU-085 `model.final_norm.output` caller-output-storage blocker for the Qwen3 Q8_0 Kaby performance lane. A valid slice may add a behavior-preserving final-norm output-storage helper or record a more precise machine-checkable LayerNorm/RMSNorm API blocker, but any runtime change must preserve the established Qwen3 Q8_0 appliance behavior oracle: model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity where applicable, and fallback_used=false. The slice must not move to residual-add/layer-output runtime changes, promote packed_q8_sidecar, claim speedup without before/after receipts, claim sustained throughput, broaden Q4/Q5 support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
