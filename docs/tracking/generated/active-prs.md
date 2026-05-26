<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-126 | #774 | `codex/slm-cpu-126-qproj-allocation-timing-experiment` | Use the accepted exact-boundary SLM-CPU-125 numeric gate as a prerequisite for the next Qwen3/Qwen2.5 before/after packed-Q8 sidecar allocation or timing experiment. A valid slice must collect fresh strict CPU receipts before and after the experiment, preserve model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook identity, fallback_used=false, and the accepted layer-0 attention.q_proj_output_pre_optional_qnorm numeric gate, then either record a bounded allocation/timing result or fail closed with the exact blocker. It must not promote packed-Q8 sidecar to default runtime or claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server/GPU/NPU/OpenVINO/UHD 620 execution, Qwen3.5 support, or BitNet QK256 changes without matching receipts. |
