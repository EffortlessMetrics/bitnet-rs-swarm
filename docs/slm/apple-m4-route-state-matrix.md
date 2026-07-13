# Apple M4 Route-State Matrix

This document is the operator-facing companion to the model-free matrix emitted
by `bitnet mac evidence --json`. The CLI output is authoritative for runtime
validation; this page records the same current claim boundary in a stable,
reviewable location.

## Current routes

| Model family | Surface | State | Operator class | Boundary |
|---|---|---|---|---|
| Dense SLM | `ask` | `enabled` | `interactive` or `advisory` by model | Supported Qwen identities on `apple-m4-cpu-neon`, with `fallback_used=false`. |
| Dense SLM | `chat` | `enabled` | `interactive` or `advisory` by model | Resident dense chat only; this does not enable BitNet chat or broad chat-quality claims. |
| Dense SLM | `warm_session` | `enabled` | `interactive` or `advisory` by model | Recorded resident profiles only; no broad performance or full-residency claim. |
| Dense SLM | `serve` | `enabled` | `advisory` | Loopback local service only; not production hosting or full OpenAI compatibility. |
| Dense SLM | `streaming` | `enabled` | `advisory` | Bounded dense streaming semantics only; not BitNet streaming proof. |
| Dense SLM | `long_context` | `batch_only` | `batch` | Recorded context profiles only; requests beyond the envelope are unsupported. |
| BitNet | `ask` | `enabled` | `batch` | Accepted Microsoft I2_S artifact and external tokenizer, exact CPU/NEON profile only. |
| BitNet | `warm_session` | `enabled` | `batch` | Exact-profile warm evidence; no BitNet chat, serve, broad quality, or speedup claim. |
| BitNet | `chat` | `disabled_without_ready_gate` | `diagnostic` | A ready `bitnet_apple_m4_chat_gate` and matching chat-session receipts are required. |
| BitNet | `serve` | `disabled_without_ready_gate` | `diagnostic` | A ready serve gate plus chat, streaming, failure, readiness, and per-request receipts are required. |
| BitNet | `streaming` | `disabled_without_ready_gate` | `diagnostic` | Separate BitNet chat/serve gate and streaming evidence are required. |
| BitNet | `long_context` | `batch_only` | `batch` | Requests beyond the recorded prompt envelope are unsupported. |
| All | Full Metal inference | `unsupported` | `diagnostic` | Existing Metal evidence is phase/subgraph proof only, not full autoregressive inference. |
| All | QK256, Neural Engine, MPSGraph, MacBook, broad Apple Silicon | `unsupported` | `unsupported` | Separate receipt-backed campaigns are required; current M4 Mac mini CPU/NEON evidence does not prove these routes. |

## Contract

- `enabled` means the route may run when its cache and gate preconditions pass.
- `disabled_without_ready_gate` means the route exists but must fail closed
  until a ready gate receipt is supplied.
- `batch_only` means exact-profile work is valid but should be treated as
  unattended rather than promised as interactive.
- `unsupported` means no accepted M4 Mac mini appliance receipt supports the
  route or claim.
- Dense SLM evidence and BitNet evidence remain separate families.
- This matrix is model-free and does not download models, run live inference,
  or enable disabled routes.

The machine-readable source is `apple_m4_route_state_matrix` inside
`bitnet mac evidence --json`; regenerate and validate it with the release-bundle
commands in `docs/slm/apple-m4-release-go-no-go.md`.
