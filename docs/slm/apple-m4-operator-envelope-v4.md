# Apple M4 Operator Envelope V4

This is the `M4-HARDEN-006` refresh of the local Apple M4 Mac mini operator
envelope. It records the repeat/variance contract and separates operator class
from evidence health. It is still an exact CPU/NEON appliance envelope; it is
not a broad Apple Silicon performance or model-quality claim.

## Variance evidence

The committed dense variance receipts use two repeats across the nine
`slm-benchmark-v2` profiles and preserve p50, p90, p99, minimum, maximum, and
raw samples for timing and memory metrics:

| Evidence | Repeat/sample contract | Representative recorded fields | Evidence health |
|---|---:|---|---|
| `ci/hardware/apple-m4-mac-mini/2026-05-19T1125Z/benchmark-variance/qwen2.5-0.5b-instruct-q8_0/summary.json` | 2 repeats / 18 profile samples | TTFT p50/p90/p99/min/max `2586/3641/3641/2586/3641 ms`; peak memory p50/p90/p99/min/max `3655.094/3753.938/3753.938/3655.094/3753.938 MB`; memory drift all `0 MB` | `diagnostic`: `context_4k` exceeded its 720-second comparison boundary |
| `ci/hardware/apple-m4-mac-mini/2026-05-19T1125Z/benchmark-variance/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | 2 repeats / 18 profile samples | Same complete statistic contract, with raw samples retained | `diagnostic`: `context_4k` exceeded its 720-second comparison boundary |
| `ci/hardware/apple-m4-mac-mini/2026-05-19T1125Z/benchmark-variance/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | 2 repeats / 18 profile samples | Same complete statistic contract, with raw samples retained | `diagnostic`: long/context profiles exceeded recorded timeout boundaries |
| `ci/hardware/apple-m4-mac-mini/2026-05-19T2245Z/bitnet-benchmark-variance/summary.json` | 2 child runs / 4 aggregate paths | Fixed-warm TTFT p50/p90/p99/min/max `7491/8486/8486/7480/8486 ms`; decode p50/p90/p99/min/max `2.082/2.083/2.083/1.937/2.083 tok/s`; peak memory p50/p90/p99/min/max `4322.875/4327.438/4327.438/4322.875/4327.438 MB` | `batch`: exact accepted artifact/tokenizer evidence; not a chat or serve gate |

The dashboard counts only top-level aggregate receipts. Child summaries under
`summary-runs/` are inputs to their parent and are not additional history
points. A single aggregate remains `insufficient_history` for trend purposes.

## Threshold and outlier policy

Timing and memory drift are advisory by default. The dashboard and variance
receipts expose the following structured policy:

- timing drift advisory threshold: 15% for the standard timing class;
- peak-memory advisory threshold: 10%;
- memory-drift advisory threshold: 15%;
- hard blockers: identity mismatch, timeout or `not_run`, fallback use, missing
  required receipt fields, and claim-boundary violations;
- `--fail-on-drift` is an explicit opt-in for scheduled or release gates;
- raw samples are retained, never silently trimmed, and p99 versus p50 plus
  min/max are the review signals for suspected outliers.

The statistic object also records population standard deviation and coefficient
of variation (`stddev`, `cv_pct`) so timing variance is numeric and reproducible.
These fields describe observed samples; they do not promise speed or establish
a release threshold by themselves.

## Operator classes

`operator_class` is separate from route `state` and evidence-health
`operator_status`:

| Class | Meaning in this envelope |
|---|---|
| `interactive` | Bounded short dense Qwen 0.5B ask/chat use on the recorded CPU/NEON identity. |
| `advisory` | Usable local evidence with receipt review required, including the dense 1.5B route and dense loopback server. |
| `batch` | Exact-profile BitNet ask and warm-session work; quality remains repair-first. |
| `diagnostic` | Metal phase/subgraph proof, incomplete matching history, or timeout-invalid variance evidence. It is not a supported user route. |
| `unsupported` | Neural Engine, QK256-on-Apple, MPSGraph model inference, MacBook transfer, and broad Apple Silicon claims. |

BitNet chat and BitNet serve remain `disabled_without_ready_gate` regardless of
the dense server receipts, BitNet ask receipts, or warm-session evidence. Full
Metal remains unsupported as a model-inference route; only phase/subgraph proof
is diagnostic. No route in this refresh enables chat, serve, QK256, Neural
Engine, MPSGraph model inference, or a broad performance claim.

The machine-readable route matrix now accepts `diagnostic` as an operator class
while preserving the existing disabled and unsupported route states. The
model-free `mac regression-dashboard` receipt exposes the variance summaries,
operator class, class rationale, and threshold contract for each group.

## Required validation

```text
cargo fmt --all -- --check
cargo test --locked -p bitnet-cli benchmark -- --nocapture
cargo test --locked -p bitnet-cli variance -- --nocapture
cargo test --locked -p bitnet-cli operator -- --nocapture
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli --bin bitnet -- mac regression-dashboard --json-out target/apple-m4-post-excellence-hardening/regression-dashboard.json --markdown-out target/apple-m4-post-excellence-hardening/regression-dashboard.md --json
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-post-excellence-hardening
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```
