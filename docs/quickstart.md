# BitNet-rs Usable Preview Quickstart

**Build BitNet-rs, verify a supported model, run a bounded local answer, and inspect the receipt.**

This guide uses the supported-preview path only. Before assuming a model,
device, speed, or server claim, check [status/SUPPORT_MATRIX.md](status/SUPPORT_MATRIX.md)
and the receipt emitted by your run. For comprehensive development setup, see
[development/](development/).

## Prerequisites (1 minute)

```bash
# Check Rust version (1.95.0+ required)
rustc --version

# Clone repository
git clone https://github.com/EffortlessMetrics/BitNet-rs
cd BitNet-rs
```

## The Supported-Preview Path

`docs/release/V0_3_USABLE_PREVIEW.md` defines the usable-preview release in
terms of six commands. They are the shortest route from a fresh clone to a
local answer you can actually inspect:

```bash
bitnet model status
bitnet model fetch <supported-model>
bitnet model verify <supported-model>
bitnet ask --model <path> --device <supported-device> "What is 2+2?"
bitnet receipts explain --latest
bitnet support bundle --latest --device <supported-device>
```

Steps 1-6 below walk that path with a real supported artifact. Every command
and every quoted output in those steps was executed against
`qwen2.5-0.5b-instruct-q8_0` on a plain `cpu` device.

## Step 1: Build BitNet-rs

```bash
# CPU inference (fastest setup); full-cli enables model/ask/receipts/support
cargo build --release --no-default-features --features cpu,full-cli

# Optional exact CUDA rows only; check status first
cargo build --release --no-default-features --features gpu,full-cli
```

`bitnet-cli` declares `default = ["cpu", "full-cli"]`, so it does build
without those flags. Spell the features out anyway: it is the repo-wide
convention, it keeps the command correct if the defaults change, and the
root `bitnet` package and most other crates *do* have empty defaults, where
omitting them fails.

Then put the binary on `PATH` for the rest of this guide:

```bash
export PATH="$PWD/target/release:$PATH"
```

Every `bitnet ...` command below assumes that. Without it, use the full
`./target/release/bitnet ...` path.

## Step 2: See What Is Supported

`model status` is a read-only view of the coverage matrix. It does not probe
hardware, so it works before you own any of the devices it lists.

```bash
bitnet model list                    # supported ids + cache state
bitnet model status --device cpu     # posture for one device
bitnet model status --format json    # machine-readable
```

`bitnet model list` prints the ids this build accepts:

```text
ID                                       Cache         Quant        M4 CPU      Contract
microsoft-bitnet-b1.58-2B-4T-i2s         missing       I2_S/QK256   no          microsoft_bitnet_b158_2b_4t_i2s
qwen2.5-0.5b-instruct-q8_0               missing       Q8_0         supported   -
qwen2.5-0.5b-instruct-q4_k_m             missing       Q4_K_M       supported   -
qwen2.5-1.5b-instruct-q4_k_m             missing       Q4_K_M       supported   -
```

> Bare `bitnet model status` summarizes the canonical CUDA lane
> (`nvidia-rtx-5070-ti-cuda`) regardless of your hardware. Pass `--device` for
> the device you actually have.

## Step 3: Fetch a Supported Artifact

This guide uses the 0.5B dense SLM: it is ~676 MB and answers in seconds,
where the 2B BitNet QK256 artifact runs at roughly 0.1 tok/s on scalar
kernels.

```bash
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
```

Fetch reports cache location and byte identity, and is explicit that artifact
verification is not answer readiness:

```text
downloaded: qwen2.5-0.5b-instruct-q8_0 at /root/.cache/bitnet-rs/models/... (675.71 MB, verified=true)
expected: bytes=675710816, sha256=ca59ca7f...
actual:   bytes=675710816, sha256=ca59ca7f...
artifact verification: passed
answer ready: not proven by model verify; use `bitnet model status` and receipts for answer claims
tokenizer authority: qwen2 (embedded_gguf_metadata_bound_to_model_sha256)
prompt authority: qwen2.5 (GGUF tokenizer.chat_template / Qwen2.5 ChatML identity)
```

## Step 4: Verify the Artifact

```bash
bitnet model verify qwen2.5-0.5b-instruct-q8_0
```

Verify re-checks byte identity against the expected sha256 and restates the
tokenizer and prompt authority. It proves *provenance*, not answer quality.

## Step 5: Ask One Question

`ask` takes a **path**, not a model id. Use the cache path that `fetch`
printed:

```bash
MODEL=~/.cache/bitnet-rs/models/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf

bitnet ask --model "$MODEL" --device cpu --max-tokens 24 "What is 2+2?"
```

`ask` writes a receipt by default and prints a proof block:

```text
Generated 24 tokens in 11518ms (2.1 tok/s)
Wrote target/bitnet/receipts/ask/ask-latest.json
Proof:
  model: Qwen/Qwen2.5-0.5B-Instruct / qwen2.5-0.5b-instruct-q8_0.gguf
  backend: cpu-rust
  runtime: cpu
  fallback: false
  quality: true
  speed claim: false
  receipt: target/bitnet/receipts/ask/ask-latest.json
```

`speed claim: false` is deliberate. A tok/s number printed here is local
timing, not a promoted speedup claim.

## Step 6: Inspect the Receipt and Bundle Support

```bash
bitnet receipts explain --latest
bitnet support bundle --latest --device cpu --format text
```

`receipts explain` maps the run back to the coverage matrix, and says so
plainly when nothing matches:

```text
Backend:
  requested: cpu
  selected: cpu-rust
  runtime: cpu
  fallback: false
Quality:
  answer_quality_passed: true
Model Coverage:
  warnings: no model coverage row matched this receipt
```

That warning is the honest outcome, not a bug. The matrix promotes this model
on `dense_regular_llm_cuda` and on Apple CPU/NEON — **not** on a generic `cpu`
device. So the run executed and passed its quality gate, while
`support bundle` correctly reports the promotion fields as `not_available`:

```text
selected_backend: cpu-rust
fallback_used: false
quality_gate: passed
speedup_claim: false
model_coverage_row: not_available
current_tier: not_available
server_ready: not_available
```

Read that as: *the code ran and answered, on a route no support row promotes.*
For a promoted row, use the exact model/device pairs named in
[status/SUPPORT_MATRIX.md](status/SUPPORT_MATRIX.md).

The bundle is designed to be safe to attach to an issue.

## Optional: CPU Validation Tuning

For a local CPU validation run with native optimizations:

```bash
# Build with native CPU optimizations
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# Run with full CPU parallelization and reduced log noise
RAYON_NUM_THREADS=$(nproc) RUST_LOG=warn \
  cargo run --release -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "Explain 1-bit quantization" --max-tokens 128 --temperature 0.7

# Deterministic math sanity check (validates model correctness)
RAYON_NUM_THREADS=1 RUST_LOG=warn \
  cargo run --release -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "Answer with a single digit: 2+2=" --max-tokens 1 \
  --temperature 0.0 --greedy
```

**Expected output from math check:** `4`

**Performance Tuning:**

- `RUSTFLAGS="-C target-cpu=native"`: Enable all CPU instructions (AVX2/AVX-512/NEON)
- `-C opt-level=3`: Maximum optimization (aggressive inlining, vectorization)
- `-C lto=thin`: Link-time optimization for better performance
- `RAYON_NUM_THREADS=$(nproc)`: Use all CPU cores for a local preview run
- `RAYON_NUM_THREADS=1`: Single-threaded (deterministic results for validation)
- `RUST_LOG=warn`: Reduce logging overhead (shows only warnings/errors)

## Status And Performance Boundaries (Read This First!)

**Before you start, separate model support from speed claims:**

| Quantization Format | Release posture | Speed posture | Use Case |
|---------------------|-----------------|---------------|----------|
| **I2_S BitNet32-F16 / QK256** | Supported preview for exact matrix rows | Not claimed here | Bounded local answer and receipt validation |
| **Dense SLM rows** | Supported preview only for exact matrix rows | Not claimed here | Exact CPU/CUDA/Apple rows named by status docs |
| **TL1/TL2 and other routes** | Candidate or diagnostic unless promoted | Not claimed here | Research and proof work |

**The microsoft/bitnet-b1.58-2B-4T-gguf model uses QK256 format.**
Treat QK256 quickstart runs as bounded validation unless the support matrix and
receipt for your exact row say more.

### QK256 Performance Guidance

**If you're using QK256 models (like microsoft/bitnet-b1.58-2B-4T-gguf):**

```bash
# Quick validation (4-16 tokens) - recommended for this guide
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 8  # Keep this small for QK256

# Long generation is outside this quickstart's claim boundary
```

**Why is QK256 slow?**

- Some paths use validation-first kernels rather than optimized kernels
- Speedup requires exact benchmark-qualified receipts and support-matrix promotion
- Slow validation runs are not themselves a correctness failure

Use only the exact model/device rows marked supported preview in the support
matrix.

## Optional: Benchmark Performance

```bash
# Benchmark inference throughput with CPU optimization
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli
RAYON_NUM_THREADS=$(nproc) RUST_LOG=warn \
  cargo run --release --no-default-features -p xtask -- benchmark \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokens 16  # Reduced for QK256
```

**Benchmark posture:**

- A benchmark command produces local evidence; it does not create a speedup claim.
- A speedup claim requires an exact benchmark-qualified receipt and support-matrix row.
- Memory usage depends on the selected model, backend, and runtime profile.

## QK256 Strict Mode Validation

For supported-preview validation with QK256 models, use strict loader mode to
ensure proper model loading:

```bash
# Enable strict loader (fail-fast on model loading errors)
export BITNET_DISABLE_MINIMAL_LOADER=1

# Verify model loads correctly with enhanced GGUF loader
cargo run --no-default-features -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Run inference with strict validation
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --prompt "What is 2+2?" \
  --max-tokens 16
```

**Why Strict Mode?** The strict loader prevents silent fallback to the minimal
loader, which may use incorrect default values (for example, 32 layers or
0 kv_heads) if the enhanced loader fails. This keeps the local answer path tied
to the selected model dimensions.

## Using QK256 Models (GGML I2_S)

QK256 is a GGML-compatible I2_S quantization format with 256-element blocks and
separate scale tensors. BitNet-rs provides automatic format detection and
strict validation modes for supported-preview validation.

### Automatic Format Detection

The loader automatically detects QK256 format based on tensor size patterns. When a tensor's size matches the QK256 quantization scheme (256-element blocks with separate scales), the loader routes to QK256-specific kernels without requiring explicit configuration.

**How it works:**

1. Loader examines tensor dimensions during GGUF parsing
2. Calculates expected size for different quantization formats
3. Prioritizes QK256 (GgmlQk256NoScale) for close matches
4. Routes to appropriate dequantization kernels automatically

**Benefits:**

- Zero configuration required for standard QK256 models
- Seamless compatibility with GGML ecosystem
- Receipt-visible routing; use strict mode when a hidden fallback would be misleading

### Strict Loader Mode

Enforce exact QK256 alignment (reject tensors with >0.1% size deviation) for
supported-preview validation:

```bash
# Enable strict loader with BITNET_DISABLE_MINIMAL_LOADER environment variable
export BITNET_DISABLE_MINIMAL_LOADER=1

# Run inference with strict validation
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
  --strict-loader \
  --prompt "Test" \
  --max-tokens 16
```

**Use strict mode when:**

- Validating model exports for supported-preview use
- Debugging model loading issues
- Running CI/CD parity tests

**What strict mode enforces:**

- Exact tensor size alignment (no tolerance for size mismatches)
- Fail-fast on quantization format detection errors
- Prevents silent fallback to minimal loader defaults

**Learn more:** See [howto/use-qk256-models.md](howto/use-qk256-models.md) for comprehensive QK256 usage guide.

## Receipt Validation Workflow

BitNet-rs generates receipts for every inference run, proving real computation with kernel IDs:

```bash
# 1. Run parity validation (generates receipt)
scripts/parity_smoke.sh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# 2. Check receipt location (automatically created with timestamp)
# Receipt path: docs/baselines/<YYYY-MM-DD>/parity-bitnetcpp.json

# 3. View receipt summary (if jq installed)
jq '{parity, tokenizer, validation}' docs/baselines/$(date +%Y-%m-%d)/parity-bitnetcpp.json

# 4. Verify parity metrics
# - cosine_similarity: ≥0.99 (Rust vs C++ agreement)
# - exact_match_rate: token-level agreement percentage
# - status: "ok" (parity passed) or "rust_only" (C++ unavailable)
```

**Receipt Fields:**

- `validation.compute`: `"rust"` (pure Rust kernels) or `"cpp"` (FFI fallback)
- `parity.status`: `"ok"` (validated), `"rust_only"` (no C++ ref), or `"failed"`
- `parity.cpp_available`: `true` if C++ reference was used for validation
- `tokenizer.source`: `"rust"` (always Rust tokenizer, even with FFI compute)

### Cross-Validation Against C++ Reference

Verify QK256 implementation against the Microsoft BitNet C++ reference:

```bash
# Set up C++ reference path
export BITNET_CPP_DIR=/path/to/bitnet.cpp

# Run comprehensive cross-validation
cargo run --no-default-features -p xtask -- crossval

# Or use quick parity smoke test
./scripts/parity_smoke.sh models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

**Receipt validation:**

```bash
# View parity metrics from generated receipt
jq '.parity' docs/baselines/*/parity-bitnetcpp.json

# Expected output:
# {
#   "cpp_available": true,
#   "cosine_similarity": 0.9923,
#   "exact_match_rate": 1.0,
#   "status": "ok"
# }
```

**Cross-validation ensures:**

- Numerical equivalence between Rust and C++ implementations
- Cosine similarity ≥0.99 for output tensors
- Token-level agreement for autoregressive generation
- Receipt-based proof of parity validation

## What Just Happened?

Following Steps 1-6, you:

1. **Built the CLI** with explicit features, since default features are empty
2. **Read the support posture** with `bitnet model list` / `model status`, without needing the hardware present
3. **Fetched a supported artifact** and saw its cache path, byte count, and sha256 verified against the expected identity
4. **Verified provenance** with `bitnet model verify` — byte identity plus tokenizer and prompt authority, explicitly *not* answer readiness
5. **Ran one local answer** with `bitnet ask`, which wrote a receipt and reported backend, fallback, quality gate, and `speed claim: false`
6. **Inspected that receipt** with `receipts explain` and produced an issue-safe `support bundle`

The most important thing you saw is step 6's `no model coverage row matched
this receipt`. Running successfully on a device and being a *promoted*
model/device row are separate facts, and the tooling keeps them separate.

## Next Steps

- **QK256 Deep Dive**: Comprehensive QK256 usage guide in [howto/use-qk256-models.md](howto/use-qk256-models.md)
- **I2_S Architecture**: Understand dual-flavor quantization in [explanation/i2s-dual-flavor.md](explanation/i2s-dual-flavor.md)
- **Tokenizer Discovery**: Learn about automatic tokenizer discovery in [reference/tokenizer-discovery-api.md](reference/tokenizer-discovery-api.md)
- **API Integration**: See [reference/real-model-api-contracts.md](reference/real-model-api-contracts.md) for Rust API usage
- **Model Formats**: Learn about GGUF, I2_S, TL1, TL2 quantization in [explanation/](explanation/)
- **CUDA Status and Setup**: Check exact CUDA rows before using [development/gpu-setup-guide.md](development/gpu-setup-guide.md)
- **Troubleshooting**: Common issues in [troubleshooting.md](troubleshooting/troubleshooting.md)

## Quick Commands Reference

```bash
# Supported-preview path (Steps 1-6 above)
bitnet model list
bitnet model status --device cpu
bitnet model fetch qwen2.5-0.5b-instruct-q8_0
bitnet model verify qwen2.5-0.5b-instruct-q8_0
bitnet ask --model PATH --device cpu --max-tokens 24 "What is 2+2?"
bitnet receipts explain --latest
bitnet support bundle --latest --device cpu --format text

# CPU build and test
cargo build --no-default-features --features cpu
cargo test --workspace --no-default-features --features cpu

# Exact CUDA row build and test, after checking support status
cargo build --no-default-features --features gpu
cargo test --workspace --no-default-features --features gpu

# Download and verify model (automatic tokenizer discovery)
cargo run --no-default-features -p xtask -- download-model
cargo run --no-default-features -p xtask -- verify --model PATH

# Neural network inference with automatic tokenizer
cargo run --no-default-features -p xtask -- infer --model PATH --prompt "TEXT" --deterministic
cargo run --no-default-features -p xtask -- benchmark --model PATH --tokens 128

# Explicit tokenizer specification (optional)
cargo run --no-default-features -p xtask -- verify --model PATH --tokenizer PATH
cargo run --no-default-features -p xtask -- infer --model PATH --tokenizer PATH --prompt "TEXT"
```

Total time is about 5 minutes to a bounded local preview run and receipt.
