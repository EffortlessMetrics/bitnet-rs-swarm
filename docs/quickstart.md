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

## Step 1: Build BitNet-rs (1 minute)

```bash
# CPU inference (fastest setup)
cargo build --release --no-default-features --features cpu

# Optional exact CUDA rows only; check status first
cargo build --release --no-default-features --features gpu
```

## Step 2: Download BitNet Model (1 minute)

```bash
# Download Microsoft's 1.58-bit quantized model (QK256 GGML I2_S format)
cargo run --no-default-features -p xtask -- download-model --id microsoft/bitnet-b1.58-2B-4T-gguf --file ggml-model-i2_s.gguf
```

**What is QK256?** This model uses GGML-compatible I2_S quantization with 256-element blocks and separate scale tensors. BitNet-rs automatically detects the quantization flavor and routes to the appropriate kernels.

## Step 3: Automatic Tokenizer Discovery (30 seconds)

BitNet-rs automatically discovers and loads tokenizers from GGUF files:

```bash
# Verify GGUF model with automatic tokenizer discovery
cargo run --no-default-features -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf

# Or specify tokenizer explicitly if needed
cargo run --no-default-features -p xtask -- verify --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json
```

**What Just Happened?**

- BitNet-rs extracted tokenizer metadata from GGUF file
- Detected model architecture (BitNet, LLaMA, GPT-2, etc.)
- Resolved vocabulary size (32K, 128K, or custom)
- Applied model-specific tokenizer configuration

## Step 4: Run Neural Network Inference (30 seconds)

```bash
# Generate text with automatic tokenizer discovery
cargo run --no-default-features -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --prompt "BitNet is a neural network architecture that" --deterministic

# Stream inference (real-time generation) with automatic tokenizer
cargo run --no-default-features -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --prompt "Explain 1-bit quantization:" --stream

# Or specify tokenizer explicitly if needed
cargo run --no-default-features -p xtask -- infer --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json --prompt "Test" --deterministic
```

## Step 5: CPU Validation Tuning (Optional)

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
cargo run -p bitnet-cli --features cpu,full-cli -- run \
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

## Step 6: Benchmark Performance

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

You've successfully:

1. **Built BitNet-rs** with device-aware quantization and complete transformer implementation
2. **Downloaded a QK256 model** (Microsoft's 1.58-bit GGUF in GGML I2_S format) with automatic flavor detection
3. **Automatic tokenizer discovery** extracted tokenizer from GGUF metadata, detected model architecture, and applied optimal configuration
4. **Verified model compatibility** with enhanced GGUF loader, strict mode validation, and comprehensive tensor validation
5. **Ran bounded preview inference** with QK256 kernels, real transformer weights, and autoregressive generation
6. **Prepared benchmark evidence** - run `cargo run --no-default-features -p xtask -- benchmark --model <path> --tokens 128` to produce a local receipt before making any speed claim
7. **Generated validation receipts** with parity metrics, kernel IDs, and reproducible baselines in `docs/baselines/`

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
