//! Feature gate consistency tests for Issue #439 (baseline reconciled per #1709)
//!
//! Validates that *new* GPU code uses the unified predicate
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]` rather than standalone
//! cuda-only gates, to prevent compile-time drift.
//!
//! ## Why a baseline?
//!
//! The workspace already contains many standalone `#[cfg(feature = "cuda")]`
//! gates. The overwhelming majority live under
//! `crates/bitnet-kernels/src/cuda/**` and other CUDA-specific surfaces where a
//! bare `feature = "cuda"` gate is *correct*: unifying them to
//! `any(feature = "gpu", feature = "cuda")` would drag CUDA-only code into a
//! non-CUDA `gpu` backend build and fail to compile. Rewriting all of them is
//! therefore neither safe nor desirable.
//!
//! Instead we grandfather the existing findings into
//! [`tests/cuda_gate_baseline.tsv`] and keep the check strict for *new* debt:
//! any standalone cuda gate beyond the recorded per-identity counts fails.
//!
//! Regenerate the baseline after an intentional change:
//!
//! ```text
//! BLESS_CUDA_GATE_BASELINE=1 \
//!   cargo test --locked -p bitnet-kernels --no-default-features \
//!   --test feature_gate_consistency -- --test-threads=1
//! ```
//!
//! See `docs/reference/cuda-gate-baseline.md` for the policy.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// Helper to find workspace root by walking up to the `.git` directory.
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root (no .git directory found)");
        }
    }
    path
}

// ---------------------------------------------------------------------------
// Baseline machinery (issue #1709)
// ---------------------------------------------------------------------------

/// Runtime `cfg!` gates that are intentionally CUDA-specific and therefore
/// exempt from the unified-predicate requirement. These are authoritative
/// sources that deliberately distinguish `feature = "cuda"` (CUDA-specific)
/// from `feature = "gpu"` (umbrella).
const ALLOWED_CFG_MACRO_EXCEPTIONS: &[&str] =
    &["kernel_registry.rs", "backend_selection.rs", "bitnet-runtime-feature-flags/src/lib.rs"];

/// Path to the committed baseline of grandfathered findings.
fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cuda_gate_baseline.tsv")
}

fn bless_requested() -> bool {
    std::env::var_os("BLESS_CUDA_GATE_BASELINE").is_some_and(|v| v == "1")
}

/// Run ripgrep from the workspace root and return its stdout. ripgrep exits
/// with status 1 when there are no matches, which is not an error here.
fn run_rg(args: &[&str]) -> String {
    let output = Command::new("rg")
        .args(args)
        .current_dir(workspace_root())
        .output()
        .expect("Failed to run ripgrep - ensure 'rg' is installed");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Split a `path:match` ripgrep line (from `-o -H`, i.e. only the matched
/// construct, prefixed with the file path) into its `(path, match)` parts.
/// The path segment never contains a colon in this repo, so the first colon is
/// always the boundary.
fn parse_finding(line: &str) -> Option<(&str, &str)> {
    let (path, matched) = line.split_once(':')?;
    let matched = matched.trim();
    if matched.is_empty() {
        return None;
    }
    Some((path, matched))
}

/// Build the stable `path\tmatch` identity used for baseline comparison.
///
/// Because ripgrep runs with `--only-matching`, `matched` is exactly the gate
/// construct (e.g. `#[cfg(feature = "cuda")]`) with no surrounding code,
/// indentation, or trailing comment — so an incidental `any(feature ...)` in a
/// comment cannot mask a real gate, and each occurrence on a shared line is
/// counted separately.
fn identity(path: &str, matched: &str) -> String {
    format!("{path}\t{matched}")
}

/// All standalone `#[cfg(feature = "cuda")]` attribute findings across
/// `crates/` (those not already using the unified `any(feature = ...)` form).
fn cfg_attr_findings() -> Vec<String> {
    // `-o` (only-matching) guarantees the regex — which matches only the
    // standalone `#[cfg(feature = "cuda")]` form, never the unified
    // `#[cfg(any(feature = ...))]` — is the entire captured text, so no
    // "already unified" post-filter is needed. `-H` forces the path prefix even
    // when a single file matches.
    run_rg(&[
        r#"#\[cfg\(feature\s*=\s*"cuda"\)\]"#,
        "-o",
        "-H",
        "--no-heading",
        "--no-line-number",
        "--color=never",
        "--glob",
        "*.rs",
        "--glob",
        "!Cargo.lock",
        // This test file mentions the pattern in its docs; don't scan itself.
        "--glob",
        "!**/feature_gate_consistency.rs",
        "crates/",
    ])
    .lines()
    .filter_map(parse_finding)
    .map(|(path, matched)| identity(path, matched))
    .collect()
}

/// All standalone `cfg!(feature = "cuda")` runtime findings across `crates/`,
/// excluding this test file and the intentionally CUDA-specific exceptions.
fn cfg_macro_findings() -> Vec<String> {
    run_rg(&[
        r#"cfg!\(feature\s*=\s*"cuda"\)"#,
        "-o",
        "-H",
        "--no-heading",
        "--no-line-number",
        "--color=never",
        "--glob",
        "*.rs",
        "--glob",
        "!Cargo.lock",
        "--glob",
        "!**/feature_gate_consistency.rs",
        "crates/",
    ])
    .lines()
    .filter_map(parse_finding)
    // The exceptions are file allowlist entries: match the PATH only, never the
    // matched text, so an unrelated finding cannot be exempted by a comment.
    .filter(|(path, _)| !ALLOWED_CFG_MACRO_EXCEPTIONS.iter().any(|exc| path.contains(exc)))
    .map(|(path, matched)| identity(path, matched))
    .collect()
}

/// Load the recorded identities for one category from the baseline file.
fn load_baseline(category: &str) -> Vec<String> {
    let text = std::fs::read_to_string(baseline_path()).unwrap_or_default();
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let (cat, rest) = l.split_once('\t')?;
            (cat == category).then(|| rest.to_string())
        })
        .collect()
}

/// Return every `current` occurrence that exceeds the baselined count for its
/// identity (i.e. genuinely new debt). Removing debt — a `current` count below
/// the baseline — is always allowed and returns nothing for that identity.
///
/// This is a pure function so the no-new-debt policy can be regression-tested
/// without touching the filesystem or ripgrep.
fn new_debt(current: &[String], baseline: &[String]) -> Vec<String> {
    let mut allowed: HashMap<&str, i64> = HashMap::new();
    for b in baseline {
        *allowed.entry(b.as_str()).or_insert(0) += 1;
    }
    let mut seen: HashMap<&str, i64> = HashMap::new();
    let mut excess = Vec::new();
    for c in current {
        let n = seen.entry(c.as_str()).or_insert(0);
        *n += 1;
        if *n > allowed.get(c.as_str()).copied().unwrap_or(0) {
            excess.push(c.clone());
        }
    }
    excess
}

/// Regenerate the committed baseline from the current findings. Writes
/// atomically (temp file + rename) so a `--test-threads=1` bless run cannot
/// leave a half-written file even if invoked from more than one test.
fn regenerate_baseline() -> std::io::Result<()> {
    let mut lines = vec![
        "# CUDA feature-gate baseline (issue #1709).".to_string(),
        "# Grandfathered pre-existing standalone `#[cfg(feature = \"cuda\")]` and".to_string(),
        "# `cfg!(feature = \"cuda\")` findings. New debt beyond these counts fails".to_string(),
        "# the feature_gate_consistency tests.".to_string(),
        "#".to_string(),
        "# Regenerate after an intentional change:".to_string(),
        "#   BLESS_CUDA_GATE_BASELINE=1 cargo test --locked -p bitnet-kernels \\".to_string(),
        "#     --no-default-features --test feature_gate_consistency -- --test-threads=1"
            .to_string(),
        "#".to_string(),
        "# Columns: <category>\\t<path>\\t<trimmed content>".to_string(),
    ];
    let mut entries: Vec<String> = Vec::new();
    entries.extend(cfg_attr_findings().into_iter().map(|id| format!("cfg-attr\t{id}")));
    entries.extend(cfg_macro_findings().into_iter().map(|id| format!("cfg-macro\t{id}")));
    entries.sort();
    lines.extend(entries);
    let mut body = lines.join("\n");
    body.push('\n');

    let final_path = baseline_path();
    let tmp_path = final_path.with_extension("tsv.tmp");
    std::fs::write(&tmp_path, body)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

/// Format a set of new-debt identities into an actionable assertion message.
fn debt_message(kind: &str, findings: &[String]) -> String {
    let list = findings.iter().map(|f| f.replace('\t', "  ->  ")).collect::<Vec<_>>().join("\n");
    format!(
        "Found {} new standalone {kind} (AC1) not covered by the baseline:\n{list}\n\n\
         Fix: use the unified predicate `any(feature = \"gpu\", feature = \"cuda\")`.\n\
         If this gate is intentionally CUDA-specific (e.g. under \
         crates/bitnet-kernels/src/cuda/**), re-bless the baseline:\n  \
         BLESS_CUDA_GATE_BASELINE=1 cargo test --locked -p bitnet-kernels \
         --no-default-features --test feature_gate_consistency -- --test-threads=1",
        findings.len(),
    )
}

// ---------------------------------------------------------------------------
// AC:1 — kernel gate unification
// ---------------------------------------------------------------------------

/// AC:1 - No *new* standalone cuda feature gates in `bitnet-kernels/src`.
///
/// Existing CUDA-specific gates (mostly under `src/cuda/**`) are grandfathered
/// via the baseline; new ones must use the unified predicate.
#[test]
fn ac1_no_standalone_cuda_gates_in_kernels() -> Result<(), Box<dyn std::error::Error>> {
    if bless_requested() {
        regenerate_baseline()?;
        return Ok(());
    }
    let is_kernels = |id: &String| id.starts_with("crates/bitnet-kernels/");
    let current: Vec<String> = cfg_attr_findings().into_iter().filter(is_kernels).collect();
    let baseline: Vec<String> = load_baseline("cfg-attr").into_iter().filter(is_kernels).collect();

    let debt = new_debt(&current, &baseline);
    assert!(debt.is_empty(), "{}", debt_message("cuda feature gate", &debt));
    Ok(())
}

/// AC:1 - Verify GPU-specific modules use the unified predicate.
///
/// Uses `gpu/validation.rs` as a representative critical GPU module.
#[test]
fn ac1_gpu_validation_module_uses_unified_predicate() -> Result<(), Box<dyn std::error::Error>> {
    let validation_path = workspace_root().join("crates/bitnet-kernels/src/gpu/validation.rs");

    if !validation_path.exists() {
        println!("Note: gpu/validation.rs not found - this test validates it uses unified gates");
        return Ok(());
    }

    let validation_rs = std::fs::read_to_string(&validation_path)?;

    let unified_pattern = r#"#[cfg(any(feature = "gpu", feature = "cuda"))]"#;

    // If the file has bare feature cfg gates, ensure the unified predicate is used.
    if validation_rs.contains("#[cfg(feature") {
        assert!(
            validation_rs.contains(unified_pattern)
                || validation_rs.contains(r#"any(feature = "gpu", feature = "cuda")"#),
            "gpu/validation.rs must use unified GPU predicate (AC1)"
        );
    }
    Ok(())
}

/// AC:1 - No *new* standalone cuda feature gates anywhere under `crates/`.
#[test]
fn ac1_workspace_wide_cuda_gate_consistency() -> Result<(), Box<dyn std::error::Error>> {
    if bless_requested() {
        regenerate_baseline()?;
        return Ok(());
    }
    let current = cfg_attr_findings();
    let baseline = load_baseline("cfg-attr");

    let debt = new_debt(&current, &baseline);
    assert!(debt.is_empty(), "{}", debt_message("cuda feature gate", &debt));
    Ok(())
}

/// AC:1 - Verify build.rs uses unified GPU detection.
///
/// Ensures the build script checks both `CARGO_FEATURE_GPU` and
/// `CARGO_FEATURE_CUDA` for GPU feature detection.
#[test]
fn ac1_build_scripts_check_both_gpu_features() -> Result<(), Box<dyn std::error::Error>> {
    let build_rs_path = workspace_root().join("crates/bitnet-kernels/build.rs");

    if !build_rs_path.exists() {
        println!("Note: bitnet-kernels/build.rs not found - AC2 will validate unified detection");
        return Ok(());
    }

    let build_rs = std::fs::read_to_string(&build_rs_path)?;

    let has_unified_detection = build_rs.contains("CARGO_FEATURE_GPU")
        && build_rs.contains("CARGO_FEATURE_CUDA")
        && (build_rs.contains("||") || build_rs.contains("is_some()"));

    assert!(
        has_unified_detection,
        "build.rs must check both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA (AC1)\n\
         Expected pattern: CARGO_FEATURE_GPU.is_some() || CARGO_FEATURE_CUDA.is_some()"
    );
    Ok(())
}

#[cfg(test)]
mod gpu_runtime_checks {
    use super::*;

    /// AC:1 - No *new* standalone `cfg!(feature = "cuda")` runtime checks.
    ///
    /// Intentional CUDA-specific runtime checks are either named in
    /// [`ALLOWED_CFG_MACRO_EXCEPTIONS`] or grandfathered via the baseline.
    #[test]
    fn ac1_cfg_macro_uses_unified_predicate() -> Result<(), Box<dyn std::error::Error>> {
        if bless_requested() {
            regenerate_baseline()?;
            return Ok(());
        }
        let current = cfg_macro_findings();
        let baseline = load_baseline("cfg-macro");

        let debt = new_debt(&current, &baseline);
        assert!(
            debt.is_empty(),
            "{}",
            debt_message("cfg!(feature = \"cuda\") runtime check", &debt)
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Regression tests for the no-new-debt matcher (pure, no filesystem/ripgrep)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod baseline_matcher {
    use super::new_debt;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn covered_findings_are_not_debt() {
        let baseline = v(&["a\tgate", "b\tgate"]);
        let current = v(&["a\tgate", "b\tgate"]);
        assert!(new_debt(&current, &baseline).is_empty());
    }

    #[test]
    fn a_new_identity_is_flagged() {
        let baseline = v(&["a\tgate"]);
        let current = v(&["a\tgate", "b\tgate"]);
        assert_eq!(new_debt(&current, &baseline), v(&["b\tgate"]));
    }

    #[test]
    fn an_extra_occurrence_of_a_known_identity_is_flagged() {
        // Two occurrences where the baseline only grandfathered one.
        let baseline = v(&["a\tgate"]);
        let current = v(&["a\tgate", "a\tgate"]);
        assert_eq!(new_debt(&current, &baseline), v(&["a\tgate"]));
    }

    #[test]
    fn removing_debt_is_allowed() {
        // Fewer occurrences than baselined is fine (debt was fixed).
        let baseline = v(&["a\tgate", "a\tgate", "b\tgate"]);
        let current = v(&["a\tgate"]);
        assert!(new_debt(&current, &baseline).is_empty());
    }

    #[test]
    fn empty_baseline_flags_all_current() {
        let baseline = v(&[]);
        let current = v(&["a\tgate", "b\tgate"]);
        assert_eq!(new_debt(&current, &baseline), v(&["a\tgate", "b\tgate"]));
    }
}
