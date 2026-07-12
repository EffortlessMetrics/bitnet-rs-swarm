use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use regex::Regex;
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

use crate::model_coverage;

#[derive(Args)]
pub struct ReleaseReadyArgs {
    /// Release-readiness profile to evaluate.
    #[arg(long, value_enum)]
    pub profile: ReleaseReadyProfile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ReleaseReadyProfile {
    /// v0.3 usable-preview release bar.
    UsablePreview,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckState {
    Pass,
    Fail,
    Unknown,
}

impl CheckState {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug)]
struct Check {
    name: &'static str,
    state: CheckState,
    detail: String,
}

impl Check {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, state: CheckState::Pass, detail: detail.into() }
    }

    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, state: CheckState::Fail, detail: detail.into() }
    }

    fn unknown(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, state: CheckState::Unknown, detail: detail.into() }
    }
}

#[derive(Debug)]
struct Report {
    profile: ReleaseReadyProfile,
    checks: Vec<Check>,
}

impl Report {
    fn failures(&self) -> usize {
        self.checks.iter().filter(|check| check.state == CheckState::Fail).count()
    }

    fn unknowns(&self) -> usize {
        self.checks.iter().filter(|check| check.state == CheckState::Unknown).count()
    }

    fn is_ready(&self) -> bool {
        self.failures() == 0 && self.unknowns() == 0
    }

    fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(match self.profile {
            ReleaseReadyProfile::UsablePreview => "release-ready profile: usable-preview\n",
        });
        for check in &self.checks {
            out.push_str(&format!("{} {}: {}\n", check.state.label(), check.name, check.detail));
        }
        out.push_str(&format!(
            "summary: {} pass, {} fail, {} unknown\n",
            self.checks.iter().filter(|check| check.state == CheckState::Pass).count(),
            self.failures(),
            self.unknowns()
        ));
        out
    }
}

pub fn run(args: ReleaseReadyArgs) -> Result<()> {
    let root = std::env::current_dir().context("resolve current directory")?;
    let report = evaluate(&root, args.profile);
    print!("{}", report.render());
    if !report.is_ready() {
        bail!(
            "release readiness blocked: {} failed, {} unknown",
            report.failures(),
            report.unknowns()
        );
    }
    Ok(())
}

fn evaluate(root: &Path, profile: ReleaseReadyProfile) -> Report {
    let mut checks = Vec::new();
    match profile {
        ReleaseReadyProfile::UsablePreview => {
            checks.push(check_required_files(root));
            checks.push(check_readme_links(root));
            checks.push(check_model_coverage(root));
            checks.push(check_false_badges(root));
            checks.push(check_release_lane_pr_open_rows(root));
            checks.push(check_critical_ask_path(root));
            checks.push(check_server_state(root));
            checks.push(check_speed_claims(root));
        }
    }
    Report { profile, checks }
}

fn check_required_files(root: &Path) -> Check {
    const REQUIRED: &[&str] = &[
        "README.md",
        "docs/quickstart.md",
        "docs/release/V0_3_USABLE_PREVIEW.md",
        "docs/release/notes/v0.3.0-preview.md",
        "docs/status/SUPPORT_MATRIX.md",
        "docs/status/BITNET_CAPABILITY_MATRIX.md",
        "docs/status/CUDA_CAPABILITY_MATRIX.md",
        "docs/status/APPLE_CAPABILITY_MATRIX.md",
        "docs/status/KNOWN_LIMITATIONS.md",
        "docs/model-artifacts/MODEL_COVERAGE_MATRIX.md",
        "ci/model-artifacts/model-coverage-matrix.toml",
    ];

    let missing: Vec<_> =
        REQUIRED.iter().filter(|path| !root.join(path).is_file()).copied().collect();
    if missing.is_empty() {
        Check::pass("required-files", format!("{} release/status files present", REQUIRED.len()))
    } else {
        Check::fail("required-files", format!("missing {}", missing.join(", ")))
    }
}

fn check_readme_links(root: &Path) -> Check {
    let readme = root.join("README.md");
    let raw = match fs::read_to_string(&readme) {
        Ok(raw) => raw,
        Err(err) => {
            return Check::fail("readme-links", format!("cannot read README.md: {err}"));
        }
    };
    let mut missing = Vec::new();
    for target in markdown_link_targets(&raw) {
        let Some(path) = local_link_path(root, &readme, &target) else {
            continue;
        };
        if !path.exists() {
            missing.push(target);
        }
    }

    if missing.is_empty() {
        Check::pass("readme-links", "repo-local README links resolve")
    } else {
        Check::fail("readme-links", format!("missing local targets: {}", missing.join(", ")))
    }
}

fn markdown_link_targets(raw: &str) -> Vec<String> {
    let Ok(re) = Regex::new(r"!?\[[^\]]*\]\(([^)]+)\)") else {
        return Vec::new();
    };
    re.captures_iter(raw)
        .filter_map(|captures| captures.get(1))
        .map(|target| normalize_markdown_target(target.as_str()))
        .filter(|target| !target.is_empty())
        .collect()
}

fn normalize_markdown_target(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix('<') {
        return rest.split('>').next().unwrap_or("").to_string();
    }
    trimmed.split_whitespace().next().unwrap_or("").trim_matches('"').to_string()
}

fn local_link_path(root: &Path, source: &Path, target: &str) -> Option<PathBuf> {
    let lower = target.to_ascii_lowercase();
    if target.starts_with('#')
        || lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || lower.starts_with("tel:")
    {
        return None;
    }

    let path_part = target.split('#').next().unwrap_or(target);
    if path_part.is_empty() {
        return None;
    }

    let path = Path::new(path_part);
    if path.is_absolute() {
        return Some(root.join(path.strip_prefix(Path::new("/")).unwrap_or(path)));
    }
    Some(source.parent().unwrap_or(root).join(path))
}

fn check_model_coverage(root: &Path) -> Check {
    let path = root.join("ci/model-artifacts/model-coverage-matrix.toml");
    match model_coverage::validate_file(&path) {
        Ok(entries) => {
            Check::pass("model-coverage", format!("matrix validates with {entries} entries"))
        }
        Err(err) => Check::fail("model-coverage", err.to_string()),
    }
}

fn check_false_badges(root: &Path) -> Check {
    let readme = root.join("README.md");
    let raw = match fs::read_to_string(&readme) {
        Ok(raw) => raw,
        Err(err) => return Check::fail("false-badges", format!("cannot read README.md: {err}")),
    };

    let offending: Vec<_> = raw
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let lower = line.to_ascii_lowercase();
            let badge_like = lower.contains("badge")
                || lower.contains("img.shields.io")
                || lower.contains("shields.io/");
            let release_registry_like = lower.contains("crates.io")
                || lower.contains("docs.rs")
                || lower.contains("/crates/v/")
                || lower.contains("/docsrs/");
            if badge_like && release_registry_like {
                Some(format!("README.md:{}", index + 1))
            } else {
                None
            }
        })
        .collect();

    if offending.is_empty() {
        Check::pass("false-badges", "no crates.io/docs.rs release badges in README")
    } else {
        Check::fail("false-badges", format!("registry-looking badges at {}", offending.join(", ")))
    }
}

fn check_release_lane_pr_open_rows(root: &Path) -> Check {
    let tracking = root.join("docs/tracking");
    if !tracking.exists() {
        return Check::unknown("release-pr-open", "docs/tracking is absent");
    }

    let mut rows = Vec::new();
    for entry in WalkDir::new(&tracking).into_iter().filter_map(|entry| entry.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !matches!(path.extension().and_then(|ext| ext.to_str()), Some("md" | "toml")) {
            continue;
        }
        let Ok(raw) = fs::read_to_string(path) else {
            continue;
        };
        for (index, line) in raw.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("pr_open") && is_release_lane_text(&lower) {
                let display = path.strip_prefix(root).unwrap_or(path);
                rows.push(format!("{}:{}", display.display(), index + 1));
            }
        }
    }

    if rows.is_empty() {
        Check::pass("release-pr-open", "no release-lane pr_open tracker rows found")
    } else {
        Check::unknown(
            "release-pr-open",
            format!(
                "release-lane pr_open rows require live GitHub reconciliation: {}",
                rows.join(", ")
            ),
        )
    }
}

fn is_release_lane_text(lower: &str) -> bool {
    lower.contains("usable-release")
        || lower.contains("usable preview")
        || lower.contains("usable-preview")
        || lower.contains("release-ready")
        || lower.contains("v0.3 usable")
}

fn check_critical_ask_path(root: &Path) -> Check {
    const PROOF_MANIFESTS: &[&str] =
        &["docs/release/usable-preview-proof.toml", "ci/release/usable-preview-proof.toml"];
    let existing: Vec<_> =
        PROOF_MANIFESTS.iter().filter(|path| root.join(path).is_file()).copied().collect();
    if existing.is_empty() {
        return Check::unknown(
            "critical-ask-path",
            "no usable-preview proof manifest found; run the exact ask path or commit an accepted receipt manifest",
        );
    }

    let mut accepted = Vec::new();
    let mut failures = Vec::new();
    for manifest in existing {
        match validate_ask_proof_manifest(root, manifest) {
            Ok(receipt) => accepted.push(format!("{manifest} -> {}", receipt.display())),
            Err(err) => failures.push(format!("{manifest}: {err}")),
        }
    }

    if failures.is_empty() {
        Check::pass("critical-ask-path", accepted.join("; "))
    } else {
        Check::fail("critical-ask-path", failures.join("; "))
    }
}

#[derive(Debug, Deserialize)]
struct AskProofManifest {
    critical_ask_receipt: PathBuf,
    model_id: String,
    device: String,
    fallback_used: bool,
    quality_gate_passed: bool,
    receipt_explain_checked: bool,
}

fn validate_ask_proof_manifest(root: &Path, manifest: &str) -> Result<PathBuf> {
    let path = root.join(manifest);
    let raw = fs::read_to_string(&path).with_context(|| format!("read {manifest}"))?;
    let proof: AskProofManifest =
        toml::from_str(&raw).with_context(|| format!("parse {manifest}"))?;
    if proof.model_id.trim().is_empty() {
        bail!("model_id is empty");
    }
    if proof.device.trim().is_empty() {
        bail!("device is empty");
    }
    if proof.fallback_used {
        bail!("fallback_used must be false for the release-critical ask path");
    }
    if !proof.quality_gate_passed {
        bail!("quality_gate_passed must be true for the release-critical ask path");
    }
    if !proof.receipt_explain_checked {
        bail!("receipt_explain_checked must be true");
    }
    let receipt = root.join(&proof.critical_ask_receipt);
    if !receipt.is_file() {
        bail!("critical ask receipt is missing: {}", proof.critical_ask_receipt.display());
    }
    Ok(proof.critical_ask_receipt)
}

#[derive(Debug, Deserialize)]
struct ClaimMatrix {
    entry: Vec<ClaimEntry>,
}

#[derive(Debug, Deserialize)]
struct ClaimEntry {
    id: String,
    claims: ClaimFlags,
}

#[derive(Debug, Deserialize)]
struct ClaimFlags {
    benchmark_qualified: bool,
    server_ready: bool,
    speedup_claim: bool,
}

fn load_claim_matrix(root: &Path) -> Result<ClaimMatrix> {
    let path = root.join("ci/model-artifacts/model-coverage-matrix.toml");
    let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn check_server_state(root: &Path) -> Check {
    let matrix = match load_claim_matrix(root) {
        Ok(matrix) => matrix,
        Err(err) => return Check::fail("server-state", err.to_string()),
    };
    let server_ready: BTreeSet<_> = matrix
        .entry
        .iter()
        .filter(|entry| entry.claims.server_ready)
        .map(|entry| entry.id.as_str())
        .collect();
    if server_ready.is_empty() {
        return Check::pass("server-state", "no server_ready=true model coverage rows");
    }

    let mut docs = String::new();
    for path in ["docs/status/SUPPORT_MATRIX.md", "docs/status/CUDA_CAPABILITY_MATRIX.md"] {
        match fs::read_to_string(root.join(path)) {
            Ok(raw) => {
                docs.push_str(&raw.to_ascii_lowercase());
                docs.push('\n');
            }
            Err(err) => return Check::fail("server-state", format!("cannot read {path}: {err}")),
        }
    }

    let mut missing = Vec::new();
    for row in &server_ready {
        if !docs.contains(&row.to_ascii_lowercase()) {
            missing.push((*row).to_string());
        }
    }
    let exact_profile_named = docs.contains("exact-profile") || docs.contains("exact profile only");
    if !missing.is_empty() {
        Check::fail(
            "server-state",
            format!("server_ready rows missing from status docs: {}", missing.join(", ")),
        )
    } else if !exact_profile_named {
        Check::fail("server-state", "server_ready rows are not bounded to exact-profile wording")
    } else {
        Check::pass(
            "server-state",
            format!("server_ready rows are exact-profile bounded: {}", join_set(&server_ready)),
        )
    }
}

fn check_speed_claims(root: &Path) -> Check {
    let matrix = match load_claim_matrix(root) {
        Ok(matrix) => matrix,
        Err(err) => return Check::fail("speed-claims", err.to_string()),
    };
    let mut invalid = Vec::new();
    let mut speed_rows = Vec::new();
    for entry in &matrix.entry {
        if entry.claims.speedup_claim {
            speed_rows.push(entry.id.clone());
            if !entry.claims.benchmark_qualified {
                invalid.push(entry.id.clone());
            }
        }
    }
    if !invalid.is_empty() {
        return Check::fail(
            "speed-claims",
            format!("speedup_claim without benchmark_qualified: {}", invalid.join(", ")),
        );
    }

    let docs = [
        "docs/status/SUPPORT_MATRIX.md",
        "docs/status/CUDA_CAPABILITY_MATRIX.md",
        "docs/release/notes/v0.3.0-preview.md",
        "docs/status/KNOWN_LIMITATIONS.md",
    ];
    let mut combined = String::new();
    for path in docs {
        match fs::read_to_string(root.join(path)) {
            Ok(raw) => {
                combined.push_str(&raw.to_ascii_lowercase());
                combined.push('\n');
            }
            Err(err) => return Check::fail("speed-claims", format!("cannot read {path}: {err}")),
        }
    }

    if speed_rows.is_empty() {
        if combined.contains("speedup_claim=true") {
            Check::fail("speed-claims", "docs mention speedup_claim=true but no matrix row does")
        } else {
            Check::pass("speed-claims", "no benchmark-qualified speedup rows are promoted")
        }
    } else {
        Check::pass(
            "speed-claims",
            format!("speedup rows are benchmark-qualified: {}", speed_rows.join(", ")),
        )
    }
}

fn join_set(values: &BTreeSet<&str>) -> String {
    values.iter().copied().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readme_link_check_catches_missing_local_target() -> Result<()> {
        let temp = tempfile::tempdir()?;
        fs::write(temp.path().join("README.md"), "[missing](docs/missing.md)\n")?;
        let check = check_readme_links(temp.path());
        assert_eq!(check.state, CheckState::Fail);
        assert!(check.detail.contains("docs/missing.md"));
        Ok(())
    }

    #[test]
    fn false_badge_check_rejects_registry_badges() -> Result<()> {
        let temp = tempfile::tempdir()?;
        fs::write(
            temp.path().join("README.md"),
            "![Crates.io badge](https://img.shields.io/crates/v/bitnet)\n",
        )?;
        let check = check_false_badges(temp.path());
        assert_eq!(check.state, CheckState::Fail);
        assert!(check.detail.contains("README.md:1"));
        Ok(())
    }

    #[test]
    fn unknown_critical_ask_path_blocks_readiness() {
        let report = Report {
            profile: ReleaseReadyProfile::UsablePreview,
            checks: vec![Check::unknown("critical-ask-path", "missing")],
        };
        assert!(!report.is_ready());
        assert_eq!(report.unknowns(), 1);
    }
}
