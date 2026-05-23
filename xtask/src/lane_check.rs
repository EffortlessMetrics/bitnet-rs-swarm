use anyhow::{Context, Result, bail};
use clap::Args;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Args)]
pub struct LaneCheckArgs {
    /// Markdown PR body to validate. If omitted, only changed-file checks run.
    #[arg(long)]
    pr_body: Option<PathBuf>,
    /// Base ref used to compute changed files.
    #[arg(long, default_value = "origin/main")]
    base: String,
    /// Head ref used to compute changed files.
    #[arg(long, default_value = "HEAD")]
    head: String,
    /// Branch name to validate. If omitted, uses `Branch:` from the PR body.
    #[arg(long)]
    branch: Option<String>,
}

#[derive(Debug, Default)]
struct CheckReport {
    changed_files: Vec<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

pub fn run(args: LaneCheckArgs) -> Result<()> {
    let changed_files = changed_files(&args.base, &args.head)?;
    let mut report = CheckReport { changed_files: changed_files.clone(), ..CheckReport::default() };

    let pr_body = match args.pr_body.as_deref() {
        Some(path) => Some(
            fs::read_to_string(path)
                .with_context(|| format!("failed to read PR body: {}", path.display()))?,
        ),
        None => {
            report
                .warnings
                .push("no --pr-body provided; PR metadata checks were skipped".to_string());
            None
        }
    };

    if let Some(body) = pr_body.as_deref() {
        validate_pr_body(body, args.branch.as_deref(), &changed_files, &mut report);
    }
    validate_changed_files(&changed_files, &mut report);

    print_report(&report);
    if !report.errors.is_empty() {
        bail!("lane-check failed with {} error(s)", report.errors.len());
    }
    Ok(())
}

fn changed_files(base: &str, head: &str) -> Result<Vec<String>> {
    let range = format!("{base}..{head}");
    let output = Command::new("git")
        .args(["diff", "--name-only", &range])
        .output()
        .with_context(|| format!("failed to list changed files for `{range}`"))?;
    if !output.status.success() {
        bail!(
            "git diff exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(normalize_path)
        .collect())
}

fn validate_pr_body(
    body: &str,
    branch_arg: Option<&str>,
    changed_files: &[String],
    report: &mut CheckReport,
) {
    for field in [
        "Lane",
        "Campaign",
        "Work item",
        "Orchestrator",
        "Branch",
        "Base main SHA",
        "Closeout required",
        "Source promotion needed",
        "Model/hardware/proof claims added",
        "Claims explicitly not promoted",
        "Rollback",
    ] {
        if scalar_field_value(body, field).is_none() {
            report.errors.push(format!("PR body must fill `{field}:`"));
        }
    }

    for field in ["Allowed paths", "Shared surfaces touched", "Commands run", "Validation gaps"] {
        if list_field_values(body, field).is_empty() {
            report.errors.push(format!("PR body must list `{field}:`"));
        }
    }

    let branch = branch_arg.map(ToOwned::to_owned).or_else(|| scalar_field_value(body, "Branch"));
    if let Some(branch) = branch {
        if !valid_branch_name(&branch) {
            report.errors.push(format!(
                "branch `{branch}` must start with codex/<lane>/, claude/<lane>/, droid/<lane>/, or dependabot/"
            ));
        }
    }

    let shared_surfaces = shared_surface_paths(changed_files);
    if !shared_surfaces.is_empty() {
        let declared = list_field_values(body, "Shared surfaces touched");
        if declared.iter().any(|value| is_none_value(value)) {
            report.errors.push(format!(
                "shared surfaces changed but PR body declares none: {}",
                shared_surfaces.join(", ")
            ));
        }
    }

    if touches_repo_boundary(changed_files) {
        for field in [
            "Promotion or sync packet path",
            "Source repo commit",
            "Swarm base commit",
            "Merge method",
            "Source impact",
            "Release/publish/signing impact",
            "Excluded work",
        ] {
            if scalar_field_value(body, field).is_none() {
                report.errors.push(format!("repo-boundary PR body must fill `{field}:`"));
            }
        }
    }

    if touches_generated_dashboard(changed_files)
        && !body.contains("campaign generate --check")
        && !body.contains("campaign generate")
    {
        report.warnings.push(
            "generated dashboards changed; PR body should include campaign generator evidence"
                .to_string(),
        );
    }
}

fn validate_changed_files(changed_files: &[String], report: &mut CheckReport) {
    let generated = generated_dashboard_paths(changed_files);
    if !generated.is_empty() && !touches_campaign_source(changed_files) {
        report.errors.push(format!(
            "generated dashboard changes require campaign source changes: {}",
            generated.join(", ")
        ));
    }

    for path in changed_files {
        if is_tracking_or_generated(path) && file_contains_conflict_marker(Path::new(path.as_str()))
        {
            report.errors.push(format!("conflict marker found in `{path}`"));
        }
    }
}

fn scalar_field_value(body: &str, field: &str) -> Option<String> {
    let prefix = format!("{field}:");
    body.lines().find_map(|line| {
        let trimmed = line.trim();
        let value = trimmed.strip_prefix(&prefix)?.trim();
        if value.is_empty() || is_placeholder(value) { None } else { Some(value.to_string()) }
    })
}

fn list_field_values(body: &str, field: &str) -> Vec<String> {
    let header = format!("{field}:");
    let lines: Vec<&str> = body.lines().collect();
    let Some(start) = lines.iter().position(|line| line.trim() == header) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    for line in lines.iter().skip(start + 1) {
        let trimmed = line.trim();
        if is_body_field_line(trimmed) && !trimmed.starts_with('-') {
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.trim_start_matches('-').trim();
        if !value.is_empty() && !is_placeholder(value) {
            values.push(value.to_string());
        }
    }
    values
}

fn is_placeholder(value: &str) -> bool {
    value.contains("<!--") || value == "_" || value.eq_ignore_ascii_case("todo")
}

fn is_body_field_line(line: &str) -> bool {
    if line.starts_with('#') || line.starts_with("<!--") {
        return false;
    }
    let Some((name, _)) = line.split_once(':') else {
        return false;
    };
    let name = name.trim();
    !name.is_empty()
        && name.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '/' | '-'))
}

fn is_none_value(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "none" | "n/a" | "not applicable")
}

fn valid_branch_name(branch: &str) -> bool {
    branch.starts_with("codex/")
        || branch.starts_with("claude/")
        || branch.starts_with("droid/")
        || branch.starts_with("dependabot/")
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn shared_surface_paths(changed_files: &[String]) -> Vec<String> {
    changed_files.iter().filter(|path| is_shared_surface(path)).cloned().collect()
}

fn is_shared_surface(path: &str) -> bool {
    path == "AGENTS.md"
        || path == "README.md"
        || path == "Cargo.toml"
        || path == "Cargo.lock"
        || path.starts_with(".github/")
        || path.starts_with("xtask/")
        || path.starts_with("policy/")
        || path.starts_with("docs/tracking/generated/")
        || path.starts_with("ci/model-artifacts/")
        || path.starts_with("crates/bitnet-cli/")
        || path.starts_with("crates/bitnet-server/")
        || path == "docs/development/SWARM_DEVELOPMENT_AUTHORITY.md"
        || path == "docs/development/SWARM_HISTORY_REPAIR.md"
        || path == "docs/release/PROMOTE_TO_BITNET_RS.md"
        || path.starts_with("docs/release/promotion-packets/")
        || path == "docs/tracking/LANE_OWNERSHIP.md"
}

fn touches_repo_boundary(changed_files: &[String]) -> bool {
    changed_files.iter().any(|path| {
        path == "AGENTS.md"
            || path == "README.md"
            || path.starts_with(".github/")
            || path == "docs/development/SWARM_DEVELOPMENT_AUTHORITY.md"
            || path == "docs/development/SWARM_HISTORY_REPAIR.md"
            || path == "docs/release/PROMOTE_TO_BITNET_RS.md"
            || path.starts_with("docs/release/promotion-packets/")
            || path == "docs/tracking/LANE_OWNERSHIP.md"
            || path == "policy/repo-boundary.toml"
    })
}

fn generated_dashboard_paths(changed_files: &[String]) -> Vec<String> {
    changed_files.iter().filter(|path| is_generated_dashboard(path)).cloned().collect()
}

fn touches_generated_dashboard(changed_files: &[String]) -> bool {
    changed_files.iter().any(|path| is_generated_dashboard(path))
}

fn is_generated_dashboard(path: &str) -> bool {
    path.starts_with("docs/tracking/generated/")
        || (path.starts_with("docs/tracking/campaigns/") && path.contains("/generated/"))
}

fn touches_campaign_source(changed_files: &[String]) -> bool {
    changed_files.iter().any(|path| {
        path.starts_with("docs/tracking/campaigns/")
            && (path.ends_with("/active.toml") || path.contains("/events/"))
    })
}

fn is_tracking_or_generated(path: &str) -> bool {
    path.starts_with("docs/tracking/") || path.starts_with("target/pr-review/")
}

fn file_contains_conflict_marker(path: &Path) -> bool {
    let Ok(raw) = fs::read_to_string(path) else {
        return false;
    };
    raw.lines().any(|line| {
        line.starts_with("<<<<<<< ") || line.starts_with("=======") || line.starts_with(">>>>>>> ")
    })
}

fn print_report(report: &CheckReport) {
    let status = if report.errors.is_empty() {
        if report.warnings.is_empty() { "ok" } else { "warn" }
    } else {
        "fail"
    };
    println!("lane-check status: {status}");
    println!("changed_files: {}", report.changed_files.len());
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    for error in &report.errors {
        println!("error: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_filled_pr_body() {
        let body = r#"Lane: repo-boundary
Campaign: none
Work item: LANE-CHECK-001
Orchestrator: codex
Branch: codex/repo-boundary/LANE-CHECK-001
Base main SHA: abc123
Allowed paths:
- xtask/**
Shared surfaces touched:
- xtask/**
Closeout required: no
Source promotion needed: no
Model/hardware/proof claims added: none
Claims explicitly not promoted: release, publish, signing, runtime, model, hardware, speed
Commands run:
- cargo test --locked -p xtask --no-default-features lane_check
Validation gaps:
- none
Rollback: revert this PR
Promotion or sync packet path: n/a
Source repo commit: n/a
Swarm base commit: n/a
Merge method: squash
Source impact: swarm-only
Release/publish/signing impact: none
Excluded work: none
"#;
        let mut report = CheckReport::default();
        validate_pr_body(body, None, &["xtask/src/lane_check.rs".to_string()], &mut report);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }

    #[test]
    fn rejects_template_placeholders() {
        let body = r#"Lane:
Campaign:
Work item:
Orchestrator:
Branch:
Base main SHA:
Allowed paths:
- <!-- path or none -->
Shared surfaces touched:
- <!-- shared surface or none -->
Closeout required:
Source promotion needed:
Model/hardware/proof claims added:
Claims explicitly not promoted:
Commands run:
- <!-- command or none -->
Validation gaps:
- <!-- gap or none -->
Rollback:
"#;
        let mut report = CheckReport::default();
        validate_pr_body(body, None, &[], &mut report);
        assert!(report.errors.iter().any(|error| error.contains("Lane")));
        assert!(report.errors.iter().any(|error| error.contains("Allowed paths")));
        assert!(report.errors.iter().any(|error| error.contains("Source promotion needed")));
        assert!(report.errors.iter().any(|error| error.contains("Rollback")));
    }

    #[test]
    fn generated_dashboards_require_campaign_source() {
        let mut report = CheckReport::default();
        validate_changed_files(
            &["docs/tracking/generated/global-dashboard.md".to_string()],
            &mut report,
        );
        assert!(
            report.errors.iter().any(|error| error.contains("generated dashboard changes require"))
        );
    }

    #[test]
    fn campaign_source_allows_generated_dashboard_changes() {
        let mut report = CheckReport::default();
        validate_changed_files(
            &[
                "docs/tracking/campaigns/intel-a770/events/example.toml".to_string(),
                "docs/tracking/generated/global-dashboard.md".to_string(),
            ],
            &mut report,
        );
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }
}
