use anyhow::{Result, anyhow};
use std::{collections::HashSet, path::Path};
use walkdir::WalkDir;

pub fn lint_workflows() -> Result<()> {
    let workflows_dir = Path::new(".github/workflows");

    if !workflows_dir.exists() {
        return Err(anyhow!(".github/workflows directory not found"));
    }

    let mut files: Vec<_> = WalkDir::new(workflows_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "yml"))
        .map(|e| e.path().to_path_buf())
        .collect();

    files.sort();

    let mut failed = false;

    for path in files {
        match check_file(&path) {
            Ok(()) => println!("✓ {}", path.display()),
            Err(e) => {
                eprintln!("❌ {}: {}", path.display(), e);
                failed = true;
            }
        }

        // Warn-only tripwire (does not fail the lint): surface jobs eligible for
        // the 8 GB cx33 tiny runners during the routing stabilization pass.
        if let Ok(content) = std::fs::read_to_string(&path) {
            for (line, value) in unpinned_self_hosted_warnings(&content) {
                println!(
                    "::warning file={},line={}::runs-on {} targets the generic self-hosted pool with no capacity label, so it stays eligible for the 8 GB cx33 tiny runners. Add em-ci + a capacity label (rust-tiny/rust-small/rust-medium/rust-large) or pin a host class (cx43/cx53).",
                    path.display(),
                    line,
                    value,
                );
            }
        }
    }

    if failed {
        return Err(anyhow!("Some workflows have validation errors"));
    }

    println!("\n✓ All workflows valid (no duplicate keys)");
    Ok(())
}

/// Warn-only bridge for the cx33 routing contract.
///
/// Returns `(line_number, runs_on_value)` for every inline `runs-on: [...]` that
/// targets the generic `self-hosted` + `linux` + `x64` pool without any label
/// that would exclude the 8 GB cx33 tiny runners. A capacity label
/// (`rust-tiny`/`rust-small`/`rust-medium`/`rust-large`), an explicit host class
/// (`cx33`/`cx43`/`cx53`/`cpx42`), or a specialized-hardware label
/// (`gpu`/`intel-gpu`/`rocm`/`a770`/`metal`/`cuda`) all make a job ineligible for
/// the generic pool, so those are treated as safe and not flagged.
///
/// This is intentionally non-failing: the full migration of pre-existing bare
/// self-hosted jobs is tracked separately. The annotations are a tripwire only.
fn unpinned_self_hosted_warnings(content: &str) -> Vec<(usize, String)> {
    const SAFE_LABELS: &[&str] = &[
        "rust-tiny",
        "rust-small",
        "rust-medium",
        "rust-large",
        "cx33",
        "cx43",
        "cx53",
        "cpx42",
        "gpu",
        "intel-gpu",
        "rocm",
        "a770",
        "metal",
        "cuda",
    ];

    let mut warnings = Vec::new();

    for (line_index, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        let Some(rest) = line.strip_prefix("runs-on:") else {
            continue;
        };
        let value = rest.trim();

        // Only the inline array form carries the bare generic pool; the block
        // (`group:` / `labels:`) form in this repo always pins capacity labels.
        let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) else {
            continue;
        };

        let labels: Vec<String> = inner
            .split(',')
            .map(|s| s.trim().trim_matches(['"', '\'']).to_ascii_lowercase())
            .collect();

        let is_generic_linux_pool = labels.iter().any(|l| l == "self-hosted")
            && labels.iter().any(|l| l == "linux")
            && labels.iter().any(|l| l == "x64");
        if !is_generic_linux_pool {
            continue;
        }

        let excluded_from_cx33 = labels.iter().any(|l| SAFE_LABELS.contains(&l.as_str()));
        if !excluded_from_cx33 {
            warnings.push((line_index + 1, value.to_string()));
        }
    }

    warnings
}

fn check_file(path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    check_duplicate_keys(&content)?;
    let _: serde_yaml::Value =
        serde_yaml::from_str(&content).map_err(|e| anyhow!("YAML parse error: {}", e))?;
    Ok(())
}

fn check_duplicate_keys(content: &str) -> Result<()> {
    let mut frames = Vec::<MappingFrame>::new();
    let mut block_scalar_indent = None;

    for (line_index, raw_line) in content.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim_end_matches('\r');
        let Some(indent) = line.find(|ch| ch != ' ') else {
            continue;
        };
        let trimmed = &line[indent..];

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(block_indent) = block_scalar_indent {
            if indent > block_indent {
                continue;
            }
            block_scalar_indent = None;
        }

        let (key_indent, key_source, starts_sequence_item) =
            if let Some(rest) = trimmed.strip_prefix("- ") {
                (indent + 2, rest, true)
            } else {
                (indent, trimmed, false)
            };

        let Some((key, value)) = parse_mapping_key(key_source) else {
            continue;
        };

        frames.retain(|frame| frame.indent <= key_indent);
        if starts_sequence_item {
            frames.retain(|frame| frame.indent < key_indent);
        }

        let frame_index = match frames.iter().position(|frame| frame.indent == key_indent) {
            Some(index) => index,
            None => {
                frames.push(MappingFrame::new(key_indent));
                frames.len() - 1
            }
        };

        if !frames[frame_index].keys.insert(key.to_string()) {
            return Err(anyhow!("duplicate key '{}' at line {}", key, line_number));
        }

        if value.trim_start().starts_with(['|', '>']) {
            block_scalar_indent = Some(key_indent);
        }
    }

    Ok(())
}

fn parse_mapping_key(line: &str) -> Option<(&str, &str)> {
    let colon_index = if line.starts_with('"') || line.starts_with('\'') {
        quoted_key_end(line).and_then(|end| {
            line[end..].char_indices().find_map(|(offset, ch)| (ch == ':').then_some(end + offset))
        })?
    } else {
        line.find(':')?
    };

    let after_colon = &line[colon_index + 1..];
    if !after_colon.is_empty()
        && !after_colon.starts_with(char::is_whitespace)
        && !after_colon.starts_with(['|', '>', '#'])
    {
        return None;
    }

    let key = line[..colon_index].trim().trim_matches(['"', '\'']);
    (!key.is_empty()).then_some((key, after_colon))
}

fn quoted_key_end(line: &str) -> Option<usize> {
    let mut chars = line.char_indices();
    let (_, quote) = chars.next()?;
    let mut escaped = false;

    for (index, ch) in chars {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == '"' && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == quote {
            return Some(index + ch.len_utf8());
        }
    }

    None
}

struct MappingFrame {
    indent: usize,
    keys: HashSet<String>,
}

impl MappingFrame {
    fn new(indent: usize) -> Self {
        Self { indent, keys: HashSet::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::check_duplicate_keys;

    #[test]
    fn rejects_duplicate_top_level_keys() {
        let err = check_duplicate_keys("name: CI\non: push\non: pull_request\n").unwrap_err();
        assert!(err.to_string().contains("duplicate key 'on' at line 3"));
    }

    #[test]
    fn rejects_duplicate_nested_keys() {
        let err = check_duplicate_keys(
            "jobs:\n  build:\n    runs-on: ubuntu-22.04\n    runs-on: ubuntu-24.04\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate key 'runs-on' at line 4"));
    }

    #[test]
    fn allows_same_key_in_separate_sequence_items() {
        check_duplicate_keys("steps:\n  - name: Checkout\n    uses: actions/checkout@v4\n  - name: Test\n    run: cargo test\n")
            .unwrap();
    }

    #[test]
    fn rejects_duplicate_key_in_same_sequence_item() {
        let err =
            check_duplicate_keys("steps:\n  - name: Checkout\n    name: Duplicate\n").unwrap_err();
        assert!(err.to_string().contains("duplicate key 'name' at line 3"));
    }

    #[test]
    fn ignores_mapping_like_lines_inside_block_scalars() {
        check_duplicate_keys("jobs:\n  build:\n    steps:\n      - run: |\n          echo 'on: push'\n          echo 'on: pull_request'\n        shell: bash\n")
            .unwrap();
    }

    use super::unpinned_self_hosted_warnings;

    #[test]
    fn warns_on_bare_generic_self_hosted_pool() {
        let w = unpinned_self_hosted_warnings(
            "jobs:\n  build:\n    runs-on: [self-hosted, linux, x64]\n",
        );
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, 3);
    }

    #[test]
    fn does_not_warn_when_capacity_label_present() {
        assert!(
            unpinned_self_hosted_warnings(
                "    runs-on: [self-hosted, linux, x64, em-ci, rust-medium, trusted-pr]\n",
            )
            .is_empty()
        );
    }

    #[test]
    fn does_not_warn_on_specialized_hardware_runners() {
        // intel-gpu, gpu, and rocm runners are not the generic cx33-eligible pool.
        assert!(
            unpinned_self_hosted_warnings(
                "    runs-on: [self-hosted, intel-gpu]\n    runs-on: [self-hosted, linux, x64, gpu]\n    runs-on: [self-hosted, linux, rocm]\n",
            )
            .is_empty()
        );
    }

    #[test]
    fn ignores_github_hosted_and_block_form_runners() {
        // ubuntu-latest is not self-hosted; the block form pins capacity itself.
        assert!(
            unpinned_self_hosted_warnings("    runs-on: ubuntu-latest\n    runs-on:\n").is_empty()
        );
    }

    #[test]
    fn warns_when_em_ci_present_but_capacity_label_missing() {
        // em-ci alone does not exclude cx33 (cx33 carries em-ci), so still flag.
        let w = unpinned_self_hosted_warnings(
            "    runs-on: [self-hosted, linux, x64, em-ci, trusted-pr]\n",
        );
        assert_eq!(w.len(), 1);
    }
}
