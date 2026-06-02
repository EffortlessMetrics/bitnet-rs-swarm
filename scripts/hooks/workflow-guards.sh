#!/usr/bin/env bash
# Pre-commit workflow guards — shift-left mirror of guards-nightly.yml.
#
# Runs the cheap, deterministic ripgrep-based GitHub Actions guards locally so
# they never have to consume a CI runner under a saturated self-hosted fleet:
#   1. Actions pinned to an immutable 40-hex commit SHA (no floating tags)
#   2. No floating action refs (@v1, @main, @stable, @latest)
#   3. cargo/cross invocations use --locked
#
# Mirrors the patterns enforced in .github/workflows/guards-nightly.yml so the
# CI guard and the local guard cannot drift.
set -euo pipefail

WF_DIR=".github/workflows"
[ -d "$WF_DIR" ] || exit 0

if ! command -v rg >/dev/null 2>&1; then
  echo "⚠️  ripgrep (rg) not found; skipping workflow guards (CI will still enforce them)"
  exit 0
fi

rc=0

# 1. Actions pinned to a 40-hex SHA (ignore local ./ actions and guards.yml self-refs)
if rg --pcre2 -n --glob '!guards.yml' \
     '^\s*uses:\s*(?!\./)[^ @]+/[^ @]+@(?![0-9a-f]{40}\b)' "$WF_DIR"; then
  echo "❌ Non-immutable action pin detected (use a 40-hex commit SHA)"
  rc=1
else
  echo "✅ Actions pinned to 40-hex SHA"
fi

# 2. No floating action refs
if rg -n --glob '!guards.yml' 'uses:.*@v[0-9]|uses:.*@(main|stable|latest)' "$WF_DIR"; then
  echo "❌ Floating action ref detected (@vN/@main/@stable/@latest)"
  rc=1
else
  echo "✅ No floating action refs"
fi

# 3. cargo/cross commands use --locked
violations="$(rg -n --glob '*.yml' --glob '!guards.yml' \
  '\b(cargo|cross)\s+(build|test|run|bench|clippy)\b' "$WF_DIR" \
  | grep -v -- '--locked' || true)"
if [ -n "$violations" ]; then
  echo "❌ Missing --locked in workflow cargo/cross command(s):"
  echo "$violations"
  rc=1
else
  echo "✅ All cargo/cross commands use --locked"
fi

exit "$rc"
