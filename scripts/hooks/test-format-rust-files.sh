#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
formatter="$repo_root/scripts/hooks/format-rust-files.sh"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

mkdir -p "$tmp_dir/src"
printf '%s\n' 'fn main(){println!("formatted");}' >"$tmp_dir/src/main.rs"
printf '%s\n' 'fn child(){println!("unchanged child");}' >"$tmp_dir/src/child.rs"
printf '%s\n' 'not rust' >"$tmp_dir/notes.txt"

"$formatter" "$tmp_dir/src/main.rs" "$tmp_dir/notes.txt" "$tmp_dir/missing.rs"

grep -Fq 'fn main() {' "$tmp_dir/src/main.rs"
grep -Fq 'fn child(){println!("unchanged child");}' "$tmp_dir/src/child.rs"
grep -Fxq 'not rust' "$tmp_dir/notes.txt"

echo "diff-scoped rustfmt hook tests passed"

hook_repo="$tmp_dir/hook-repo"
mkdir -p "$hook_repo/.githooks" "$hook_repo/scripts/hooks" \
  "$hook_repo/crates/bitnet-inference/tests"
cp "$repo_root/.githooks/pre-commit" "$hook_repo/.githooks/pre-commit"
cp "$formatter" "$hook_repo/scripts/hooks/format-rust-files.sh"
(
  cd "$hook_repo"
  git init -q
  git config user.name test
  git config user.email test@example.invalid
  printf '%s\n' 'fn main(){println!("staged");}' >staged.rs
  git add staged.rs
  .githooks/pre-commit
  git show :staged.rs | grep -Fq 'fn main() {'

  printf '%s\n' 'fn main(){println!("index");}' >partial.rs
  git add partial.rs
  printf '%s\n' 'fn main(){println!("working tree");}' >partial.rs
  if .githooks/pre-commit >hook-output.txt 2>&1; then
    echo "partially staged Rust file unexpectedly passed" >&2
    exit 1
  fi
  grep -Fq 'Cannot autofix partially staged Rust file: partial.rs' hook-output.txt
  git show :partial.rs | grep -Fq 'fn main(){println!("index");}'
)

echo "pre-commit staged autofix tests passed"
