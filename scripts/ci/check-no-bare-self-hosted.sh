#!/usr/bin/env bash
#
# M3 runner-contract guard: no bare self-hosted routing.
#
# Fails if any workflow routes to the generic self-hosted pool without an
# explicit runner group (group: em-ci-...) and capacity labels. Once a repo is
# granted access to the em-ci-review (CX23) pool, a bare route like
# `runs-on: [self-hosted, linux, x64]` becomes eligible to land on the review
# runners, so the M3 contract requires every self-hosted route to declare an
# explicit group + capacity label.
#
# STATUS (bitnet-rs-swarm): NOT yet wired as a blocking CI gate. This repo still
# has ~100 pre-existing bare routes that the staged M3 migration (route every
# bare self-hosted job to an explicit em-ci-* group) will fix. Today the
# warn-only tripwire in `cargo run -p xtask -- lint-workflows` surfaces the same
# signal without failing CI. Flip this script to a required gate only after the
# bare-route migration lands, otherwise it red-lights every workflow at once.
#
# Usage: scripts/ci/check-no-bare-self-hosted.sh [workflows_dir]
set -euo pipefail

dir="${1:-.github/workflows}"

echo "Checking for unsafe self-hosted routing in ${dir}..."

bad=0

# 1) Bare inline arrays: runs-on: [self-hosted, linux, x64]
if grep -RInE 'runs-on:[[:space:]]*\[[^]]*self-hosted[^]]*linux[^]]*x64[^]]*\]' "$dir"; then
  echo "Bare inline self-hosted/linux/x64 route found." >&2
  bad=1
fi

# 2) Multiline blocks: a `- self-hosted` labels list that resolves to
#    linux + x64 but declares neither an em-ci-* group nor a capacity label.
while IFS=: read -r file line _; do
  window="$(sed -n "${line},$((line + 16))p" "$file")"

  if printf '%s\n' "$window" | grep -qE '^[[:space:]]*-[[:space:]]*linux[[:space:]]*$' &&
    printf '%s\n' "$window" | grep -qE '^[[:space:]]*-[[:space:]]*x64[[:space:]]*$' &&
    ! printf '%s\n' "$window" | grep -qE 'group:[[:space:]]*em-ci-' &&
    ! printf '%s\n' "$window" | grep -qE '^[[:space:]]*-[[:space:]]*(em-ci|review-nano|droid-review|llm-review|ci-nano|policy-nano|workflow-nano|rust-tiny|rust-medium|rust-large|rust-16gb|cx23|cx33|cx43|cx53|cpx42)[[:space:]]*$'; then
    echo "$file:$line: bare self-hosted block lacks group/capacity labels" >&2
    bad=1
  fi
done < <(grep -RInE '^[[:space:]]*-[[:space:]]*self-hosted[[:space:]]*$' "$dir" || true)

if [ "$bad" -eq 0 ]; then
  echo "OK: no bare self-hosted routes."
fi

exit "$bad"
