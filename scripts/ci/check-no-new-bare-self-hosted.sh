#!/usr/bin/env bash
#
# Differential M3 guard: fail only on NEWLY introduced bare self-hosted routes.
#
# The repo has a large pre-existing backlog of bare `runs-on: [self-hosted,
# linux, x64]` routes (and multiline equivalents) that the staged M3 migration
# will burn down. Flipping a full-repo guard to blocking today would red-light
# every workflow at once, so this guard is differential: it compares the bare
# route count per workflow file between the PR base and head, fails only when a
# PR *adds* bare routes, and reports the pre-existing backlog as informational.
#
# A route is "safe" (not bare) when its block declares an explicit
# `group: em-ci-*` and capacity/class labels (em-ci, review-nano, cx43, ...).
#
# Usage: scripts/ci/check-no-new-bare-self-hosted.sh <base-ref-or-sha> [head]
#   base : git ref/sha to treat as the baseline (e.g. origin/main). If empty or
#          unresolvable, the script reports the working-tree backlog and passes
#          (advisory) because "newly introduced" cannot be computed.
#   head : working tree is scanned directly; this arg is accepted for symmetry.
set -euo pipefail

base="${1:-}"
wfdir=".github/workflows"

# Emit one workflow basename per bare-route occurrence found under $1.
scan() {
  local dir="$1"
  [ -d "$dir" ] || return 0

  # 1) Inline arrays: runs-on: [self-hosted, linux, x64]
  #    Anchored to the start of the line (after indentation) so the same
  #    literal appearing inside a comment or string is not matched.
  grep -RInE '^[[:space:]]*runs-on:[[:space:]]*\[[^]]*self-hosted[^]]*linux[^]]*x64[^]]*\]' "$dir" 2>/dev/null \
    | awk -F: '{print $1}' | while read -r f; do basename "$f"; done

  # 2) Multiline blocks: a `- self-hosted` list resolving to linux + x64 that
  #    declares neither an em-ci-* group nor a capacity/class label.
  while IFS=: read -r f line _; do
    window="$(sed -n "${line},$((line + 16))p" "$f")"
    if printf '%s\n' "$window" | grep -qE '^[[:space:]]*-[[:space:]]*linux[[:space:]]*$' &&
      printf '%s\n' "$window" | grep -qE '^[[:space:]]*-[[:space:]]*x64[[:space:]]*$' &&
      ! printf '%s\n' "$window" | grep -qE 'group:[[:space:]]*em-ci-' &&
      ! printf '%s\n' "$window" | grep -qE '^[[:space:]]*-[[:space:]]*(em-ci|review-nano|droid-review|llm-review|ci-nano|policy-nano|workflow-nano|rust-tiny|rust-medium|rust-large|rust-16gb|cx23|cx33|cx43|cx53|cpx42)[[:space:]]*$'; then
      basename "$f"
    fi
  done < <(grep -RInE '^[[:space:]]*-[[:space:]]*self-hosted[[:space:]]*$' "$dir" 2>/dev/null || true)
}

# "<file> <count>" per file that has at least one bare route.
counts() { sed '/^$/d' | sort | uniq -c | awk '{print $2" "$1}'; }

head_list="$(scan "$wfdir" || true)"
head_total=$(printf '%s\n' "$head_list" | sed '/^$/d' | wc -l | tr -d ' ')

if [ -z "${base:-}" ] || ! git cat-file -e "$base" 2>/dev/null; then
  echo "No usable base ref ('${base:-}'); cannot compute newly-introduced routes."
  echo "Working-tree bare self-hosted routes (backlog): ${head_total}"
  printf '%s\n' "$head_list" | counts | sort -k2 -rn || true
  echo "Advisory pass (no baseline to diff against)."
  exit 0
fi

basetmp="$(mktemp -d)"
trap 'rm -rf "$basetmp"' EXIT
if ! git archive "$base" -- "$wfdir" 2>/dev/null | tar -x -C "$basetmp" 2>/dev/null; then
  echo "Could not materialize base workflows from '${base}'; advisory pass."
  echo "Working-tree bare self-hosted routes (backlog): ${head_total}"
  exit 0
fi

base_list="$(scan "$basetmp/$wfdir" || true)"
base_total=$(printf '%s\n' "$base_list" | sed '/^$/d' | wc -l | tr -d ' ')

declare -A BASEC HEADC
while read -r f c; do [ -n "$f" ] && BASEC["$f"]="$c"; done < <(printf '%s\n' "$base_list" | counts)
while read -r f c; do [ -n "$f" ] && HEADC["$f"]="$c"; done < <(printf '%s\n' "$head_list" | counts)

new=0
for f in "${!HEADC[@]}"; do
  hc="${HEADC[$f]}"
  bc="${BASEC[$f]:-0}"
  if [ "$hc" -gt "$bc" ]; then
    echo "::error file=${wfdir}/${f}::introduces $((hc - bc)) new bare self-hosted route(s) (base=${bc}, head=${hc}). Use an explicit 'group: em-ci-*' plus capacity labels (e.g. cx43 + rust-medium)."
    new=1
  fi
done

echo "----"
echo "Pre-existing bare self-hosted routes (backlog, base=${base_total}): informational, tracked by the staged M3 migration."
if [ "$new" -eq 0 ]; then
  echo "OK: this PR introduces no new bare self-hosted routes."
fi
exit "$new"
